from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing import TypedDict
import subprocess
from openai import OpenAI
import textwrap
from typing_extensions import Annotated
from langchain.chat_models import init_chat_model
import operator
import base64

llm = init_chat_model(model="gpt-4o-mini")


class State(TypedDict):
    video_file: str
    audio_file: str
    transciption: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]
    thumbnail_sketches: Annotated[list[str], operator.add]
    final_summary: str


def extract_audio(state: State) -> State:
    output_file = state["video_file"].replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-i",
        state["video_file"],
        "-filter:a",
        "atempo=2.0",
        output_file,
    ]
    subprocess.run(command)
    return {
        "audio_file": output_file,
    }


def transcribe_audio(state: State) -> State:
    client = OpenAI()
    with open(state["audio_file"], "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en",
        )
        return {"transciption": transcription}


def dispatch_summarisers(state: State):
    transcription = state["transciption"]
    chunks = {}
    for i, chunk in enumerate(textwrap.wrap(transcription, 500)):
        chunks.append(
            {
                "id": i + 1,
                "chunk": chunk,
            }
        )
    return [Send("summarise_chunk", chunk) for chunk in chunks]


def summarise_chunk(chunk):
    chunk_id = chunk["id"]
    chunk_text = chunk["chunk"]
    response = llm.invoke(
        f"""
        You are a helpful assistant that summarises chunks of text.
        You will be given a chunk of text and you need to summarise it.
        The chunk of text is:
        {chunk_text}
        """
    )
    summary = f"Summary of chunk {chunk_id}: {response.content}"
    return {
        "summaries": [summary],
    }


def mega_summarise(state: State):
    all_summaries = "\n".join(state["summaries"])
    prompt = f""" 
        You are a helpful assistant that summarises chunks of text.
        You will be given a list of summaries of chunks of text.
        The summaries are:
        {all_summaries}
        Create a comprehensive summary of the entire video based on the summaries.
        """
    response = llm.invoke(prompt)
    return {
        "final_summary": response.content,
    }


def dispatch_artists(state: State):
    return [
        Send("generate_thumbnail", {"id": i, "summary": state["final_summary"]})
        for i in [1, 2, 3, 4, 5]
    ]


def generate_thumbnail(args):
    id = args["id"]
    summary = args["summary"]
    prompt = f"""
    Based on video summary , create a prompt for a youtube thumbnail.
    The summary is:
    {summary}
    The prompt should be in the following format:
    <main visual elements>
    <color palette>
    <font>
    <text>
    the thumbnail should be a youtube thumbnail
    """
    response = llm.invoke(prompt)
    thumbnail_prompt = response.content.split

    client = OpenAI()

    result = client.images.generate(
        model="gpt-image-1",
        prompt=thumbnail_prompt,
        quality="low",
        moderation="low",
        size="auto",
    )
    image_bytes = base64.b64encode(result.data[0].b64_json)

    filename = f"thumbnail_{id}.jpg"
    with open(filename, "wb") as file:
        file.write(image_bytes)
    return {
        "thumbnail_prompts": [thumbnail_prompt],
        "thumbnail_sketches": [filename],
    }


graph_builder = StateGraph(State)
graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarise_chunk", summarise_chunk)
graph_builder.add_node("mega_summarise", mega_summarise)

graph_builder.add_node("generate_thumbnail", generate_thumbnail)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarisers, ["summarise_chunk"]
)
graph_builder.add_edge("summarise_chunk", mega_summarise)
graph_builder.add_conditional_edges(
    "mega_summarise", dispatch_artists, ["generate_thumbnail"]
)
graph_builder.add_edge("generate_thumbnail", END)
graph = graph_builder.compile()
