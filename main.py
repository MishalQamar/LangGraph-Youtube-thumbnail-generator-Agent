from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing import TypedDict
import subprocess
from openai import OpenAI
import textwrap
from typing_extensions import Annotated
from langchain.chat_models import init_chat_model
import operator

llm = init_chat_model(model="gpt-4o-mini")


class State(TypedDict):
    video_file: str
    audio_file: str
    transciption: str
    summaries: Annotated[list[str], operator.add]


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


graph_builder = StateGraph(State)
graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarise_chunk", summarise_chunk)
graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarisers, ["summarise_chunk"]
)
graph_builder.add_edge("summarise_chunk", END)

graph = graph_builder.compile()
