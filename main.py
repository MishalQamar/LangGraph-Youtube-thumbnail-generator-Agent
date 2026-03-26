import base64
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="YouTube Thumbnail Generator", page_icon="🎬", layout="wide")
st.title("🎬 YouTube Thumbnail Generator")
st.caption("Upload a video, generate thumbnail options, choose one, then refine to final HD.")


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def normalize_api_key(raw_value: str) -> str:
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return ""
    key = cleaned.splitlines()[0].strip().strip('"').strip("'")
    if key.startswith("OPENAI_API_KEY="):
        key = key.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def is_valid_api_key_format(key: str) -> bool:
    return key.startswith("sk-") and " " not in key and len(key) > 20


def extract_audio_with_ffmpeg(video_path: str) -> str:
    audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(audio_fd)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    return audio_path


def transcribe_audio(client: OpenAI, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
            language="en",
        )


def summarize_text(client: OpenAI, text: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=(
            "You are a professional content summarization assistant. "
            "Summarize the following transcript into a concise, actionable summary "
            "for generating YouTube thumbnail ideas:\n\n"
            f"{text}"
        ),
    )
    return response.output_text.strip()


def create_thumbnail_prompts(client: OpenAI, context_text: str, count: int) -> list[str]:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=(
            "You are a professional YouTube thumbnail strategist.\n"
            f"Create {count} distinct high-converting thumbnail prompts.\n"
            "Return plain text only as a numbered list.\n"
            "Include: subject, composition, text overlay, typography style, color palette, "
            "lighting, mood, and visual hierarchy.\n"
            "Create fresh concepts from the provided video context.\n\n"
            f"Context:\n{context_text}"
        ),
    )
    raw = response.output_text.strip()
    prompts = []
    for line in raw.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate[0].isdigit() and "." in candidate[:4]:
            candidate = candidate.split(".", 1)[1].strip()
        if candidate:
            prompts.append(candidate)
    if not prompts:
        prompts = [raw]
    return prompts[:count]


def generate_image(client: OpenAI, prompt: str, quality: str = "medium") -> bytes:
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        quality=quality,
        size="1536x1024",
    )
    b64_image = result.data[0].b64_json
    return base64.b64decode(b64_image)


def refine_prompt(client: OpenAI, chosen_prompt: str, feedback: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=(
            "You are a professional thumbnail prompt engineer.\n"
            "Refine this draft prompt using user feedback and output one final production-grade prompt.\n\n"
            f"Draft prompt:\n{chosen_prompt}\n\n"
            f"User feedback:\n{feedback}"
        ),
    )
    return response.output_text.strip()


if "thumbnail_candidates" not in st.session_state:
    st.session_state.thumbnail_candidates: list[dict[str, Any]] = []
if "final_image" not in st.session_state:
    st.session_state.final_image = None
if "final_prompt" not in st.session_state:
    st.session_state.final_prompt = ""


with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    api_key = normalize_api_key(api_key_input)
    num_options = st.slider("Thumbnail options", min_value=2, max_value=4, value=2)
    st.caption("Paste only the raw key (starts with `sk-`).")


uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "webm"])

extra_context = st.text_area(
    "Optional context",
    placeholder="Describe target audience, title text, style direction, colors, etc.",
)

if st.button("Generate thumbnail options", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please add your OpenAI API key in the sidebar.")
        st.stop()
    if not is_valid_api_key_format(api_key):
        st.error("Invalid API key format. Paste only the raw key starting with `sk-`.")
        st.stop()
    if uploaded_video is None:
        st.error("Please upload a video file first.")
        st.stop()

    client = get_client(api_key)

    with st.spinner("Generating options..."):
        progress = st.progress(0, text="Preparing video...")
        context = extra_context.strip()
        temp_files: list[str] = []
        try:
            video_ext = Path(uploaded_video.name).suffix or ".mp4"
            fd, video_path = tempfile.mkstemp(suffix=video_ext)
            os.close(fd)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            temp_files.append(video_path)
            progress.progress(10, text="Extracting audio...")

            audio_path = extract_audio_with_ffmpeg(video_path)
            temp_files.append(audio_path)
            progress.progress(30, text="Transcribing audio...")

            transcript = transcribe_audio(client, audio_path)
            progress.progress(55, text="Summarizing transcript...")
            summary = summarize_text(client, transcript)
            context = f"{summary}\n\nAdditional user context:\n{context}" if context else summary
            progress.progress(70, text="Creating prompt ideas...")

            candidates: list[dict[str, Any]] = []
            prompt_candidates = create_thumbnail_prompts(client, context, num_options)
            for idx, prompt in enumerate(prompt_candidates, start=1):
                progress_value = 70 + int((idx / max(len(prompt_candidates), 1)) * 28)
                progress.progress(progress_value, text=f"Generating image {idx}/{len(prompt_candidates)}...")
                image_bytes = generate_image(client, prompt=prompt, quality="low")
                candidates.append({"prompt": prompt, "image_bytes": image_bytes})

            st.session_state.thumbnail_candidates = candidates
            st.session_state.final_image = None
            st.session_state.final_prompt = ""
            progress.progress(100, text="Done.")
            st.success("Generated thumbnail options. Choose one below.")
        except Exception as exc:
            st.error(f"Failed to generate thumbnails: {exc}")
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.remove(path)


candidates = st.session_state.thumbnail_candidates
if candidates:
    st.subheader("Choose a thumbnail option")
    cols = st.columns(2)
    for index, candidate in enumerate(candidates):
        with cols[index % 2]:
            st.image(candidate["image_bytes"], use_container_width=True)
            with st.expander(f"Prompt {index + 1}"):
                st.write(candidate["prompt"])

    selected_idx = st.selectbox(
        "Select preferred option",
        options=list(range(len(candidates))),
        format_func=lambda x: f"Option {x + 1}",
    )
    user_feedback = st.text_area(
        "Feedback for final version",
        placeholder="Example: Make text bigger, add stronger contrast, use red + black palette...",
    )

    if st.button("Generate final HD thumbnail", use_container_width=True):
        if not api_key:
            st.error("Please add your OpenAI API key in the sidebar.")
            st.stop()
        if not is_valid_api_key_format(api_key):
            st.error("Invalid API key format. Paste only the raw key starting with `sk-`.")
            st.stop()
        client = get_client(api_key)
        chosen_prompt = candidates[selected_idx]["prompt"]

        with st.spinner("Refining and generating HD thumbnail..."):
            try:
                final_prompt = refine_prompt(client, chosen_prompt, user_feedback.strip())
                final_image = generate_image(client, final_prompt, quality="high")
                st.session_state.final_prompt = final_prompt
                st.session_state.final_image = final_image
                st.success("Final HD thumbnail generated.")
            except Exception as exc:
                st.error(f"Failed to generate final HD thumbnail: {exc}")


if st.session_state.final_image:
    st.subheader("Final HD thumbnail")
    st.image(st.session_state.final_image, use_container_width=True)
    with st.expander("Final prompt used"):
        st.write(st.session_state.final_prompt)
    st.download_button(
        "Download final thumbnail",
        data=st.session_state.final_image,
        file_name="hd_thumbnail.png",
        mime="image/png",
        use_container_width=True,
    )