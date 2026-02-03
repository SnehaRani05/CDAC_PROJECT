import streamlit as st
import os
import subprocess
import sys

st.set_page_config(page_title="Speaker Diarization", layout="centered")
st.title("🎙️ Speaker Diarization UI")

UPLOAD_DIR = "test_audio"
SPEAKER_DIR = "outputs/speakers"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    audio_path = os.path.join(UPLOAD_DIR, "audio.wav")

    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(audio_path)

    if st.button("🚀 Run Diarization"):
        with st.spinner("Running diarization..."):
            subprocess.run(
                [sys.executable, "diarize.py"],
                check=True,
                env=os.environ
            )

            subprocess.run(
                [sys.executable, "listen_verify.py"],
                check=True,
                env=os.environ
            )

        st.success("Diarization complete!")

        for file in os.listdir(SPEAKER_DIR):
            if file.endswith(".wav"):
                path = os.path.join(SPEAKER_DIR, file)
                st.audio(path)
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download {file}",
                        data=f,
                        file_name=file,
                        mime="audio/wav"
                    )
