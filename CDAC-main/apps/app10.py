# app.py
import os
import subprocess
import tempfile
import streamlit as st

st.set_page_config("Speaker Diarization", layout="wide")
st.title("🎙 Speaker Diarization + Cleaning + DER")

UPLOAD_DIR = "uploads"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MANUAL_RTTM_DIR = "manual_rttm"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

uploaded = st.file_uploader("Upload WAV audio", type=["wav"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=UPLOAD_DIR) as tmp:
        tmp.write(uploaded.read())
        raw_audio = tmp.name

    file_id = os.path.splitext(uploaded.name)[0]
    cleaned_audio = os.path.join(INPUT_DIR, file_id + "_clean.wav")

    st.audio(raw_audio)

    if st.button("🧹 Clean Audio"):
        res = subprocess.run(
            ["python", "process.py", raw_audio, cleaned_audio],
            capture_output=True,
            text=True
        )
        st.code(res.stdout)
        st.audio(cleaned_audio)

    if os.path.exists(cleaned_audio) and st.button("🚀 Run Diarization"):
        manual_rttm = os.path.join(MANUAL_RTTM_DIR, file_id + ".rttm")
        if not os.path.exists(manual_rttm):
            manual_rttm = None

        res = subprocess.run(
            [
                "python", "diarize.py",
                "--audio", cleaned_audio,
                "--file_id", file_id,
                "--output_dir", OUTPUT_DIR
            ] + (["--manual_rttm", manual_rttm] if manual_rttm else []),
            capture_output=True,
            text=True
        )

        st.subheader("📜 Logs")
        st.code(res.stdout)

        speaker_dir = os.path.join(OUTPUT_DIR, "speaker_outputs")
        if os.path.exists(speaker_dir):
            st.subheader("🔊 Speaker Segments")
            for f in sorted(os.listdir(speaker_dir)):
                if f.endswith(".wav"):
                    st.audio(os.path.join(speaker_dir, f))
