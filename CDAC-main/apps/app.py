import streamlit as st
import os
import subprocess
import sys
import shutil

st.set_page_config(page_title="Speaker Diarization + DER", layout="centered")
st.title("🎙️ Speaker Diarization with DER")

UPLOAD_DIR = "test_audio"
OUTPUT_DIR = "outputs"
SPEAKER_DIR = "outputs/speakers"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)

# ==========================
# Upload audio
# ==========================
audio_file = st.file_uploader("Upload WAV file", type=["wav"])
audio_name = None

if audio_file:
    audio_name = audio_file.name
    temp_path = os.path.join(UPLOAD_DIR, audio_name)

    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Normalize filename for backend
    final_audio_path = os.path.join(UPLOAD_DIR, "audio.wav")
    shutil.move(temp_path, final_audio_path)

    st.audio(final_audio_path)

# ==========================
# Run pipeline
# ==========================
if st.button("🚀 Run Diarization") and audio_file:
    with st.spinner("Running diarization, DER & segmentation..."):

        # Clean old speaker segments
        for f in os.listdir(SPEAKER_DIR):
            os.remove(os.path.join(SPEAKER_DIR, f))

        # 1️⃣ Diarization → RTTM
        subprocess.run(
            [sys.executable, "diarize.py"],
            check=True,
            env=os.environ
        )

        # 2️⃣ DER calculation (backend RTTM)
        result = subprocess.run(
            [sys.executable, "der.py", audio_name],
            capture_output=True,
            text=True,
            check=True
        )

        der = float(result.stdout.strip())

        # 3️⃣ Speaker segmentation
        subprocess.run(
            [sys.executable, "listen_verify.py"],
            check=True
        )

    st.success("✅ Diarization pipeline completed")

    # ==========================
    # Show DER
    # ==========================
    st.metric("📊 Diarization Error Rate (DER)", f"{der:.2f}%")

    # ==========================
    # Speaker Segments
    # ==========================
    st.subheader("🗣️ Speaker Segments")

    speaker_files = sorted(os.listdir(SPEAKER_DIR))
    if not speaker_files:
        st.warning("No speaker segments found.")
    else:
        for file in speaker_files:
            if file.endswith(".wav"):
                path = os.path.join(SPEAKER_DIR, file)
                st.audio(path)
                with open(path, "rb") as f:
                    st.download_button(
                        f"Download {file}",
                        f,
                        file_name=file,
                        mime="audio/wav"
                    )

    # ==========================
    # RTTM download
    # ==========================
    st.subheader("📄 RTTM Output")
    with open("outputs/output.rttm", "rb") as f:
        st.download_button(
            "Download Hypothesis RTTM",
            f,
            file_name="output.rttm",
            mime="text/plain"
        )
