import os
import streamlit as st
import librosa
import soundfile as sf
import sounddevice as sd

from pyannote.audio import Pipeline

# ================= CONFIG =================
SR = 16000
BASE_DIR = "output"
AUDIO_PATH = os.path.join(BASE_DIR, "input.wav")
RTTM_PATH = os.path.join(BASE_DIR, "output.rttm")
SPEAKER_DIR = os.path.join(BASE_DIR, "speaker_audio")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)

# ================= UTILS =================
def clear_previous_outputs():
    # Clear old speaker audio
    if os.path.exists(SPEAKER_DIR):
        for f in os.listdir(SPEAKER_DIR):
            os.remove(os.path.join(SPEAKER_DIR, f))

    # Clear old RTTM
    if os.path.exists(RTTM_PATH):
        os.remove(RTTM_PATH)

# ================= PAGE =================
st.set_page_config(
    page_title="Speaker Diarization Studio",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Speaker Diarization Studio (Clean Run Mode)")
st.caption("Mic / WAV → Diarization → RTTM → Speaker Audio")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

record_sec = st.sidebar.slider(
    "Recording duration (seconds)",
    5, 30, 10
)

expected_speakers = st.sidebar.slider(
    "Expected number of speakers",
    1, 6, 2,
    help="Model uses ±1 range to avoid missing speakers"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_pipeline():
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=True
    )

# ================= INPUT =================
tab1, tab2 = st.tabs(["🎤 Live Recording", "📂 Upload WAV"])

with tab1:
    if st.button("🎙 Start Recording"):
        st.info("Recording...")
        audio = sd.rec(
            int(record_sec * SR),
            samplerate=SR,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        sf.write(AUDIO_PATH, audio, SR)
        st.success("Recording saved")
        st.audio(AUDIO_PATH)

with tab2:
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded:
        audio, _ = librosa.load(uploaded, sr=SR, mono=True)
        sf.write(AUDIO_PATH, audio, SR)
        st.success("File uploaded")
        st.audio(AUDIO_PATH)

# ================= RUN DIARIZATION =================
st.divider()
st.subheader("▶ Run Speaker Diarization")

if st.button("🚀 Run Diarization"):
    if not os.path.exists(AUDIO_PATH):
        st.error("Please upload or record audio first")
        st.stop()

    status = st.empty()
    progress = st.progress(0)

    # ---------- CLEAN OLD OUTPUTS ----------
    clear_previous_outputs()

    # ---------- LOAD AUDIO ----------
    status.info("Loading audio...")
    audio_np, _ = librosa.load(AUDIO_PATH, sr=SR)
    duration = len(audio_np) / SR
    progress.progress(20)

    # ---------- DIARIZATION ----------
    status.info("Running diarization (speaker range mode)...")
    pipeline = load_pipeline()

    diarization = pipeline(
        AUDIO_PATH,
        min_speakers=max(1, expected_speakers - 1),
        max_speakers=expected_speakers + 1
    )
    progress.progress(60)

    # ---------- SAVE RESULTS ----------
    status.info("Saving speaker segments...")
    speakers = set()
    rttm_lines = []
    segment_id = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)

        start = int(turn.start * SR)
        end = int(turn.end * SR)

        sf.write(
            os.path.join(SPEAKER_DIR, f"{speaker}_{segment_id}.wav"),
            audio_np[start:end],
            SR
        )

        rttm_lines.append(
            f"SPEAKER session 1 {turn.start:.3f} "
            f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )
        segment_id += 1

    with open(RTTM_PATH, "w") as f:
        f.writelines(rttm_lines)

    progress.progress(100)
    status.success("Diarization completed successfully ✅")

    # ================= RESULTS =================
    st.divider()
    st.subheader("📊 Results")

    st.metric("Audio Duration (sec)", f"{duration:.2f}")
    st.metric("Speakers Detected", len(speakers))

    st.subheader("🧑‍🤝‍🧑 Speaker Segments (RTTM)")
    for line in rttm_lines:
        st.code(line.strip())

    st.subheader("🔊 Speaker-wise Audio")
    for file in sorted(os.listdir(SPEAKER_DIR)):
        st.write(file)
        st.audio(os.path.join(SPEAKER_DIR, file))

    st.subheader("📄 RTTM File Content")
    st.code("".join(rttm_lines))
