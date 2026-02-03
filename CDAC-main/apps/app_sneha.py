import streamlit as st
import os
import torch
import torchaudio
import librosa
import soundfile as sf
import sounddevice as sd

from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from silero_vad import load_silero_vad, get_speech_timestamps

# ================= CONFIG =================
SR = 16000
BASE_DIR = "output"
os.makedirs(BASE_DIR, exist_ok=True)
INPUT_AUDIO = os.path.join(BASE_DIR, "input.wav")

# ================= PAGE =================
st.set_page_config(
    page_title="Speaker Diarization Pipeline",
    page_icon="🎙",
    layout="wide"
)

st.title("🎙 Speaker Diarization Pipeline")
st.caption("Mic / WAV → VAD → Diarization → RTTM → DER")

# ================= SIDEBAR =================
st.sidebar.title("⚙ Controls")

record_seconds = st.sidebar.slider(
    "Recording duration (seconds)", 5, 30, 10
)

vad_threshold = st.sidebar.slider(
    "VAD threshold", 0.1, 0.9, 0.5
)

# ================= LOAD MODELS (CACHED) =================
@st.cache_resource
def load_diarization_pipeline():
    return Pipeline.from_pretrained("pyannote/speaker-diarization")

@st.cache_resource
def load_vad_model():
    return load_silero_vad()

# ================= INPUT TABS =================
tab1, tab2 = st.tabs(["🎤 Record Audio", "📂 Upload WAV"])

with tab1:
    st.subheader("Record from Microphone")

    if st.button("Start Recording"):
        st.info("Recording...")

        audio = sd.rec(
            int(record_seconds * SR),
            samplerate=SR,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        sf.write(INPUT_AUDIO, audio, SR)
        st.success("Recording saved")
        st.audio(INPUT_AUDIO)

with tab2:
    st.subheader("Upload WAV File")

    uploaded = st.file_uploader("Upload WAV", type=["wav"])
    if uploaded is not None:
        audio_np, _ = librosa.load(uploaded, sr=SR, mono=True)
        sf.write(INPUT_AUDIO, audio_np, SR)
        st.success("File uploaded")
        st.audio(INPUT_AUDIO)

# ================= AUDIO INFO =================
st.divider()

if os.path.exists(INPUT_AUDIO):
    audio_np, _ = librosa.load(INPUT_AUDIO, sr=SR)
    st.metric("Audio duration (sec)", f"{len(audio_np)/SR:.2f}")
else:
    st.info("No audio available yet")

# ================= RUN PIPELINE =================
st.divider()
st.subheader("Run Pipeline")

if st.button("▶ Run Full Pipeline"):
    if not os.path.exists(INPUT_AUDIO):
        st.error("Please record or upload audio first")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    file_id = "session"

    # -------- STEP 1: LOAD AUDIO --------
    status.info("Loading audio...")
    wav, sr = torchaudio.load(INPUT_AUDIO)
    if wav.dim() > 1:
        wav = wav.mean(dim=0)

    progress.progress(10)

    # -------- STEP 2: VAD --------
    status.info("Running VAD...")
    vad_model = load_vad_model()

    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sr,
        threshold=vad_threshold
    )

    if len(speech_timestamps) == 0:
        st.error("No speech detected. Lower VAD threshold.")
        st.stop()

    vad_audio = torch.cat([
        wav[s["start"]:s["end"]] for s in speech_timestamps
    ])

    vad_path = os.path.join(BASE_DIR, "vad.wav")
    sf.write(vad_path, vad_audio.numpy(), sr)

    progress.progress(35)

    # -------- STEP 3: DIARIZATION --------
    status.info("Running diarization...")
    pipeline = load_diarization_pipeline()
    diarization = pipeline(vad_path)

    progress.progress(65)

    # -------- STEP 4: RTTM --------
    status.info("Generating RTTM...")
    hyp_rttm = os.path.join(BASE_DIR, "hypothesis.rttm")

    rttm_lines = []
    speaker_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        rttm_lines.append(
            f"SPEAKER {file_id} 1 {turn.start:.3f} "
            f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )
        speaker_segments.append((speaker, turn.start, turn.end))

    with open(hyp_rttm, "w") as f:
        f.writelines(rttm_lines)

    progress.progress(85)

    # -------- STEP 5: DER (DEMO) --------
    status.info("Computing DER (demo reference)...")

    audio_np, _ = librosa.load(INPUT_AUDIO, sr=SR)
    ref_rttm = os.path.join(BASE_DIR, "reference.rttm")

    with open(ref_rttm, "w") as f:
        f.write(
            f"SPEAKER {file_id} 1 0.000 {len(audio_np)/SR:.3f} "
            f"<NA> <NA> speaker_0 <NA> <NA>\n"
        )

    reference = load_rttm(ref_rttm)[file_id]
    hypothesis = load_rttm(hyp_rttm)[file_id]

    der = DiarizationErrorRate()(reference, hypothesis)

    progress.progress(100)
    status.success("Pipeline completed")

    # ================= RESULTS =================
    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detected speakers", len(set(s for s, _, _ in speaker_segments)))
        st.metric("DER (demo)", f"{der:.2%}")

    with col2:
        st.warning(
            "DER shown is illustrative only.\n"
            "Use manually annotated RTTM for real evaluation."
        )

    st.subheader("Speaker Segments")
    for s, stt, end in speaker_segments:
        st.write(f"**{s}** : {stt:.2f}s → {end:.2f}s")

    st.subheader("Generated RTTM")
    st.code("".join(rttm_lines))
