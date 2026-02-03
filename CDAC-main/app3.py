# # import streamlit as st
# # import os
# # import torch
# # import torchaudio
# # import soundfile as sf
# # import sounddevice as sd

# # from pyannote.audio import Pipeline
# # from pyannote.metrics.diarization import DiarizationErrorRate
# # from pyannote.database.util import load_rttm


# # # ================= CONFIG =================

# # BASE_DIR = "output"
# # SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")
# # INPUT_AUDIO = os.path.join(BASE_DIR, "input.wav")

# # os.makedirs(BASE_DIR, exist_ok=True)
# # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # SR = 16000


# # # ================= PAGE =================

# # st.set_page_config(layout="wide")
# # st.title("High Accuracy Speaker Diarization")
# # st.caption("No external VAD • Neural segmentation • Speaker playback")


# # # ================= SIDEBAR =================

# # st.sidebar.header("Controls")

# # record_seconds = st.sidebar.slider("Recording Duration", 5, 30, 10)

# # expected_speakers = st.sidebar.selectbox(
# #     "Expected Speakers",
# #     [None, 1, 2, 3, 4],
# #     index=2
# # )

# # min_segment_len = st.sidebar.slider(
# #     "Min Segment Length (sec)",
# #     0.3, 2.0, 0.7
# # )


# # # ================= LOAD MODEL =================

# # @st.cache_resource
# # def load_pipeline():

# #     pipeline = Pipeline.from_pretrained(
# #         "pyannote/speaker-diarization-3.1"
# #     )

# #     pipeline.to(torch.device("cpu"))

# #     return pipeline


# # # ================= AUDIO INPUT =================

# # tab1, tab2 = st.tabs(["Record", "Upload"])


# # # ---------- RECORD ----------
# # with tab1:

# #     if st.button("Start Recording"):

# #         st.info("Recording... Speak clearly.")

# #         audio = sd.rec(
# #             int(record_seconds * SR),
# #             samplerate=SR,
# #             channels=1,
# #             dtype="float32"
# #         )
# #         sd.wait()

# #         sf.write(INPUT_AUDIO, audio, SR, subtype="PCM_16")

# #         st.success("Saved!")
# #         st.audio(INPUT_AUDIO)


# # # ---------- UPLOAD ----------
# # with tab2:

# #     uploaded = st.file_uploader("Upload WAV", type=["wav"])

# #     if uploaded is not None:

# #         with open(INPUT_AUDIO, "wb") as f:
# #             f.write(uploaded.read())

# #         st.success("Uploaded!")
# #         st.audio(INPUT_AUDIO)


# # # ================= INFO =================

# # if os.path.exists(INPUT_AUDIO):

# #     wav, sr = torchaudio.load(INPUT_AUDIO)
# #     duration = wav.shape[1] / sr

# #     st.metric("Audio Duration", f"{duration:.2f} sec")

# # else:
# #     st.info("Provide audio to begin.")


# # # ================= RUN PIPELINE =================

# # if st.button("Run High-Accuracy Diarization"):

# #     if not os.path.exists(INPUT_AUDIO):
# #         st.error("Please upload or record audio.")
# #         st.stop()

# #     progress = st.progress(0)
# #     status = st.empty()

# #     # ---------- LOAD AUDIO ----------
# #     status.info("Loading audio...")

# #     wav, sr = torchaudio.load(INPUT_AUDIO)

# #     if wav.shape[0] > 1:
# #         wav = wav.mean(dim=0, keepdim=True)

# #     progress.progress(10)

# #     # ---------- LOAD MODEL ----------
# #     status.info("Loading diarization model...")
# #     pipeline = load_pipeline()

# #     progress.progress(30)

# #     # ---------- RUN DIARIZATION ----------
# #     status.info("Running neural speaker segmentation...")

# #     if expected_speakers is None:

# #         diarization = pipeline(
# #             INPUT_AUDIO,
# #             min_speakers=1,
# #             max_speakers=5
# #         )

# #     else:

# #         diarization = pipeline(
# #             INPUT_AUDIO,
# #             num_speakers=expected_speakers
# #         )

# #     progress.progress(65)

# #     # ---------- CLEAR OLD FILES ----------
# #     for f in os.listdir(SPEAKER_DIR):
# #         os.remove(os.path.join(SPEAKER_DIR, f))

# #     # ---------- SAVE SPEAKERS ----------
# #     full_audio = wav.squeeze().numpy()

# #     speaker_segments = []
# #     rttm_lines = []

# #     file_id = "session"

# #     for i, (turn, _, speaker) in enumerate(
# #         diarization.itertracks(yield_label=True)
# #     ):

# #         duration = turn.end - turn.start

# #         # ⭐ Remove micro segments (huge quality boost)
# #         if duration < min_segment_len:
# #             continue

# #         start = int(turn.start * sr)
# #         end = int(turn.end * sr)

# #         segment = full_audio[start:end]

# #         if len(segment) == 0:
# #             continue

# #         path = os.path.join(
# #             SPEAKER_DIR,
# #             f"{speaker}_{i}.wav"
# #         )

# #         sf.write(path, segment, sr, subtype="PCM_16")

# #         speaker_segments.append({
# #             "speaker": speaker,
# #             "start": turn.start,
# #             "end": turn.end,
# #             "path": path
# #         })

# #         rttm_lines.append(
# #             f"SPEAKER {file_id} 1 {turn.start:.3f} "
# #             f"{duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# #         )

# #     hyp_rttm = os.path.join(BASE_DIR, "hypothesis.rttm")

# #     with open(hyp_rttm, "w") as f:
# #         f.writelines(rttm_lines)

# #     progress.progress(100)
# #     status.success("Diarization Complete!")


# #     # ================= RESULTS =================

# #     st.subheader("Results")

# #     st.metric(
# #         "Detected Speakers",
# #         len(set(seg["speaker"] for seg in speaker_segments))
# #     )

# #     # ---------- PLAYER ----------
# #     st.subheader("Speaker Playback")

# #     cols = st.columns(2)

# #     for i, seg in enumerate(speaker_segments):

# #         with cols[i % 2]:

# #             st.markdown(f"### 🎤 {seg['speaker']}")
# #             st.write(f"{seg['start']:.2f}s → {seg['end']:.2f}s")

# #             st.audio(seg["path"])

# #     # ---------- RTTM ----------
# #     st.subheader("Generated RTTM")
# #     st.code("".join(rttm_lines))




# import streamlit as st
# import os
# import torch
# import torchaudio
# import soundfile as sf
# import sounddevice as sd

# from pyannote.audio import Pipeline
# from pyannote.database.util import load_rttm
# from pyannote.metrics.diarization import DiarizationErrorRate

# # ================= PATH CONFIG =================

# BASE_DIR = "output"
# SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")
# HYP_RTTM = os.path.join(BASE_DIR, "hypothesis.rttm")

# REF_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

# os.makedirs(BASE_DIR, exist_ok=True)
# os.makedirs(SPEAKER_DIR, exist_ok=True)

# SR = 16000

# # ================= PAGE =================

# st.set_page_config(layout="wide")
# st.title("High Accuracy Speaker Diarization + DER")
# st.caption("Automatic RTTM matching • Neural diarization • Evaluation")

# # ================= SIDEBAR =================

# st.sidebar.header("Controls")

# record_seconds = st.sidebar.slider("Recording Duration (sec)", 5, 30, 10)

# expected_speakers = st.sidebar.selectbox(
#     "Expected Speakers",
#     [None, 1, 2, 3, 4, 5],
#     index=2
# )

# min_segment_len = st.sidebar.slider(
#     "Min Segment Length (sec)",
#     0.3, 2.0, 0.7
# )

# # ================= MODEL =================

# @st.cache_resource
# def load_pipeline():
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1"
#     )
#     pipeline.to(torch.device("cpu"))
#     return pipeline

# # ================= HELPERS =================

# def find_reference_rttm(file_id, rttm_dir):
#     for f in os.listdir(rttm_dir):
#         if f.lower() == f"{file_id.lower()}.rttm":
#             return os.path.join(rttm_dir, f)
#     return None

# # ================= AUDIO INPUT =================

# tab1, tab2 = st.tabs(["🎙 Record", "📁 Upload"])

# # ---------- RECORD ----------
# with tab1:
#     if st.button("Start Recording"):
#         audio = sd.rec(
#             int(record_seconds * SR),
#             samplerate=SR,
#             channels=1,
#             dtype="float32"
#         )
#         sd.wait()

#         audio_path = os.path.join(BASE_DIR, "recorded_audio.wav")
#         sf.write(audio_path, audio, SR, subtype="PCM_16")

#         st.session_state["audio_path"] = audio_path

#         st.success("Recording saved")
#         st.audio(audio_path)

# # ---------- UPLOAD ----------
# with tab2:
#     uploaded = st.file_uploader("Upload WAV file", type=["wav"])

#     if uploaded:
#         audio_path = os.path.join(BASE_DIR, uploaded.name)

#         with open(audio_path, "wb") as f:
#             f.write(uploaded.read())

#         st.session_state["audio_path"] = audio_path

#         st.success("Audio uploaded")
#         st.audio(audio_path)

# # ================= AUDIO INFO =================

# if "audio_path" in st.session_state:
#     wav, sr = torchaudio.load(st.session_state["audio_path"])
#     duration = wav.shape[1] / sr
#     st.metric("Audio Duration", f"{duration:.2f} sec")
# else:
#     st.info("Upload or record audio to begin")

# # ================= RUN DIARIZATION =================

# if st.button("🚀 Run Diarization"):

#     if "audio_path" not in st.session_state:
#         st.error("No audio provided")
#         st.stop()

#     audio_path = st.session_state["audio_path"]
#     file_id = os.path.splitext(os.path.basename(audio_path))[0]

#     progress = st.progress(0)
#     status = st.empty()

#     # ---------- LOAD AUDIO ----------
#     status.info("Loading audio")
#     wav, sr = torchaudio.load(audio_path)
#     if wav.shape[0] > 1:
#         wav = wav.mean(dim=0, keepdim=True)
#     full_audio = wav.squeeze().numpy()
#     progress.progress(20)

#     # ---------- LOAD MODEL ----------
#     status.info("Loading diarization model")
#     pipeline = load_pipeline()
#     progress.progress(40)

#     # ---------- RUN DIARIZATION ----------
#     status.info("Running neural diarization")

#     if expected_speakers is None:
#         diarization = pipeline(audio_path, min_speakers=1, max_speakers=6)
#     else:
#         diarization = pipeline(audio_path, num_speakers=expected_speakers)

#     progress.progress(60)

#     # ---------- CLEAN SPEAKER FILES ----------
#     for f in os.listdir(SPEAKER_DIR):
#         os.remove(os.path.join(SPEAKER_DIR, f))

#     speaker_segments = []
#     rttm_lines = []

#     # ---------- SAVE SEGMENTS + RTTM ----------
#     for i, (turn, _, speaker) in enumerate(
#         diarization.itertracks(yield_label=True)
#     ):
#         dur = turn.end - turn.start
#         if dur < min_segment_len:
#             continue

#         start = int(turn.start * sr)
#         end = int(turn.end * sr)
#         segment = full_audio[start:end]

#         if len(segment) == 0:
#             continue

#         seg_path = os.path.join(SPEAKER_DIR, f"{speaker}_{i}.wav")
#         sf.write(seg_path, segment, sr, subtype="PCM_16")

#         speaker_segments.append({
#             "speaker": speaker,
#             "start": turn.start,
#             "end": turn.end,
#             "path": seg_path
#         })

#         rttm_lines.append(
#             f"SPEAKER {file_id} 1 {turn.start:.3f} "
#             f"{dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
#         )

#     with open(HYP_RTTM, "w") as f:
#         f.writelines(rttm_lines)

#     progress.progress(80)

#     # ================= DER =================

#     status.info("Calculating DER")

#     ref_rttm_path = find_reference_rttm(file_id, REF_RTTM_DIR)

#     if ref_rttm_path is None:
#         st.warning(
#             f"No reference RTTM found for '{file_id}.wav'. "
#             "DER skipped."
#         )
#     else:
#         try:
#             reference = load_rttm(ref_rttm_path)[file_id]
#             hypothesis = load_rttm(HYP_RTTM)[file_id]

#             metric = DiarizationErrorRate()
#             der = metric(reference, hypothesis)

#             st.subheader("📊 Diarization Error Rate")
#             st.metric("DER", f"{der * 100:.2f}%")

#         except Exception as e:
#             st.error("DER calculation failed")
#             st.exception(e)

#     progress.progress(100)
#     status.success("Diarization complete")

#     # ================= RESULTS =================

#     st.subheader("🎧 Speaker Segments")

#     st.metric(
#         "Detected Speakers",
#         len(set(s["speaker"] for s in speaker_segments))
#     )

#     cols = st.columns(2)
#     for i, seg in enumerate(speaker_segments):
#         with cols[i % 2]:
#             st.markdown(f"### 🎤 {seg['speaker']}")
#             st.write(f"{seg['start']:.2f}s → {seg['end']:.2f}s")
#             st.audio(seg["path"])

#     st.subheader("📄 Hypothesis RTTM")
#     st.code("".join(rttm_lines))





import streamlit as st
import os
import torch
import torchaudio
import soundfile as sf
import sounddevice as sd

from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

# ======== ADDED IMPORTS (NO LOGIC CHANGE) ========
import matplotlib.pyplot as plt
import numpy as np
# ===============================================

# ================= PATH CONFIG =================

BASE_DIR = "output"
SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")
HYP_RTTM = os.path.join(BASE_DIR, "hypothesis.rttm")

REF_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)

SR = 16000

# ================= PAGE =================

st.set_page_config(layout="wide")
st.title("High Accuracy Speaker Diarization + DER")
st.caption("Automatic RTTM matching • Neural diarization • Evaluation")

# ================= SIDEBAR =================

st.sidebar.header("Controls")

record_seconds = st.sidebar.slider("Recording Duration (sec)", 5, 30, 10)

expected_speakers = st.sidebar.selectbox(
    "Expected Speakers",
    [None, 1, 2, 3, 4, 5],
    index=2
)

min_segment_len = st.sidebar.slider(
    "Min Segment Length (sec)",
    0.3, 2.0, 0.7
)

# ================= MODEL =================

@st.cache_resource
def load_pipeline():
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )
    pipeline.to(torch.device("cpu"))
    return pipeline

# ================= HELPERS =================

def find_reference_rttm(file_id, rttm_dir):
    for f in os.listdir(rttm_dir):
        if f.lower() == f"{file_id.lower()}.rttm":
            return os.path.join(rttm_dir, f)
    return None

# ======== ADDED: TIMELINE FUNCTION ========

def plot_speaker_timeline(speaker_segments):
    speakers = sorted(set(seg["speaker"] for seg in speaker_segments))
    speaker_to_y = {spk: i for i, spk in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(14, 1.2 * len(speakers)))

    colors = plt.cm.tab10.colors
    speaker_colors = {
        spk: colors[i % len(colors)] for i, spk in enumerate(speakers)
    }

    for seg in speaker_segments:
        y = speaker_to_y[seg["speaker"]]
        ax.barh(
            y=y,
            width=seg["end"] - seg["start"],
            left=seg["start"],
            height=0.5,
            color=speaker_colors[seg["speaker"]],
            edgecolor="black"
        )

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title("Speaker Diarization Timeline")

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    return fig

# =========================================

# ================= AUDIO INPUT =================

tab1, tab2 = st.tabs(["🎙 Record", "📁 Upload"])

# ---------- RECORD ----------
with tab1:
    if st.button("Start Recording"):
        audio = sd.rec(
            int(record_seconds * SR),
            samplerate=SR,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        audio_path = os.path.join(BASE_DIR, "recorded_audio.wav")
        sf.write(audio_path, audio, SR, subtype="PCM_16")

        st.session_state["audio_path"] = audio_path

        st.success("Recording saved")
        st.audio(audio_path)

# ---------- UPLOAD ----------
with tab2:
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded:
        audio_path = os.path.join(BASE_DIR, uploaded.name)

        with open(audio_path, "wb") as f:
            f.write(uploaded.read())

        st.session_state["audio_path"] = audio_path

        st.success("Audio uploaded")
        st.audio(audio_path)

# ================= AUDIO INFO =================

if "audio_path" in st.session_state:
    wav, sr = torchaudio.load(st.session_state["audio_path"])
    duration = wav.shape[1] / sr
    st.metric("Audio Duration", f"{duration:.2f} sec")
else:
    st.info("Upload or record audio to begin")

# ================= RUN DIARIZATION =================

if st.button("🚀 Run Diarization"):

    if "audio_path" not in st.session_state:
        st.error("No audio provided")
        st.stop()

    audio_path = st.session_state["audio_path"]
    file_id = os.path.splitext(os.path.basename(audio_path))[0]

    progress = st.progress(0)
    status = st.empty()

    # ---------- LOAD AUDIO ----------
    status.info("Loading audio")
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    full_audio = wav.squeeze().numpy()
    progress.progress(20)

    # ---------- LOAD MODEL ----------
    status.info("Loading diarization model")
    pipeline = load_pipeline()
    progress.progress(40)

    # ---------- RUN DIARIZATION ----------
    status.info("Running neural diarization")

    if expected_speakers is None:
        diarization = pipeline(audio_path, min_speakers=1, max_speakers=6)
    else:
        diarization = pipeline(audio_path, num_speakers=expected_speakers)

    progress.progress(60)

    # ---------- CLEAN SPEAKER FILES ----------
    for f in os.listdir(SPEAKER_DIR):
        os.remove(os.path.join(SPEAKER_DIR, f))

    speaker_segments = []
    rttm_lines = []

    # ---------- SAVE SEGMENTS + RTTM ----------
    for i, (turn, _, speaker) in enumerate(
        diarization.itertracks(yield_label=True)
    ):
        dur = turn.end - turn.start
        if dur < min_segment_len:
            continue

        start = int(turn.start * sr)
        end = int(turn.end * sr)
        segment = full_audio[start:end]

        if len(segment) == 0:
            continue

        seg_path = os.path.join(SPEAKER_DIR, f"{speaker}_{i}.wav")
        sf.write(seg_path, segment, sr, subtype="PCM_16")

        speaker_segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "path": seg_path
        })

        rttm_lines.append(
            f"SPEAKER {file_id} 1 {turn.start:.3f} "
            f"{dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )

    with open(HYP_RTTM, "w") as f:
        f.writelines(rttm_lines)

    progress.progress(80)

    # ================= DER =================

    status.info("Calculating DER")

    ref_rttm_path = find_reference_rttm(file_id, REF_RTTM_DIR)

    if ref_rttm_path is None:
        st.warning(f"No reference RTTM found for '{file_id}.wav'. DER skipped.")
    else:
        reference = load_rttm(ref_rttm_path)[file_id]
        hypothesis = load_rttm(HYP_RTTM)[file_id]

        metric = DiarizationErrorRate()
        der = metric(reference, hypothesis)

        st.subheader("📊 Diarization Error Rate")
        st.metric("DER", f"{der * 100:.2f}%")

    progress.progress(100)
    status.success("Diarization complete")

    # ================= RESULTS =================

    st.subheader("🎧 Speaker Segments")

    st.metric(
        "Detected Speakers",
        len(set(s["speaker"] for s in speaker_segments))
    )

    cols = st.columns(2)
    for i, seg in enumerate(speaker_segments):
        with cols[i % 2]:
            st.markdown(f"### 🎤 {seg['speaker']}")
            st.write(f"{seg['start']:.2f}s → {seg['end']:.2f}s")
            st.audio(seg["path"])

    # ======== ADDED: TIMELINE DISPLAY ========

    st.subheader("🕒 Speaker Timeline")
    if speaker_segments:
        fig = plot_speaker_timeline(speaker_segments)
        st.pyplot(fig)
    else:
        st.info("No segments to display")

    # ========================================

    st.subheader("📄 Hypothesis RTTM")
    st.code("".join(rttm_lines))
