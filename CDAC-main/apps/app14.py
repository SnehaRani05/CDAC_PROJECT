# import os
# import streamlit as st
# import librosa
# import soundfile as sf
# import sounddevice as sd
# import pandas as pd

# from pyannote.audio import Pipeline

# # ================= CONFIG =================
# SR = 16000
# BASE_DIR = "output"
# AUDIO_PATH = os.path.join(BASE_DIR, "input.wav")
# SPEAKER_DIR = os.path.join(BASE_DIR, "speaker_audio")
# RTTM_PATH = os.path.join(BASE_DIR, "output.rttm")

# os.makedirs(BASE_DIR, exist_ok=True)
# os.makedirs(SPEAKER_DIR, exist_ok=True)

# # ================= SESSION STATE =================
# if "audio_ready" not in st.session_state:
#     st.session_state.audio_ready = False

# # ================= UTILS =================
# def clear_outputs():
#     for f in os.listdir(SPEAKER_DIR):
#         os.remove(os.path.join(SPEAKER_DIR, f))
#     if os.path.exists(RTTM_PATH):
#         os.remove(RTTM_PATH)

# # ================= PAGE =================
# st.set_page_config("Speaker Diarization", "🎙️", layout="wide")
# st.title("🎙️ Speaker Diarization Studio (AUTO)")
# st.caption("Live / WAV → Speaker Segmentation → Timestamps")

# # ================= SIDEBAR =================
# threshold = st.sidebar.slider(
#     "Speaker separation sensitivity",
#     0.45, 0.75, 0.52,
#     help="Lower = more speakers | Higher = fewer speakers"
# )

# # ================= LOAD MODEL =================
# @st.cache_resource
# def load_pipeline(threshold):
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization",
#         use_auth_token=True
#     )

#     # 🔥 ONLY IMPORTANT FIX IS HERE
#     pipeline.instantiate({
#         "segmentation": {
#             "min_duration_off": 0.15
#         },
#         "clustering": {
#             "method": "complete",   # <<< CRITICAL FIX
#             "threshold": threshold
#         }
#     })
#     return pipeline

# # ================= INPUT =================
# tab1, tab2 = st.tabs(["🎤 Live Recording", "📂 Upload WAV"])

# with tab1:
#     if st.button("🎙 Record 10 seconds"):
#         audio = sd.rec(
#             int(10 * SR),
#             samplerate=SR,
#             channels=1,
#             dtype="float32"
#         )
#         sd.wait()
#         sf.write(AUDIO_PATH, audio, SR)
#         st.audio(AUDIO_PATH)
#         st.session_state.audio_ready = True

# with tab2:
#     uploaded = st.file_uploader("Upload WAV file", type=["wav"])
#     if uploaded is not None:
#         audio, _ = librosa.load(uploaded, sr=SR, mono=True)
#         sf.write(AUDIO_PATH, audio, SR)
#         st.audio(AUDIO_PATH)
#         st.session_state.audio_ready = True

# # ================= RUN =================
# st.divider()

# if st.session_state.audio_ready:
#     if st.button("🚀 Run Speaker Diarization"):
#         clear_outputs()

#         st.info("Running diarization… please wait")

#         audio_np, _ = librosa.load(AUDIO_PATH, sr=SR)
#         pipeline = load_pipeline(threshold)
#         diarization = pipeline(AUDIO_PATH)

#         segments = []
#         rttm = []
#         seg_id = 0

#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             start, end = round(turn.start, 2), round(turn.end, 2)

#             segments.append({
#                 "Speaker": speaker,
#                 "Start (s)": start,
#                 "End (s)": end,
#                 "Duration (s)": round(end - start, 2)
#             })

#             sf.write(
#                 os.path.join(SPEAKER_DIR, f"{speaker}_{seg_id}.wav"),
#                 audio_np[int(turn.start * SR):int(turn.end * SR)],
#                 SR
#             )

#             rttm.append(
#                 f"SPEAKER session 1 {start:.3f} {end-start:.3f} "
#                 f"<NA> <NA> {speaker} <NA> <NA>\n"
#             )
#             seg_id += 1

#         with open(RTTM_PATH, "w") as f:
#             f.writelines(rttm)

#         st.success("Diarization completed ✅")

#         # ================= OUTPUT =================
#         st.subheader("📊 Speaker Timeline")
#         st.dataframe(pd.DataFrame(segments), use_container_width=True)

#         st.subheader("🔊 Speaker Segments")
#         for f in sorted(os.listdir(SPEAKER_DIR)):
#             st.audio(os.path.join(SPEAKER_DIR, f))

#         st.subheader("📄 RTTM")
#         st.code("".join(rttm))
# else:
#     st.warning("Please upload or record audio first")



import os
import streamlit as st
import librosa
import soundfile as sf
import sounddevice as sd
import pandas as pd
import numpy as np
import noisereduce as nr

from pyannote.audio import Pipeline

# ================= CONFIG =================
SR = 16000
BASE_DIR = "output"
AUDIO_PATH = os.path.join(BASE_DIR, "input.wav")
SPEAKER_DIR = os.path.join(BASE_DIR, "speaker_audio")
RTTM_PATH = os.path.join(BASE_DIR, "output.rttm")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)

# ================= SESSION STATE =================
if "audio_ready" not in st.session_state:
    st.session_state.audio_ready = False

# ================= UTILS =================
def clear_outputs():
    for f in os.listdir(SPEAKER_DIR):
        os.remove(os.path.join(SPEAKER_DIR, f))
    if os.path.exists(RTTM_PATH):
        os.remove(RTTM_PATH)

def remove_noise(audio, sr):
    """Simple & stable noise reduction"""
    noise_profile = audio[: int(0.5 * sr)]
    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_profile,
        prop_decrease=0.9
    )
    return reduced.astype(np.float32)

# ================= PAGE =================
st.set_page_config("Speaker Diarization", "🎙️", layout="wide")
st.title("🎙️ Speaker Diarization Studio (AUTO)")
st.caption("Live / WAV → Noise Reduction → Speaker Segmentation → RTTM")

# ================= SIDEBAR =================
threshold = st.sidebar.slider(
    "Speaker separation sensitivity",
    0.45, 0.75, 0.52,
    help="Lower = more speakers | Higher = fewer speakers"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_pipeline(threshold):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=True
    )

    pipeline.instantiate({
        "segmentation": {
            "min_duration_off": 0.15
        },
        "clustering": {
            "method": "complete",
            "threshold": threshold
        }
    })
    return pipeline

# ================= INPUT =================
tab1, tab2 = st.tabs(["🎤 Live Recording", "📂 Upload WAV"])

with tab1:
    if st.button("🎙 Record 10 seconds"):
        audio = sd.rec(
            int(10 * SR),
            samplerate=SR,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        audio = audio.squeeze()
        audio = remove_noise(audio, SR)

        sf.write(AUDIO_PATH, audio, SR)
        st.audio(AUDIO_PATH)
        st.session_state.audio_ready = True

with tab2:
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded is not None:
        audio, _ = librosa.load(uploaded, sr=SR, mono=True)

        audio = remove_noise(audio, SR)

        sf.write(AUDIO_PATH, audio, SR)
        st.audio(AUDIO_PATH)
        st.session_state.audio_ready = True

# ================= RUN =================
st.divider()

if st.session_state.audio_ready:
    if st.button("🚀 Run Speaker Diarization"):
        clear_outputs()
        st.info("Running diarization… please wait")

        audio_np, _ = librosa.load(AUDIO_PATH, sr=SR)
        pipeline = load_pipeline(threshold)
        diarization = pipeline(AUDIO_PATH)

        segments = []
        rttm = []
        seg_id = 0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start, end = round(turn.start, 2), round(turn.end, 2)

            segments.append({
                "Speaker": speaker,
                "Start (s)": start,
                "End (s)": end,
                "Duration (s)": round(end - start, 2)
            })

            sf.write(
                os.path.join(SPEAKER_DIR, f"{speaker}_{seg_id}.wav"),
                audio_np[int(turn.start * SR):int(turn.end * SR)],
                SR
            )

            rttm.append(
                f"SPEAKER session 1 {start:.3f} {end-start:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )
            seg_id += 1

        with open(RTTM_PATH, "w") as f:
            f.writelines(rttm)

        st.success("Diarization completed ✅")

        # ================= OUTPUT =================
        st.subheader("📊 Speaker Timeline")
        st.dataframe(pd.DataFrame(segments), use_container_width=True)

        st.subheader("🔊 Speaker Segments")
        for f in sorted(os.listdir(SPEAKER_DIR)):
            st.audio(os.path.join(SPEAKER_DIR, f))

        st.subheader("📄 RTTM")
        st.code("".join(rttm))
else:
    st.warning("Please upload or record audio first")
