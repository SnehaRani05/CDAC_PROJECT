# # # # import os
# # # # import sys
# # # # import subprocess
# # # # import streamlit as st
# # # # import librosa
# # # # import soundfile as sf
# # # # import pandas as pd
# # # # import matplotlib.pyplot as plt

# # # # from pyannote.audio import Pipeline

# # # # # =================================================
# # # # # ENV FIXES
# # # # # =================================================
# # # # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# # # # os.environ["SB_LOCAL_STRATEGY"] = "copy"

# # # # HF_TOKEN = os.getenv("HF_TOKEN")
# # # # if HF_TOKEN is None:
# # # #     st.error("❌ HF_TOKEN not set")
# # # #     st.stop()

# # # # # =================================================
# # # # # DIRECTORIES
# # # # # =================================================
# # # # UPLOAD_DIR = "test_audio"
# # # # OUTPUT_DIR = "outputs"

# # # # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # # # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # # # =================================================
# # # # # LOAD PYANNOTE (CACHED)
# # # # # =================================================
# # # # @st.cache_resource
# # # # def load_pipeline():
# # # #     return Pipeline.from_pretrained(
# # # #         "pyannote/speaker-diarization",
# # # #         use_auth_token=HF_TOKEN
# # # #     )

# # # # # =================================================
# # # # # AUDIO CONVERSION (PURE PYTHON)
# # # # # =================================================
# # # # def convert_to_16k_mono(input_path, output_path):
# # # #     audio, _ = librosa.load(input_path, sr=16000, mono=True)
# # # #     sf.write(output_path, audio, 16000)

# # # # # =================================================
# # # # # DIARIZATION (WITH RTTM CACHE)
# # # # # =================================================
# # # # def wav_to_rttm(audio_path):
# # # #     pipeline = load_pipeline()
# # # #     file_id = os.path.splitext(os.path.basename(audio_path))[0]
# # # #     rttm_path = os.path.join(OUTPUT_DIR, f"{file_id}.rttm")

# # # #     # Use cached RTTM if available
# # # #     if os.path.exists(rttm_path):
# # # #         return rttm_path

# # # #     diarization = pipeline(audio_path)

# # # #     with open(rttm_path, "w") as f:
# # # #         for turn, _, speaker in diarization.itertracks(yield_label=True):
# # # #             f.write(
# # # #                 f"SPEAKER {file_id} 1 "
# # # #                 f"{turn.start:.3f} "
# # # #                 f"{turn.end - turn.start:.3f} "
# # # #                 f"<NA> <NA> {speaker} <NA> <NA>\n"
# # # #             )

# # # #     return rttm_path

# # # # # =================================================
# # # # # RTTM UTILITIES
# # # # # =================================================
# # # # def rttm_to_df(path):
# # # #     rows = []
# # # #     with open(path) as f:
# # # #         for line in f:
# # # #             p = line.split()
# # # #             start = float(p[3])
# # # #             dur = float(p[4])
# # # #             rows.append({
# # # #                 "speaker": p[7],
# # # #                 "start": start,
# # # #                 "end": start + dur
# # # #             })
# # # #     return pd.DataFrame(rows)

# # # # def plot_timeline(df):
# # # #     speakers = sorted(df["speaker"].unique())
# # # #     smap = {s: i for i, s in enumerate(speakers)}

# # # #     fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))
# # # #     for _, r in df.iterrows():
# # # #         ax.barh(smap[r["speaker"]], r["end"] - r["start"], left=r["start"])

# # # #     ax.set_yticks(list(smap.values()))
# # # #     ax.set_yticklabels(speakers)
# # # #     ax.set_xlabel("Time (seconds)")
# # # #     ax.set_ylabel("Speaker")
# # # #     ax.set_title("Speaker Timeline")
# # # #     ax.grid(axis="x", linestyle="--", alpha=0.5)
# # # #     return fig

# # # # def count_speakers(rttm):
# # # #     return len({line.split()[7] for line in open(rttm)})

# # # # # =================================================
# # # # # RUN DER SCRIPT (INDEPENDENT)
# # # # # =================================================
# # # # def run_der(audio_name):
# # # #     try:
# # # #         result = subprocess.run(
# # # #             [sys.executable, "der.py", audio_name],
# # # #             capture_output=True,
# # # #             text=True,
# # # #             check=True
# # # #         )
# # # #         return result.stdout.strip(), None
# # # #     except subprocess.CalledProcessError as e:
# # # #         return None, e.stderr

# # # # # =================================================
# # # # # UI
# # # # # =================================================
# # # # st.set_page_config(page_title="Speaker Diarization + DER", layout="wide")
# # # # st.title("🎙️ Speaker Diarization (Real-Time + Upload + DER)")

# # # # tab1, tab2, tab3 = st.tabs([
# # # #     "🎤 Record / Upload",
# # # #     "🧠 Diarization",
# # # #     "📊 Results"
# # # # ])

# # # # # =================================================
# # # # # TAB 1 — INPUT
# # # # # =================================================
# # # # with tab1:
# # # #     st.subheader("🎤 Real-time Microphone")

# # # #     audio_file = st.audio_input("Click to record")

# # # #     if audio_file:
# # # #         raw = os.path.join(UPLOAD_DIR, "mic_raw.wav")
# # # #         with open(raw, "wb") as f:
# # # #             f.write(audio_file.getbuffer())

# # # #         conv = os.path.join(UPLOAD_DIR, "mic_16k.wav")
# # # #         convert_to_16k_mono(raw, conv)

# # # #         st.audio(conv)

# # # #         st.session_state["audio"] = conv
# # # #         st.session_state["audio_name"] = "mic_16k.wav"   # 🔥 FIXED

# # # #     st.divider()
# # # #     st.subheader("📁 Upload WAV")

# # # #     uploaded = st.file_uploader("Choose WAV file", type=["wav"])

# # # #     if uploaded:
# # # #         raw = os.path.join(UPLOAD_DIR, uploaded.name)
# # # #         with open(raw, "wb") as f:
# # # #             f.write(uploaded.getbuffer())

# # # #         conv = os.path.join(
# # # #             UPLOAD_DIR,
# # # #             os.path.splitext(uploaded.name)[0] + "_16k.wav"
# # # #         )
# # # #         convert_to_16k_mono(raw, conv)

# # # #         st.audio(conv)

# # # #         st.session_state["audio"] = conv
# # # #         st.session_state["audio_name"] = os.path.basename(conv)

# # # # # =================================================
# # # # # TAB 2 — DIARIZATION
# # # # # =================================================
# # # # with tab2:
# # # #     if st.button("🚀 Run Diarization"):
# # # #         if "audio" not in st.session_state:
# # # #             st.error("❌ Upload or record audio first")
# # # #         else:
# # # #             with st.spinner("Running speaker diarization..."):
# # # #                 rttm = wav_to_rttm(st.session_state["audio"])
# # # #                 st.session_state["rttm"] = rttm
# # # #             st.success("✅ Diarization completed")

# # # # # =================================================
# # # # # TAB 3 — RESULTS + DER
# # # # # =================================================
# # # # with tab3:
# # # #     if "rttm" in st.session_state:
# # # #         rttm = st.session_state["rttm"]

# # # #         st.metric("🗣️ Speakers Detected", count_speakers(rttm))

# # # #         df = rttm_to_df(rttm)
# # # #         st.pyplot(plot_timeline(df))
# # # #         plt.close("all")

# # # #         st.divider()
# # # #         st.subheader("🎯 Diarization Error Rate (DER)")

# # # #         der, err = run_der(st.session_state["audio_name"])

# # # #         if err:
# # # #             st.warning("⚠️ Reference RTTM not found → DER not computed")
# # # #         else:
# # # #             st.metric("DER (%)", der)

# # # #         with open(rttm, "rb") as f:
# # # #             st.download_button(
# # # #                 "📄 Download RTTM",
# # # #                 f,
# # # #                 file_name=os.path.basename(rttm),
# # # #                 mime="text/plain"
# # # #             )
# # # #     else:
# # # #         st.info("Run diarization to see results")



# # # # import os
# # # # import sys
# # # # import subprocess
# # # # import streamlit as st
# # # # import pandas as pd
# # # # import matplotlib.pyplot as plt

# # # # # =================================================
# # # # # ENV FIXES
# # # # # =================================================
# # # # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# # # # os.environ["SB_LOCAL_STRATEGY"] = "copy"

# # # # HF_TOKEN = os.getenv("HF_TOKEN")
# # # # if HF_TOKEN is None:
# # # #     st.error("❌ HF_TOKEN not set")
# # # #     st.stop()

# # # # # =================================================
# # # # # DIRECTORIES
# # # # # =================================================
# # # # UPLOAD_DIR = "uploads"
# # # # CLEAN_DIR = "cleaned_audio"
# # # # OUTPUT_DIR = "outputs"
# # # # DATASET_AUDIO_DIR = "processed/dataset/audio"
# # # # DATASET_RTTM_DIR = "processed/dataset/rttm"

# # # # for d in [UPLOAD_DIR, CLEAN_DIR, OUTPUT_DIR]:
# # # #     os.makedirs(d, exist_ok=True)

# # # # # =================================================
# # # # # BACKEND CALLS
# # # # # =================================================
# # # # def run_preprocess(raw, cleaned):
# # # #     subprocess.run(
# # # #         [sys.executable, "process.py", raw, cleaned],
# # # #         check=True
# # # #     )

# # # # def run_diarization(cleaned):
# # # #     subprocess.run(
# # # #         [sys.executable, "diarize.py", cleaned, OUTPUT_DIR],
# # # #         check=True
# # # #     )
# # # #     base = os.path.splitext(os.path.basename(cleaned))[0]
# # # #     rttm = os.path.join(OUTPUT_DIR, base + ".rttm")
# # # #     if not os.path.exists(rttm):
# # # #         raise FileNotFoundError("RTTM not created")
# # # #     return rttm

# # # # def run_der(audio_name):
# # # #     result = subprocess.run(
# # # #         [sys.executable, "der.py", audio_name],
# # # #         capture_output=True,
# # # #         text=True
# # # #     )
# # # #     if result.returncode != 0:
# # # #         return None
# # # #     return result.stdout.strip()

# # # # # =================================================
# # # # # RTTM UTILS
# # # # # =================================================
# # # # def rttm_to_df(path):
# # # #     rows = []
# # # #     with open(path) as f:
# # # #         for line in f:
# # # #             p = line.split()
# # # #             rows.append({
# # # #                 "speaker": p[7],
# # # #                 "start": float(p[3]),
# # # #                 "end": float(p[3]) + float(p[4])
# # # #             })
# # # #     return pd.DataFrame(rows)

# # # # def plot_timeline(df):
# # # #     speakers = sorted(df.speaker.unique())
# # # #     smap = {s: i for i, s in enumerate(speakers)}

# # # #     fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))
# # # #     for _, r in df.iterrows():
# # # #         ax.barh(smap[r.speaker], r.end - r.start, left=r.start)

# # # #     ax.set_yticks(list(smap.values()))
# # # #     ax.set_yticklabels(speakers)
# # # #     ax.set_xlabel("Time (s)")
# # # #     ax.set_ylabel("Speaker")
# # # #     ax.set_title("Speaker Diarization Timeline")
# # # #     ax.grid(axis="x", alpha=0.4)
# # # #     return fig

# # # # def count_speakers(rttm):
# # # #     return len({l.split()[7] for l in open(rttm)})

# # # # # =================================================
# # # # # UI
# # # # # =================================================
# # # # st.set_page_config("Speaker Diarization", layout="wide")
# # # # st.title("🎙️ Speaker Diarization (Dataset + Real-Time)")

# # # # mode = st.radio(
# # # #     "Select mode",
# # # #     ["📦 Dataset (with RTTM & DER)", "🎤 Real-Time / Unseen Audio"]
# # # # )

# # # # # =================================================
# # # # # DATASET MODE
# # # # # =================================================
# # # # if mode.startswith("📦"):
# # # #     st.subheader("Dataset Audio")

# # # #     files = sorted(os.listdir(DATASET_AUDIO_DIR))
# # # #     audio_name = st.selectbox("Choose audio", files)

# # # #     if st.button("🚀 Run Dataset Diarization"):
# # # #         raw = os.path.join(DATASET_AUDIO_DIR, audio_name)
# # # #         cleaned = os.path.join(CLEAN_DIR, audio_name.replace(".wav", "_cleaned.wav"))

# # # #         run_preprocess(raw, cleaned)
# # # #         rttm = run_diarization(cleaned)

# # # #         st.audio(raw)
# # # #         st.metric("Speakers", count_speakers(rttm))

# # # #         df = rttm_to_df(rttm)
# # # #         st.pyplot(plot_timeline(df))

# # # #         st.divider()
# # # #         der = run_der(audio_name)
# # # #         if der:
# # # #             st.metric("DER (%)", der)

# # # # # =================================================
# # # # # REAL-TIME MODE
# # # # # =================================================
# # # # else:
# # # #     st.subheader("Real-Time / Unseen Audio")

# # # #     audio_file = st.audio_input("🎤 Record audio")
# # # #     uploaded = st.file_uploader("📁 Upload WAV", type=["wav"])

# # # #     raw = None
# # # #     name = None

# # # #     if audio_file:
# # # #         name = "mic.wav"
# # # #         raw = os.path.join(UPLOAD_DIR, name)
# # # #         with open(raw, "wb") as f:
# # # #             f.write(audio_file.getbuffer())

# # # #     if uploaded:
# # # #         name = uploaded.name
# # # #         raw = os.path.join(UPLOAD_DIR, name)
# # # #         with open(raw, "wb") as f:
# # # #             f.write(uploaded.getbuffer())

# # # #     if raw and st.button("🚀 Run Diarization"):
# # # #         cleaned = os.path.join(CLEAN_DIR, name.replace(".wav", "_cleaned.wav"))

# # # #         run_preprocess(raw, cleaned)
# # # #         rttm = run_diarization(cleaned)

# # # #         st.audio(raw)
# # # #         st.metric("Speakers", count_speakers(rttm))

# # # #         df = rttm_to_df(rttm)
# # # #         st.pyplot(plot_timeline(df))

# # # #         st.info("ℹ️ DER not available for unseen / real-time audio")

# # # #         with open(rttm, "rb") as f:
# # # #             st.download_button("📄 Download RTTM", f)


# # # import os
# # # import sys
# # # import subprocess
# # # import streamlit as st
# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # # =================================================
# # # # WINDOWS FIXES (IMPORTANT)
# # # # =================================================
# # # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# # # os.environ["SB_LOCAL_STRATEGY"] = "copy"

# # # HF_TOKEN = os.getenv("HF_TOKEN")
# # # if not HF_TOKEN:
# # #     st.error("❌ HF_TOKEN not set")
# # #     st.stop()

# # # # =================================================
# # # # DIRECTORIES
# # # # =================================================
# # # DATASET_AUDIO_DIR = "processed/dataset/audio"
# # # DATASET_RTTM_DIR  = "processed/dataset/rttm"

# # # UPLOAD_DIR  = "uploads"
# # # CLEAN_DIR   = "cleaned_audio"
# # # OUTPUT_DIR  = "outputs"

# # # for d in [UPLOAD_DIR, CLEAN_DIR, OUTPUT_DIR]:
# # #     os.makedirs(d, exist_ok=True)

# # # # =================================================
# # # # BACKEND PIPELINE (sys + subprocess)
# # # # =================================================
# # # def run_preprocess(raw_path, cleaned_path):
# # #     result = subprocess.run(
# # #         [sys.executable, "process.py", raw_path, cleaned_path],
# # #         capture_output=True,
# # #         text=True,
# # #         env=os.environ.copy()
# # #     )
# # #     if result.returncode != 0:
# # #         raise RuntimeError(result.stderr or result.stdout)

# # #     if not os.path.exists(cleaned_path):
# # #         raise FileNotFoundError(f"Cleaned audio not created: {cleaned_path}")

# # #     return cleaned_path


# # # def run_diarization(cleaned_path):
# # #     if not os.path.exists(cleaned_path):
# # #         raise FileNotFoundError(f"Audio not found: {cleaned_path}")

# # #     result = subprocess.run(
# # #         [sys.executable, "diarize.py", cleaned_path, OUTPUT_DIR],
# # #         capture_output=True,
# # #         text=True,
# # #         env=os.environ.copy()
# # #     )
# # #     if result.returncode != 0:
# # #         raise RuntimeError(result.stderr or result.stdout)

# # #     base = os.path.splitext(os.path.basename(cleaned_path))[0]
# # #     rttm_path = os.path.join(OUTPUT_DIR, base + ".rttm")

# # #     if not os.path.exists(rttm_path):
# # #         raise FileNotFoundError(f"RTTM not created: {rttm_path}")

# # #     return rttm_path


# # # def run_der(audio_name):
# # #     result = subprocess.run(
# # #         [sys.executable, "der.py", audio_name],
# # #         capture_output=True,
# # #         text=True,
# # #         env=os.environ.copy()
# # #     )
# # #     if result.returncode != 0:
# # #         return None
# # #     return result.stdout.strip()

# # # # =================================================
# # # # RTTM UTILITIES
# # # # =================================================
# # # def rttm_to_df(path):
# # #     rows = []
# # #     with open(path) as f:
# # #         for line in f:
# # #             p = line.split()
# # #             rows.append({
# # #                 "speaker": p[7],
# # #                 "start": float(p[3]),
# # #                 "end": float(p[3]) + float(p[4])
# # #             })
# # #     return pd.DataFrame(rows)


# # # def plot_timeline(df):
# # #     speakers = sorted(df["speaker"].unique())
# # #     smap = {s: i for i, s in enumerate(speakers)}

# # #     fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))
# # #     for _, r in df.iterrows():
# # #         ax.barh(
# # #             smap[r["speaker"]],
# # #             r["end"] - r["start"],
# # #             left=r["start"]
# # #         )

# # #     ax.set_yticks(list(smap.values()))
# # #     ax.set_yticklabels(speakers)
# # #     ax.set_xlabel("Time (seconds)")
# # #     ax.set_ylabel("Speaker")
# # #     ax.set_title("Speaker Diarization Timeline")
# # #     ax.grid(axis="x", alpha=0.4)
# # #     return fig


# # # def count_speakers(rttm):
# # #     return len({l.split()[7] for l in open(rttm)})

# # # # =================================================
# # # # STREAMLIT UI
# # # # =================================================
# # # st.set_page_config("Speaker Diarization", layout="wide")
# # # st.title("🎙️ Speaker Diarization (Dataset + Real-Time)")

# # # mode = st.radio(
# # #     "Select mode",
# # #     ["📦 Dataset Audio (DER available)", "🎤 Real-Time / Unseen Audio"]
# # # )

# # # # =================================================
# # # # DATASET MODE
# # # # =================================================
# # # if mode.startswith("📦"):
# # #     st.subheader("Dataset Audio")

# # #     files = sorted(os.listdir(DATASET_AUDIO_DIR))
# # #     audio_name = st.selectbox("Choose dataset audio", files)

# # #     if st.button("🚀 Run Diarization"):
# # #         raw = os.path.join(DATASET_AUDIO_DIR, audio_name)
# # #         base = os.path.splitext(audio_name)[0]
# # #         cleaned = os.path.join(CLEAN_DIR, base + "_cleaned.wav")

# # #         with st.spinner("Processing dataset audio..."):
# # #             run_preprocess(raw, cleaned)
# # #             rttm = run_diarization(cleaned)

# # #         st.audio(raw)
# # #         st.metric("🗣️ Speakers Detected", count_speakers(rttm))

# # #         df = rttm_to_df(rttm)
# # #         st.pyplot(plot_timeline(df))
# # #         plt.close("all")

# # #         st.divider()
# # #         der = run_der(audio_name)
# # #         if der:
# # #             st.metric("🎯 DER (%)", der)

# # # # =================================================
# # # # REAL-TIME / UNSEEN MODE
# # # # =================================================
# # # else:
# # #     st.subheader("Real-Time / Unseen Audio")

# # #     audio_file = st.audio_input("🎤 Record audio")
# # #     uploaded   = st.file_uploader("📁 Upload WAV", type=["wav"])

# # #     raw = None
# # #     name = None

# # #     if audio_file:
# # #         name = "mic.wav"
# # #         raw = os.path.join(UPLOAD_DIR, name)
# # #         with open(raw, "wb") as f:
# # #             f.write(audio_file.getbuffer())

# # #     if uploaded:
# # #         name = uploaded.name
# # #         raw = os.path.join(UPLOAD_DIR, name)
# # #         with open(raw, "wb") as f:
# # #             f.write(uploaded.getbuffer())

# # #     if raw and st.button("🚀 Run Diarization"):
# # #         base = os.path.splitext(name)[0]
# # #         cleaned = os.path.join(CLEAN_DIR, base + "_cleaned.wav")

# # #         with st.spinner("Processing real-time audio..."):
# # #             run_preprocess(raw, cleaned)
# # #             rttm = run_diarization(cleaned)

# # #         st.audio(raw)
# # #         st.metric("🗣️ Speakers Detected", count_speakers(rttm))

# # #         df = rttm_to_df(rttm)
# # #         st.pyplot(plot_timeline(df))
# # #         plt.close("all")

# # #         st.info("ℹ️ DER is not computed for unseen / real-time audio")

# # #         with open(rttm, "rb") as f:
# # #             st.download_button("📄 Download RTTM", f)



import os
import sys
import subprocess
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

# =========================
# WINDOWS FIXES
# =========================
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["SB_LOCAL_STRATEGY"] = "copy"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ HF_TOKEN not set")
    st.stop()

# =========================
# DIRECTORIES
# =========================
UPLOAD_DIR = "uploads"
CLEAN_DIR = "cleaned_audio"
OUTPUT_DIR = "outputs"
SPEAKER_DIR = "speaker_audio"

DATASET_AUDIO_DIR = "processed/dataset/audio"

for d in [UPLOAD_DIR, CLEAN_DIR, OUTPUT_DIR, SPEAKER_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================
# BACKEND CALLS
# =========================
def run_preprocess(raw_path, cleaned_path):
    subprocess.run(
        [sys.executable, "process.py", raw_path, cleaned_path],
        check=True,
        env=os.environ.copy()
    )
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned audio not created: {cleaned_path}")

def run_diarization(cleaned_path):
    subprocess.run(
        [sys.executable, "integrated.py", cleaned_path, OUTPUT_DIR],
        check=True,
        env=os.environ.copy()
    )
    base = os.path.splitext(os.path.basename(cleaned_path))[0]
    rttm = os.path.join(OUTPUT_DIR, base + ".rttm")
    if not os.path.exists(rttm):
        raise FileNotFoundError("RTTM not created")
    return rttm

# =========================
# RTTM UTILITIES
# =========================
def read_rttm(rttm_path):
    rows = []
    with open(rttm_path) as f:
        for line in f:
            p = line.strip().split()
            start = float(p[3])
            dur = float(p[4])
            rows.append({
                "speaker": p[7],
                "start": start,
                "end": start + dur
            })
    return rows

def show_rttm_text(rttm_path):
    st.subheader("📄 RTTM Output")
    with open(rttm_path) as f:
        st.text_area("RTTM file", f.read(), height=220)

def plot_timeline(df):
    speakers = sorted(df["speaker"].unique())
    smap = {s: i for i, s in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))
    for _, r in df.iterrows():
        ax.barh(smap[r["speaker"]], r["end"] - r["start"], left=r["start"])

    ax.set_yticks(list(smap.values()))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title("Speaker Diarization Timeline")
    ax.grid(axis="x", alpha=0.4)
    return fig

def count_speakers(rttm):
    return len({l.split()[7] for l in open(rttm)})

# =========================
# SPEAKER-WISE AUDIO
# =========================
def export_speaker_audio(audio_path, segments):
    audio, sr = sf.read(audio_path)
    os.makedirs(SPEAKER_DIR, exist_ok=True)

    speaker_chunks = {}
    for seg in segments:
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        speaker_chunks.setdefault(seg["speaker"], []).append(audio[s:e])

    out_files = {}
    for speaker, chunks in speaker_chunks.items():
        merged = np.concatenate(chunks)
        out_path = os.path.join(SPEAKER_DIR, f"{speaker}.wav")
        sf.write(out_path, merged, sr)
        out_files[speaker] = out_path

    return out_files




# =========================
# UI
# =========================
st.set_page_config("Speaker Diarization", layout="wide")
st.title("🎙️ Speaker Diarization (Dataset + Real-Time)")

mode = st.radio(
    "Select mode",
    ["📦 Dataset Audio", "🎤 Real-Time / Unseen Audio"]
)

# =========================
# DATASET MODE
# =========================
if mode.startswith("📦"):
    st.subheader("Dataset Audio")

    files = sorted(os.listdir(DATASET_AUDIO_DIR))
    audio_name = st.selectbox("Choose dataset audio", files)

    if st.button("🚀 Run Diarization"):
        raw = os.path.join(DATASET_AUDIO_DIR, audio_name)
        cleaned = os.path.join(CLEAN_DIR, audio_name.replace(".wav", "_cleaned.wav"))

        with st.spinner("Processing dataset audio..."):
            run_preprocess(raw, cleaned)
            rttm = run_diarization(cleaned)

        st.audio(raw)
        st.metric("🗣️ Speakers Detected", count_speakers(rttm))

        segments = read_rttm(rttm)
        df = pd.DataFrame(segments)

        st.subheader("🕒 Timestamp Segments")
        st.dataframe(df)

        st.pyplot(plot_timeline(df))
        plt.close("all")

        show_rttm_text(rttm)

        st.subheader("🔊 Speaker-wise Audio")
        speaker_files = export_speaker_audio(cleaned, segments)
        for spk, path in speaker_files.items():
            st.markdown(f"### 🎙️ {spk}")
            st.audio(path)

# =========================
# REAL-TIME MODE
# =========================
else:
    st.subheader("Real-Time / Unseen Audio")

    audio_file = st.audio_input("🎤 Record audio")
    uploaded = st.file_uploader("📁 Upload WAV", type=["wav"])

    raw = None
    name = None

    if audio_file:
        name = "mic.wav"
        raw = os.path.join(UPLOAD_DIR, name)
        with open(raw, "wb") as f:
            f.write(audio_file.getbuffer())

    if uploaded:
        name = uploaded.name
        raw = os.path.join(UPLOAD_DIR, name)
        with open(raw, "wb") as f:
            f.write(uploaded.getbuffer())

    if raw and st.button("🚀 Run Diarization"):
        cleaned = os.path.join(CLEAN_DIR, name.replace(".wav", "_cleaned.wav"))

        with st.spinner("Processing real-time audio..."):
            run_preprocess(raw, cleaned)
            rttm = run_diarization(cleaned)

        st.audio(raw)
        st.metric("🗣️ Speakers Detected", count_speakers(rttm))

        segments = read_rttm(rttm)
        df = pd.DataFrame(segments)

        st.subheader("🕒 Timestamp Segments")
        st.dataframe(df)

        st.pyplot(plot_timeline(df))
        plt.close("all")

        show_rttm_text(rttm)

        st.subheader("🔊 Speaker-wise Audio")
        speaker_files = export_speaker_audio(cleaned, segments)
        for spk, path in speaker_files.items():
            st.markdown(f"### 🎙️ {spk}")
            st.audio(path)

        st.info("ℹ️ DER is not computed for unseen / real-time audio")

        with open(rttm, "rb") as f:
            st.download_button("📄 Download RTTM", f)
