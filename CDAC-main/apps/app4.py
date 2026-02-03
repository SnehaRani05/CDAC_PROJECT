# # import os
# # import sys
# # import subprocess
# # import streamlit as st

# # # =============================
# # # ENV FIX (Windows + pyannote)
# # # =============================
# # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# # os.environ["SB_LOCAL_STRATEGY"] = "copy"

# # if not os.getenv("HF_TOKEN"):
# #     st.error("❌ HF_TOKEN not set. Run: huggingface-cli login")
# #     st.stop()

# # # =============================
# # # DIRECTORIES
# # # =============================
# # UPLOAD_DIR = "uploads"
# # OUTPUT_DIR = "output"
# # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")

# # for d in [UPLOAD_DIR, OUTPUT_DIR, SPEAKER_DIR]:
# #     os.makedirs(d, exist_ok=True)

# # # =============================
# # # STREAMLIT UI
# # # =============================
# # st.set_page_config(page_title="Speaker Diarization", layout="wide")
# # st.title("🎙️ Speaker Diarization (Automatic N-Speakers)")

# # st.caption(
# #     "Upload any audio file. The system will automatically detect "
# #     "the number of speakers and generate speaker-wise audio."
# # )

# # uploaded = st.file_uploader(
# #     "📁 Upload audio file",
# #     type=["wav", "mp3", "m4a", "flac", "ogg"]
# # )

# # audio_path = None

# # # =============================
# # # SAVE UPLOADED AUDIO
# # # =============================
# # if uploaded:
# #     audio_path = os.path.join(UPLOAD_DIR, uploaded.name)
# #     with open(audio_path, "wb") as f:
# #         f.write(uploaded.getbuffer())

# #     st.audio(audio_path)

# # # =============================
# # # RUN DIARIZATION
# # # =============================
# # if audio_path and st.button("🚀 Run Diarization"):

# #     with st.spinner("Running speaker diarization..."):
# #         result = subprocess.run(
# #             [
# #                 sys.executable,
# #                 "diarize.py",
# #                 audio_path,
# #                 OUTPUT_DIR
# #             ],
# #             capture_output=True,
# #             text=True,
# #             env=os.environ.copy()
# #         )

# #     if result.returncode != 0:
# #         st.error("❌ Diarization failed")
# #         st.code(result.stderr)
# #         st.stop()

# #     st.success("✅ Diarization completed successfully")

# #     # =============================
# #     # SHOW RTTM
# #     # =============================
# #     rttm_path = os.path.join(
# #         OUTPUT_DIR,
# #         os.path.splitext(uploaded.name)[0] + ".rttm"
# #     )

# #     if os.path.exists(rttm_path):
# #         st.subheader("📄 RTTM Output")
# #         with open(rttm_path) as f:
# #             st.text(f.read())

# #     # =============================
# #     # SHOW SPEAKER AUDIO
# #     # =============================
# #     st.subheader("🔊 Speaker-wise Audio")

# #     speaker_files = sorted(
# #         f for f in os.listdir(SPEAKER_DIR)
# #         if f.lower().endswith(".wav")
# #     )

# #     st.metric(" Speakers Detected", len(speaker_files))

# #     for wav in speaker_files:
# #         st.markdown(f"###  {wav.replace('.wav', '')}")
# #         st.audio(os.path.join(SPEAKER_DIR, wav))




# import os
# import sys
# import subprocess
# import streamlit as st

# # =========================
# # ENV FIX (Windows + pyannote)
# # =========================
# os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# os.environ["SB_LOCAL_STRATEGY"] = "copy"

# if not os.getenv("HF_TOKEN"):
#     st.error("HF_TOKEN not set (run: huggingface-cli login)")
#     st.stop()

# # =========================
# # PATHS
# # =========================
# UPLOAD_DIR = "uploads"
# CLEAN_DIR = "cleaned_audio"
# OUTPUT_DIR = "output"
# SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")

# REF_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

# for d in [UPLOAD_DIR, CLEAN_DIR, OUTPUT_DIR, SPEAKER_DIR]:
#     os.makedirs(d, exist_ok=True)

# # =========================
# # UI
# # =========================
# st.set_page_config(page_title="Speaker Diarization", layout="wide")
# st.title("Speaker Diarization System (Automatic N-Speakers)")

# uploaded = st.file_uploader(
#     "Upload audio file",
#     type=["wav", "mp3", "m4a", "flac", "ogg"]
# )

# raw_audio = None
# clean_audio = None

# # =========================
# # SAVE UPLOADED AUDIO
# # =========================
# if uploaded:
#     raw_audio = os.path.join(UPLOAD_DIR, uploaded.name)
#     with open(raw_audio, "wb") as f:
#         f.write(uploaded.getbuffer())

#     st.audio(raw_audio)

#     clean_audio = os.path.join(
#         CLEAN_DIR,
#         os.path.splitext(uploaded.name)[0] + "_cleaned.wav"
#     )

# # =========================
# # RUN PIPELINE
# # =========================
# if raw_audio and st.button("Run Diarization"):

#     # -------- PREPROCESS --------
#     with st.spinner("Cleaning audio..."):
#         p = subprocess.run(
#             [sys.executable, "process.py", raw_audio, clean_audio],
#             capture_output=True,
#             text=True
#         )

#     if p.returncode != 0:
#         st.error("Preprocessing failed")
#         st.code(p.stderr)
#         st.stop()

#     st.success("Cleaned audio generated")
#     st.audio(clean_audio)

#     # -------- DIARIZATION --------
#     with st.spinner("Running diarization..."):
#         d = subprocess.run(
#             [sys.executable, "diarize.py", clean_audio, OUTPUT_DIR],
#             capture_output=True,
#             text=True,
#             env=os.environ.copy()
#         )

#     if d.returncode != 0:
#         st.error("Diarization failed")
#         st.code(d.stderr)
#         st.stop()

#     st.success("Diarization completed")

#     # =========================
#     # RTTM DISPLAY
#     # =========================
#     rttm_path = os.path.join(
#         OUTPUT_DIR,
#         os.path.splitext(os.path.basename(clean_audio))[0] + ".rttm"
#     )

#     st.subheader("RTTM Output")
#     st.text(open(rttm_path).read())

#     # =========================
#     # SPEAKER AUDIO
#     # =========================
#     st.subheader("Speaker-wise Audio")

#     speaker_files = sorted(
#         f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")
#     )

#     st.metric("Speakers Detected", len(speaker_files))

#     for wav in speaker_files:
#         st.markdown(f"{wav.replace('.wav', '')}")
#         st.audio(os.path.join(SPEAKER_DIR, wav))

#     # =========================
#     # DER (DATASET ONLY)
#     # =========================
#     st.subheader("Diarization Error Rate (DER)")

#     with st.spinner("Computing DER..."):
#         der = subprocess.run(
#             [
#                 sys.executable,
#                 "der.py",
#                 os.path.basename(clean_audio),
#                 REF_RTTM_DIR
#             ],
#             capture_output=True,
#             text=True
#         )

#     if der.returncode == 0 and der.stdout.strip():
#         st.metric("DER (%)", der.stdout.strip())
#     else:
#         st.info("Reference RTTM not found — DER not computed")



import os
import sys
import subprocess
import streamlit as st

# =========================
# ENV FIX (Windows + pyannote)
# =========================
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["SB_LOCAL_STRATEGY"] = "copy"

if not os.getenv("HF_TOKEN"):
    st.error("HF_TOKEN not set (run: huggingface-cli login)")
    st.stop()

# =========================
# PATHS
# =========================
UPLOAD_DIR = "uploads"
CLEAN_DIR = "cleaned_audio"
OUTPUT_DIR = "output"
SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")

REF_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

for d in [UPLOAD_DIR, CLEAN_DIR, OUTPUT_DIR, SPEAKER_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================
# UI
# =========================
st.set_page_config(page_title="Speaker Diarization", layout="wide")
st.title("Speaker Diarization System")

uploaded = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg"]
)

st.subheader("Or record audio")
recorded_audio = st.audio_input("Record from microphone")

raw_audio = None
clean_audio = None
is_recorded = False

# =========================
# SAVE INPUT AUDIO
# =========================
if uploaded:
    raw_audio = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(raw_audio, "wb") as f:
        f.write(uploaded.getbuffer())

    st.audio(raw_audio)

    clean_audio = os.path.join(
        CLEAN_DIR,
        os.path.splitext(uploaded.name)[0] + "_cleaned.wav"
    )

elif recorded_audio:
    raw_audio = os.path.join(UPLOAD_DIR, "recorded_audio.wav")
    with open(raw_audio, "wb") as f:
        f.write(recorded_audio.getbuffer())

    st.audio(raw_audio)

    clean_audio = os.path.join(
        CLEAN_DIR,
        "recorded_audio_cleaned.wav"
    )

    is_recorded = True

# =========================
# RUN PIPELINE
# =========================
if raw_audio and st.button("Run Diarization"):

    # -------- PREPROCESS --------
    with st.spinner("Cleaning audio..."):
        p = subprocess.run(
            [sys.executable, "process.py", raw_audio, clean_audio],
            capture_output=True,
            text=True
        )

    if p.returncode != 0:
        st.error("Preprocessing failed")
        st.code(p.stderr)
        st.stop()

    st.success("Cleaned audio generated")
    st.audio(clean_audio)

    # -------- DIARIZATION --------
    with st.spinner("Running diarization..."):
        d = subprocess.run(
            [sys.executable, "diarize.py", clean_audio, OUTPUT_DIR],
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )

    if d.returncode != 0:
        st.error("Diarization failed")
        st.code(d.stderr)
        st.stop()

    st.success("Diarization completed")

    # =========================
    # RTTM DISPLAY
    # =========================
    rttm_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(os.path.basename(clean_audio))[0] + ".rttm"
    )

    st.subheader("RTTM Output")
    st.text(open(rttm_path).read())

    # =========================
    # SPEAKER SEGMENTS
    # =========================
    st.subheader("Speaker Segments")

    speaker_files = sorted(
        f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")
    )

    st.metric("Segments Detected", len(speaker_files))

    for wav in speaker_files:
        st.markdown(wav)
        st.audio(os.path.join(SPEAKER_DIR, wav))

    # =========================
    # DER (ONLY FOR DATASET AUDIO)
    # =========================
    st.subheader("Diarization Error Rate (DER)")

    if is_recorded:
        st.info("DER not computed for recorded audio (no reference RTTM).")
    else:
        with st.spinner("Computing DER..."):
            der = subprocess.run(
                [
                    sys.executable,
                    "der.py",
                    os.path.basename(clean_audio),
                    REF_RTTM_DIR
                ],
                capture_output=True,
                text=True
            )

        if der.returncode == 0 and der.stdout.strip():
            st.metric("DER (%)", der.stdout.strip())
        else:
            st.info("Reference RTTM not found — DER not computed")
