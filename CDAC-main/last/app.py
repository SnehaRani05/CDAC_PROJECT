# # # # import streamlit as st
# # # # import os
# # # # import tempfile
# # # # import soundfile as sf
# # # # import numpy as np

# # # # from pyannote.audio import Pipeline, Model

# # # # # ===============================
# # # # # CONFIG
# # # # # ===============================
# # # # CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_22\checkpoints\epoch=4-step=2440.ckpt"
# # # # OUTPUT_DIR = "outputs"
# # # # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # # st.set_page_config(page_title="Speaker Diarization", layout="centered")
# # # # st.title("🎙️ Speaker Diarization App")

# # # # # ===============================
# # # # # LOAD PIPELINE (CACHED)
# # # # # ===============================
# # # # @st.cache_resource
# # # # def load_pipeline():
# # # #     # Load your trained segmentation model
# # # #     segmentation_model = Model.from_pretrained(
# # # #         CHECKPOINT,
# # # #         strict=False
# # # #     )

# # # #     # Load official diarization pipeline
# # # #     pipeline = Pipeline.from_pretrained(
# # # #         "pyannote/speaker-diarization-3.1",
# # # #         use_auth_token=False
# # # #     )

# # # #     # 🔴 Replace ONLY segmentation model
# # # #     pipeline._segmentation.model = segmentation_model

# # # #     # 🔴 REQUIRED parameters (fixes ParamDict errors)
# # # #     pipeline.segmentation.threshold = 0.4
# # # #     pipeline.segmentation.min_duration_on = 0.0
# # # #     pipeline.segmentation.min_duration_off = 0.0

# # # #     # 🔴 Allow more speakers
# # # #     pipeline.clustering.min_num_speakers = 1
# # # #     pipeline.clustering.max_num_speakers = 10

# # # #     return pipeline


# # # # pipeline = load_pipeline()

# # # # # ===============================
# # # # # AUDIO INPUT
# # # # # ===============================
# # # # st.subheader("Upload or Record Audio")

# # # # uploaded_file = st.file_uploader(
# # # #     "Upload WAV audio",
# # # #     type=["wav"]
# # # # )

# # # # audio_path = None

# # # # if uploaded_file:
# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
# # # #         tmp.write(uploaded_file.read())
# # # #         audio_path = tmp.name

# # # #     st.audio(audio_path)

# # # # # ===============================
# # # # # RUN DIARIZATION
# # # # # ===============================
# # # # if audio_path and st.button("▶ Run Diarization"):
# # # #     with st.spinner("Running diarization..."):
# # # #         diarization = pipeline(audio_path)

# # # #     st.success("Diarization complete!")

# # # #     # ===============================
# # # #     # DISPLAY RESULTS
# # # #     # ===============================
# # # #     st.subheader("Speaker Segments")

# # # #     segments = []
# # # #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# # # #         segments.append((turn.start, turn.end, speaker))
# # # #         st.write(f"🗣️ **{speaker}** : {turn.start:.2f}s → {turn.end:.2f}s")

# # # #     # ===============================
# # # #     # SAVE RTTM
# # # #     # ===============================
# # # #     rttm_path = os.path.join(OUTPUT_DIR, "predicted.rttm")
# # # #     with open(rttm_path, "w") as f:
# # # #         diarization.write_rttm(f)

# # # #     st.success(f"RTTM saved at: `{rttm_path}`")

# # # #     with open(rttm_path, "r") as f:
# # # #         st.download_button(
# # # #             label="⬇ Download RTTM",
# # # #             data=f.read(),
# # # #             file_name="predicted.rttm",
# # # #             mime="text/plain"
# # # #         )



# # # import os
# # # import tempfile
# # # import streamlit as st
# # # import sounddevice as sd
# # # import soundfile as sf
# # # import numpy as np

# # # from pyannote.audio import Pipeline, Model
# # # from pyannote.metrics.diarization import DiarizationErrorRate
# # # from pyannote.database.util import load_rttm

# # # # =========================
# # # # CONFIG
# # # # =========================
# # # CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_22\checkpoints\epoch=4-step=2440.ckpt"
# # # REFERENCE_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\last\dataset\rttm"

# # # UPLOAD_DIR = "uploads"
# # # RECORD_DIR = "recordings"
# # # OUT_DIR = "infer/outputs"

# # # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # # os.makedirs(RECORD_DIR, exist_ok=True)
# # # os.makedirs(OUT_DIR, exist_ok=True)

# # # st.set_page_config(page_title="Speaker Diarization App", layout="wide")
# # # st.title("🎙️ Speaker Diarization (Upload / Record)")

# # # # =========================
# # # # LOAD MODELS (ONCE)
# # # # =========================
# # # @st.cache_resource
# # # def load_pipeline():
# # #     segmentation_model = Model.from_pretrained(CHECKPOINT, strict=False)

# # #     pipeline = Pipeline.from_pretrained(
# # #         "pyannote/speaker-diarization-3.1",
# # #         use_auth_token=False
# # #     )

# # #     # IMPORTANT FIX (threshold error)
# # #     pipeline.segmentation = segmentation_model
# # #     pipeline.segmentation.threshold = 0.5
# # #     pipeline.segmentation.min_duration_off = 0.1

# # #     return pipeline

# # # pipeline = load_pipeline()

# # # # =========================
# # # # AUDIO INPUT
# # # # =========================
# # # st.sidebar.header("Audio Input")

# # # mode = st.sidebar.radio("Choose input method", ["Upload audio", "Record from mic"])

# # # audio_path = None
# # # audio_name = None

# # # # ---------- UPLOAD ----------
# # # if mode == "Upload audio":
# # #     uploaded = st.file_uploader("Upload WAV audio", type=["wav"])

# # #     if uploaded:
# # #         audio_name = os.path.splitext(uploaded.name)[0]
# # #         audio_path = os.path.join(UPLOAD_DIR, uploaded.name)

# # #         with open(audio_path, "wb") as f:
# # #             f.write(uploaded.read())

# # #         st.success(f"Uploaded: {uploaded.name}")

# # # # ---------- RECORD ----------
# # # if mode == "Record from mic":
# # #     duration = st.slider("Recording duration (seconds)", 3, 30, 10)
# # #     audio_name = st.text_input("Recording name", "recorded_audio")

# # #     if st.button("🎙️ Start Recording"):
# # #         fs = 16000
# # #         st.info("Recording...")
# # #         audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# # #         sd.wait()

# # #         audio_path = os.path.join(RECORD_DIR, f"{audio_name}.wav")
# # #         sf.write(audio_path, audio, fs)

# # #         st.success(f"Recording saved: {audio_path}")

# # # # =========================
# # # # RUN DIARIZATION
# # # # =========================
# # # if audio_path and st.button("▶️ Run Diarization"):

# # #     st.info("Running diarization...")
# # #     diarization = pipeline(audio_path)

# # #     # ---------- DISPLAY RESULTS ----------
# # #     st.subheader("🗣️ Speaker Segments")
# # #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# # #         st.write(f"{turn.start:.2f}s → {turn.end:.2f}s : {speaker}")

# # #     # ---------- SAVE RTTM ----------
# # #     pred_rttm = os.path.join(OUT_DIR, f"{audio_name}.rttm")
# # #     with open(pred_rttm, "w") as f:
# # #         diarization.write_rttm(f)

# # #     st.success(f"Predicted RTTM saved: {pred_rttm}")

# # #     # =========================
# # #     # DER COMPUTATION
# # #     # =========================
# # #     ref_rttm_path = os.path.join(REFERENCE_RTTM_DIR, f"{audio_name}.rttm")

# # #     st.subheader("📊 Diarization Error Rate (DER)")

# # #     if not os.path.exists(ref_rttm_path):
# # #         st.warning(f"Reference RTTM not found: {ref_rttm_path}")
# # #     else:
# # #         reference = load_rttm(ref_rttm_path)
# # #         hypothesis = load_rttm(pred_rttm)

# # #         uri = list(reference.keys())[0]

# # #         der = DiarizationErrorRate()
# # #         score = der(reference[uri], hypothesis[uri])
# # #         der_percent = score * 100

# # #         st.success(f"DER = {der_percent:.2f}%")

# # # # =========================
# # # # FOOTER
# # # # =========================
# # # st.markdown("---")
# # # st.caption("CDAC Speaker Diarization | PyAnnote + Streamlit")



# # import os
# # import streamlit as st
# # import sounddevice as sd
# # import soundfile as sf
# # import numpy as np
# # import matplotlib.pyplot as plt

# # from pyannote.audio import Pipeline, Model
# # from pyannote.metrics.diarization import DiarizationErrorRate
# # from pyannote.database.util import load_rttm

# # # =========================
# # # CONFIG
# # # =========================
# # CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_22\checkpoints\epoch=4-step=2440.ckpt"
# # REFERENCE_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\last\dataset\rttm"

# # UPLOAD_DIR = "uploads"
# # RECORD_DIR = "recordings"
# # OUT_DIR = "infer/outputs"

# # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # os.makedirs(RECORD_DIR, exist_ok=True)
# # os.makedirs(OUT_DIR, exist_ok=True)

# # # =========================
# # # STREAMLIT STATE INIT
# # # =========================
# # if "audio_path" not in st.session_state:
# #     st.session_state.audio_path = None

# # if "audio_name" not in st.session_state:
# #     st.session_state.audio_name = None

# # if "diarization" not in st.session_state:
# #     st.session_state.diarization = None

# # # =========================
# # # PAGE SETUP
# # # =========================
# # st.set_page_config(page_title="Speaker Diarization", layout="wide")
# # st.title("🎙️ Speaker Diarization (Upload / Record)")

# # # =========================
# # # LOAD PIPELINE (ONCE)
# # # =========================
# # @st.cache_resource
# # def load_pipeline():
# #     segmentation_model = Model.from_pretrained(CHECKPOINT, strict=False)

# #     pipeline = Pipeline.from_pretrained(
# #         "pyannote/speaker-diarization-3.1",
# #         use_auth_token=False
# #     )

# #     # ✅ correct replacement (NO ParamDict error)
# #     pipeline._segmentation.model = segmentation_model

# #     # safe defaults
# #     pipeline.segmentation.threshold = 0.5
# #     pipeline.segmentation.min_duration_on = 0.0
# #     pipeline.segmentation.min_duration_off = 0.0

# #     pipeline.clustering.min_num_speakers = 1
# #     pipeline.clustering.max_num_speakers = 10

# #     return pipeline

# # pipeline = load_pipeline()

# # # =========================
# # # AUDIO INPUT
# # # =========================
# # st.sidebar.header("Audio Input")
# # mode = st.sidebar.radio("Choose input method", ["Upload audio", "Record from mic"])

# # # ---------- UPLOAD ----------
# # if mode == "Upload audio":
# #     uploaded = st.file_uploader("Upload WAV file", type=["wav"])

# #     if uploaded:
# #         st.session_state.audio_name = os.path.splitext(uploaded.name)[0]
# #         st.session_state.audio_path = os.path.join(UPLOAD_DIR, uploaded.name)

# #         with open(st.session_state.audio_path, "wb") as f:
# #             f.write(uploaded.read())

# #         st.audio(st.session_state.audio_path)
# #         st.success(f"Uploaded: {uploaded.name}")

# # # ---------- RECORD ----------
# # if mode == "Record from mic":
# #     duration = st.slider("Recording duration (seconds)", 3, 30, 10)
# #     name = st.text_input("Recording name", "recorded_audio")

# #     if st.button("🎙️ Start Recording"):
# #         fs = 16000
# #         st.info("Recording...")
# #         audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# #         sd.wait()

# #         st.session_state.audio_name = name
# #         st.session_state.audio_path = os.path.join(RECORD_DIR, f"{name}.wav")

# #         sf.write(st.session_state.audio_path, audio, fs)
# #         st.audio(st.session_state.audio_path)
# #         st.success("Recording completed")

# # # =========================
# # # RUN DIARIZATION
# # # =========================
# # if st.session_state.audio_path and st.button("▶️ Run Diarization"):
# #     with st.spinner("Running diarization..."):
# #         st.session_state.diarization = pipeline(st.session_state.audio_path)

# # # =========================
# # # SHOW RESULTS (ALWAYS)
# # # =========================
# # if st.session_state.diarization is not None:
# #     diarization = st.session_state.diarization

# #     st.subheader("🗣️ Speaker Segments")

# #     segments = []
# #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# #         segments.append((turn.start, turn.end, speaker))
# #         st.write(f"{turn.start:.2f}s → {turn.end:.2f}s : **{speaker}**")

# #     # =========================
# #     # SAVE RTTM (CLEAN NAME)
# #     # =========================
# #     pred_rttm = os.path.join(OUT_DIR, f"{st.session_state.audio_name}.rttm")
# #     with open(pred_rttm, "w") as f:
# #         diarization.write_rttm(f)

# #     st.success(f"Predicted RTTM saved: {pred_rttm}")

# #     # =========================
# #     # TIME vs SPEAKER PLOT
# #     # =========================
# #     st.subheader("📊 Time vs Speaker")

# #     speakers = sorted(set(s for _, _, s in segments))
# #     speaker_to_y = {s: i for i, s in enumerate(speakers)}

# #     fig, ax = plt.subplots(figsize=(12, 3))

# #     for start, end, speaker in segments:
# #         ax.barh(
# #             speaker_to_y[speaker],
# #             end - start,
# #             left=start,
# #             height=0.5
# #         )

# #     ax.set_yticks(list(speaker_to_y.values()))
# #     ax.set_yticklabels(speakers)
# #     ax.set_xlabel("Time (seconds)")
# #     ax.set_ylabel("Speaker")
# #     ax.set_title("Speaker Diarization Timeline")
# #     ax.grid(True)

# #     st.pyplot(fig)

# #     # =========================
# #     # DER COMPUTATION
# #     # =========================
# #     st.subheader("📉 Diarization Error Rate (DER)")

# #     ref_rttm = os.path.join(
# #         REFERENCE_RTTM_DIR,
# #         f"{st.session_state.audio_name}.rttm"
# #     )

# #     if not os.path.exists(ref_rttm):
# #         st.warning(f"Reference RTTM not found:\n{ref_rttm}")
# #     else:
# #         reference = load_rttm(ref_rttm)
# #         hypothesis = load_rttm(pred_rttm)

# #         uri = list(reference.keys())[0]
# #         der = DiarizationErrorRate()
# #         score = der(reference[uri], hypothesis[uri]) * 100

# #         st.success(f"✅ DER = **{score:.3f}%**")

# #         with open(pred_rttm, "r") as f:
# #             st.download_button(
# #                 "⬇️ Download Predicted RTTM",
# #                 f.read(),
# #                 file_name=f"{st.session_state.audio_name}.rttm",
# #                 mime="text/plain"
# #             )

# # # =========================
# # # FOOTER
# # # =========================
# # st.markdown("---")
# # st.caption("CDAC Internship | Speaker Diarization | PyAnnote + Streamlit")




# # import streamlit as st
# # import os
# # import tempfile
# # import soundfile as sf
# # import numpy as np

# # from pyannote.audio import Pipeline, Model

# # # ===============================
# # # CONFIG
# # # ===============================
# # CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_22\checkpoints\epoch=4-step=2440.ckpt"
# # OUTPUT_DIR = "outputs"
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # st.set_page_config(page_title="Speaker Diarization", layout="centered")
# # st.title("🎙️ Speaker Diarization App")

# # # ===============================
# # # LOAD PIPELINE (CACHED)
# # # ===============================
# # @st.cache_resource
# # def load_pipeline():
# #     # Load your trained segmentation model
# #     segmentation_model = Model.from_pretrained(
# #         CHECKPOINT,
# #         strict=False
# #     )

# #     # Load official diarization pipeline
# #     pipeline = Pipeline.from_pretrained(
# #         "pyannote/speaker-diarization-3.1",
# #         use_auth_token=False
# #     )

# #     # 🔴 Replace ONLY segmentation model
# #     pipeline._segmentation.model = segmentation_model

# #     # 🔴 REQUIRED parameters (fixes ParamDict errors)
# #     pipeline.segmentation.threshold = 0.4
# #     pipeline.segmentation.min_duration_on = 0.0
# #     pipeline.segmentation.min_duration_off = 0.0

# #     # 🔴 Allow more speakers
# #     pipeline.clustering.min_num_speakers = 1
# #     pipeline.clustering.max_num_speakers = 10

# #     return pipeline


# # pipeline = load_pipeline()

# # # ===============================
# # # AUDIO INPUT
# # # ===============================
# # st.subheader("Upload or Record Audio")

# # uploaded_file = st.file_uploader(
# #     "Upload WAV audio",
# #     type=["wav"]
# # )

# # audio_path = None

# # if uploaded_file:
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
# #         tmp.write(uploaded_file.read())
# #         audio_path = tmp.name

# #     st.audio(audio_path)

# # # ===============================
# # # RUN DIARIZATION
# # # ===============================
# # if audio_path and st.button("▶ Run Diarization"):
# #     with st.spinner("Running diarization..."):
# #         diarization = pipeline(audio_path)

# #     st.success("Diarization complete!")

# #     # ===============================
# #     # DISPLAY RESULTS
# #     # ===============================
# #     st.subheader("Speaker Segments")

# #     segments = []
# #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# #         segments.append((turn.start, turn.end, speaker))
# #         st.write(f"🗣️ **{speaker}** : {turn.start:.2f}s → {turn.end:.2f}s")

# #     # ===============================
# #     # SAVE RTTM
# #     # ===============================
# #     rttm_path = os.path.join(OUTPUT_DIR, "predicted.rttm")
# #     with open(rttm_path, "w") as f:
# #         diarization.write_rttm(f)

# #     st.success(f"RTTM saved at: `{rttm_path}`")

# #     with open(rttm_path, "r") as f:
# #         st.download_button(
# #             label="⬇ Download RTTM",
# #             data=f.read(),
# #             file_name="predicted.rttm",
# #             mime="text/plain"
# #         )



# import os
# import tempfile
# import streamlit as st
# import sounddevice as sd
# import soundfile as sf
# import numpy as np

# from pyannote.audio import Pipeline, Model
# from pyannote.metrics.diarization import DiarizationErrorRate
# from pyannote.database.util import load_rttm

# # =========================
# # CONFIG
# # =========================
# CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_24\checkpoints\epoch=14-step=7320.ckpt"
# REFERENCE_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\last\dataset\rttm"

# UPLOAD_DIR = "uploads"
# RECORD_DIR = "recordings"
# OUT_DIR = "infer/outputs"

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(RECORD_DIR, exist_ok=True)
# os.makedirs(OUT_DIR, exist_ok=True)

# st.set_page_config(page_title="Speaker Diarization App", layout="wide")
# st.title("🎙️ Speaker Diarization (Upload / Record)")

# # =========================
# # LOAD MODELS (ONCE)
# # =========================
# @st.cache_resource
# def load_pipeline():
#     segmentation_model = Model.from_pretrained(CHECKPOINT, strict=False)

#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=False
#     )

#     # IMPORTANT FIX (threshold error)
#     pipeline.segmentation = segmentation_model
#     pipeline.segmentation.threshold = 0.5
#     pipeline.segmentation.min_duration_off = 0.1

#     return pipeline

# pipeline = load_pipeline()

# # =========================
# # AUDIO INPUT
# # =========================
# st.sidebar.header("Audio Input")

# mode = st.sidebar.radio("Choose input method", ["Upload audio", "Record from mic"])

# audio_path = None
# audio_name = None

# # ---------- UPLOAD ----------
# if mode == "Upload audio":
#     uploaded = st.file_uploader("Upload WAV audio", type=["wav"])

#     if uploaded:
#         audio_name = os.path.splitext(uploaded.name)[0]
#         audio_path = os.path.join(UPLOAD_DIR, uploaded.name)

#         with open(audio_path, "wb") as f:
#             f.write(uploaded.read())

#         st.success(f"Uploaded: {uploaded.name}")

# # ---------- RECORD ----------
# if mode == "Record from mic":
#     duration = st.slider("Recording duration (seconds)", 3, 30, 10)
#     audio_name = st.text_input("Recording name", "recorded_audio")

#     if st.button("🎙️ Start Recording"):
#         fs = 16000
#         st.info("Recording...")
#         audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#         sd.wait()

#         audio_path = os.path.join(RECORD_DIR, f"{audio_name}.wav")
#         sf.write(audio_path, audio, fs)

#         st.success(f"Recording saved: {audio_path}")

# # =========================
# # RUN DIARIZATION
# # =========================
# if audio_path and st.button("▶️ Run Diarization"):

#     st.info("Running diarization...")
#     diarization = pipeline(audio_path)

#     # ---------- DISPLAY RESULTS ----------
#     st.subheader("🗣️ Speaker Segments")
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         st.write(f"{turn.start:.2f}s → {turn.end:.2f}s : {speaker}")

#     # ---------- SAVE RTTM ----------
#     pred_rttm = os.path.join(OUT_DIR, f"{audio_name}.rttm")
#     with open(pred_rttm, "w") as f:
#         diarization.write_rttm(f)

#     st.success(f"Predicted RTTM saved: {pred_rttm}")

    

#     # =========================
#     # DER COMPUTATION
#     # =========================
#     ref_rttm_path = os.path.join(REFERENCE_RTTM_DIR, f"{audio_name}.rttm")

#     st.subheader("📊 Diarization Error Rate (DER)")

#     if not os.path.exists(ref_rttm_path):
#         st.warning(f"Reference RTTM not found: {ref_rttm_path}")
#     else:
#         reference = load_rttm(ref_rttm_path)
#         hypothesis = load_rttm(pred_rttm)

#         uri = list(reference.keys())[0]

#         der = DiarizationErrorRate()
#         score = der(reference[uri], hypothesis[uri])
#         der_percent = score * 100

#         st.success(f"DER = {der_percent:.2f}%")

# # =========================
# # FOOTER
# # =========================
# st.markdown("---")
# st.caption("CDAC Speaker Diarization | PyAnnote + Streamlit")





import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from pyannote.audio import Pipeline, Model
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

# =========================
# CONFIG
# =========================
CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_24\checkpoints\epoch=14-step=7320.ckpt"
REFERENCE_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\last\dataset\rttm"

UPLOAD_DIR = "uploads"
RECORD_DIR = "recordings"
OUT_DIR = "infer/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# STREAMLIT STATE
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_name" not in st.session_state:
    st.session_state.audio_name = None

if "diarization" not in st.session_state:
    st.session_state.diarization = None

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Speaker Diarization App", layout="wide")
st.title("🎙️ Speaker Diarization (Upload / Record)")

# =========================
# LOAD PIPELINE (ONCE)
# =========================
@st.cache_resource
def load_pipeline():
    segmentation_model = Model.from_pretrained(
        CHECKPOINT,
        strict=False
    )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=False
    )

    # ✅ Correct replacement (NO ParamDict error)
    pipeline._segmentation.model = segmentation_model

    pipeline.segmentation.threshold = 0.5
    pipeline.segmentation.min_duration_on = 0.0
    pipeline.segmentation.min_duration_off = 0.0

    pipeline.clustering.min_num_speakers = 1
    pipeline.clustering.max_num_speakers = 10

    return pipeline

pipeline = load_pipeline()

# =========================
# AUDIO INPUT
# =========================
st.sidebar.header("Audio Input")
mode = st.sidebar.radio("Choose input method", ["Upload audio", "Record from mic"])

# ---------- UPLOAD ----------
if mode == "Upload audio":
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded:
        st.session_state.audio_name = os.path.splitext(uploaded.name)[0]
        st.session_state.audio_path = os.path.join(UPLOAD_DIR, uploaded.name)

        with open(st.session_state.audio_path, "wb") as f:
            f.write(uploaded.read())

        st.audio(st.session_state.audio_path)
        st.success(f"Uploaded: {uploaded.name}")

# ---------- RECORD ----------
if mode == "Record from mic":
    duration = st.slider("Recording duration (seconds)", 3, 30, 10)
    name = st.text_input("Recording name", "recorded_audio")

    if st.button("🎙️ Start Recording"):
        fs = 16000
        st.info("Recording...")
        audio = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype=np.float32
        )
        sd.wait()

        st.session_state.audio_name = name
        st.session_state.audio_path = os.path.join(
            RECORD_DIR, f"{name}.wav"
        )

        sf.write(st.session_state.audio_path, audio, fs)
        st.audio(st.session_state.audio_path)

        st.success("Recording completed")

# =========================
# RUN DIARIZATION
# =========================
if st.session_state.audio_path and st.button("▶️ Run Diarization"):
    with st.spinner("Running diarization..."):
        st.session_state.diarization = pipeline(
            st.session_state.audio_path
        )

# =========================
# SHOW RESULTS
# =========================
if st.session_state.diarization is not None:
    diarization = st.session_state.diarization

    st.subheader("🗣️ Speaker Segments")

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
        st.write(
            f"{turn.start:.2f}s → {turn.end:.2f}s : **{speaker}**"
        )

    # =========================
    # SAVE RTTM
    # =========================
    pred_rttm = os.path.join(
        OUT_DIR, f"{st.session_state.audio_name}.rttm"
    )

    with open(pred_rttm, "w") as f:
        diarization.write_rttm(f)

    st.success(f"Predicted RTTM saved at: {pred_rttm}")

    with open(pred_rttm, "r") as f:
        st.download_button(
            "⬇️ Download Predicted RTTM",
            f.read(),
            file_name=f"{st.session_state.audio_name}.rttm",
            mime="text/plain"
        )

    # =========================
    # TIME vs SPEAKER PLOT
    # =========================
    st.subheader("📊 Time vs Speaker")

    speakers = sorted(set(s for _, _, s in segments))
    speaker_to_y = {s: i for i, s in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, 3))

    for start, end, speaker in segments:
        ax.barh(
            speaker_to_y[speaker],
            end - start,
            left=start,
            height=0.5
        )

    ax.set_yticks(list(speaker_to_y.values()))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title("Speaker Diarization Timeline")
    ax.grid(True)

    st.pyplot(fig)

    # =========================
    # DER COMPUTATION
    # =========================
    st.subheader("📉 Diarization Error Rate (DER)")

    ref_rttm = os.path.join(
        REFERENCE_RTTM_DIR,
        f"{st.session_state.audio_name}.rttm"
    )

    if not os.path.exists(ref_rttm):
        st.warning(f"Reference RTTM not found:\n{ref_rttm}")
    else:
        reference = load_rttm(ref_rttm)
        hypothesis = load_rttm(pred_rttm)

        uri = list(reference.keys())[0]
        der = DiarizationErrorRate()
        score = der(reference[uri], hypothesis[uri]) * 100

        st.success(f"✅ DER = **{score:.3f}%**")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("CDAC Internship | Speaker Diarization | PyAnnote + Streamlit")
