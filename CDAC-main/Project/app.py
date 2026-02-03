# # # # # # # # # # import streamlit as st
# # # # # # # # # # import os
# # # # # # # # # # from diarize import run_diarization
# # # # # # # # # # from pyannote.database.util import load_rttm
# # # # # # # # # # from pyannote.metrics.diarization import DiarizationErrorRate
# # # # # # # # # # from paths import AUDIO_DIR, RTTM_DIR, HYP_RTTM

# # # # # # # # # # st.set_page_config(layout="wide")
# # # # # # # # # # st.title("Speaker Diarization – CDAC Dataset")

# # # # # # # # # # audio_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
# # # # # # # # # # selected = st.selectbox("Select audio", audio_files)

# # # # # # # # # # if st.button("Run Diarization"):

# # # # # # # # # #     audio_path = os.path.join(AUDIO_DIR, selected)
# # # # # # # # # #     run_diarization(audio_path)

# # # # # # # # # #     st.success("Diarization completed")

# # # # # # # # # #     # Speaker playback
# # # # # # # # # #     st.subheader("Speaker Segments")
# # # # # # # # # #     for f in os.listdir("output/speakers"):
# # # # # # # # # #         st.audio(os.path.join("output/speakers", f))

# # # # # # # # # #     # DER
# # # # # # # # # #     ref_rttm = os.path.join(
# # # # # # # # # #         RTTM_DIR,
# # # # # # # # # #         os.path.splitext(selected)[0] + ".rttm"
# # # # # # # # # #     )

# # # # # # # # # #     ref = load_rttm(ref_rttm)
# # # # # # # # # #     file_id = list(ref.keys())[0]
# # # # # # # # # #     hyp = load_rttm(HYP_RTTM)[file_id]

# # # # # # # # # #     der = DiarizationErrorRate()(ref[file_id], hyp)
# # # # # # # # # #     st.metric("DER", f"{der*100:.2f}%")


# # # # # # # # # import streamlit as st
# # # # # # # # # import os
# # # # # # # # # from collections import defaultdict

# # # # # # # # # from diarize import run_diarization
# # # # # # # # # from pyannote.database.util import load_rttm
# # # # # # # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # # # # # # ================= PATHS =================

# # # # # # # # # BASE_PATH = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject"

# # # # # # # # # AUDIO_DIR = os.path.join(BASE_PATH, "processed", "dataset", "audio")
# # # # # # # # # RTTM_DIR = os.path.join(BASE_PATH, "processed", "dataset", "rttm")

# # # # # # # # # OUTPUT_DIR = os.path.join(BASE_PATH, "output")
# # # # # # # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # # # # # # HYP_RTTM = os.path.join(OUTPUT_DIR, "hypothesis.rttm")

# # # # # # # # # os.makedirs(OUTPUT_DIR, exist_ok=True)
# # # # # # # # # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # # # # # # # ================= PAGE =================

# # # # # # # # # st.set_page_config(layout="wide")
# # # # # # # # # st.title("🎤 Speaker Diarization (CDAC Dataset)")
# # # # # # # # # st.caption("Neural diarization • Speaker segments • DER evaluation")

# # # # # # # # # # ================= UTIL =================

# # # # # # # # # def parse_rttm(rttm_path):
# # # # # # # # #     segments = []
# # # # # # # # #     with open(rttm_path) as f:
# # # # # # # # #         for line in f:
# # # # # # # # #             p = line.strip().split()
# # # # # # # # #             start = float(p[3])
# # # # # # # # #             dur = float(p[4])
# # # # # # # # #             speaker = p[7]
# # # # # # # # #             segments.append({
# # # # # # # # #                 "speaker": speaker,
# # # # # # # # #                 "start": start,
# # # # # # # # #                 "end": start + dur,
# # # # # # # # #                 "duration": dur
# # # # # # # # #             })
# # # # # # # # #     return segments


# # # # # # # # # # ================= AUDIO SELECTION =================

# # # # # # # # # audio_files = sorted(
# # # # # # # # #     f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")
# # # # # # # # # )

# # # # # # # # # selected_audio = st.selectbox("Select audio file", audio_files)

# # # # # # # # # audio_path = os.path.join(AUDIO_DIR, selected_audio)
# # # # # # # # # st.audio(audio_path)

# # # # # # # # # # ================= RUN DIARIZATION =================

# # # # # # # # # if st.button("Run Diarization"):

# # # # # # # # #     with st.spinner("Running diarization..."):
# # # # # # # # #         run_diarization(audio_path)

# # # # # # # # #     st.success("Diarization completed")

# # # # # # # # #     # ================= SHOW SEGMENTS =================

# # # # # # # # #     st.subheader("Detected Speaker Segments")

# # # # # # # # #     if not os.path.exists(HYP_RTTM):
# # # # # # # # #         st.error("Hypothesis RTTM not found")
# # # # # # # # #         st.stop()

# # # # # # # # #     segments = parse_rttm(HYP_RTTM)

# # # # # # # # #     speaker_files = sorted(
# # # # # # # # #         f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")
# # # # # # # # #     )

# # # # # # # # #     if not segments or not speaker_files:
# # # # # # # # #         st.warning("No segments created.")
# # # # # # # # #         st.stop()

# # # # # # # # #     # Group segments by speaker
# # # # # # # # #     grouped = defaultdict(list)
# # # # # # # # #     for seg, wav in zip(segments, speaker_files):
# # # # # # # # #         grouped[seg["speaker"]].append((seg, wav))

# # # # # # # # #     for speaker, items in grouped.items():
# # # # # # # # #         st.markdown(f"## 🎤 {speaker}")

# # # # # # # # #         for i, (seg, wav) in enumerate(items):
# # # # # # # # #             st.markdown(
# # # # # # # # #                 f"""
# # # # # # # # #                 **Segment {i+1}**  
# # # # # # # # #                 ⏱️ {seg['start']:.2f}s → {seg['end']:.2f}s  
# # # # # # # # #                 🧮 Duration: {seg['duration']:.2f}s
# # # # # # # # #                 """
# # # # # # # # #             )
# # # # # # # # #             st.audio(os.path.join(SPEAKER_DIR, wav))

# # # # # # # # #     # ================= DER =================

# # # # # # # # #     st.subheader("📊 Diarization Error Rate (DER)")

# # # # # # # # #     ref_rttm = os.path.join(
# # # # # # # # #         RTTM_DIR,
# # # # # # # # #         os.path.splitext(selected_audio)[0] + ".rttm"
# # # # # # # # #     )

# # # # # # # # #     if not os.path.exists(ref_rttm):
# # # # # # # # #         st.warning("Reference RTTM not found — DER skipped.")
# # # # # # # # #     else:
# # # # # # # # #         ref = load_rttm(ref_rttm)
# # # # # # # # #         file_id = list(ref.keys())[0]

# # # # # # # # #         hyp = load_rttm(HYP_RTTM)[file_id]

# # # # # # # # #         metric = DiarizationErrorRate(collar=0.25)
# # # # # # # # #         der = metric(ref[file_id], hyp)

# # # # # # # # #         st.metric("DER", f"{der * 100:.2f}%")



# # # # # # # # import streamlit as st
# # # # # # # # import os
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # import pandas as pd

# # # # # # # # from diarize import run_diarization
# # # # # # # # from pyannote.database.util import load_rttm
# # # # # # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # # # # # ================= PATHS =================

# # # # # # # # BASE_PATH = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject"

# # # # # # # # AUDIO_DIR = os.path.join(BASE_PATH, "processed", "dataset", "audio")
# # # # # # # # RTTM_DIR = os.path.join(BASE_PATH, "processed", "dataset", "rttm")

# # # # # # # # OUTPUT_DIR = os.path.join(BASE_PATH, "output")
# # # # # # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # # # # # HYP_RTTM = os.path.join(OUTPUT_DIR, "hypothesis.rttm")

# # # # # # # # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # # # # # # ================= PAGE =================

# # # # # # # # st.set_page_config(layout="wide")
# # # # # # # # st.title("🎤 Speaker Diarization – CDAC Dataset")
# # # # # # # # st.caption("Segment-accurate diarization • Timeline • DER")

# # # # # # # # # ================= UTILITIES =================

# # # # # # # # def parse_rttm(rttm_path):
# # # # # # # #     segments = []
# # # # # # # #     with open(rttm_path) as f:
# # # # # # # #         for line in f:
# # # # # # # #             p = line.strip().split()
# # # # # # # #             start = float(p[3])
# # # # # # # #             dur = float(p[4])
# # # # # # # #             speaker = p[7]
# # # # # # # #             segments.append({
# # # # # # # #                 "speaker": speaker,
# # # # # # # #                 "start": start,
# # # # # # # #                 "end": start + dur,
# # # # # # # #                 "duration": dur
# # # # # # # #             })
# # # # # # # #     return segments


# # # # # # # # def segment_index(filename):
# # # # # # # #     # SPEAKER_00_12.wav -> 12
# # # # # # # #     return int(os.path.splitext(filename)[0].split("_")[-1])


# # # # # # # # def plot_timeline(segments, title):
# # # # # # # #     if not segments:
# # # # # # # #         st.warning("No segments to plot.")
# # # # # # # #         return

# # # # # # # #     df = pd.DataFrame(segments)
# # # # # # # #     speakers = sorted(df["speaker"].unique())
# # # # # # # #     y_map = {spk: i for i, spk in enumerate(speakers)}

# # # # # # # #     fig, ax = plt.subplots(figsize=(14, 3))

# # # # # # # #     for _, row in df.iterrows():
# # # # # # # #         ax.barh(
# # # # # # # #             y_map[row["speaker"]],
# # # # # # # #             row["end"] - row["start"],
# # # # # # # #             left=row["start"]
# # # # # # # #         )

# # # # # # # #     ax.set_yticks(list(y_map.values()))
# # # # # # # #     ax.set_yticklabels(list(y_map.keys()))
# # # # # # # #     ax.set_xlabel("Time (seconds)")
# # # # # # # #     ax.set_ylabel("Speaker")
# # # # # # # #     ax.set_title(title)
# # # # # # # #     ax.grid(axis="x", linestyle="--", alpha=0.5)

# # # # # # # #     st.pyplot(fig)

# # # # # # # # # ================= AUDIO SELECTION =================

# # # # # # # # audio_files = sorted(
# # # # # # # #     f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")
# # # # # # # # )

# # # # # # # # selected_audio = st.selectbox("Select audio file", audio_files)
# # # # # # # # audio_path = os.path.join(AUDIO_DIR, selected_audio)

# # # # # # # # st.audio(audio_path)

# # # # # # # # # ================= RUN DIARIZATION =================

# # # # # # # # if st.button("Run Diarization"):

# # # # # # # #     with st.spinner("Running diarization..."):
# # # # # # # #         run_diarization(audio_path)

# # # # # # # #     st.success("Diarization completed")

# # # # # # # #     # ================= LOAD OUTPUT =================

# # # # # # # #     if not os.path.exists(HYP_RTTM):
# # # # # # # #         st.error("Hypothesis RTTM not found.")
# # # # # # # #         st.stop()

# # # # # # # #     segments = parse_rttm(HYP_RTTM)

# # # # # # # #     speaker_files = sorted(
# # # # # # # #         [f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")],
# # # # # # # #         key=segment_index
# # # # # # # #     )

# # # # # # # #     if not segments or not speaker_files:
# # # # # # # #         st.warning("No speaker segments created.")
# # # # # # # #         st.stop()

# # # # # # # #     # ================= SHOW SEGMENTS (NO MERGING) =================

# # # # # # # #     st.subheader("🎧 Detected Speaker Segments")

# # # # # # # #     for i, seg in enumerate(segments):

# # # # # # # #         st.markdown(
# # # # # # # #             f"""
# # # # # # # #             ### 🎤 {seg['speaker']} — Segment {i+1}
# # # # # # # #             ⏱️ {seg['start']:.2f}s → {seg['end']:.2f}s  
# # # # # # # #             🧮 Duration: {seg['duration']:.2f}s
# # # # # # # #             """
# # # # # # # #         )

# # # # # # # #         if i < len(speaker_files):
# # # # # # # #             st.audio(os.path.join(SPEAKER_DIR, speaker_files[i]))

# # # # # # # #     # ================= TIMELINE GRAPHS =================

# # # # # # # #     st.subheader("📈 Predicted Speaker Timeline")
# # # # # # # #     plot_timeline(segments, "Predicted Diarization Timeline")

# # # # # # # #     st.subheader("📊 Manual RTTM Timeline")

# # # # # # # #     ref_rttm = os.path.join(
# # # # # # # #         RTTM_DIR,
# # # # # # # #         os.path.splitext(selected_audio)[0] + ".rttm"
# # # # # # # #     )

# # # # # # # #     if os.path.exists(ref_rttm):
# # # # # # # #         manual_segments = parse_rttm(ref_rttm)
# # # # # # # #         plot_timeline(manual_segments, "Manual RTTM Timeline")
# # # # # # # #     else:
# # # # # # # #         st.warning("Manual RTTM not found.")

# # # # # # # #     # ================= DER =================

# # # # # # # #     st.subheader("📐 Diarization Error Rate (DER)")

# # # # # # # #     if os.path.exists(ref_rttm):
# # # # # # # #         ref = load_rttm(ref_rttm)
# # # # # # # #         file_id = list(ref.keys())[0]

# # # # # # # #         hyp = load_rttm(HYP_RTTM)[file_id]

# # # # # # # #         metric = DiarizationErrorRate(collar=0.25)
# # # # # # # #         der = metric(ref[file_id], hyp)

# # # # # # # #         st.metric("DER", f"{der * 100:.2f}%")
# # # # # # # #     else:
# # # # # # # #         st.warning("DER skipped (no reference RTTM).")



# # # # # # # import streamlit as st
# # # # # # # import os
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # import pandas as pd
# # # # # # # from collections import defaultdict

# # # # # # # from diarize import run_diarization
# # # # # # # from pyannote.database.util import load_rttm
# # # # # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # # # # ================= PATHS =================

# # # # # # # BASE_PATH = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject"

# # # # # # # AUDIO_DIR = os.path.join(BASE_PATH, "processed", "dataset", "audio")
# # # # # # # RTTM_DIR = os.path.join(BASE_PATH, "processed", "dataset", "rttm")

# # # # # # # OUTPUT_DIR = os.path.join(BASE_PATH, "output")
# # # # # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # # # # HYP_RTTM = os.path.join(OUTPUT_DIR, "hypothesis.rttm")

# # # # # # # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # # # # # ================= PAGE =================

# # # # # # # st.set_page_config(layout="wide")
# # # # # # # st.title("🎤 Speaker Diarization – CDAC Dataset")
# # # # # # # st.caption("Speaker-wise segments • Timeline • DER")

# # # # # # # # ================= UTILITIES =================

# # # # # # # def parse_rttm(rttm_path):
# # # # # # #     segments = []
# # # # # # #     with open(rttm_path) as f:
# # # # # # #         for idx, line in enumerate(f):
# # # # # # #             p = line.strip().split()
# # # # # # #             start = float(p[3])
# # # # # # #             dur = float(p[4])
# # # # # # #             speaker = p[7]
# # # # # # #             segments.append({
# # # # # # #                 "index": idx,          # 🔑 segment index
# # # # # # #                 "speaker": speaker,
# # # # # # #                 "start": start,
# # # # # # #                 "end": start + dur,
# # # # # # #                 "duration": dur
# # # # # # #             })
# # # # # # #     return segments


# # # # # # # def segment_index(filename):
# # # # # # #     # SPEAKER_00_12.wav -> 12
# # # # # # #     return int(os.path.splitext(filename)[0].split("_")[-1])


# # # # # # # def plot_timeline(segments, title):
# # # # # # #     if not segments:
# # # # # # #         st.warning("No segments to plot.")
# # # # # # #         return

# # # # # # #     df = pd.DataFrame(segments)
# # # # # # #     speakers = sorted(df["speaker"].unique())
# # # # # # #     y_map = {spk: i for i, spk in enumerate(speakers)}

# # # # # # #     fig, ax = plt.subplots(figsize=(14, 3))

# # # # # # #     for _, row in df.iterrows():
# # # # # # #         ax.barh(
# # # # # # #             y_map[row["speaker"]],
# # # # # # #             row["end"] - row["start"],
# # # # # # #             left=row["start"]
# # # # # # #         )

# # # # # # #     ax.set_yticks(list(y_map.values()))
# # # # # # #     ax.set_yticklabels(list(y_map.keys()))
# # # # # # #     ax.set_xlabel("Time (seconds)")
# # # # # # #     ax.set_ylabel("Speaker")
# # # # # # #     ax.set_title(title)
# # # # # # #     ax.grid(axis="x", linestyle="--", alpha=0.5)

# # # # # # #     st.pyplot(fig)

# # # # # # # # ================= AUDIO SELECTION =================

# # # # # # # audio_files = sorted(
# # # # # # #     f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")
# # # # # # # )

# # # # # # # selected_audio = st.selectbox("Select audio file", audio_files)
# # # # # # # audio_path = os.path.join(AUDIO_DIR, selected_audio)

# # # # # # # st.audio(audio_path)

# # # # # # # # ================= RUN DIARIZATION =================

# # # # # # # if st.button("Run Diarization"):

# # # # # # #     with st.spinner("Running diarization..."):
# # # # # # #         run_diarization(audio_path)

# # # # # # #     st.success("Diarization completed")

# # # # # # #     if not os.path.exists(HYP_RTTM):
# # # # # # #         st.error("Hypothesis RTTM not found.")
# # # # # # #         st.stop()

# # # # # # #     # ================= LOAD SEGMENTS =================

# # # # # # #     segments = parse_rttm(HYP_RTTM)

# # # # # # #     speaker_files = sorted(
# # # # # # #         [f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")],
# # # # # # #         key=segment_index
# # # # # # #     )

# # # # # # #     if not segments or not speaker_files:
# # # # # # #         st.warning("No speaker segments created.")
# # # # # # #         st.stop()

# # # # # # #     # ================= GROUP BY SPEAKER (NO MERGING) =================

# # # # # # #     grouped = defaultdict(list)
# # # # # # #     for seg in segments:
# # # # # # #         grouped[seg["speaker"]].append(seg)

# # # # # # #     # Sort segments of each speaker by time
# # # # # # #     for spk in grouped:
# # # # # # #         grouped[spk] = sorted(grouped[spk], key=lambda x: x["start"])

# # # # # # #     st.subheader("🎧 Speaker-wise Segments")

# # # # # # #     for speaker, segs in grouped.items():
# # # # # # #         st.markdown(f"## 🎤 {speaker}")

# # # # # # #         for i, seg in enumerate(segs):
# # # # # # #             st.markdown(
# # # # # # #                 f"""
# # # # # # #                 **Segment {i+1}**  
# # # # # # #                 ⏱️ {seg['start']:.2f}s → {seg['end']:.2f}s  
# # # # # # #                 🧮 Duration: {seg['duration']:.2f}s
# # # # # # #                 """
# # # # # # #             )

# # # # # # #             idx = seg["index"]
# # # # # # #             if idx < len(speaker_files):
# # # # # # #                 st.audio(os.path.join(SPEAKER_DIR, speaker_files[idx]))

# # # # # # #     # ================= TIMELINE GRAPHS =================

# # # # # # #     st.subheader("📈 Predicted Speaker Timeline")
# # # # # # #     plot_timeline(segments, "Predicted Diarization Timeline")

# # # # # # #     st.subheader("📊 Manual RTTM Timeline")

# # # # # # #     ref_rttm = os.path.join(
# # # # # # #         RTTM_DIR,
# # # # # # #         os.path.splitext(selected_audio)[0] + ".rttm"
# # # # # # #     )

# # # # # # #     if os.path.exists(ref_rttm):
# # # # # # #         manual_segments = parse_rttm(ref_rttm)
# # # # # # #         plot_timeline(manual_segments, "Manual RTTM Timeline")
# # # # # # #     else:
# # # # # # #         st.warning("Manual RTTM not found.")

# # # # # # #     # ================= DER =================

# # # # # # #     st.subheader("📐 Diarization Error Rate (DER)")

# # # # # # #     if os.path.exists(ref_rttm):
# # # # # # #         ref = load_rttm(ref_rttm)
# # # # # # #         file_id = list(ref.keys())[0]

# # # # # # #         hyp = load_rttm(HYP_RTTM)[file_id]

# # # # # # #         metric = DiarizationErrorRate(collar=0.25)
# # # # # # #         der = metric(ref[file_id], hyp)

# # # # # # #         st.metric("DER", f"{der * 100:.2f}%")
# # # # # # #     else:
# # # # # # #         st.warning("DER skipped (no reference RTTM).")



# # # # # # import streamlit as st
# # # # # # import os
# # # # # # import matplotlib.pyplot as plt
# # # # # # import pandas as pd
# # # # # # from collections import defaultdict

# # # # # # from diarize import run_diarization
# # # # # # from pyannote.database.util import load_rttm
# # # # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # # # ================= PATHS =================

# # # # # # BASE_PATH = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject"

# # # # # # AUDIO_DIR = os.path.join(BASE_PATH, "processed", "dataset", "audio")
# # # # # # RTTM_DIR = os.path.join(BASE_PATH, "processed", "dataset", "rttm")

# # # # # # OUTPUT_DIR = os.path.join(BASE_PATH, "output")
# # # # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # # # HYP_RTTM = os.path.join(OUTPUT_DIR, "hypothesis.rttm")

# # # # # # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # # # # ================= PAGE =================

# # # # # # st.set_page_config(layout="wide")
# # # # # # st.title("🎤 Speaker Diarization – CDAC Dataset")
# # # # # # st.caption("Speaker-wise segments • Color-consistent timelines • DER")

# # # # # # # ================= UTILITIES =================

# # # # # # def parse_rttm(rttm_path):
# # # # # #     segments = []
# # # # # #     with open(rttm_path) as f:
# # # # # #         for idx, line in enumerate(f):
# # # # # #             p = line.strip().split()
# # # # # #             start = float(p[3])
# # # # # #             dur = float(p[4])
# # # # # #             speaker = p[7]
# # # # # #             segments.append({
# # # # # #                 "index": idx,
# # # # # #                 "speaker": speaker,
# # # # # #                 "start": start,
# # # # # #                 "end": start + dur,
# # # # # #                 "duration": dur
# # # # # #             })
# # # # # #     return segments


# # # # # # def segment_index(filename):
# # # # # #     return int(os.path.splitext(filename)[0].split("_")[-1])


# # # # # # def build_color_map(speakers):
# # # # # #     """
# # # # # #     Assign ONE color per speaker (consistent everywhere)
# # # # # #     """
# # # # # #     cmap = plt.get_cmap("tab10")
# # # # # #     return {spk: cmap(i % 10) for i, spk in enumerate(sorted(speakers))}


# # # # # # def plot_timeline(segments, title, color_map):
# # # # # #     if not segments:
# # # # # #         st.warning("No segments to plot.")
# # # # # #         return

# # # # # #     df = pd.DataFrame(segments)
# # # # # #     speakers = sorted(df["speaker"].unique())
# # # # # #     y_map = {spk: i for i, spk in enumerate(speakers)}

# # # # # #     fig, ax = plt.subplots(figsize=(14, 3))

# # # # # #     for _, row in df.iterrows():
# # # # # #         ax.barh(
# # # # # #             y_map[row["speaker"]],
# # # # # #             row["end"] - row["start"],
# # # # # #             left=row["start"],
# # # # # #             color=color_map[row["speaker"]]
# # # # # #         )

# # # # # #     ax.set_yticks(list(y_map.values()))
# # # # # #     ax.set_yticklabels(list(y_map.keys()))
# # # # # #     ax.set_xlabel("Time (seconds)")
# # # # # #     ax.set_ylabel("Speaker")
# # # # # #     ax.set_title(title)
# # # # # #     ax.grid(axis="x", linestyle="--", alpha=0.4)

# # # # # #     st.pyplot(fig)

# # # # # # # ================= AUDIO SELECTION =================

# # # # # # audio_files = sorted(
# # # # # #     f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")
# # # # # # )

# # # # # # selected_audio = st.selectbox("Select audio file", audio_files)
# # # # # # audio_path = os.path.join(AUDIO_DIR, selected_audio)

# # # # # # st.audio(audio_path)

# # # # # # # ================= RUN DIARIZATION =================

# # # # # # if st.button("Run Diarization"):

# # # # # #     with st.spinner("Running diarization..."):
# # # # # #         run_diarization(audio_path)

# # # # # #     st.success("Diarization completed")

# # # # # #     if not os.path.exists(HYP_RTTM):
# # # # # #         st.error("Hypothesis RTTM not found.")
# # # # # #         st.stop()

# # # # # #     # ================= LOAD SEGMENTS =================

# # # # # #     predicted_segments = parse_rttm(HYP_RTTM)

# # # # # #     speaker_files = sorted(
# # # # # #         [f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")],
# # # # # #         key=segment_index
# # # # # #     )

# # # # # #     if not predicted_segments or not speaker_files:
# # # # # #         st.warning("No speaker segments created.")
# # # # # #         st.stop()

# # # # # #     # ================= BUILD COLOR MAP (ONCE) =================

# # # # # #     all_speakers = {seg["speaker"] for seg in predicted_segments}

# # # # # #     ref_rttm_path = os.path.join(
# # # # # #         RTTM_DIR,
# # # # # #         os.path.splitext(selected_audio)[0] + ".rttm"
# # # # # #     )

# # # # # #     if os.path.exists(ref_rttm_path):
# # # # # #         manual_segments = parse_rttm(ref_rttm_path)
# # # # # #         all_speakers |= {seg["speaker"] for seg in manual_segments}
# # # # # #     else:
# # # # # #         manual_segments = []

# # # # # #     color_map = build_color_map(all_speakers)

# # # # # #     # ================= SPEAKER-WISE SEGMENTS =================

# # # # # #     st.subheader("🎧 Speaker-wise Segments (Not Merged)")

# # # # # #     grouped = defaultdict(list)
# # # # # #     for seg in predicted_segments:
# # # # # #         grouped[seg["speaker"]].append(seg)

# # # # # #     for spk in grouped:
# # # # # #         grouped[spk] = sorted(grouped[spk], key=lambda x: x["start"])

# # # # # #     for speaker, segs in grouped.items():
# # # # # #         st.markdown(
# # # # # #             f"<h3 style='color: rgb{tuple(int(c*255) for c in color_map[speaker][:3])}'>🎤 {speaker}</h3>",
# # # # # #             unsafe_allow_html=True
# # # # # #         )

# # # # # #         for i, seg in enumerate(segs):
# # # # # #             st.markdown(
# # # # # #                 f"""
# # # # # #                 **Segment {i+1}**  
# # # # # #                 ⏱️ {seg['start']:.2f}s → {seg['end']:.2f}s  
# # # # # #                 🧮 Duration: {seg['duration']:.2f}s
# # # # # #                 """
# # # # # #             )

# # # # # #             idx = seg["index"]
# # # # # #             if idx < len(speaker_files):
# # # # # #                 st.audio(os.path.join(SPEAKER_DIR, speaker_files[idx]))

# # # # # #     # ================= TIMELINE GRAPHS =================

# # # # # #     st.subheader("📈 Predicted RTTM Timeline")
# # # # # #     plot_timeline(
# # # # # #         predicted_segments,
# # # # # #         "Predicted Speaker Diarization",
# # # # # #         color_map
# # # # # #     )

# # # # # #     st.subheader("📊 Manual RTTM Timeline (Ground Truth)")

# # # # # #     if manual_segments:
# # # # # #         plot_timeline(
# # # # # #             manual_segments,
# # # # # #             "Manual RTTM (Ground Truth)",
# # # # # #             color_map
# # # # # #         )
# # # # # #     else:
# # # # # #         st.warning("Manual RTTM not found.")

# # # # # #     # ================= DER =================

# # # # # #     st.subheader("📐 Diarization Error Rate (DER)")

# # # # # #     if os.path.exists(ref_rttm_path):
# # # # # #         ref = load_rttm(ref_rttm_path)
# # # # # #         file_id = list(ref.keys())[0]

# # # # # #         hyp = load_rttm(HYP_RTTM)[file_id]

# # # # # #         metric = DiarizationErrorRate(collar=0.25)
# # # # # #         der = metric(ref[file_id], hyp)

# # # # # #         st.metric("DER", f"{der * 100:.2f}%")
# # # # # #     else:
# # # # # #         st.warning("DER skipped (no reference RTTM).")



# # # # # import streamlit as st
# # # # # import os
# # # # # import matplotlib.pyplot as plt
# # # # # import pandas as pd
# # # # # from collections import defaultdict

# # # # # from diarize import run_diarization
# # # # # from pyannote.database.util import load_rttm
# # # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # # ================= PATHS =================

# # # # # BASE_PATH = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject"

# # # # # AUDIO_DIR = os.path.join(BASE_PATH, "processed", "dataset", "audio")
# # # # # RTTM_DIR = os.path.join(BASE_PATH, "processed", "dataset", "rttm")

# # # # # OUTPUT_DIR = os.path.join(BASE_PATH, "output")
# # # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # # HYP_RTTM = os.path.join(OUTPUT_DIR, "hypothesis.rttm")

# # # # # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # # # ================= PAGE =================

# # # # # st.set_page_config(layout="wide")
# # # # # st.title("🎤 Speaker Diarization – CDAC Dataset")
# # # # # st.caption(
# # # # #     "Speaker-wise segments • Color-consistent timelines • Hypothesis RTTM table • DER"
# # # # # )

# # # # # # ================= UTILITIES =================

# # # # # def parse_rttm(rttm_path):
# # # # #     segments = []
# # # # #     with open(rttm_path) as f:
# # # # #         for idx, line in enumerate(f):
# # # # #             p = line.strip().split()
# # # # #             start = float(p[3])
# # # # #             dur = float(p[4])
# # # # #             speaker = p[7]
# # # # #             segments.append({
# # # # #                 "index": idx,              # segment index (RTTM order)
# # # # #                 "speaker": speaker,
# # # # #                 "start": start,
# # # # #                 "end": start + dur,
# # # # #                 "duration": dur
# # # # #             })
# # # # #     return segments


# # # # # def segment_index(filename):
# # # # #     # SPEAKER_00_12.wav -> 12
# # # # #     return int(os.path.splitext(filename)[0].split("_")[-1])


# # # # # def build_color_map(speakers):
# # # # #     cmap = plt.get_cmap("tab10")
# # # # #     return {spk: cmap(i % 10) for i, spk in enumerate(sorted(speakers))}


# # # # # def plot_timeline(segments, title, color_map):
# # # # #     if not segments:
# # # # #         st.warning("No segments to plot.")
# # # # #         return

# # # # #     df = pd.DataFrame(segments)
# # # # #     speakers = sorted(df["speaker"].unique())
# # # # #     y_map = {spk: i for i, spk in enumerate(speakers)}

# # # # #     fig, ax = plt.subplots(figsize=(14, 3))

# # # # #     for _, row in df.iterrows():
# # # # #         ax.barh(
# # # # #             y_map[row["speaker"]],
# # # # #             row["end"] - row["start"],
# # # # #             left=row["start"],
# # # # #             color=color_map[row["speaker"]]
# # # # #         )

# # # # #     ax.set_yticks(list(y_map.values()))
# # # # #     ax.set_yticklabels(list(y_map.keys()))
# # # # #     ax.set_xlabel("Time (seconds)")
# # # # #     ax.set_ylabel("Speaker")
# # # # #     ax.set_title(title)
# # # # #     ax.grid(axis="x", linestyle="--", alpha=0.4)

# # # # #     st.pyplot(fig)


# # # # # def rttm_to_dataframe(segments):
# # # # #     rows = []
# # # # #     for i, seg in enumerate(segments, start=1):
# # # # #         rows.append({
# # # # #             "Segment": i,
# # # # #             "Speaker": seg["speaker"],
# # # # #             "Start (s)": round(seg["start"], 2),
# # # # #             "End (s)": round(seg["end"], 2),
# # # # #             "Duration (s)": round(seg["duration"], 2)
# # # # #         })
# # # # #     return pd.DataFrame(rows)

# # # # # # ================= AUDIO SELECTION =================

# # # # # audio_files = sorted(
# # # # #     f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")
# # # # # )

# # # # # selected_audio = st.selectbox("Select audio file", audio_files)
# # # # # audio_path = os.path.join(AUDIO_DIR, selected_audio)

# # # # # st.audio(audio_path)

# # # # # # ================= RUN DIARIZATION =================

# # # # # if st.button("Run Diarization"):

# # # # #     with st.spinner("Running diarization..."):
# # # # #         run_diarization(audio_path)

# # # # #     st.success("Diarization completed")

# # # # #     if not os.path.exists(HYP_RTTM):
# # # # #         st.error("Hypothesis RTTM not found.")
# # # # #         st.stop()

# # # # #     # ================= LOAD SEGMENTS =================

# # # # #     predicted_segments = parse_rttm(HYP_RTTM)

# # # # #     speaker_files = sorted(
# # # # #         [f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")],
# # # # #         key=segment_index
# # # # #     )

# # # # #     if not predicted_segments or not speaker_files:
# # # # #         st.warning("No speaker segments created.")
# # # # #         st.stop()

# # # # #     # ================= LOAD MANUAL RTTM =================

# # # # #     ref_rttm_path = os.path.join(
# # # # #         RTTM_DIR,
# # # # #         os.path.splitext(selected_audio)[0] + ".rttm"
# # # # #     )

# # # # #     if os.path.exists(ref_rttm_path):
# # # # #         manual_segments = parse_rttm(ref_rttm_path)
# # # # #     else:
# # # # #         manual_segments = []

# # # # #     # ================= COLOR MAP (GLOBAL) =================

# # # # #     all_speakers = {seg["speaker"] for seg in predicted_segments}
# # # # #     all_speakers |= {seg["speaker"] for seg in manual_segments}

# # # # #     color_map = build_color_map(all_speakers)

# # # # #     # ================= SPEAKER-WISE SEGMENTS =================

# # # # #     st.subheader("🎧 Speaker-wise Segments (Not Merged)")

# # # # #     grouped = defaultdict(list)
# # # # #     for seg in predicted_segments:
# # # # #         grouped[seg["speaker"]].append(seg)

# # # # #     for spk in grouped:
# # # # #         grouped[spk] = sorted(grouped[spk], key=lambda x: x["start"])

# # # # #     for speaker, segs in grouped.items():
# # # # #         rgb = tuple(int(c * 255) for c in color_map[speaker][:3])
# # # # #         st.markdown(
# # # # #             f"<h3 style='color: rgb{rgb}'>🎤 {speaker}</h3>",
# # # # #             unsafe_allow_html=True
# # # # #         )

# # # # #         for i, seg in enumerate(segs):
# # # # #             st.markdown(
# # # # #                 f"""
# # # # #                 **Segment {i+1}**  
# # # # #                 ⏱️ {seg['start']:.2f}s → {seg['end']:.2f}s  
# # # # #                 🧮 Duration: {seg['duration']:.2f}s
# # # # #                 """
# # # # #             )

# # # # #             idx = seg["index"]
# # # # #             if idx < len(speaker_files):
# # # # #                 st.audio(os.path.join(SPEAKER_DIR, speaker_files[idx]))

# # # # #     # ================= HYPOTHESIS RTTM TABLE =================

# # # # #     st.subheader("📋 Hypothesis RTTM Table (Model Output)")

# # # # #     hyp_df = rttm_to_dataframe(predicted_segments)

# # # # #     st.dataframe(
# # # # #         hyp_df,
# # # # #         use_container_width=True,
# # # # #         hide_index=True
# # # # #     )

# # # # #     # ================= TIMELINE GRAPHS =================

# # # # #     st.subheader("📈 Predicted RTTM Timeline")
# # # # #     plot_timeline(
# # # # #         predicted_segments,
# # # # #         "Predicted Speaker Diarization Timeline",
# # # # #         color_map
# # # # #     )

# # # # #     st.subheader("📊 Manual RTTM Timeline (Ground Truth)")

# # # # #     if manual_segments:
# # # # #         plot_timeline(
# # # # #             manual_segments,
# # # # #             "Manual RTTM Timeline",
# # # # #             color_map
# # # # #         )
# # # # #     else:
# # # # #         st.warning("Manual RTTM not found.")

# # # # #     # ================= DER =================

# # # # #     st.subheader("📐 Diarization Error Rate (DER)")

# # # # #     if os.path.exists(ref_rttm_path):
# # # # #         ref = load_rttm(ref_rttm_path)
# # # # #         file_id = list(ref.keys())[0]

# # # # #         hyp = load_rttm(HYP_RTTM)[file_id]

# # # # #         metric = DiarizationErrorRate(collar=0.25)
# # # # #         der = metric(ref[file_id], hyp)

# # # # #         st.metric("DER", f"{der * 100:.2f}%")
# # # # #     else:
# # # # #         st.warning("DER skipped (no reference RTTM).")



# # # # import streamlit as st
# # # # import os
# # # # import matplotlib.pyplot as plt
# # # # import pandas as pd
# # # # from collections import defaultdict
# # # # import shutil

# # # # from diarize import run_diarization
# # # # from pyannote.database.util import load_rttm
# # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # ================= PATHS =================

# # # # BASE_PATH = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject"

# # # # DATASET_AUDIO_DIR = os.path.join(BASE_PATH, "processed", "dataset", "audio")
# # # # RTTM_DIR = os.path.join(BASE_PATH, "processed", "dataset", "rttm")

# # # # OUTPUT_DIR = os.path.join(BASE_PATH, "output")
# # # # UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploaded")
# # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # HYP_RTTM = os.path.join(OUTPUT_DIR, "hypothesis.rttm")

# # # # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # # # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # # ================= PAGE =================

# # # # st.set_page_config(layout="wide")
# # # # st.title("🎤 Speaker Diarization System")
# # # # st.caption(
# # # #     "Upload any audio • Speaker-wise segments • RTTM timelines • DER"
# # # # )

# # # # # ================= UTILITIES =================

# # # # def parse_rttm(rttm_path):
# # # #     segments = []
# # # #     with open(rttm_path) as f:
# # # #         for idx, line in enumerate(f):
# # # #             p = line.strip().split()
# # # #             start = float(p[3])
# # # #             dur = float(p[4])
# # # #             speaker = p[7]
# # # #             segments.append({
# # # #                 "index": idx,
# # # #                 "speaker": speaker,
# # # #                 "start": start,
# # # #                 "end": start + dur,
# # # #                 "duration": dur
# # # #             })
# # # #     return segments


# # # # def segment_index(filename):
# # # #     return int(os.path.splitext(filename)[0].split("_")[-1])


# # # # def build_color_map(speakers):
# # # #     cmap = plt.get_cmap("tab10")
# # # #     return {spk: cmap(i % 10) for i, spk in enumerate(sorted(speakers))}


# # # # def plot_timeline(segments, title, color_map):
# # # #     if not segments:
# # # #         st.warning("No segments to plot.")
# # # #         return

# # # #     df = pd.DataFrame(segments)
# # # #     speakers = sorted(df["speaker"].unique())
# # # #     y_map = {spk: i for i, spk in enumerate(speakers)}

# # # #     fig, ax = plt.subplots(figsize=(14, 3))

# # # #     for _, row in df.iterrows():
# # # #         ax.barh(
# # # #             y_map[row["speaker"]],
# # # #             row["end"] - row["start"],
# # # #             left=row["start"],
# # # #             color=color_map[row["speaker"]]
# # # #         )

# # # #     ax.set_yticks(list(y_map.values()))
# # # #     ax.set_yticklabels(list(y_map.keys()))
# # # #     ax.set_xlabel("Time (seconds)")
# # # #     ax.set_ylabel("Speaker")
# # # #     ax.set_title(title)
# # # #     ax.grid(axis="x", linestyle="--", alpha=0.4)

# # # #     st.pyplot(fig)


# # # # def rttm_to_dataframe(segments):
# # # #     return pd.DataFrame([
# # # #         {
# # # #             "Segment": i + 1,
# # # #             "Speaker": s["speaker"],
# # # #             "Start (s)": round(s["start"], 2),
# # # #             "End (s)": round(s["end"], 2),
# # # #             "Duration (s)": round(s["duration"], 2)
# # # #         }
# # # #         for i, s in enumerate(segments)
# # # #     ])

# # # # # ================= AUDIO INPUT =================

# # # # st.subheader("🎵 Audio Input")

# # # # input_mode = st.radio(
# # # #     "Choose audio source",
# # # #     ["Upload Audio", "Use Dataset Audio"]
# # # # )

# # # # if input_mode == "Upload Audio":

# # # #     uploaded = st.file_uploader(
# # # #         "Upload audio file",
# # # #         type=["wav", "mp3", "flac", "ogg"]
# # # #     )

# # # #     if uploaded is None:
# # # #         st.stop()

# # # #     uploaded_audio_path = os.path.join(UPLOAD_DIR, uploaded.name)

# # # #     with open(uploaded_audio_path, "wb") as f:
# # # #         f.write(uploaded.read())

# # # #     audio_path = uploaded_audio_path
# # # #     st.audio(audio_path)

# # # #     reference_rttm = None  # no ground truth for uploaded audio

# # # # else:
# # # #     audio_files = sorted(
# # # #         f for f in os.listdir(DATASET_AUDIO_DIR) if f.endswith(".wav")
# # # #     )

# # # #     selected_audio = st.selectbox("Select dataset audio", audio_files)
# # # #     audio_path = os.path.join(DATASET_AUDIO_DIR, selected_audio)
# # # #     st.audio(audio_path)

# # # #     reference_rttm = os.path.join(
# # # #         RTTM_DIR,
# # # #         os.path.splitext(selected_audio)[0] + ".rttm"
# # # #     )

# # # # # ================= RUN DIARIZATION =================

# # # # if st.button("Run Diarization"):

# # # #     with st.spinner("Running diarization..."):
# # # #         run_diarization(audio_path)

# # # #     st.success("Diarization completed")

# # # #     if not os.path.exists(HYP_RTTM):
# # # #         st.error("Hypothesis RTTM not found.")
# # # #         st.stop()

# # # #     # ================= LOAD SEGMENTS =================

# # # #     predicted_segments = parse_rttm(HYP_RTTM)

# # # #     speaker_files = sorted(
# # # #         [f for f in os.listdir(SPEAKER_DIR) if f.endswith(".wav")],
# # # #         key=segment_index
# # # #     )

# # # #     if not predicted_segments or not speaker_files:
# # # #         st.warning("No speaker segments created.")
# # # #         st.stop()

# # # #     # ================= LOAD MANUAL RTTM (IF EXISTS) =================

# # # #     if reference_rttm and os.path.exists(reference_rttm):
# # # #         manual_segments = parse_rttm(reference_rttm)
# # # #     else:
# # # #         manual_segments = []

# # # #     # ================= COLOR MAP =================

# # # #     all_speakers = {s["speaker"] for s in predicted_segments}
# # # #     all_speakers |= {s["speaker"] for s in manual_segments}

# # # #     color_map = build_color_map(all_speakers)

# # # #     # ================= SPEAKER-WISE SEGMENTS =================

# # # #     st.subheader("🎧 Speaker-wise Segments")

# # # #     grouped = defaultdict(list)
# # # #     for seg in predicted_segments:
# # # #         grouped[seg["speaker"]].append(seg)

# # # #     for spk in grouped:
# # # #         grouped[spk] = sorted(grouped[spk], key=lambda x: x["start"])

# # # #     for speaker, segs in grouped.items():
# # # #         rgb = tuple(int(c * 255) for c in color_map[speaker][:3])
# # # #         st.markdown(
# # # #             f"<h3 style='color: rgb{rgb}'>🎤 {speaker}</h3>",
# # # #             unsafe_allow_html=True
# # # #         )

# # # #         for i, seg in enumerate(segs):
# # # #             st.markdown(
# # # #                 f"""
# # # #                 **Segment {i+1}**  
# # # #                 ⏱️ {seg['start']:.2f}s → {seg['end']:.2f}s  
# # # #                 🧮 Duration: {seg['duration']:.2f}s
# # # #                 """
# # # #             )
# # # #             idx = seg["index"]
# # # #             if idx < len(speaker_files):
# # # #                 st.audio(os.path.join(SPEAKER_DIR, speaker_files[idx]))

# # # #     # ================= RTTM TABLE =================

# # # #     st.subheader("📋 Hypothesis RTTM Table")

# # # #     st.dataframe(
# # # #         rttm_to_dataframe(predicted_segments),
# # # #         use_container_width=True,
# # # #         hide_index=True
# # # #     )

# # # #     # ================= TIMELINE GRAPHS =================

# # # #     st.subheader("📈 Predicted RTTM Timeline")
# # # #     plot_timeline(
# # # #         predicted_segments,
# # # #         "Predicted Speaker Diarization",
# # # #         color_map
# # # #     )

# # # #     if manual_segments:
# # # #         st.subheader("📊 Manual RTTM Timeline")
# # # #         plot_timeline(
# # # #             manual_segments,
# # # #             "Manual RTTM (Ground Truth)",
# # # #             color_map
# # # #         )

# # # #     # ================= DER =================

# # # #     if manual_segments:
# # # #         st.subheader("📐 Diarization Error Rate (DER)")
# # # #         ref = load_rttm(reference_rttm)
# # # #         file_id = list(ref.keys())[0]

# # # #         hyp = load_rttm(HYP_RTTM)[file_id]
# # # #         der = DiarizationErrorRate(collar=0.25)(ref[file_id], hyp)

# # # #         st.metric("DER", f"{der * 100:.2f}%")
# # # #     else:
# # # #         st.info("DER not available for uploaded audio (no manual RTTM).")



# # # import streamlit as st
# # # import os
# # # import random
# # # import matplotlib.pyplot as plt
# # # import pandas as pd
# # # from collections import defaultdict

# # # from diarize import run_diarization

# # # # ================= PATHS =================

# # # BASE_DIR = "output"
# # # UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded")
# # # SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")
# # # HYP_RTTM = os.path.join(BASE_DIR, "hypothesis.rttm")

# # # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # ================= PAGE =================

# # # st.set_page_config(layout="wide")
# # # st.title("🎤 Automatic Speaker Diarization")
# # # st.caption("Unknown speakers • Any audio • Timeline • RTTM table")

# # # # ================= UTILITIES =================

# # # def parse_rttm(path):
# # #     segments = []
# # #     with open(path) as f:
# # #         for idx, line in enumerate(f):
# # #             p = line.strip().split()
# # #             start = float(p[3])
# # #             dur = float(p[4])
# # #             speaker = p[7]
# # #             segments.append({
# # #                 "index": idx,
# # #                 "speaker": speaker,
# # #                 "start": start,
# # #                 "end": start + dur,
# # #                 "duration": dur
# # #             })
# # #     return segments


# # # def build_color_map(speakers):
# # #     random.seed(42)
# # #     return {
# # #         spk: (random.random(), random.random(), random.random())
# # #         for spk in speakers
# # #     }


# # # def plot_timeline(segments, title, color_map):
# # #     df = pd.DataFrame(segments)
# # #     speakers = sorted(df["speaker"].unique())
# # #     y_map = {s: i for i, s in enumerate(speakers)}

# # #     fig, ax = plt.subplots(figsize=(14, 3))

# # #     for _, row in df.iterrows():
# # #         ax.barh(
# # #             y_map[row["speaker"]],
# # #             row["end"] - row["start"],
# # #             left=row["start"],
# # #             color=color_map[row["speaker"]]
# # #         )

# # #     ax.set_yticks(list(y_map.values()))
# # #     ax.set_yticklabels(list(y_map.keys()))
# # #     ax.set_xlabel("Time (seconds)")
# # #     ax.set_ylabel("Speaker")
# # #     ax.set_title(title)
# # #     ax.grid(axis="x", linestyle="--", alpha=0.4)

# # #     st.pyplot(fig)


# # # def rttm_table(segments):
# # #     return pd.DataFrame([
# # #         {
# # #             "Segment": i + 1,
# # #             "Speaker": s["speaker"],
# # #             "Start (s)": round(s["start"], 2),
# # #             "End (s)": round(s["end"], 2),
# # #             "Duration (s)": round(s["duration"], 2)
# # #         }
# # #         for i, s in enumerate(segments)
# # #     ])

# # # # ================= AUDIO UPLOAD =================

# # # uploaded = st.file_uploader(
# # #     "Upload audio file",
# # #     type=["wav", "mp3", "flac", "ogg"]
# # # )

# # # if uploaded is None:
# # #     st.stop()

# # # audio_path = os.path.join(UPLOAD_DIR, uploaded.name)

# # # with open(audio_path, "wb") as f:
# # #     f.write(uploaded.read())

# # # st.audio(audio_path)

# # # # ================= RUN DIARIZATION =================

# # # if st.button("Run Diarization"):

# # #     with st.spinner("Running diarization..."):
# # #         run_diarization(audio_path)

# # #     st.success("Diarization complete")

# # #     segments = parse_rttm(HYP_RTTM)

# # #     if not segments:
# # #         st.warning("No speech segments detected.")
# # #         st.stop()

# # #     color_map = build_color_map({s["speaker"] for s in segments})

# # #     # ================= SPEAKER-WISE SEGMENTS =================

# # #     st.subheader("🎧 Speaker-wise Segments")

# # #     grouped = defaultdict(list)
# # #     for s in segments:
# # #         grouped[s["speaker"]].append(s)

# # #     for spk, segs in grouped.items():
# # #         st.markdown(
# # #             f"<h3 style='color: rgb{tuple(int(c*255) for c in color_map[spk])}'>🎤 {spk}</h3>",
# # #             unsafe_allow_html=True
# # #         )

# # #         for seg in segs:
# # #             st.write(
# # #                 f"{seg['start']:.2f}s → {seg['end']:.2f}s "
# # #                 f"({seg['duration']:.2f}s)"
# # #             )
# # #             audio_file = os.path.join(
# # #                 SPEAKER_DIR,
# # #                 f"{spk}_{seg['index']}.wav"
# # #             )
# # #             if os.path.exists(audio_file):
# # #                 st.audio(audio_file)

# # #     # ================= RTTM TABLE =================

# # #     st.subheader("📋 Hypothesis RTTM Table")
# # #     st.dataframe(rttm_table(segments), use_container_width=True)

# # #     # ================= TIMELINE =================

# # #     st.subheader("📈 Speaker Timeline")
# # #     plot_timeline(segments, "Predicted Speaker Timeline", color_map)



# # import streamlit as st
# # import os
# # import torchaudio
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # from diarize import run_diarization

# # # ================= CONFIG =================

# # BASE_DIR = "output"
# # UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# # SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")
# # RTTM_PATH = os.path.join(BASE_DIR, "hypothesis.rttm")

# # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # SR = 16000

# # # ================= PAGE =================

# # st.set_page_config(layout="wide")
# # st.title("🎙️ Speaker Diarization with LLM Correction")
# # st.caption("Pyannote diarization + LLM-based logical correction")

# # # ================= UPLOAD =================

# # st.subheader("📤 Upload Audio")

# # uploaded = st.file_uploader("Upload WAV audio", type=["wav"])

# # if uploaded is None:
# #     st.info("Please upload an audio file to continue.")
# #     st.stop()

# # audio_path = os.path.join(UPLOAD_DIR, uploaded.name)

# # with open(audio_path, "wb") as f:
# #     f.write(uploaded.read())

# # st.success("Audio uploaded successfully")
# # st.audio(audio_path)

# # # ================= AUDIO INFO =================

# # wav, sr = torchaudio.load(audio_path)
# # duration = wav.shape[1] / sr

# # st.metric("Audio Duration (sec)", f"{duration:.2f}")

# # # ================= RUN DIARIZATION =================

# # st.divider()

# # if st.button("🚀 Run Speaker Diarization"):

# #     with st.spinner("Running diarization + LLM correction..."):
# #         rttm_path = run_diarization(audio_path)

# #     st.success("Diarization complete!")

# #     # ================= LOAD RTTM =================

# #     segments = []

# #     with open(rttm_path) as f:
# #         for line in f:
# #             parts = line.strip().split()
# #             segments.append({
# #                 "speaker": parts[7],
# #                 "start": float(parts[3]),
# #                 "duration": float(parts[4]),
# #                 "end": float(parts[3]) + float(parts[4])
# #             })

# #     df = pd.DataFrame(segments)

# #     # ================= METRICS =================

# #     st.subheader("📊 Summary")

# #     st.metric("Detected Speakers", df["speaker"].nunique())
# #     st.metric("Total Segments", len(df))

# #     # ================= TIMELINE =================

# #     st.subheader("🧭 Speaker Timeline")

# #     fig, ax = plt.subplots(figsize=(12, 3))

# #     speakers = df["speaker"].unique()
# #     color_map = {spk: f"C{i}" for i, spk in enumerate(speakers)}
# #     y_map = {spk: i for i, spk in enumerate(speakers)}

# #     for _, row in df.iterrows():
# #         ax.barh(
# #             y_map[row["speaker"]],
# #             row["duration"],
# #             left=row["start"],
# #             color=color_map[row["speaker"]]
# #         )

# #     ax.set_yticks(list(y_map.values()))
# #     ax.set_yticklabels(list(y_map.keys()))
# #     ax.set_xlabel("Time (seconds)")
# #     ax.set_ylabel("Speaker")
# #     ax.set_title("Time vs Speaker")

# #     st.pyplot(fig)

# #     # ================= HYPOTHESIS RTTM TABLE =================

# #     st.subheader("📋 Hypothesis RTTM Table")

# #     st.dataframe(
# #         df[["speaker", "start", "end", "duration"]],
# #         use_container_width=True,
# #         hide_index=True
# #     )

# #     # ================= SPEAKER AUDIO =================

# #     st.subheader("🎧 Speaker Segments")

# #     speaker_files = sorted(os.listdir(SPEAKER_DIR))
# #     cols = st.columns(2)

# #     for i, fname in enumerate(speaker_files):
# #         with cols[i % 2]:
# #             st.markdown(f"**{fname}**")
# #             st.audio(os.path.join(SPEAKER_DIR, fname))

# #     # ================= RAW RTTM =================

# #     st.subheader("📝 Generated RTTM")

# #     with open(rttm_path) as f:
# #         st.code(f.read())



# import streamlit as st
# import os
# import torchaudio
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import soundfile as sf

# from st_audiorec import st_audiorec
# from diarize import run_diarization

# # ================= CONFIG =================

# BASE_DIR = "output"
# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# RECORD_DIR = os.path.join(BASE_DIR, "recordings")
# SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(RECORD_DIR, exist_ok=True)
# os.makedirs(SPEAKER_DIR, exist_ok=True)

# SR = 16000

# # ================= PAGE =================

# st.set_page_config(layout="wide")
# st.title("🎙️ Speaker Diarization (Upload or Record)")
# st.caption("Upload WAV or record live audio using microphone")

# # ================= INPUT MODE =================

# tab_upload, tab_record = st.tabs(["📤 Upload Audio", "🎤 Record Audio"])

# audio_path = None

# # ================= UPLOAD TAB =================

# with tab_upload:
#     uploaded = st.file_uploader("Upload WAV audio", type=["wav"])

#     if uploaded is not None:
#         audio_path = os.path.join(UPLOAD_DIR, uploaded.name)
#         with open(audio_path, "wb") as f:
#             f.write(uploaded.read())

#         st.success("Audio uploaded successfully")
#         st.audio(audio_path)

# # ================= RECORD TAB =================

# with tab_record:
#     st.info("Click ▶️ to start recording. Click ⏹️ to stop.")

#     wav_audio_data = st_audiorec()

#     if wav_audio_data is not None:
#         audio_path = os.path.join(RECORD_DIR, "mic_recording.wav")

#         # Convert numpy float32 → WAV
#         sf.write(audio_path, wav_audio_data, SR)

#         st.success("Recording saved")
#         st.audio(audio_path)

# # ================= GUARD =================

# if audio_path is None:
#     st.info("Upload or record audio to continue.")
#     st.stop()

# # ================= AUDIO INFO =================

# wav, sr = torchaudio.load(audio_path)
# duration = wav.shape[1] / sr
# st.metric("Audio Duration (seconds)", f"{duration:.2f}")

# # ================= RUN DIARIZATION =================

# st.divider()

# if st.button("🚀 Run Speaker Diarization"):

#     with st.spinner("Running diarization..."):
#         rttm_path = run_diarization(audio_path)

#     st.success("Diarization complete!")

#     # ================= LOAD RTTM =================

#     segments = []
#     with open(rttm_path) as f:
#         for line in f:
#             p = line.strip().split()
#             segments.append({
#                 "speaker": p[7],
#                 "start": float(p[3]),
#                 "duration": float(p[4]),
#                 "end": float(p[3]) + float(p[4])
#             })

#     df = pd.DataFrame(segments)

#     # ================= METRICS =================

#     st.subheader("📊 Summary")
#     st.metric("Detected Speakers", df["speaker"].nunique())
#     st.metric("Total Segments", len(df))

#     # ================= TIMELINE =================

#     st.subheader("🧭 Speaker Timeline")

#     fig, ax = plt.subplots(figsize=(12, 3))

#     speakers = df["speaker"].unique()
#     y_map = {spk: i for i, spk in enumerate(speakers)}
#     color_map = {spk: f"C{i}" for i, spk in enumerate(speakers)}

#     for _, row in df.iterrows():
#         ax.barh(
#             y_map[row["speaker"]],
#             row["duration"],
#             left=row["start"],
#             color=color_map[row["speaker"]]
#         )

#     ax.set_yticks(list(y_map.values()))
#     ax.set_yticklabels(list(y_map.keys()))
#     ax.set_xlabel("Time (seconds)")
#     ax.set_ylabel("Speaker")
#     ax.set_title("Time vs Speaker")

#     st.pyplot(fig)

#     # ================= RTTM TABLE =================

#     st.subheader("📋 Hypothesis RTTM Table")
#     st.dataframe(
#         df[["speaker", "start", "end", "duration"]],
#         use_container_width=True,
#         hide_index=True
#     )

#     # ================= SPEAKER AUDIO =================

#     st.subheader("🎧 Speaker Segments")

#     cols = st.columns(2)
#     files = sorted(os.listdir(SPEAKER_DIR))

#     for i, f in enumerate(files):
#         with cols[i % 2]:
#             st.markdown(f"**{f}**")
#             st.audio(os.path.join(SPEAKER_DIR, f))
import streamlit as st
import os
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf

from diarize import run_diarization

# ================= CONFIG =================

BASE_DIR = "output"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RECORD_DIR = os.path.join(BASE_DIR, "recordings")
SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)

# ================= PAGE =================

st.set_page_config(layout="wide")
st.title("🎙️ Speaker Diarization System")
st.caption("Upload audio or record from microphone | Unknown speakers supported")

# ================= INPUT MODE =================

tab_upload, tab_record = st.tabs(["📤 Upload Audio", "🎤 Record Audio"])

audio_path = None

# ================= UPLOAD =================

with tab_upload:
    uploaded = st.file_uploader("Upload WAV audio", type=["wav"])

    if uploaded is not None:
        audio_path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())

        st.success("Audio uploaded")
        st.audio(audio_path)

# ================= MIC RECORD =================

with tab_record:
    st.info("Click Record → Speak → Stop")

    audio_bytes = st.audio_input("🎤 Record from microphone")

    if audio_bytes is not None:
        audio_path = os.path.join(RECORD_DIR, "mic_recording.wav")

        audio_np, sr = sf.read(audio_bytes)
        sf.write(audio_path, audio_np, sr)

        st.success("Recording saved")
        st.audio(audio_path)

# ================= GUARD =================

if audio_path is None:
    st.info("Upload or record audio to continue.")
    st.stop()

# ================= AUDIO INFO =================

wav, sr = torchaudio.load(audio_path)
duration = wav.shape[1] / sr
st.metric("Audio Duration (sec)", f"{duration:.2f}")

# ================= RUN DIARIZATION =================

st.divider()

if st.button("🚀 Run Speaker Diarization"):

    with st.spinner("Running diarization..."):
        rttm_path = run_diarization(audio_path)

    st.success("Diarization complete")

    # ================= LOAD RTTM =================

    segments = []
    with open(rttm_path) as f:
        for line in f:
            p = line.strip().split()
            segments.append({
                "speaker": p[7],
                "start": float(p[3]),
                "duration": float(p[4]),
                "end": float(p[3]) + float(p[4])
            })

    df = pd.DataFrame(segments)

    # ================= METRICS =================

    st.subheader("📊 Summary")
    st.metric("Detected Speakers", df["speaker"].nunique())
    st.metric("Total Segments", len(df))

    # ================= TIMELINE =================

    st.subheader("🧭 Speaker Timeline")

    fig, ax = plt.subplots(figsize=(12, 3))

    speakers = df["speaker"].unique()
    y_map = {spk: i for i, spk in enumerate(speakers)}
    color_map = {spk: f"C{i}" for i, spk in enumerate(speakers)}

    for _, row in df.iterrows():
        ax.barh(
            y_map[row["speaker"]],
            row["duration"],
            left=row["start"],
            color=color_map[row["speaker"]]
        )

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title("Time vs Speaker")

    st.pyplot(fig)

    # ================= RTTM TABLE =================

    st.subheader("📋 Hypothesis RTTM Table")

    st.dataframe(
        df[["speaker", "start", "end", "duration"]],
        use_container_width=True,
        hide_index=True
    )

    # ================= SPEAKER AUDIO =================

    st.subheader("🎧 Speaker Segments")

    cols = st.columns(2)
    files = sorted(os.listdir(SPEAKER_DIR))

    for i, f in enumerate(files):
        with cols[i % 2]:
            st.markdown(f"**{f}**")
            st.audio(os.path.join(SPEAKER_DIR, f))

    # ================= RAW RTTM =================

    st.subheader("📝 Generated RTTM")

    with open(rttm_path) as f:
        st.code(f.read())
