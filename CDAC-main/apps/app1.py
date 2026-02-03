# # # import streamlit as st
# # # import os
# # # import subprocess
# # # import sys
# # # import shutil

# # # # =====================
# # # # Helper function
# # # # =====================
# # # def get_reference_rttm(original_audio_name):
# # #     """
# # #     Return reference RTTM path if it exists, else None
# # #     """
# # #     REF_RTTM_DIR = "processed/dataset/rttm"
# # #     base = os.path.splitext(original_audio_name)[0]
# # #     ref_path = os.path.join(REF_RTTM_DIR, base + ".rttm")

# # #     if os.path.exists(ref_path):
# # #         return ref_path
# # #     return None


# # # # =====================
# # # # Page config
# # # # =====================
# # # st.set_page_config(page_title="Speech Processing Pipeline", layout="wide")
# # # st.title("🎙️ Speech Processing Pipeline")

# # # UPLOAD_DIR = "test_audio"
# # # OUTPUT_DIR = "outputs"
# # # SPEAKER_DIR = "outputs/speakers"
# # # REF_RTTM_DIR = "processed/dataset/rttm"

# # # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # # os.makedirs(OUTPUT_DIR, exist_ok=True)
# # # os.makedirs(SPEAKER_DIR, exist_ok=True)
# # # os.makedirs(REF_RTTM_DIR, exist_ok=True)

# # # # =====================
# # # # Tabs
# # # # =====================
# # # tab1, tab2, tab3 = st.tabs([
# # #     "🔇 Noise Reduction & VAD",
# # #     "🧠 Diarization & DER",
# # #     "📊 Results"
# # # ])

# # # # Store original filename across tabs
# # # if "original_audio_name" not in st.session_state:
# # #     st.session_state["original_audio_name"] = None

# # # # =====================================================
# # # # TAB 1 — Noise Reduction + VAD
# # # # =====================================================
# # # with tab1:
# # #     st.header("🔇 Noise Reduction & Voice Activity Detection")

# # #     audio_file = st.file_uploader("Upload WAV file", type=["wav"])

# # #     if audio_file:
# # #         original_audio_name = audio_file.name
# # #         st.session_state["original_audio_name"] = original_audio_name

# # #         temp_path = os.path.join(UPLOAD_DIR, original_audio_name)
# # #         with open(temp_path, "wb") as f:
# # #             f.write(audio_file.getbuffer())

# # #         # Normalize ONLY for backend processing
# # #         final_audio = os.path.join(UPLOAD_DIR, "audio.wav")
# # #         shutil.move(temp_path, final_audio)

# # #         st.audio(final_audio)

# # #         if st.button("Run Noise Reduction + VAD"):
# # #             with st.spinner("Running noise reduction and VAD..."):
# # #                 subprocess.run(
# # #                     [sys.executable, "process.py"],
# # #                     check=True
# # #                 )
# # #             st.success("✅ Noise reduction & VAD completed")

# # #             if os.path.exists("outputs/clean.wav"):
# # #                 st.audio("outputs/clean.wav")
# # #                 st.caption("Cleaned Audio")

# # # # =====================================================
# # # # TAB 2 — Diarization + DER
# # # # =====================================================
# # # with tab2:
# # #     st.header("🧠 Speaker Diarization & DER")

# # #     if st.button("Run Diarization + DER"):
# # #         if not st.session_state["original_audio_name"]:
# # #             st.error("❌ Please upload audio in Tab 1 first.")
# # #         else:
# # #             with st.spinner("Running diarization and DER..."):

# # #                 # 1️⃣ Run diarization (creates outputs/output.rttm)
# # #                 subprocess.run(
# # #                     [sys.executable, "diarize.py"],
# # #                     check=True,
# # #                     env=os.environ
# # #                 )

# # #                 original_audio_name = st.session_state["original_audio_name"]
# # #                 ref_rttm = get_reference_rttm(original_audio_name)

# # #                 # 2️⃣ Compute DER if reference RTTM exists
# # #                 if ref_rttm:
# # #                     st.info(f"✅ Reference RTTM found: {os.path.basename(ref_rttm)}")

# # #                     result = subprocess.run(
# # #                         [sys.executable, "der.py", original_audio_name],
# # #                         capture_output=True,
# # #                         text=True
# # #                     )

# # #                     output = result.stdout.strip()

# # #                     if output == "NO_REF_RTTM":
# # #                         st.error("❌ Reference RTTM could not be read.")
# # #                     else:
# # #                         st.session_state["DER"] = float(output)
# # #                         st.success("✅ DER computed successfully")

# # #                 else:
# # #                     st.warning("⚠️ Reference RTTM not found. DER skipped.")

# # #             st.success("✅ Diarization completed")

# # # # =====================================================
# # # # TAB 3 — Results
# # # # =====================================================
# # # with tab3:
# # #     st.header("📊 Results")

# # #     # Show DER
# # #     if "DER" in st.session_state:
# # #         st.metric("Diarization Error Rate (DER)", f"{st.session_state['DER']:.2f}%")

# # #     # Speaker segments
# # #     st.subheader("🗣️ Speaker Segments")

# # #     if st.button("Generate Speaker Segments"):
# # #         subprocess.run(
# # #             [sys.executable, "segment_speakers.py"],
# # #             check=True
# # #         )

# # #     if os.path.exists(SPEAKER_DIR):
# # #         for file in sorted(os.listdir(SPEAKER_DIR)):
# # #             if file.endswith(".wav"):
# # #                 path = os.path.join(SPEAKER_DIR, file)
# # #                 st.audio(path)
# # #                 with open(path, "rb") as f:
# # #                     st.download_button(
# # #                         f"Download {file}",
# # #                         f,
# # #                         file_name=file,
# # #                         mime="audio/wav"
# # #                     )

# # #     # RTTM download
# # #     if os.path.exists("outputs/output.rttm"):
# # #         with open("outputs/output.rttm", "rb") as f:
# # #             st.download_button(
# # #                 "📄 Download Hypothesis RTTM",
# # #                 f,
# # #                 file_name="output.rttm",
# # #                 mime="text/plain"
# # #             )


# # import streamlit as st
# # import os
# # import subprocess
# # import sys
# # import shutil
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # =====================
# # # Helper functions
# # # =====================
# # def get_reference_rttm(original_audio_name):
# #     """
# #     Return reference RTTM path if it exists, else None
# #     """
# #     REF_RTTM_DIR = "processed/dataset/rttm"
# #     base = os.path.splitext(original_audio_name)[0]
# #     ref_path = os.path.join(REF_RTTM_DIR, base + ".rttm")

# #     if os.path.exists(ref_path):
# #         return ref_path
# #     return None


# # def rttm_to_dataframe(rttm_path):
# #     rows = []
# #     with open(rttm_path) as f:
# #         for line in f:
# #             p = line.strip().split()
# #             start = float(p[3])
# #             dur = float(p[4])
# #             speaker = p[7]
# #             rows.append({
# #                 "speaker": speaker,
# #                 "start": start,
# #                 "end": start + dur
# #             })
# #     return pd.DataFrame(rows)


# # def plot_speaker_timeline(df, title):
# #     speakers = sorted(df["speaker"].unique())
# #     speaker_to_y = {spk: i for i, spk in enumerate(speakers)}

# #     fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))

# #     for _, row in df.iterrows():
# #         y = speaker_to_y[row["speaker"]]
# #         ax.barh(
# #             y=y,
# #             width=row["end"] - row["start"],
# #             left=row["start"]
# #         )

# #     ax.set_yticks(list(speaker_to_y.values()))
# #     ax.set_yticklabels(speakers)
# #     ax.set_xlabel("Time (seconds)")
# #     ax.set_ylabel("Speaker")
# #     ax.set_title(title)
# #     ax.grid(True, axis="x", linestyle="--", alpha=0.5)

# #     return fig


# # # =====================
# # # Page config
# # # =====================
# # st.set_page_config(page_title="Speech Processing Pipeline", layout="wide")
# # st.title("🎙️ Speech Processing Pipeline")

# # UPLOAD_DIR = "test_audio"
# # OUTPUT_DIR = "outputs"
# # SPEAKER_DIR = "outputs/speakers"
# # REF_RTTM_DIR = "processed/dataset/rttm"

# # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # os.makedirs(OUTPUT_DIR, exist_ok=True)
# # os.makedirs(SPEAKER_DIR, exist_ok=True)
# # os.makedirs(REF_RTTM_DIR, exist_ok=True)

# # # =====================
# # # Tabs
# # # =====================
# # tab1, tab2, tab3 = st.tabs([
# #     "🔇 Noise Reduction & VAD",
# #     "🧠 Diarization & DER",
# #     "📊 Results"
# # ])

# # if "original_audio_name" not in st.session_state:
# #     st.session_state["original_audio_name"] = None

# # # =====================================================
# # # TAB 1 — Noise Reduction + VAD
# # # =====================================================
# # with tab1:
# #     st.header("🔇 Noise Reduction & Voice Activity Detection")

# #     audio_file = st.file_uploader("Upload WAV file", type=["wav"])

# #     if audio_file:
# #         original_audio_name = audio_file.name
# #         st.session_state["original_audio_name"] = original_audio_name

# #         temp_path = os.path.join(UPLOAD_DIR, original_audio_name)
# #         with open(temp_path, "wb") as f:
# #             f.write(audio_file.getbuffer())

# #         final_audio = os.path.join(UPLOAD_DIR, "audio.wav")
# #         shutil.move(temp_path, final_audio)

# #         st.audio(final_audio)

# #         if st.button("Run Noise Reduction + VAD"):
# #             with st.spinner("Running noise reduction and VAD..."):
# #                 subprocess.run(
# #                     [sys.executable, "process.py"],
# #                     check=True
# #                 )
# #             st.success("✅ Noise reduction & VAD completed")

# #             if os.path.exists("outputs/clean.wav"):
# #                 st.audio("outputs/clean.wav")
# #                 st.caption("Cleaned Audio")

# # # =====================================================
# # # TAB 2 — Diarization + DER
# # # =====================================================
# # with tab2:
# #     st.header("🧠 Speaker Diarization & DER")

# #     if st.button("Run Diarization + DER"):
# #         if not st.session_state["original_audio_name"]:
# #             st.error("❌ Please upload audio in Tab 1 first.")
# #         else:
# #             with st.spinner("Running diarization and DER..."):

# #                 subprocess.run(
# #                     [sys.executable, "diarize.py"],
# #                     check=True,
# #                     env=os.environ
# #                 )

# #                 original_audio_name = st.session_state["original_audio_name"]
# #                 ref_rttm = get_reference_rttm(original_audio_name)

# #                 if ref_rttm:
# #                     st.info(f"✅ Reference RTTM found: {os.path.basename(ref_rttm)}")

# #                     result = subprocess.run(
# #                         [sys.executable, "der.py", original_audio_name],
# #                         capture_output=True,
# #                         text=True
# #                     )

# #                     output = result.stdout.strip()

# #                     if output == "NO_REF_RTTM":
# #                         st.error("❌ Reference RTTM could not be read.")
# #                     else:
# #                         st.session_state["DER"] = float(output)
# #                         st.success("✅ DER computed successfully")

# #                 else:
# #                     st.warning("⚠️ Reference RTTM not found. DER skipped.")

# #             st.success("✅ Diarization completed")

# # # =====================================================
# # # TAB 3 — Results
# # # =====================================================
# # with tab3:
# #     st.header("📊 Results")

# #     # DER
# #     if "DER" in st.session_state:
# #         st.metric("Diarization Error Rate (DER)", f"{st.session_state['DER']:.2f}%")

# #     # Speaker Segments
# #     st.subheader("🗣️ Speaker Segments")

# #     if st.button("Generate Speaker Segments"):
# #         subprocess.run(
# #             [sys.executable, "segment_speakers.py"],
# #             check=True
# #         )

# #     if os.path.exists(SPEAKER_DIR):
# #         for file in sorted(os.listdir(SPEAKER_DIR)):
# #             if file.endswith(".wav"):
# #                 path = os.path.join(SPEAKER_DIR, file)
# #                 st.audio(path)
# #                 with open(path, "rb") as f:
# #                     st.download_button(
# #                         f"Download {file}",
# #                         f,
# #                         file_name=file,
# #                         mime="audio/wav"
# #                     )

# #     # RTTM download
# #     if os.path.exists("outputs/output.rttm"):
# #         with open("outputs/output.rttm", "rb") as f:
# #             st.download_button(
# #                 "📄 Download Hypothesis RTTM",
# #                 f,
# #                 file_name="output.rttm",
# #                 mime="text/plain"
# #             )

# #     # =========================
# #     # Speaker vs Time Graphs
# #     # =========================
# #     st.subheader("📈 Speaker vs Time")

# #     HYP_RTTM = "outputs/output.rttm"
# #     original_audio_name = st.session_state.get("original_audio_name")
# #     ref_rttm = get_reference_rttm(original_audio_name) if original_audio_name else None

# #     if os.path.exists(HYP_RTTM):
# #         st.markdown("### 🤖 Hypothesis RTTM")
# #         hyp_df = rttm_to_dataframe(HYP_RTTM)
# #         st.pyplot(plot_speaker_timeline(hyp_df, "Hypothesis Speaker Timeline"))

# #     if ref_rttm:
# #         st.markdown("### 👤 Manual (Reference) RTTM")
# #         ref_df = rttm_to_dataframe(ref_rttm)
# #         st.pyplot(plot_speaker_timeline(ref_df, "Manual Speaker Timeline"))


# import streamlit as st
# import os
# import subprocess
# import sys
# import shutil
# import pandas as pd
# import matplotlib.pyplot as plt

# # =====================
# # Helper functions
# # =====================
# def get_reference_rttm(original_audio_name):
#     """
#     Return reference RTTM path if it exists, else None
#     """
#     REF_RTTM_DIR = "processed/dataset/rttm"
#     base = os.path.splitext(original_audio_name)[0]
#     ref_path = os.path.join(REF_RTTM_DIR, base + ".rttm")
#     return ref_path if os.path.exists(ref_path) else None


# def rttm_to_dataframe(rttm_path):
#     rows = []
#     with open(rttm_path) as f:
#         for line in f:
#             p = line.strip().split()
#             start = float(p[3])
#             dur = float(p[4])
#             speaker = p[7]
#             rows.append({
#                 "speaker": speaker,
#                 "start": start,
#                 "end": start + dur
#             })
#     return pd.DataFrame(rows)


# def plot_speaker_time(df, title):
#     speakers = sorted(df["speaker"].unique())
#     speaker_map = {spk: i for i, spk in enumerate(speakers)}

#     fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))

#     for _, row in df.iterrows():
#         ax.barh(
#             speaker_map[row["speaker"]],
#             row["end"] - row["start"],
#             left=row["start"]
#         )

#     ax.set_yticks(list(speaker_map.values()))
#     ax.set_yticklabels(speakers)
#     ax.set_xlabel("Time (seconds)")
#     ax.set_ylabel("Speaker")
#     ax.set_title(title)
#     ax.grid(axis="x", linestyle="--", alpha=0.5)

#     return fig


# # =====================
# # Page config
# # =====================
# st.set_page_config(page_title="Speech Processing Pipeline", layout="wide")
# st.title("🎙️ Speech Processing Pipeline")

# UPLOAD_DIR = "test_audio"
# OUTPUT_DIR = "outputs"
# REF_RTTM_DIR = "processed/dataset/rttm"

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(REF_RTTM_DIR, exist_ok=True)

# # =====================
# # Tabs
# # =====================
# tab1, tab2, tab3 = st.tabs([
#     "🔇 Noise Reduction & VAD",
#     "🧠 Diarization & DER",
#     "📊 Results"
# ])

# if "original_audio_name" not in st.session_state:
#     st.session_state["original_audio_name"] = None

# # =====================================================
# # TAB 1 — Noise Reduction + VAD
# # =====================================================
# with tab1:
#     st.header("🔇 Noise Reduction & Voice Activity Detection")

#     audio_file = st.file_uploader("Upload WAV file", type=["wav"])

#     if audio_file:
#         original_audio_name = audio_file.name
#         st.session_state["original_audio_name"] = original_audio_name

#         temp_path = os.path.join(UPLOAD_DIR, original_audio_name)
#         with open(temp_path, "wb") as f:
#             f.write(audio_file.getbuffer())

#         final_audio = os.path.join(UPLOAD_DIR, "audio.wav")
#         shutil.move(temp_path, final_audio)

#         st.audio(final_audio)

#         if st.button("Run Noise Reduction + VAD"):
#             with st.spinner("Running noise reduction and VAD..."):
#                 subprocess.run(
#                     [sys.executable, "process.py"],
#                     check=True
#                 )
#             st.success("✅ Noise reduction & VAD completed")

#             if os.path.exists("outputs/clean.wav"):
#                 st.audio("outputs/clean.wav")
#                 st.caption("Cleaned Audio")

# # =====================================================
# # TAB 2 — Diarization + DER
# # =====================================================
# with tab2:
#     st.header("🧠 Speaker Diarization & DER")

#     if st.button("Run Diarization + DER"):
#         if not st.session_state["original_audio_name"]:
#             st.error("❌ Please upload audio in Tab 1 first.")
#         else:
#             with st.spinner("Running diarization and DER..."):

#                 subprocess.run(
#                     [sys.executable, "diarize.py"],
#                     check=True,
#                     env=os.environ
#                 )

#                 original_audio_name = st.session_state["original_audio_name"]
#                 ref_rttm = get_reference_rttm(original_audio_name)

#                 if ref_rttm:
#                     st.info(f"✅ Reference RTTM found: {os.path.basename(ref_rttm)}")

#                     result = subprocess.run(
#                         [sys.executable, "der.py", original_audio_name],
#                         capture_output=True,
#                         text=True
#                     )

#                     output = result.stdout.strip()

#                     if output == "NO_REF_RTTM":
#                         st.error("❌ Reference RTTM could not be read.")
#                     else:
#                         st.session_state["DER"] = float(output)
#                         st.success("✅ DER computed successfully")

#                 else:
#                     st.warning("⚠️ Reference RTTM not found. DER skipped.")

#             st.success("✅ Diarization completed")

# # =====================================================
# # TAB 3 — Results
# # =====================================================
# with tab3:
#     st.header("📊 Results")

#     # DER
#     if "DER" in st.session_state:
#         st.metric("Diarization Error Rate (DER)", f"{st.session_state['DER']:.2f}%")

#     # RTTM download
#     if os.path.exists("outputs/output.rttm"):
#         with open("outputs/output.rttm", "rb") as f:
#             st.download_button(
#                 "📄 Download Hypothesis RTTM",
#                 f,
#                 file_name="output.rttm",
#                 mime="text/plain"
#             )

#     # =========================
#     # Speaker vs Time Graphs
#     # =========================
#     st.subheader("📈 Speaker vs Time Graph")

#     HYP_RTTM = "outputs/output.rttm"
#     original_audio_name = st.session_state.get("original_audio_name")
#     ref_rttm = get_reference_rttm(original_audio_name) if original_audio_name else None

#     if os.path.exists(HYP_RTTM):
#         hyp_df = rttm_to_dataframe(HYP_RTTM)
#         st.markdown("### 🤖 Hypothesis RTTM")
#         st.pyplot(plot_speaker_time(hyp_df, "Hypothesis Speaker Timeline"))

#     if ref_rttm:
#         ref_df = rttm_to_dataframe(ref_rttm)
#         st.markdown("### 👤 Manual (Reference) RTTM")
#         st.pyplot(plot_speaker_time(ref_df, "Manual Speaker Timeline"))
#     else:
#         st.warning("Reference RTTM not available for this audio.")


import streamlit as st
import os
import subprocess
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Helper functions
# =====================
def get_reference_rttm(original_audio_name):
    REF_RTTM_DIR = "processed/dataset/rttm"
    base = os.path.splitext(original_audio_name)[0]
    ref_path = os.path.join(REF_RTTM_DIR, base + ".rttm")
    return ref_path if os.path.exists(ref_path) else None


def rttm_to_dataframe(rttm_path):
    rows = []
    with open(rttm_path) as f:
        for line in f:
            p = line.strip().split()
            start = float(p[3])
            dur = float(p[4])
            speaker = p[7]
            rows.append({
                "speaker": speaker,
                "start": start,
                "end": start + dur
            })
    return pd.DataFrame(rows)


def plot_speaker_time(df, title):
    speakers = sorted(df["speaker"].unique())
    speaker_map = {spk: i for i, spk in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))

    for _, row in df.iterrows():
        ax.barh(
            speaker_map[row["speaker"]],
            row["end"] - row["start"],
            left=row["start"]
        )

    ax.set_yticks(list(speaker_map.values()))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    return fig


def count_speakers_from_rttm(rttm_path):
    speakers = set()
    with open(rttm_path) as f:
        for line in f:
            speakers.add(line.strip().split()[7])
    return len(speakers), sorted(speakers)


# =====================
# Page config
# =====================
st.set_page_config(page_title="Speech Processing Pipeline", layout="wide")
st.title("🎙️ Speech Processing Pipeline")

UPLOAD_DIR = "test_audio"
OUTPUT_DIR = "outputs"
REF_RTTM_DIR = "processed/dataset/rttm"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_RTTM_DIR, exist_ok=True)

# =====================
# Tabs
# =====================
tab1, tab2, tab3 = st.tabs([
    "🔇 Noise Reduction & VAD",
    "🧠 Diarization & DER",
    "📊 Results"
])

if "original_audio_name" not in st.session_state:
    st.session_state["original_audio_name"] = None

# =====================================================
# TAB 1 — Noise Reduction + VAD
# =====================================================
with tab1:
    st.header("🔇 Noise Reduction & Voice Activity Detection")

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file:
        original_audio_name = audio_file.name
        st.session_state["original_audio_name"] = original_audio_name

        temp_path = os.path.join(UPLOAD_DIR, original_audio_name)
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())

        final_audio = os.path.join(UPLOAD_DIR, "audio.wav")
        shutil.move(temp_path, final_audio)

        st.audio(final_audio)

        if st.button("Run Noise Reduction + VAD"):
            with st.spinner("Running noise reduction and VAD..."):
                subprocess.run(
                    [sys.executable, "process.py"],
                    check=True
                )
            st.success("✅ Noise reduction & VAD completed")

# =====================================================
# TAB 2 — Diarization + DER
# =====================================================
with tab2:
    st.header("🧠 Speaker Diarization & DER")

    if st.button("Run Diarization + DER"):
        if not st.session_state["original_audio_name"]:
            st.error("❌ Please upload audio in Tab 1 first.")
        else:
            with st.spinner("Running diarization and DER..."):

                subprocess.run(
                    [sys.executable, "diarize.py"],
                    check=True,
                    env=os.environ
                )

                original_audio_name = st.session_state["original_audio_name"]
                ref_rttm = get_reference_rttm(original_audio_name)

                if ref_rttm:
                    result = subprocess.run(
                        [sys.executable, "der.py", original_audio_name],
                        capture_output=True,
                        text=True
                    )

                    output = result.stdout.strip()
                    if output != "NO_REF_RTTM":
                        st.session_state["DER"] = float(output)
                        st.success("✅ DER computed")

                else:
                    st.warning("⚠️ Reference RTTM not found. DER skipped.")

            st.success("✅ Diarization completed")

# =====================================================
# TAB 3 — Results
# =====================================================
with tab3:
    st.header("📊 Results")

    # DER
    if "DER" in st.session_state:
        st.metric("Diarization Error Rate (DER)", f"{st.session_state['DER']:.2f}%")

    # =========================
    # Speaker Count (ADDED)
    # =========================
    HYP_RTTM = "outputs/output.rttm"

    if os.path.exists(HYP_RTTM):
        num_speakers, speaker_list = count_speakers_from_rttm(HYP_RTTM)
        st.metric("🗣️ Number of Speakers Detected", num_speakers)

        with st.expander("Show detected speaker labels"):
            for spk in speaker_list:
                st.write(f"• {spk}")

    # RTTM download
    if os.path.exists(HYP_RTTM):
        with open(HYP_RTTM, "rb") as f:
            st.download_button(
                "📄 Download Hypothesis RTTM",
                f,
                file_name="output.rttm",
                mime="text/plain"
            )

    # =========================
    # Speaker vs Time Graphs
    # =========================
    st.subheader("📈 Speaker vs Time")

    original_audio_name = st.session_state.get("original_audio_name")
    ref_rttm = get_reference_rttm(original_audio_name) if original_audio_name else None

    if os.path.exists(HYP_RTTM):
        hyp_df = rttm_to_dataframe(HYP_RTTM)
        st.markdown("### 🤖 Hypothesis RTTM")
        st.pyplot(plot_speaker_time(hyp_df, "Hypothesis Speaker Timeline"))

    if ref_rttm:
        ref_df = rttm_to_dataframe(ref_rttm)
        st.markdown("### 👤 Manual (Reference) RTTM")
        st.pyplot(plot_speaker_time(ref_df, "Manual Speaker Timeline"))
