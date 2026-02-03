import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pyannote.audio import Pipeline

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Multilingual Speaker Diarization",
    layout="wide"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🎙️ Speaker Diarization")

audio_file = st.sidebar.file_uploader(
    "Upload WAV Audio",
    type=["wav"]
)

language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Hindi", "Regional"]
)

run_btn = st.sidebar.button("▶ Run Diarization")

st.sidebar.markdown("---")
st.sidebar.info(
    "This UI demonstrates end-to-end speaker diarization:\n"
    "Audio → Segments → Speakers → Timeline"
)

# ------------------ MAIN TITLE ------------------
st.title("Multilingual Speaker Diarization System")

# ------------------ AUDIO PLAYER ------------------
if audio_file is not None:
    st.subheader("🔊 Uploaded Audio")
    st.audio(audio_file)

# ------------------ RUN DIARIZATION ------------------
if run_btn and audio_file is not None:

    st.info("Running speaker diarization... please wait ⏳")

    # Save uploaded audio temporarily
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    # Load pretrained pyannote pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )

    diarization = pipeline("temp.wav")

    # Convert diarization output to dataframe
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "Speaker": speaker,
            "Start (s)": round(turn.start, 2),
            "End (s)": round(turn.end, 2),
            "Duration (s)": round(turn.end - turn.start, 2)
        })

    df = pd.DataFrame(segments)

    st.success("✅ Diarization completed")

    # ------------------ RESULTS TABLE ------------------
    st.subheader("🗂️ Speaker Segments")
    st.dataframe(df, use_container_width=True)

    # ------------------ TIMELINE VISUALIZATION ------------------
    st.subheader("📊 Speaker Timeline")

    fig, ax = plt.subplots(figsize=(12, 3))

    speakers = df["Speaker"].unique()
    speaker_map = {spk: i for i, spk in enumerate(speakers)}

    for _, row in df.iterrows():
        ax.barh(
            speaker_map[row["Speaker"]],
            row["Duration (s)"],
            left=row["Start (s)"]
        )

    ax.set_yticks(list(speaker_map.values()))
    ax.set_yticklabels(list(speaker_map.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speakers")
    ax.set_title(f"Speaker Timeline ({language})")

    st.pyplot(fig)

    # ------------------ METRICS ------------------
    st.subheader("📉 Evaluation")
    st.metric(
        label="Diarization Error Rate (DER)",
        value="Not computed",
        help="RTTM ground truth required"
    )

    # Cleanup temp file
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption(
    "Built using Streamlit + PyAnnote | Multilingual Speaker Diarization Project"
)
