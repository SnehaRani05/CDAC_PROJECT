import os
import streamlit as st
import librosa
import pandas as pd
import matplotlib.pyplot as plt

from pyannote.audio import Pipeline

# =========================
# CONFIG
# =========================
SR = 16000
OUTPUT_DIR = "output"
RTTM_PATH = os.path.join(OUTPUT_DIR, "diarization.rttm")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("🎙 Speaker Diarization Viewer (RTTM + Timeline)")

uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"])

if uploaded_audio is not None:

    # -------------------------
    # Save uploaded audio
    # -------------------------
    audio_path = os.path.join(OUTPUT_DIR, uploaded_audio.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())

    st.audio(audio_path)

    # -------------------------
    # Run diarization
    # -------------------------
    with st.spinner("Running speaker diarization..."):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=os.getenv("HF_TOKEN")
        )

        diarization = pipeline(audio_path)

    # -------------------------
    # Save RTTM (NO MERGING)
    # -------------------------
    rttm_lines = []
    records = []

    file_id = os.path.splitext(uploaded_audio.name)[0]

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        duration = turn.end - turn.start

        rttm_lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>\n"
        )

        records.append({
            "speaker": speaker,
            "start": start,
            "end": turn.end,
            "duration": duration
        })

    with open(RTTM_PATH, "w") as f:
        f.writelines(rttm_lines)

    st.success("RTTM generated successfully (segments NOT merged)")

    # -------------------------
    # RTTM TABLE
    # -------------------------
    df = pd.DataFrame(records).sort_values("start")

    st.subheader("📄 RTTM Segments (Speaker-wise)")
    st.dataframe(df, use_container_width=True)

    # -------------------------
    # TIMELINE PLOT
    # -------------------------
    st.subheader("📊 Time vs Speaker Timeline")

    speakers = df["speaker"].unique()
    speaker_to_y = {spk: i for i, spk in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, 3 + len(speakers)))

    colors = plt.cm.tab10.colors  # fixed distinct colors

    for i, speaker in enumerate(speakers):
        spk_df = df[df["speaker"] == speaker]
        for _, row in spk_df.iterrows():
            ax.barh(
                speaker_to_y[speaker],
                row["duration"],
                left=row["start"],
                color=colors[i % len(colors)],
                edgecolor="black"
            )

    ax.set_yticks(list(speaker_to_y.values()))
    ax.set_yticklabels(list(speaker_to_y.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speaker")
    ax.set_title("Speaker Diarization Timeline")

    st.pyplot(fig)

    # -------------------------
    # SHOW RAW RTTM
    # -------------------------
    st.subheader("📝 Raw RTTM File")
    with open(RTTM_PATH) as f:
        st.code(f.read(), language="text")
