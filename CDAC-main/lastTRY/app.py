import streamlit as st
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyannote.audio import Pipeline, Inference
from sklearn.metrics.pairwise import cosine_similarity

from pyannote.audio import Model
from pyannote.audio import Inference

# ================= CONFIG =================

BASE_DIR = "output"
INPUT_AUDIO = os.path.join(BASE_DIR, "input.wav")
SPEAKER_DIR = os.path.join(BASE_DIR, "speakers")
RTTM_PATH = os.path.join(BASE_DIR, "output.rttm")

os.makedirs(SPEAKER_DIR, exist_ok=True)

SR = 16000

# ================= PAGE =================

st.set_page_config(layout="wide")
st.title("Speaker Diarization (Unknown Speaker Count)")
st.caption("pyannote + speaker embeddings + automatic merging")

# ================= SIDEBAR =================

st.sidebar.header("Controls")

min_segment_len = st.sidebar.slider(
    "Min segment length (sec)",
    0.3, 2.0, 0.7
)

merge_threshold = st.sidebar.slider(
    "Speaker merge threshold (cosine)",
    0.70, 0.90, 0.80
)

# ================= MODELS =================

@st.cache_resource
def load_diarization_pipeline():
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )
    pipe.to(torch.device("cpu"))
    return pipe




@st.cache_resource
def load_embedding_model():
    return Inference(
        "pyannote/embedding",
        window="whole",
        device=torch.device("cpu")   # 🔥 THIS FIXES THE CRASH
    )


# ================= UPLOAD =================

uploaded = st.file_uploader(
    "Upload WAV audio",
    type=["wav"]
)

if uploaded:
    with open(INPUT_AUDIO, "wb") as f:
        f.write(uploaded.read())

    st.success("Audio uploaded")
    st.audio(INPUT_AUDIO)

    wav, sr = torchaudio.load(INPUT_AUDIO)
    duration = wav.shape[1] / sr
    st.metric("Audio duration", f"{duration:.2f} sec")

# ================= MERGE FUNCTION =================

def merge_speakers(segments, threshold):

    speakers = list({s["speaker"] for s in segments})

    mean_emb = {}
    for spk in speakers:
        embs = [s["embedding"] for s in segments if s["speaker"] == spk]
        mean_emb[spk] = np.mean(embs, axis=0)

    mapping = {}
    used = set()

    for spk in speakers:
        if spk in used:
            continue

        mapping[spk] = spk
        used.add(spk)

        for other in speakers:
            if other in used:
                continue

            sim = cosine_similarity(
                mean_emb[spk].reshape(1, -1),
                mean_emb[other].reshape(1, -1)
            )[0][0]

            if sim >= threshold:
                mapping[other] = spk
                used.add(other)

    for s in segments:
        s["speaker"] = mapping[s["speaker"]]

    return segments

# ================= RUN =================

if st.button("Run Diarization"):

    if not os.path.exists(INPUT_AUDIO):
        st.error("Please upload audio first")
        st.stop()

    # Clear old files
    for f in os.listdir(SPEAKER_DIR):
        os.remove(os.path.join(SPEAKER_DIR, f))

    with st.spinner("Running diarization..."):

        pipeline = load_diarization_pipeline()
        embedder = load_embedding_model()

        wav, sr = torchaudio.load(INPUT_AUDIO)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        audio = wav.squeeze().numpy()

        diarization = pipeline(
            INPUT_AUDIO,
            min_speakers=1,
            max_speakers=10
        )

        segments = []

        for i, (turn, _, speaker) in enumerate(
            diarization.itertracks(yield_label=True)
        ):
            dur = turn.end - turn.start
            if dur < min_segment_len:
                continue

            start = int(turn.start * sr)
            end = int(turn.end * sr)

            path = os.path.join(SPEAKER_DIR, f"{speaker}_{i}.wav")
            sf.write(path, audio[start:end], sr)

            emb = embedder(path)
            emb = emb / np.linalg.norm(emb)

            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "path": path,
                "embedding": emb
            })

        # Merge speakers
        segments = merge_speakers(segments, merge_threshold)

        # Write RTTM
        with open(RTTM_PATH, "w") as f:
            for s in segments:
                f.write(
                    f"SPEAKER session 1 "
                    f"{s['start']:.3f} "
                    f"{s['end']-s['start']:.3f} "
                    f"<NA> <NA> {s['speaker']} <NA> <NA>\n"
                )

    # ================= RESULTS =================

    final_speakers = sorted({s["speaker"] for s in segments})

    st.subheader("Results")
    st.metric("Final speaker count", len(final_speakers))

    # Timeline
    st.subheader("Speaker timeline")

    df = pd.DataFrame(segments)
    fig, ax = plt.subplots(figsize=(12, 3))

    y_map = {s: i for i, s in enumerate(final_speakers)}

    for _, r in df.iterrows():
        ax.barh(
            y_map[r["speaker"]],
            r["end"] - r["start"],
            left=r["start"]
        )

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Time (seconds)")

    st.pyplot(fig)

    # Playback
    st.subheader("Speaker playback")

    cols = st.columns(2)
    for i, s in enumerate(segments):
        with cols[i % 2]:
            st.markdown(f"### 🎤 {s['speaker']}")
            st.write(f"{s['start']:.2f}s → {s['end']:.2f}s")
            st.audio(s["path"])

    # RTTM
    st.subheader("Generated RTTM")
    st.code(open(RTTM_PATH).read())
