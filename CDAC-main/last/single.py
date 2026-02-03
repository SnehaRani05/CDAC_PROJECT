import os
import soundfile as sf
import matplotlib.pyplot as plt

from pyannote.audio import Pipeline, Model
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

# =========================
# PATHS (FIXED FOR YOUR AUDIO)
# =========================
AUDIO_PATH = "/mnt/data/48eb4e40a20bd2a968ccf39d1fb275dc36e82a992f08907ccc0748cf.wav"
CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_24\checkpoints\epoch=14-step=7320.ckpt"
REFERENCE_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\last\dataset\rttm"

OUT_DIR = "infer/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SR = 16000
AUDIO_NAME = "test_audio"

# =========================
# LOAD PIPELINE (STABLE CONFIG)
# =========================
segmentation_model = Model.from_pretrained(
    CHECKPOINT, strict=False
)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=False
)

# replace segmentation only
pipeline._segmentation.model = segmentation_model

# IMPORTANT: speech-focused settings
pipeline.segmentation.threshold = 0.65
pipeline.segmentation.min_duration_on = 0.5
pipeline.segmentation.min_duration_off = 0.3

pipeline.clustering.method = "average"
pipeline.clustering.threshold = 0.75
pipeline.clustering.min_num_speakers = 2
pipeline.clustering.max_num_speakers = 5

# =========================
# RUN DIARIZATION (SPEECH-ONLY)
# =========================
print("Running diarization...")

diarization = pipeline(
    AUDIO_PATH,
    hook=lambda s: s[s.get_timeline().support()]  # 🔥 removes noise-only regions
)

print("Done.\n")

# =========================
# PRINT SEGMENTS
# =========================
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append((turn.start, turn.end, speaker))
    print(f"{turn.start:.2f}s → {turn.end:.2f}s : {speaker}")

# =========================
# SAVE RTTM
# =========================
pred_rttm = os.path.join(OUT_DIR, f"{AUDIO_NAME}.rttm")
with open(pred_rttm, "w") as f:
    diarization.write_rttm(f)

print(f"\nRTTM saved to: {pred_rttm}")

# =========================
# TIME vs SPEAKER PLOT
# =========================
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

plt.show()

# =========================
# OPTIONAL: DER (if RTTM exists)
# =========================
ref_rttm = os.path.join(REFERENCE_RTTM_DIR, f"{AUDIO_NAME}.rttm")

if os.path.exists(ref_rttm):
    reference = load_rttm(ref_rttm)
    hypothesis = load_rttm(pred_rttm)

    uri = list(reference.keys())[0]
    der = DiarizationErrorRate()
    score = der(reference[uri], hypothesis[uri]) * 100

    print(f"\nDER = {score:.2f}%")
else:
    print("\nNo reference RTTM (unseen audio) → DER skipped")
