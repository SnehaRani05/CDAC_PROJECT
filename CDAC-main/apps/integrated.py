import os
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

# -----------------------------
# CONFIG
# -----------------------------
AUDIO_FILE = "input/chunk_0049.wav"
MANUAL_RTTM = "output/chunk_0049.rttm"
OUTPUT_DIR = "output"
SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")
FILE_ID = "chunk_0049"
SR = 16000

os.makedirs(SPEAKER_DIR, exist_ok=True)

# -----------------------------
# LOAD AUDIO
# -----------------------------
audio, sr = librosa.load(AUDIO_FILE, sr=SR)

# -----------------------------
# DIARIZATION (NO EXTERNAL VAD)
# -----------------------------
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline(AUDIO_FILE)

# -----------------------------
# SAVE SPEAKER AUDIO + RTTM
# -----------------------------
hyp_rttm_path = os.path.join(OUTPUT_DIR, "hypothesis.rttm")
rttm_lines = []
segment_id = 0

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start = int(turn.start * SR)
    end = int(turn.end * SR)

    segment_audio = audio[start:end]
    filename = f"{speaker}_{segment_id}.wav"
    sf.write(os.path.join(SPEAKER_DIR, filename), segment_audio, SR)

    rttm_lines.append(
        f"SPEAKER {FILE_ID} 1 {turn.start:.3f} "
        f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
    )

    segment_id += 1

with open(hyp_rttm_path, "w") as f:
    f.writelines(rttm_lines)

print("Speaker audio + RTTM saved")

# -----------------------------
# DER CALCULATION
# -----------------------------
reference = load_rttm(MANUAL_RTTM)[FILE_ID]
hypothesis = load_rttm(hyp_rttm_path)[FILE_ID]

der = DiarizationErrorRate()(reference, hypothesis)
print(f"\nDER: {der:.2%}")
