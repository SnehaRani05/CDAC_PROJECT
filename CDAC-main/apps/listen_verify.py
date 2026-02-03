import soundfile as sf
import numpy as np
import os

AUDIO_FILE = "test_audio/audio.wav"
RTTM_FILE = "outputs/output.rttm"
OUT_DIR = "outputs/speakers"

os.makedirs(OUT_DIR, exist_ok=True)

audio, sr = sf.read(AUDIO_FILE)
if audio.ndim == 2:
    audio = audio.mean(axis=1)

speaker_audio = {}

with open(RTTM_FILE) as f:
    for line in f:
        parts = line.strip().split()
        start = float(parts[3])
        dur = float(parts[4])
        speaker = parts[7]

        s = int(start * sr)
        e = int((start + dur) * sr)
        chunk = audio[s:e]

        if np.mean(np.abs(chunk)) > 0.01:
            speaker_audio.setdefault(speaker, []).append(chunk)

for speaker, chunks in speaker_audio.items():
    combined = np.concatenate(chunks)
    path = os.path.join(OUT_DIR, f"{speaker}.wav")
    sf.write(path, combined, sr)
    print("🎧 Saved", path)
