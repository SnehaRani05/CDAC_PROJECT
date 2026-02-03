# segment_speakers.py
import os
import soundfile as sf
from pyannote.core import Segment

AUDIO_PATH = "test_audio/audio.wav"
RTTM_PATH = "outputs/output.rttm"
OUT_DIR = "outputs/speakers"

os.makedirs(OUT_DIR, exist_ok=True)

audio, sr = sf.read(AUDIO_PATH)

def read_rttm(path):
    segments = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            start = float(p[3])
            dur = float(p[4])
            speaker = p[7]
            segments.append((speaker, start, start + dur))
    return segments

segments = read_rttm(RTTM_PATH)

for i, (spk, start, end) in enumerate(segments):
    s = int(start * sr)
    e = int(end * sr)

    chunk = audio[s:e]
    out_path = os.path.join(OUT_DIR, f"{spk}_{i}.wav")
    sf.write(out_path, chunk, sr)

print("Speaker segments created")
