import os
import soundfile as sf
from pyannote.database.util import load_rttm

AUDIO_FILE = r"test_audio.wav"
RTTM_FILE = r"outputs/predicted.rttm"
OUT_DIR = "speakers"
URI = "test_audio"

os.makedirs(OUT_DIR, exist_ok=True)

audio, sr = sf.read(AUDIO_FILE)
annotation = load_rttm(RTTM_FILE)[URI]

speaker_buffers = {}

for segment, _, speaker in annotation.itertracks(yield_label=True):
    start = int(segment.start * sr)
    end = int(segment.end * sr)
    speaker_buffers.setdefault(speaker, []).append(audio[start:end])

for speaker, chunks in speaker_buffers.items():
    speaker_audio = sum(chunks)
    out_path = os.path.join(OUT_DIR, f"{speaker}.wav")
    sf.write(out_path, speaker_audio, sr)
    print(f"✅ Saved {out_path}")
