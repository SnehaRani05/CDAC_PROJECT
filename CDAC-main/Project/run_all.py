import os
from diarize import run_diarization
from paths import AUDIO_DIR

for audio in sorted(os.listdir(AUDIO_DIR)):
    if not audio.endswith(".wav"):
        continue

    audio_path = os.path.join(AUDIO_DIR, audio)

    print("\n==============================")
    print("Processing:", audio)
    print("==============================")

    run_diarization(audio_path)
    os.system(f"python der.py {audio}")
