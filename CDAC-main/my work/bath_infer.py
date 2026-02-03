import os
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

audio_dir = r"splits\test\audio"

for wav in os.listdir(audio_dir):
    if wav.endswith(".wav"):
        print("\n==============================")
        print("File:", wav)
        diarization = pipeline(os.path.join(audio_dir, wav))

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"{speaker}: {turn.start:.2f} → {turn.end:.2f}")
