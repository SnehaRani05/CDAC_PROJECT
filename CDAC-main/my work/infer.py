from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True   # if required
)

audio_file = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\audio\audio_00001.wav"

diarization = pipeline(audio_file)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.2f} → {turn.end:.2f}")
