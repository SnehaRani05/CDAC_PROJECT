from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
metric = DiarizationErrorRate()

audio = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\audio\audio_00001.wav"
rttm  = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\rttm\audio_00001.rttm"

hyp = pipeline(audio)
ref = load_rttm(rttm)["audio_00001"]

print("DER:", metric(ref, hyp))
