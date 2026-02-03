from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

REFERENCE_RTTM = r"dataset/rttm/audio_001.rttm"
PREDICTED_RTTM = r"outputs/predicted.rttm"
URI = "audio_001"

reference = load_rttm(REFERENCE_RTTM)[URI]
hypothesis = load_rttm(PREDICTED_RTTM)[URI]

metric = DiarizationErrorRate()
der = metric(reference, hypothesis)

print(f"\n✅ DER = {der:.4f}")
print(metric)
