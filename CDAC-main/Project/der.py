import os
import sys
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from paths import RTTM_DIR, HYP_RTTM


def read_rttm(path):
    ann = Annotation()
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            start = float(p[3])
            dur = float(p[4])
            speaker = p[7]
            ann[Segment(start, start + dur)] = speaker
    return ann


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python der.py audio_00007.wav")
        sys.exit(1)

    audio_name = sys.argv[1]
    file_id = os.path.splitext(audio_name)[0]

    ref_rttm = os.path.join(RTTM_DIR, file_id + ".rttm")

    if not os.path.exists(ref_rttm):
        print("❌ Manual RTTM not found:", ref_rttm)
        sys.exit(0)

    reference = read_rttm(ref_rttm)
    hypothesis = read_rttm(HYP_RTTM)

    metric = DiarizationErrorRate(collar=0.25)
    der = metric(reference, hypothesis)

    print(f"🎯 DER ({file_id}): {der*100:.2f}%")
