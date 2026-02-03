# from pyannote.core import Annotation, Segment
# from pyannote.metrics.diarization import DiarizationErrorRate

# def read_rttm(path):
#      ann = Annotation()
#      with open(path) as f:
#          for line in f:
#              p = line.strip().split()
#              start = float(p[3])
#              dur = float(p[4])
#              speaker = p[7]
#              ann[Segment(start, start + dur)] = speaker
#      return ann

# reference = read_rttm(
#      r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm\audio_00007.rttm"
# )
# hypothesis = read_rttm(
#      r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\outputs\output.rttm"
# )

# uem = reference.get_timeline().union(hypothesis.get_timeline())

# metric = DiarizationErrorRate(collar=0.25)
# der = metric(reference, hypothesis, uem=uem)


# print(f"🎯 DER: {der * 100:.2f}%")

import os
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


# ================= CONFIG =================

REF_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"
HYP_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\outputs"


# ================= RTTM READER =================

def read_rttm(path):
    """
    Read RTTM file and return pyannote Annotation
    """
    ann = Annotation()
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("SPEAKER"):
                continue

            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]

            ann[Segment(start, start + duration)] = speaker

    return ann


# ================= DER FOR ONE FILE =================

def compute_der(ref_rttm_path, hyp_rttm_path):
    reference = read_rttm(ref_rttm_path)
    hypothesis = read_rttm(hyp_rttm_path)

    # Reference-only evaluation map (BEST PRACTICE)
    uem = reference.get_timeline()

    metric = DiarizationErrorRate(
        collar=0.25,
        skip_overlap=True
    )

    der = metric(reference, hypothesis, uem=uem)

    return {
        "DER": der,
        "Missed": metric["missed detection"],
        "False Alarm": metric["false alarm"],
        "Confusion": metric["confusion"],
    }


# ================= DATASET DER =================

all_ders = []

print("\n========== DER RESULTS ==========")

for fname in sorted(os.listdir(REF_RTTM_DIR)):
    if not fname.endswith(".rttm"):
        continue

    ref_path = os.path.join(REF_RTTM_DIR, fname)
    hyp_path = os.path.join(HYP_RTTM_DIR, fname)

    if not os.path.exists(hyp_path):
        print(f"\n❌ Missing hypothesis RTTM for {fname}")
        continue

    scores = compute_der(ref_path, hyp_path)

    print(f"\n📄 File: {fname}")
    print(f"🎯 DER: {scores['DER'] * 100:.2f}%")
    print(
        f"Missed: {scores['Missed'] * 100:.2f}% | "
        f"False Alarm: {scores['False Alarm'] * 100:.2f}% | "
        f"Confusion: {scores['Confusion'] * 100:.2f}%"
    )

    all_ders.append(scores["DER"])


# ================= AVERAGE DER =================

if all_ders:
    avg_der = sum(all_ders) / len(all_ders)
    print("\n================================")
    print(f"📊 Average DER over {len(all_ders)} files: {avg_der * 100:.2f}%")
    print("================================")
else:
    print("\n⚠️ No valid RTTM pairs found.")
