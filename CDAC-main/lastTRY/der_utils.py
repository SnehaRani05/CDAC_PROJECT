import os
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

REFERENCE_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"


def compute_der(file_id, hyp_rttm):

    ref_path = os.path.join(REFERENCE_DIR, file_id + ".rttm")
    if not os.path.exists(ref_path):
        return None

    ref = load_rttm(ref_path)
    hyp = load_rttm(hyp_rttm)

    ref_id = list(ref.keys())[0]
    hyp_id = list(hyp.keys())[0]

    metric = DiarizationErrorRate()
    return metric(ref[ref_id], hyp[hyp_id])
