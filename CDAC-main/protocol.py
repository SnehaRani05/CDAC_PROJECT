from pathlib import Path
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm

DATASET_PATH = Path("processed/dataset")


class CustomProtocol(SpeakerDiarizationProtocol):

    def _iter(self, subset="train"):
        audio_dir = DATASET_PATH / "audio"
        rttm_dir = DATASET_PATH / "rttm"

        for wav in audio_dir.glob("*.wav"):
            rttm_file = rttm_dir / f"{wav.stem}.rttm"

            if not rttm_file.exists():
                continue

            # 🔥 THIS IS THE KEY FIX
            annotation = load_rttm(rttm_file)[wav.stem]

            yield {
                "uri": wav.stem,
                "database": "CDAC",
                "subset": subset,
                "audio": wav,
                "annotation": annotation,
                "annotated": Segment(0.0, annotation.get_timeline().extent().end),
                "scope": "file",
            }

    def train_iter(self):
        yield from self._iter("train")

    def development_iter(self):
        yield from self._iter("development")

    def test_iter(self):
        yield from self._iter("test")
