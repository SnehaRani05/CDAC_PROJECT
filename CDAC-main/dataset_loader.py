from pathlib import Path
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.core import Segment
from pyannote.database import FileFinder

DATASET_PATH = Path("processed/dataset")

class CustomProtocol(SpeakerDiarizationProtocol):

    def train_iter(self):
        for wav in (DATASET_PATH / "audio").glob("*.wav"):
            rttm = DATASET_PATH / "rttm" / f"{wav.stem}.rttm"

            yield {
                "uri": wav.stem,
                "database": "CDAC",
                "subset": "train",
                "audio": wav,
                "annotation": rttm,
                "annotated": Segment(0.0, None),  # REQUIRED
                "scope": "file",
            }

    def development_iter(self):
        return self.train_iter()

    def test_iter(self):
        return self.train_iter()
