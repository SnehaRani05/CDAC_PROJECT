import torch
import torchaudio
from pathlib import Path

from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.util import load_rttm

import pytorch_lightning as pl


# =========================
# PATHS
# =========================
AUDIO_DIR = Path("dataset/audio")
RTTM_DIR = Path("dataset/rttm")


# =========================
# PROTOCOL
# =========================
class HindiDiarizationProtocol(SpeakerDiarizationProtocol):

    name = "HindiDataset.SpeakerDiarization"
    preprocessors = {}

    def train(self):
        for wav in AUDIO_DIR.glob("*.wav"):
            uri = wav.stem
            rttm_path = RTTM_DIR / f"{uri}.rttm"

            if not rttm_path.exists():
                continue

            annotation = load_rttm(rttm_path)[uri]
            info = torchaudio.info(str(wav))

            yield {
                "database": "HindiDataset",
                "subset": "train",
                "scope": "file",
                "uri": uri,
                "audio": str(wav),
                "annotation": annotation,
                "annotated": annotation.get_timeline().support(),
                "torchaudio.info": info,
            }

    def development(self):
        return self.train()


protocol = HindiDiarizationProtocol()


# =========================
# TASK
# =========================
task = SpeakerDiarization(
    protocol=protocol,
    duration=2.0,
    batch_size=8,
    num_workers=0,        # Windows-safe
    max_speakers_per_chunk=4
)


# =========================
# MODEL
# =========================
model = Model.from_pretrained(
    "pyannote/segmentation",
    use_auth_token=True
)
model.task = task


# =========================
# TRAINER
# =========================
trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=15,
    log_every_n_steps=10
)


# =========================
# WINDOWS ENTRY POINT
# =========================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    trainer.fit(model)
