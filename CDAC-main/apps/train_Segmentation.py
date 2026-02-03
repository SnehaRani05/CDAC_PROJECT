print("🔥 TRAINING STARTED 🔥")

from pyannote.audio.tasks import Segmentation
from pyannote.audio import Model
from pytorch_lightning import Trainer
from protocol import IndianProtocol
import os

def main():
    os.makedirs("models", exist_ok=True)

    protocol = IndianProtocol()

    task = Segmentation(
        protocol=protocol,
        duration=1.9,
        max_speakers_per_frame=2,
        batch_size=8,
        num_workers=0,   # Windows-safe
        warm_up=(0.0, 0.1),
    )

    model = Model.from_pretrained("pyannote/segmentation-3.0")
    model.task = task

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=3,
        log_every_n_steps=1,
    )

    trainer.fit(model)

    # ✅ THE ONLY CORRECT SAVE LINE
    trainer.save_checkpoint("models/segmentation-indian.ckpt")

    print("✅ TRAINING FINISHED & MODEL SAVED")

if __name__ == "__main__":
    main()
