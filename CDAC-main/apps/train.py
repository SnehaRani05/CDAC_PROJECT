print("🔥 TRAIN.PY STARTED 🔥")

from pyannote.audio.tasks import Segmentation
from pyannote.audio.models.segmentation import SegmentationModel
from pytorch_lightning import Trainer

from dataset_loader import CustomProtocol


def main():
    protocol = CustomProtocol()

    task = Segmentation(
        protocol=protocol,
        duration=2.0,
        max_speakers_per_frame=2,
    )

    model = SegmentationModel(task=task)

    trainer = Trainer(
        max_epochs=1,               # keep low for CPU
        accelerator="cpu",
        devices=1,
        num_workers=0,              # 🔴 REQUIRED on Windows
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    print("🚀 ABOUT TO START TRAINING")
    trainer.fit(model)
    print("✅ TRAINING FINISHED")


if __name__ == "__main__":
    main()
