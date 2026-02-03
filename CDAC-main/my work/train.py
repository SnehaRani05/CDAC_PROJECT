from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation
from pyannote.database import get_protocol
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

protocol = get_protocol(
    "CDAC.SpeakerDiarization.MyProtocol",
    database_yml="database.yml"
)

task = Segmentation(
    protocol=protocol,
    duration=5.0,
    batch_size=16,
    num_workers=2,
)

model = Model.from_pretrained("pyannote/segmentation-3.0")
model.task = task

trainer = Trainer(
    accelerator="cpu",
    max_epochs=5,
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
    ]
)

trainer.fit(model)
