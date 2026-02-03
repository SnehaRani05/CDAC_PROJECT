import os
import random

AUDIO_DIR = "dataset/audio"
TRAIN_LST = "dataset/train.lst"
DEV_LST = "dataset/dev.lst"

files = [os.path.splitext(f)[0] for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

random.seed(42)          # reproducibility
random.shuffle(files)

split = int(0.8 * len(files))
train_files = files[:split]
dev_files = files[split:]

with open(TRAIN_LST, "w") as f:
    f.write("\n".join(train_files))

with open(DEV_LST, "w") as f:
    f.write("\n".join(dev_files))

print("Train files:", train_files)
print("Dev files:", dev_files)
