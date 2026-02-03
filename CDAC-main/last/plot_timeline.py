import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyannote.database.util import load_rttm

# =========================
# CONFIG
# =========================
RTTM_FILE = r"outputs/predicted.rttm"

# Load RTTM
rttm = load_rttm(RTTM_FILE)

# 🔴 IMPORTANT: get URI dynamically (no hardcoding)
URI = list(rttm.keys())[0]
annotation = rttm[URI]

# =========================
# PREPARE SPEAKERS
# =========================
speakers = sorted(set(label for _, _, label in annotation.itertracks(yield_label=True)))

speaker_to_y = {spk: i for i, spk in enumerate(speakers)}

# ONE fixed color per speaker
cmap = cm.get_cmap("tab10", len(speakers))
speaker_colors = {spk: cmap(i) for i, spk in enumerate(speakers)}

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(12, 3))

for segment, _, speaker in annotation.itertracks(yield_label=True):
    ax.plot(
        [segment.start, segment.end],
        [speaker_to_y[speaker]] * 2,
        linewidth=8,
        color=speaker_colors[speaker]
    )

ax.set_yticks(list(speaker_to_y.values()))
ax.set_yticklabels(speakers)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Speaker")
ax.set_title("Speaker Diarization Timeline")
ax.grid(True, axis="x")

plt.tight_layout()
plt.show()
