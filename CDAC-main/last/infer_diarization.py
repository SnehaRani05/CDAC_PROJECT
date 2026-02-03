# from pyannote.audio import Pipeline, Model

# # --------------------------------------------------
# # PATHS
# # --------------------------------------------------
# CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_22\checkpoints\epoch=4-step=2440.ckpt"
# AUDIO_FILE = r"D:\SHIVANI\INTERNSHIP\CDAC\last\sir.wav"
# OUT_RTTM = "predicted.rttm"

# # --------------------------------------------------
# # LOAD YOUR TRAINED SEGMENTATION MODEL
# # --------------------------------------------------
# segmentation_model = Model.from_pretrained(
#     CHECKPOINT,
#     strict=False
# )

# # --------------------------------------------------
# # LOAD OFFICIAL DIARIZATION PIPELINE
# # --------------------------------------------------
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1"
# )

# # --------------------------------------------------
# # 🔑 INJECT YOUR MODEL (CORRECT WAY)
# # --------------------------------------------------
# pipeline.segmentation.model = segmentation_model

# # --------------------------------------------------
# # RUN DIARIZATION
# # --------------------------------------------------
# diarization = pipeline(AUDIO_FILE)

# # --------------------------------------------------
# # PRINT RESULTS
# # --------------------------------------------------
# print("\n--- Speaker Segments ---")
# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     print(f"{turn.start:.2f} {turn.end:.2f} {speaker}")

# # --------------------------------------------------
# # SAVE RTTM
# # --------------------------------------------------
# with open(OUT_RTTM, "w") as f:
#     diarization.write_rttm(f)

# print(f"\n✅ RTTM saved as: {OUT_RTTM}")


from pyannote.audio import Pipeline, Model

CHECKPOINT = r"D:\SHIVANI\INTERNSHIP\CDAC\last\lightning_logs\version_22\checkpoints\epoch=4-step=2440.ckpt"
AUDIO_FILE = r"D:\SHIVANI\INTERNSHIP\CDAC\last\sir.wav"

# Load trained segmentation model
segmentation_model = Model.from_pretrained(
    CHECKPOINT,
    strict=False
)

# Load diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=False
)

# Inject your model
pipeline.segmentation.model = segmentation_model

# 🔥 allow more speakers
pipeline.segmentation.max_speakers_per_chunk = 6

# (optional but useful)
pipeline.clustering.threshold = 0.6

# Run diarization (AUTO speaker count)
diarization = pipeline(
    AUDIO_FILE,
    min_speakers=1,
    max_speakers=10
)

# Print results
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f} {turn.end:.2f} {speaker}")
