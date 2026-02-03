# # import os
# # import librosa
# # import soundfile as sf
# # from pyannote.audio import Pipeline
# # from pyannote.metrics.diarization import DiarizationErrorRate
# # from pyannote.database.util import load_rttm

# # # -----------------------------
# # # CONFIG
# # # -----------------------------
# # AUDIO_FILE = "input/chunk_0049.wav"
# # MANUAL_RTTM = "output/chunk_0049.rttm"
# # OUTPUT_DIR = "output"
# # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")
# # FILE_ID = "chunk_0049"
# # SR = 16000

# # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # -----------------------------
# # # LOAD AUDIO
# # # -----------------------------
# # audio, sr = librosa.load(AUDIO_FILE, sr=SR)

# # # -----------------------------
# # # DIARIZATION (NO EXTERNAL VAD)
# # # -----------------------------
# # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# # diarization = pipeline(AUDIO_FILE)

# # # -----------------------------
# # # SAVE SPEAKER AUDIO + RTTM
# # # -----------------------------
# # hyp_rttm_path = os.path.join(OUTPUT_DIR, "hypothesis.rttm")
# # rttm_lines = []
# # segment_id = 0

# # for turn, _, speaker in diarization.itertracks(yield_label=True):
# #     start = int(turn.start * SR)
# #     end = int(turn.end * SR)

# #     segment_audio = audio[start:end]
# #     filename = f"{speaker}_{segment_id}.wav"
# #     sf.write(os.path.join(SPEAKER_DIR, filename), segment_audio, SR)

# #     rttm_lines.append(
# #         f"SPEAKER {FILE_ID} 1 {turn.start:.3f} "
# #         f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# #     )

# #     segment_id += 1

# # with open(hyp_rttm_path, "w") as f:
# #     f.writelines(rttm_lines)

# # print("Speaker audio + RTTM saved")

# # # -----------------------------
# # # DER CALCULATION
# # # -----------------------------
# # reference = load_rttm(MANUAL_RTTM)[FILE_ID]
# # hypothesis = load_rttm(hyp_rttm_path)[FILE_ID]

# # der = DiarizationErrorRate()(reference, hypothesis)
# # print(f"\nDER: {der:.2%}")



# import os
# import torch
# import torchaudio
# import soundfile as sf

# from pyannote.audio import Pipeline
# from pyannote.metrics.diarization import DiarizationErrorRate
# from pyannote.database.util import load_rttm


# # -----------------------------
# # CONFIG
# # -----------------------------
# AUDIO_FILE = "input/chunk_0049.wav"
# MANUAL_RTTM = "output/chunk_0049.rttm"
# OUTPUT_DIR = "output"
# SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")
# FILE_ID = "chunk_0049"
# SR = 16000

# os.makedirs(SPEAKER_DIR, exist_ok=True)


# # -----------------------------
# # LOAD AUDIO (Better than librosa)
# # -----------------------------
# wav, sr = torchaudio.load(AUDIO_FILE)

# # convert stereo → mono safely
# if wav.shape[0] > 1:
#     wav = torch.mean(wav, dim=0, keepdim=True)

# wav = wav.squeeze(0).numpy()


# # -----------------------------
# # LOAD DIARIZATION MODEL
# # -----------------------------
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization"
#     # If gated:
#     # use_auth_token="YOUR_HF_TOKEN"
# )

# diarization = pipeline(AUDIO_FILE)


# # -----------------------------
# # SAVE SPEAKER AUDIO + RTTM
# # -----------------------------
# hyp_rttm_path = os.path.join(OUTPUT_DIR, "hypothesis.rttm")

# rttm_lines = []
# segment_id = 0

# print("\nDetected segments:\n")

# for turn, _, speaker in diarization.itertracks(yield_label=True):

#     start_sample = max(0, int(round(turn.start * sr)))
#     end_sample = min(len(wav), int(round(turn.end * sr)))

#     if end_sample <= start_sample:
#         continue

#     segment_audio = wav[start_sample:end_sample]

#     filename = f"{speaker}_{segment_id}.wav"
#     sf.write(
#         os.path.join(SPEAKER_DIR, filename),
#         segment_audio,
#         sr
#     )

#     print(f"{speaker}: {turn.start:.2f}s → {turn.end:.2f}s")

#     rttm_lines.append(
#         f"SPEAKER {FILE_ID} 1 {turn.start:.3f} "
#         f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
#     )

#     segment_id += 1


# with open(hyp_rttm_path, "w") as f:
#     f.writelines(rttm_lines)

# print("\n✅ Speaker audio + RTTM saved")


# # -----------------------------
# # DER CALCULATION
# # -----------------------------
# if not os.path.exists(MANUAL_RTTM):
#     print("\n⚠ Manual RTTM not found — skipping DER.")
#     exit()

# reference = load_rttm(MANUAL_RTTM)[FILE_ID]
# hypothesis = load_rttm(hyp_rttm_path)[FILE_ID]

# der = DiarizationErrorRate()(reference, hypothesis)

# print(f"\n🎯 DER: {der:.2%}")



import os
import sys
import torch
import torchaudio
import soundfile as sf
from pyannote.audio import Pipeline


if len(sys.argv) != 2:
    print("Usage: python integration.py cleaned.wav")
    sys.exit(1)

AUDIO_FILE = sys.argv[1]

OUTPUT_DIR = "output"
SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")
RTTM_PATH = os.path.join(OUTPUT_DIR, "output.rttm")

os.makedirs(SPEAKER_DIR, exist_ok=True)

# ✅ Clear old speakers
for f in os.listdir(SPEAKER_DIR):
    os.remove(os.path.join(SPEAKER_DIR, f))


# -------------------------
# Load audio safely
# -------------------------
try:
    wav, sr = torchaudio.load(AUDIO_FILE)
except Exception as e:
    print("Audio load failed:")
    print(e)
    sys.exit(1)

if wav.shape[0] > 1:
    wav = torch.mean(wav, dim=0, keepdim=True)

wav = wav.squeeze(0).numpy()


# -------------------------
# Load pyannote safely
# -------------------------
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization"
    )
except Exception as e:
    print("\nMODEL LOAD FAILED\n")
    print("Most likely fix:")
    print("👉 Run: huggingface-cli login")
    print("👉 Accept model terms on HuggingFace\n")
    print(e)
    sys.exit(1)


# CPU is safest for laptops
pipeline.to(torch.device("cpu"))

print("Running diarization...\n")

try:
    diarization = pipeline(AUDIO_FILE)
except Exception as e:
    print("Diarization crashed:")
    print(e)
    sys.exit(1)


# -------------------------
# Save segments
# -------------------------
file_id = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
rttm_lines = []
segment_id = 0

for turn, _, speaker in diarization.itertracks(yield_label=True):

    start = max(0, int(turn.start * sr))
    end = min(len(wav), int(turn.end * sr))

    if end <= start:
        continue

    segment_audio = wav[start:end]

    filename = f"{speaker}_{segment_id}.wav"
    sf.write(os.path.join(SPEAKER_DIR, filename), segment_audio, sr)

    rttm_lines.append(
        f"SPEAKER {file_id} 1 {turn.start:.3f} "
        f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
    )

    print(f"{speaker}: {turn.start:.2f}s → {turn.end:.2f}s")

    segment_id += 1


with open(RTTM_PATH, "w") as f:
    f.writelines(rttm_lines)

print("\nDiarization complete.")
