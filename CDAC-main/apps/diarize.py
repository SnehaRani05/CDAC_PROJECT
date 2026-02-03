# # # # # import os

# # # # # # =========================
# # # # # # WINDOWS FIXES (VERY IMPORTANT)
# # # # # # =========================
# # # # # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# # # # # os.environ["SB_LOCAL_STRATEGY"] = "copy"

# # # # # from pyannote.audio import Pipeline

# # # # # # =========================
# # # # # # CONFIG
# # # # # # =========================
# # # # # AUDIO_FILE = "test_audio/audio.wav"   # matches Streamlit upload
# # # # # OUTPUT_DIR = "outputs"

# # # # # HF_TOKEN = os.getenv("HF_TOKEN")
# # # # # if HF_TOKEN is None:
# # # # #     raise RuntimeError("HF_TOKEN not set")

# # # # # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # # # # =========================
# # # # # # LOAD PIPELINE
# # # # # # =========================
# # # # # print("🔊 Loading diarization pipeline...")

# # # # # pipeline = Pipeline.from_pretrained(
# # # # #     "pyannote/speaker-diarization-3.1",
# # # # #     use_auth_token=HF_TOKEN
# # # # # )

# # # # # # =========================
# # # # # # 🔥 DOMAIN ADAPTATION (IMPORTANT)
# # # # # # =========================

# # # # # # Reduce speaker flip-flop (helps a LOT for Indian languages)
# # # # # pipeline.segmentation.min_duration_on = 0.6
# # # # # pipeline.segmentation.min_duration_off = 0.4

# # # # # # Avoid tiny speech fragments
# # # # # pipeline.segmentation.threshold = 0.5

# # # # # # =========================
# # # # # # RUN DIARIZATION
# # # # # # =========================
# # # # # print("🧠 Running diarization...")

# # # # # diarization = pipeline(
# # # # #     AUDIO_FILE,
# # # # #     min_speakers=2,
# # # # #     max_speakers=4
# # # # # )

# # # # # # =========================
# # # # # # SAVE RTTM
# # # # # # =========================
# # # # # file_id = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
# # # # # rttm_path = os.path.join(OUTPUT_DIR, file_id + ".rttm")

# # # # # with open(rttm_path, "w") as f:
# # # # #     diarization.write_rttm(f)

# # # # # print("✅ RTTM saved at:", rttm_path)



# # # # import os
# # # # import sys
# # # # from pyannote.audio import Pipeline

# # # # # =========================
# # # # # WINDOWS FIXES
# # # # # =========================
# # # # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# # # # os.environ["SB_LOCAL_STRATEGY"] = "copy"

# # # # # =========================
# # # # # CLI ARGUMENTS
# # # # # =========================
# # # # # Usage: python diarize.py <audio.wav> <output_dir>

# # # # if len(sys.argv) < 3:
# # # #     print("Usage: python diarize.py <audio.wav> <output_dir>")
# # # #     sys.exit(1)

# # # # audio_path = sys.argv[1]
# # # # output_dir = sys.argv[2]

# # # # if not os.path.exists(audio_path):
# # # #     print(f" Audio not found: {audio_path}")
# # # #     sys.exit(1)

# # # # os.makedirs(output_dir, exist_ok=True)

# # # # # =========================
# # # # # AUTH
# # # # # =========================
# # # # HF_TOKEN = os.getenv("HF_TOKEN")
# # # # if not HF_TOKEN:
# # # #     print(" HF_TOKEN not set")
# # # #     sys.exit(1)

# # # # # =========================
# # # # # LOAD MODEL
# # # # # =========================
# # # # pipeline = Pipeline.from_pretrained(
# # # #     "pyannote/speaker-diarization-3.1",
# # # #     use_auth_token=HF_TOKEN
# # # # )

# # # # # =========================
# # # # # DOMAIN ADAPTATION
# # # # # =========================
# # # # pipeline.segmentation.min_duration_on = 0.6
# # # # pipeline.segmentation.min_duration_off = 0.4
# # # # pipeline.segmentation.threshold = 0.55

# # # # # =========================
# # # # # RUN DIARIZATION
# # # # # =========================
# # # # diarization = pipeline(audio_path)

# # # # # =========================
# # # # # WRITE RTTM (DYNAMIC NAME)
# # # # # =========================
# # # # file_id = os.path.splitext(os.path.basename(audio_path))[0]
# # # # rttm_path = os.path.join(output_dir, file_id + ".rttm")

# # # # with open(rttm_path, "w") as f:
# # # #     diarization.write_rttm(f)

# # # # # =========================
# # # # # FINAL CHECK
# # # # # =========================
# # # # if not os.path.exists(rttm_path):
# # # #     print("RTTM was not created")
# # # #     sys.exit(1)

# # # # print(f"RTTM created at: {rttm_path}")

# # # # # print(f" RTTM created: {rttm_path}")



# # # import os
# # # import sys
# # # from pyannote.audio import Pipeline

# # # # Windows fixes
# # # os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# # # os.environ["SB_LOCAL_STRATEGY"] = "copy"


# # # if len(sys.argv) < 3:
# # #     print("Usage: python diarize.py audio.wav output_dir [num_speakers]")
# # #     sys.exit(1)

# # # audio_path = sys.argv[1]
# # # output_dir = sys.argv[2]
# # # num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None

# # # HF_TOKEN = os.getenv("HF_TOKEN")
# # # if not HF_TOKEN:
# # #     raise RuntimeError("HF_TOKEN not set")

# # # os.makedirs(output_dir, exist_ok=True)

# # # print("🔊 Loading pipeline...")

# # # pipeline = Pipeline.from_pretrained(
# # #     "pyannote/speaker-diarization-community-1",  # ⭐ STRONGER
# # #     use_auth_token=HF_TOKEN
# # # )

# # # # ⭐ Stabilize segmentation
# # # pipeline.segmentation.min_duration_on = 0.8
# # # pipeline.segmentation.min_duration_off = 0.6
# # # pipeline.segmentation.threshold = 0.6


# # # print("🧠 Running diarization...")

# # # if num_speakers:
# # #     diarization = pipeline(
# # #         audio_path,
# # #         min_speakers=num_speakers,
# # #         max_speakers=num_speakers
# # #     )
# # # else:
# # #     diarization = pipeline(audio_path)


# # # file_id = os.path.splitext(os.path.basename(audio_path))[0]
# # # rttm_path = os.path.join(output_dir, file_id + ".rttm")

# # # with open(rttm_path, "w") as f:
# # #     diarization.write_rttm(f)

# # # print(f"✅ RTTM created → {rttm_path}")




# # import os
# # import sys
# # import torch
# # import torchaudio
# # import librosa
# # import soundfile as sf
# # import numpy as np
# # from pyannote.audio import Pipeline

# # # ===============================
# # # CONFIG
# # # ===============================
# # TARGET_SR = 16000
# # DEVICE = torch.device("cpu")   # safest
# # HF_TOKEN = os.getenv("HF_TOKEN")  # optional if already logged in

# # # ===============================
# # # ARGUMENTS
# # # ===============================
# # if len(sys.argv) != 3:
# #     print("Usage: python diarize.py <input_audio> <output_dir>")
# #     sys.exit(1)

# # INPUT_AUDIO = sys.argv[1]
# # OUTPUT_DIR = sys.argv[2]

# # assert os.path.exists(INPUT_AUDIO), f"Audio not found: {INPUT_AUDIO}"
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")
# # os.makedirs(SPEAKER_DIR, exist_ok=True)

# # FILE_ID = os.path.splitext(os.path.basename(INPUT_AUDIO))[0]
# # RTTM_PATH = os.path.join(OUTPUT_DIR, FILE_ID + ".rttm")

# # # ===============================
# # # LOAD + CONVERT AUDIO
# # # ===============================
# # print(" Loading audio...")

# # audio, _ = librosa.load(INPUT_AUDIO, sr=TARGET_SR, mono=True)

# # sf.write("temp_16k.wav", audio, TARGET_SR)

# # # ===============================
# # # LOAD DIARIZATION MODEL
# # # ===============================
# # print(" Loading diarization model...")

# # pipeline = Pipeline.from_pretrained(
# #     "pyannote/speaker-diarization",
# #     use_auth_token=HF_TOKEN
# # )

# # pipeline.to(DEVICE)

# # # ===============================
# # # RUN DIARIZATION (N speakers auto)
# # # ===============================
# # print(" Running diarization...")
# # diarization = pipeline("temp_16k.wav")

# # # ===============================
# # # SAVE RTTM
# # # ===============================
# # print(" Writing RTTM...")

# # with open(RTTM_PATH, "w") as f:
# #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# #         f.write(
# #             f"SPEAKER {FILE_ID} 1 "
# #             f"{turn.start:.3f} "
# #             f"{turn.end - turn.start:.3f} "
# #             f"<NA> <NA> {speaker} <NA> <NA>\n"
# #         )

# # # ===============================
# # # EXTRACT SPEAKER-WISE AUDIO (CORRECT WAY)
# # # ===============================
# # print(" Extracting speaker audio...")

# # wav, sr = torchaudio.load("temp_16k.wav")
# # wav = wav.squeeze(0).numpy()

# # # clear old files
# # for f in os.listdir(SPEAKER_DIR):
# #     os.remove(os.path.join(SPEAKER_DIR, f))

# # for speaker in diarization.labels():

# #     timeline = diarization.label_timeline(speaker)
# #     chunks = []

# #     for seg in timeline:
# #         s = int(seg.start * sr)
# #         e = int(seg.end * sr)
# #         if e > s:
# #             chunks.append(wav[s:e])

# #      if chunks:
# #         sf.write(
# #             os.path.join(SPEAKER_DIR, f"{speaker}.wav"),
# #             np.concatenate(chunks),
# #             sr
# #         )
# # # ===============================
# # # CLEANUP
# # # ===============================
# # os.remove("temp_16k.wav")

# # # ===============================
# # # SUMMARY
# # # ===============================
# # print("\n Diarization complete")
# # print(f" RTTM saved: {RTTM_PATH}")
# # print(f" Speakers detected: {len(diarization.labels())}")
# # print(f" Speaker audio saved in: {SPEAKER_DIR}")



# import os
# import sys
# import torch
# import torchaudio
# import librosa
# import soundfile as sf
# import numpy as np
# from collections import defaultdict
# from pyannote.audio import Pipeline

# TARGET_SR = 16000
# DEVICE = torch.device("cpu")
# MIN_SPEAKER_DURATION = 3.0   # seconds (filters false speakers)
# HF_TOKEN = os.getenv("HF_TOKEN")

# if len(sys.argv) != 3:
#     print("Usage: python diarize.py <cleaned_audio.wav> <output_dir>")
#     sys.exit(1)

# AUDIO_FILE = sys.argv[1]
# OUTPUT_DIR = sys.argv[2]

# if not os.path.exists(AUDIO_FILE):
#     raise FileNotFoundError(f"Audio not found: {AUDIO_FILE}")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speaker_outputs")
# os.makedirs(SPEAKER_DIR, exist_ok=True)

# FILE_ID = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
# RTTM_PATH = os.path.join(OUTPUT_DIR, FILE_ID + ".rttm")

# audio, _ = librosa.load(AUDIO_FILE, sr=TARGET_SR, mono=True)
# sf.write("temp_16k.wav", audio, TARGET_SR)

# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization",
#     use_auth_token=HF_TOKEN
# )
# pipeline.to(DEVICE)

# diarization = pipeline("temp_16k.wav")

# speaker_durations = defaultdict(float)

# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     speaker_durations[speaker] += turn.end - turn.start

# valid_speakers = {
#     speaker for speaker, dur in speaker_durations.items()
#     if dur >= MIN_SPEAKER_DURATION
# }

# with open(RTTM_PATH, "w") as f:
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         if speaker not in valid_speakers:
#             continue
#         f.write(
#             f"SPEAKER {FILE_ID} 1 "
#             f"{turn.start:.3f} "
#             f"{turn.end - turn.start:.3f} "
#             f"<NA> <NA> {speaker} <NA> <NA>\n"
#         )

# wav, sr = torchaudio.load("temp_16k.wav")
# wav = wav.squeeze(0).numpy()

# for file in os.listdir(SPEAKER_DIR):
#     os.remove(os.path.join(SPEAKER_DIR, file))

# for speaker in valid_speakers:
#     timeline = diarization.label_timeline(speaker)
#     chunks = []

#     for seg in timeline:
#         s = int(seg.start * sr)
#         e = int(seg.end * sr)
#         if e > s:
#             chunks.append(wav[s:e])

#     if chunks:
#         sf.write(
#             os.path.join(SPEAKER_DIR, f"{speaker}.wav"),
#             np.concatenate(chunks),
#             sr
#         )

# os.remove("temp_16k.wav")

# print("Diarization complete")
# print(f"Speakers detected: {len(valid_speakers)}")
# print(f"RTTM saved to: {RTTM_PATH}")
# print(f"Speaker audio saved in: {SPEAKER_DIR}")



# diarize.py
import os
import argparse
import librosa
import soundfile as sf

from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

SR = 16000


def main(args):
    audio_path = args.audio
    file_id = args.file_id
    manual_rttm = args.manual_rttm
    output_dir = args.output_dir

    speaker_dir = os.path.join(output_dir, "speaker_outputs")
    rttm_dir = os.path.join(output_dir, "rttm")

    os.makedirs(speaker_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)

    audio, _ = librosa.load(audio_path, sr=SR)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(audio_path)

    hyp_rttm = os.path.join(rttm_dir, file_id + ".rttm")
    rttm_lines = []

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        s = int(turn.start * SR)
        e = int(turn.end * SR)

        seg = audio[s:e]
        sf.write(
            os.path.join(speaker_dir, f"{speaker}_{i}.wav"),
            seg,
            SR
        )

        rttm_lines.append(
            f"SPEAKER {file_id} 1 {turn.start:.3f} "
            f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )

    with open(hyp_rttm, "w") as f:
        f.writelines(rttm_lines)

    print("✅ Diarization completed")

    if manual_rttm and os.path.exists(manual_rttm):
        ref = load_rttm(manual_rttm)[file_id]
        hyp = load_rttm(hyp_rttm)[file_id]
        der = DiarizationErrorRate()(ref, hyp)
        print(f"DER: {der:.4f}")
    else:
        print("DER skipped (no reference RTTM)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--file_id", required=True)
    parser.add_argument("--manual_rttm", default=None)
    parser.add_argument("--output_dir", default="output")

    main(parser.parse_args())
