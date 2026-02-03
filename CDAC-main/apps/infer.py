# import os
# import torch
# import torchaudio
# import soundfile as sf
# from pyannote.audio import Model
# from pyannote.core import Annotation, Segment

# # ---------------- CONFIG ----------------
# AUDIO_FILE = "test_audio/audio_00024.wav"
# MODEL_PATH = "models/segmentation-indian.ckpt"
# OUTPUT_DIR = "outputs"

# TARGET_SR = 16000
# FRAME_DURATION = 0.02
# CONF_THRESHOLD = 0.1
# MIN_SEGMENT = 0.3
# # ---------------------------------------

# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def main():
#     print("🔊 Loading trained model...")
#     model = Model.from_pretrained(MODEL_PATH)
#     model.eval()

#     print("🎙 Loading audio...")
#     waveform, sr = sf.read(AUDIO_FILE)

#     if waveform.ndim == 2:
#         waveform = waveform.mean(axis=1)

#     waveform = torch.tensor(waveform).float()

#     if sr != TARGET_SR:
#         waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

#     waveform = waveform.unsqueeze(0)

#     print("🧠 Running segmentation...")
#     with torch.no_grad():
#         scores = model(waveform)

#     scores = scores.squeeze(0)

#     uri = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
#     annotation = Annotation(uri=uri)

#     # -------- MODEL-BASED DIARIZATION --------
#     current_speaker = None
#     start_time = None

#     for i, frame_scores in enumerate(scores):
#         time = i * FRAME_DURATION

#         speaker = torch.argmax(frame_scores).item()
#         confidence = frame_scores[speaker].item()

#         if confidence < CONF_THRESHOLD:
#             if current_speaker is not None:
#                 if time - start_time >= MIN_SEGMENT:
#                     annotation[
#                         Segment(start_time, time)
#                     ] = f"SPEAKER_{current_speaker}"
#                 current_speaker = None
#                 start_time = None
#             continue

#         if current_speaker is None:
#             current_speaker = speaker
#             start_time = time

#         elif speaker != current_speaker:
#             annotation[
#                 Segment(start_time, time)
#             ] = f"SPEAKER_{current_speaker}"
#             current_speaker = speaker
#             start_time = time

#     # close last segment
#     if current_speaker is not None and time - start_time >= MIN_SEGMENT:
#         annotation[
#             Segment(start_time, time)
#         ] = f"SPEAKER_{current_speaker}"

#     # -------- OUTPUT --------
#     print("\n🧾 Speaker segments:")
#     for segment, _, speaker in annotation.itertracks(yield_label=True):
#         print(f"{speaker}: {segment.start:.2f}s → {segment.end:.2f}s")

#     rttm_path = os.path.join(OUTPUT_DIR, "output.rttm")
#     with open(rttm_path, "w") as f:
#         annotation.write_rttm(f)

#     print(f"\n✅ RTTM saved at {rttm_path}")


# if __name__ == "__main__":
#     main()


import os
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
from pyannote.audio import Model
from pyannote.core import Annotation, Segment

# ---------------- CONFIG ----------------
AUDIO_FILE = "test_audio/audio_00024.wav"
MODEL_PATH = "models/segmentation-indian.ckpt"
OUTPUT_DIR = "outputs"

TARGET_SR = 16000
CONF_THRESHOLD = 0.2      # higher = more stable
MIN_SEGMENT = 0.7         # important for unseen audio
SMOOTH_KERNEL = 7         # temporal smoothing
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("🔊 Loading trained model...")
    model = Model.from_pretrained(MODEL_PATH)
    model.eval()

    print("🎙 Loading audio...")
    waveform, sr = sf.read(AUDIO_FILE)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    waveform = torch.tensor(waveform).float()

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    waveform = waveform.unsqueeze(0)

    print("🧠 Running segmentation...")
    with torch.no_grad():
        scores = model(waveform)  # (1, frames, speakers)

    scores = scores.squeeze(0)   # (frames, speakers)

    # ---------- FIX 1: correct frame duration ----------
    duration = waveform.shape[-1] / TARGET_SR
    num_frames = scores.shape[0]
    frame_duration = duration / num_frames

    # ---------- FIX 2: temporal smoothing ----------
    scores = F.avg_pool1d(
        scores.T.unsqueeze(0),
        kernel_size=SMOOTH_KERNEL,
        stride=1,
        padding=SMOOTH_KERNEL // 2
    ).squeeze(0).T

    uri = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
    annotation = Annotation(uri=uri)

    current_speaker = None
    start_time = None
    last_time = 0.0

    # ---------- FIX 3: stable diarization ----------
    for i, frame_scores in enumerate(scores):
        time = i * frame_duration
        last_time = time

        speaker = torch.argmax(frame_scores).item()
        confidence = frame_scores[speaker].item()

        # Ignore low confidence but DON'T break segment
        if confidence < CONF_THRESHOLD:
            continue

        if current_speaker is None:
            current_speaker = speaker
            start_time = time

        elif speaker != current_speaker:
            if time - start_time >= MIN_SEGMENT:
                annotation[Segment(start_time, time)] = f"SPEAKER_{current_speaker}"
            current_speaker = speaker
            start_time = time

    # ---------- close last segment ----------
    if current_speaker is not None and last_time - start_time >= MIN_SEGMENT:
        annotation[Segment(start_time, last_time)] = f"SPEAKER_{current_speaker}"

    # ---------- OUTPUT ----------
    print("\n🧾 Speaker segments:")
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        print(f"{speaker}: {segment.start:.2f}s → {segment.end:.2f}s")

    rttm_path = os.path.join(OUTPUT_DIR, "output.rttm")
    with open(rttm_path, "w") as f:
        annotation.write_rttm(f)

    print(f"\n✅ RTTM saved at {rttm_path}")


if __name__ == "__main__":
    main()
