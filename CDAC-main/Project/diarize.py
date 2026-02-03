# # # # # import os
# # # # # import torch
# # # # # import torchaudio
# # # # # import soundfile as sf
# # # # # from pyannote.audio import Pipeline
# # # # # from paths import OUTPUT_DIR, SPEAKER_DIR, HYP_RTTM


# # # # # import os
# # # # # import torch
# # # # # import torchaudio
# # # # # import soundfile as sf
# # # # # import numpy as np
# # # # # from pyannote.audio import Pipeline
# # # # # from paths import SPEAKER_DIR, HYP_RTTM


# # # # # TARGET_SR = 16000


# # # # # def run_diarization(audio_path):

# # # # #     os.makedirs(SPEAKER_DIR, exist_ok=True)

# # # # #     # Clear old segments
# # # # #     for f in os.listdir(SPEAKER_DIR):
# # # # #         os.remove(os.path.join(SPEAKER_DIR, f))

# # # # #     # 🔥 LOAD AUDIO + RESAMPLE TO 16k
# # # # #     wav, sr = torchaudio.load(audio_path)

# # # # #     if wav.shape[0] > 1:
# # # # #         wav = wav.mean(dim=0, keepdim=True)

# # # # #     if sr != TARGET_SR:
# # # # #         resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
# # # # #         wav = resampler(wav)

# # # # #     wav = wav.squeeze(0).numpy().astype("float32")

# # # # #     # 🔥 LOAD MODEL
# # # # #     pipeline = Pipeline.from_pretrained(
# # # # #         "pyannote/speaker-diarization-3.1"
# # # # #     ).to(torch.device("cpu"))

# # # # #     diarization = pipeline(
# # # # #     audio_path,
# # # # #     min_speakers=1,
# # # # #     max_speakers=10   # you can set 6–10 safely
# # # # # )


# # # # #     file_id = os.path.splitext(os.path.basename(audio_path))[0]
# # # # #     rttm_lines = []

# # # # #     # 🔥 SLICE USING 16k AUDIO
# # # # #     for i, (turn, _, speaker) in enumerate(
# # # # #         diarization.itertracks(yield_label=True)
# # # # #     ):
# # # # #         start = int(turn.start * TARGET_SR)
# # # # #         end = int(turn.end * TARGET_SR)

# # # # #         if end <= start or end > len(wav):
# # # # #             continue

# # # # #         segment = wav[start:end]

# # # # #         if segment.size == 0:
# # # # #             continue

# # # # #         sf.write(
# # # # #             os.path.join(SPEAKER_DIR, f"{speaker}_{i}.wav"),
# # # # #             segment,
# # # # #             TARGET_SR,
# # # # #             format="WAV",
# # # # #             subtype="PCM_16"
# # # # #         )

# # # # #         rttm_lines.append(
# # # # #             f"SPEAKER {file_id} 1 {turn.start:.3f} "
# # # # #             f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# # # # #         )

# # # # #     with open(HYP_RTTM, "w") as f:
# # # # #         f.writelines(rttm_lines)

# # # # #     return diarization


# # # # import os
# # # # from sklearn import pipeline
# # # # import torch
# # # # import torchaudio
# # # # import soundfile as sf
# # # # import numpy as np
# # # # from pyannote.audio import Pipeline

# # # # TARGET_SR = 16000
# # # # MIN_SEGMENT_SEC = 0.4  # removes micro/noise segments


# # # # def run_diarization(audio_path, output_dir="output"):

# # # #     speaker_dir = os.path.join(output_dir, "speakers")
# # # #     rttm_path = os.path.join(output_dir, "hypothesis.rttm")

# # # #     os.makedirs(speaker_dir, exist_ok=True)

# # # #     # Clear old segments
# # # #     for f in os.listdir(speaker_dir):
# # # #         os.remove(os.path.join(speaker_dir, f))

# # # #     # ---------- Load & resample audio ----------
# # # #     wav, sr = torchaudio.load(audio_path)

# # # #     if wav.shape[0] > 1:
# # # #         wav = wav.mean(dim=0, keepdim=True)

# # # #     if sr != TARGET_SR:
# # # #         wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

# # # #     wav = wav.squeeze(0).numpy().astype("float32")

# # # #     # ---------- Load diarization pipeline ----------
# # # #     pipeline = Pipeline.from_pretrained(
# # # #         "pyannote/speaker-diarization-3.1"
# # # #     ).to(torch.device("cpu"))

# # # #         # 🔥 Prevent aggressive merging
# # # #     pipeline.segmentation.min_duration_off = 0.08
# # # #     pipeline.segmentation.min_duration_on = 0.0


# # # #     # 🔥 AUTO speaker detection (no fixed count)
# # # #     diarization = pipeline(
# # # #         audio_path,
# # # #         min_speakers=1,
# # # #         max_speakers=8,
# # # #       #   clustering="spectral"
# # # #     )

# # # #     file_id = os.path.splitext(os.path.basename(audio_path))[0]
# # # #     rttm_lines = []

# # # #     segment_index = 0

# # # #     for turn, _, speaker in diarization.itertracks(yield_label=True):

# # # #         duration = turn.end - turn.start
# # # #         if duration < MIN_SEGMENT_SEC:
# # # #             continue

# # # #         start = int(turn.start * TARGET_SR)
# # # #         end = int(turn.end * TARGET_SR)

# # # #         if end <= start or end > len(wav):
# # # #             continue

# # # #         segment = wav[start:end]

# # # #         sf.write(
# # # #             os.path.join(speaker_dir, f"{speaker}_{segment_index}.wav"),
# # # #             segment,
# # # #             TARGET_SR,
# # # #             format="WAV",
# # # #             subtype="PCM_16"
# # # #         )

# # # #         rttm_lines.append(
# # # #             f"SPEAKER {file_id} 1 {turn.start:.3f} "
# # # #             f"{duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# # # #         )

# # # #         segment_index += 1

# # # #     with open(rttm_path, "w") as f:
# # # #         f.writelines(rttm_lines)

# # # #     return rttm_path



# # # import os
# # # import torch
# # # import torchaudio
# # # import soundfile as sf
# # # import numpy as np
# # # from pyannote.audio import Pipeline

# # # TARGET_SR = 16000

# # # # 🔥 Aggressive over-segmentation settings
# # # MIN_SEGMENT_SEC = 0.3


# # # def run_diarization(audio_path, output_dir="output"):

# # #     speaker_dir = os.path.join(output_dir, "speakers")
# # #     rttm_path = os.path.join(output_dir, "hypothesis.rttm")

# # #     os.makedirs(speaker_dir, exist_ok=True)

# # #     # Clear old outputs
# # #     for f in os.listdir(speaker_dir):
# # #         os.remove(os.path.join(speaker_dir, f))

# # #     # ================= LOAD AUDIO =================

# # #     wav, sr = torchaudio.load(audio_path)

# # #     if wav.shape[0] > 1:
# # #         wav = wav.mean(dim=0, keepdim=True)

# # #     if sr != TARGET_SR:
# # #         wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

# # #     wav = wav.squeeze(0).numpy().astype("float32")

# # #     # ================= LOAD PIPELINE =================

# # #     pipeline = Pipeline.from_pretrained(
# # #         "pyannote/speaker-diarization-3.1"
# # #     ).to(torch.device("cpu"))

# # #     # 🔥 KEY FIX: prevent speaker collapse
# # #     pipeline.segmentation.min_duration_on = 0.15
# # #     pipeline.segmentation.min_duration_off = 0.05
# # #     MIN_SEGMENT_SEC = 0.5

# # #     # 🔥 AUTO mode but encourage more speakers
# # #     diarization = pipeline(
# # #         audio_path,
# # #         min_speakers=3,
# # #         max_speakers=8
# # #     )

# # #     file_id = os.path.splitext(os.path.basename(audio_path))[0]
# # #     rttm_lines = []

# # #     segment_index = 0

# # #     # ================= SAVE SEGMENTS =================

# # #     for turn, _, speaker in diarization.itertracks(yield_label=True):

# # #         duration = turn.end - turn.start
# # #         if duration < MIN_SEGMENT_SEC:
# # #             continue

# # #         start = int(turn.start * TARGET_SR)
# # #         end = int(turn.end * TARGET_SR)

# # #         if end <= start or end > len(wav):
# # #             continue

# # #         segment_audio = wav[start:end]

# # #         sf.write(
# # #             os.path.join(
# # #                 speaker_dir,
# # #                 f"{speaker}_{segment_index}.wav"
# # #             ),
# # #             segment_audio,
# # #             TARGET_SR,
# # #             format="WAV",
# # #             subtype="PCM_16"
# # #         )

# # #         rttm_lines.append(
# # #             f"SPEAKER {file_id} 1 {turn.start:.3f} "
# # #             f"{duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# # #         )

# # #         segment_index += 1

# # #     # ================= WRITE RTTM =================

# # #     with open(rttm_path, "w") as f:
# # #         f.writelines(rttm_lines)

# # #     return rttm_path



# # import os
# # import torch
# # import torchaudio
# # import soundfile as sf
# # from pyannote.audio import Pipeline

# # TARGET_SR = 16000

# # # ===== Balanced tuning (do NOT change further) =====
# # MIN_SEGMENT_SEC = 0.5


# # # ================= POST-PROCESSING =================

# # def merge_short_segments(segments, min_gap=0.5):
# #     """
# #     Bidirectional merge of very short mismatched segments.
# #     Fixes the last remaining edge-case mismatch.
# #     """
# #     if len(segments) <= 1:
# #         return segments

# #     merged = []
# #     i = 0

# #     while i < len(segments):
# #         seg = segments[i]

# #         # short segment → decide merge direction
# #         if seg["duration"] < min_gap and i > 0 and i < len(segments) - 1:
# #             prev_seg = merged[-1]
# #             next_seg = segments[i + 1]

# #             # If neighbors have same speaker, merge all
# #             if prev_seg["speaker"] == next_seg["speaker"]:
# #                 prev_seg["end"] = next_seg["end"]
# #                 prev_seg["duration"] = prev_seg["end"] - prev_seg["start"]
# #                 i += 2
# #                 continue

# #             # Otherwise merge into longer neighbor
# #             if (prev_seg["duration"] >= next_seg["duration"]):
# #                 prev_seg["end"] = seg["end"]
# #                 prev_seg["duration"] = prev_seg["end"] - prev_seg["start"]
# #             else:
# #                 next_seg["start"] = seg["start"]
# #                 next_seg["duration"] = next_seg["end"] - next_seg["start"]
# #                 merged.append(next_seg)
# #                 i += 2
# #                 continue

# #         else:
# #             merged.append(seg)

# #         i += 1

# #     return merged



# # # ================= MAIN FUNCTION =================

# # def run_diarization(audio_path, output_dir="output"):

# #     speaker_dir = os.path.join(output_dir, "speakers")
# #     rttm_path = os.path.join(output_dir, "hypothesis.rttm")

# #     os.makedirs(speaker_dir, exist_ok=True)

# #     # Clear old outputs
# #     for f in os.listdir(speaker_dir):
# #         os.remove(os.path.join(speaker_dir, f))

# #     # ================= LOAD AUDIO =================

# #     wav, sr = torchaudio.load(audio_path)

# #     if wav.shape[0] > 1:
# #         wav = wav.mean(dim=0, keepdim=True)

# #     if sr != TARGET_SR:
# #         wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

# #     wav = wav.squeeze(0).numpy().astype("float32")

# #     # ================= LOAD PIPELINE =================

# #     pipeline = Pipeline.from_pretrained(
# #         "pyannote/speaker-diarization-3.1"
# #     ).to(torch.device("cpu"))

# #     # 🔥 Balanced segmentation (final tuning)
# #     pipeline.segmentation.min_duration_on = 0.15
# #     pipeline.segmentation.min_duration_off = 0.05

# #     diarization = pipeline(
# #         audio_path,
# #         min_speakers=2,
# #         max_speakers=8
# #     )

# #     file_id = os.path.splitext(os.path.basename(audio_path))[0]

# #     # ================= COLLECT SEGMENTS =================

# #     raw_segments = []

# #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# #         duration = turn.end - turn.start

# #         if duration < MIN_SEGMENT_SEC:
# #             continue

# #         raw_segments.append({
# #             "speaker": speaker,
# #             "start": turn.start,
# #             "end": turn.end,
# #             "duration": duration
# #         })

# #     # ================= POST-PROCESS =================

# #     segments = merge_short_segments(raw_segments)

# #     # ================= SAVE OUTPUTS =================

# #     rttm_lines = []

# #     for idx, seg in enumerate(segments):

# #         start_sample = int(seg["start"] * TARGET_SR)
# #         end_sample = int(seg["end"] * TARGET_SR)

# #         if end_sample <= start_sample or end_sample > len(wav):
# #             continue

# #         audio_chunk = wav[start_sample:end_sample]

# #         sf.write(
# #             os.path.join(speaker_dir, f"{seg['speaker']}_{idx}.wav"),
# #             audio_chunk,
# #             TARGET_SR,
# #             format="WAV",
# #             subtype="PCM_16"
# #         )

# #         rttm_lines.append(
# #             f"SPEAKER {file_id} 1 {seg['start']:.3f} "
# #             f"{seg['duration']:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n"
# #         )

# #     with open(rttm_path, "w") as f:
# #         f.writelines(rttm_lines)

# #     return rttm_path


# import os
# import torch
# import torchaudio
# import soundfile as sf
# from pyannote.audio import Pipeline

# SR = 16000


# def run_diarization(audio_path, output_dir="output"):

#     speaker_dir = os.path.join(output_dir, "speakers")
#     rttm_path = os.path.join(output_dir, "hypothesis.rttm")

#     os.makedirs(speaker_dir, exist_ok=True)

#     # Clear old outputs
#     for f in os.listdir(speaker_dir):
#         os.remove(os.path.join(speaker_dir, f))

#     # -------------------------
#     # Load audio safely
#     # -------------------------
#     wav, sr = torchaudio.load(audio_path)

#     if wav.shape[0] > 1:
#         wav = wav.mean(dim=0, keepdim=True)

#     if sr != SR:
#         wav = torchaudio.transforms.Resample(sr, SR)(wav)

#     wav = wav.squeeze(0).numpy()

#     # -------------------------
#     # Load diarization model
#     # -------------------------
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1"
#     ).to(torch.device("cpu"))

#     diarization = pipeline(audio_path)

#     # -------------------------
#     # Save segments + RTTM
#     # -------------------------
#     file_id = os.path.splitext(os.path.basename(audio_path))[0]
#     rttm_lines = []
#     segment_id = 0

#     for turn, _, speaker in diarization.itertracks(yield_label=True):

#         start = int(turn.start * SR)
#         end = int(turn.end * SR)

#         if end <= start or end > len(wav):
#             continue

#         segment_audio = wav[start:end]

#         sf.write(
#             os.path.join(speaker_dir, f"{speaker}_{segment_id}.wav"),
#             segment_audio,
#             SR,
#             format="WAV",
#             subtype="PCM_16"
#         )

#         rttm_lines.append(
#             f"SPEAKER {file_id} 1 {turn.start:.3f} "
#             f"{turn.end - turn.start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
#         )

#         segment_id += 1

#     with open(rttm_path, "w") as f:
#         f.writelines(rttm_lines)

#     return rttm_path





import os
import torch
import torchaudio
import soundfile as sf
from pyannote.audio import Pipeline

from llm_correction import llm_correct_segments

# ================= CONFIG =================

SR = 16000
MIN_SEGMENT_SEC = 0.4        # drop micro segments
SMOOTH_GAP_SEC = 0.3         # merge same-speaker gaps


# ================= UTILITIES =================

def smooth_segments(segments):
    """
    Merge adjacent segments of the same speaker
    if the gap between them is very small.
    """
    if not segments:
        return segments

    smoothed = [segments[0]]

    for seg in segments[1:]:
        prev = smoothed[-1]

        if (
            seg["speaker"] == prev["speaker"]
            and seg["start"] - prev["end"] <= SMOOTH_GAP_SEC
        ):
            prev["end"] = seg["end"]
            prev["duration"] = prev["end"] - prev["start"]
        else:
            smoothed.append(seg)

    return smoothed


# ================= MAIN FUNCTION =================

def run_diarization(audio_path, output_dir="output"):

    speaker_dir = os.path.join(output_dir, "speakers")
    rttm_path = os.path.join(output_dir, "hypothesis.rttm")

    os.makedirs(speaker_dir, exist_ok=True)

    # Clear old outputs
    for f in os.listdir(speaker_dir):
        os.remove(os.path.join(speaker_dir, f))

    # ================= LOAD AUDIO =================

    wav, sr = torchaudio.load(audio_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)

    wav = wav.squeeze(0).numpy()

    # ================= LOAD PIPELINE =================

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    ).to(torch.device("cpu"))

    # 🔧 balanced segmentation tuning
    pipeline.segmentation.min_duration_on = 0.2
    pipeline.segmentation.min_duration_off = 0.1

    # 🔧 unknown number of speakers (encourages >2)
    diarization = pipeline(
        audio_path,
        min_speakers=2,
        max_speakers=6
    )

    # ================= COLLECT SEGMENTS =================

    file_id = os.path.splitext(os.path.basename(audio_path))[0]

    raw_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start

        if duration < MIN_SEGMENT_SEC:
            continue

        raw_segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "duration": duration
        })

    # ================= POST-PROCESS =================

    # 1️⃣ merge tiny same-speaker gaps
    segments = smooth_segments(raw_segments)

    # 2️⃣ optional LLM correction (safe & bounded)
    segments = llm_correct_segments(segments)

    # ================= SAVE OUTPUT =================

    rttm_lines = []

    for idx, seg in enumerate(segments):

        start_sample = int(seg["start"] * SR)
        end_sample = int(seg["end"] * SR)

        if end_sample <= start_sample or end_sample > len(wav):
            continue

        audio_chunk = wav[start_sample:end_sample]

        sf.write(
            os.path.join(speaker_dir, f"{seg['speaker']}_{idx}.wav"),
            audio_chunk,
            SR,
            subtype="PCM_16"
        )

        rttm_lines.append(
            f"SPEAKER {file_id} 1 {seg['start']:.3f} "
            f"{seg['duration']:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n"
        )

    with open(rttm_path, "w") as f:
        f.writelines(rttm_lines)

    return rttm_path
