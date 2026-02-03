# import os
# from pyannote.audio import Pipeline

# # =================================================
# # WINDOWS + SPEECHBRAIN FIXES
# # =================================================
# os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# os.environ["SB_LOCAL_STRATEGY"] = "copy"


# def wav_to_rttm(
#     audio_path: str,
#     output_dir: str,
#     model_name: str = "pyannote/speaker-diarization",
#     force: bool = False
# ):
#     """
#     Convert WAV audio to RTTM using pretrained diarization model.
#     Used for unseen / real-world data.
#     """

#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio not found: {audio_path}")

#     HF_TOKEN = os.getenv("HF_TOKEN")
#     if HF_TOKEN is None:
#         raise RuntimeError("HF_TOKEN not set")

#     os.makedirs(output_dir, exist_ok=True)

#     file_id = os.path.splitext(os.path.basename(audio_path))[0]
#     rttm_path = os.path.join(output_dir, f"{file_id}.rttm")

#     if os.path.exists(rttm_path) and not force:
#         return rttm_path

#     pipeline = Pipeline.from_pretrained(
#         model_name,
#         use_auth_token=HF_TOKEN
#     )

#     diarization = pipeline(audio_path)

#     with open(rttm_path, "w") as f:
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             f.write(
#                 f"SPEAKER {file_id} 1 "
#                 f"{turn.start:.3f} "
#                 f"{turn.end - turn.start:.3f} "
#                 f"<NA> <NA> {speaker} <NA> <NA>\n"
#             )

#     return rttm_path


import os
import sys
from pyannote.audio import Pipeline

# =================================================
# WINDOWS + SPEECHBRAIN FIXES
# =================================================
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["SB_LOCAL_STRATEGY"] = "copy"


def wav_to_rttm(
    audio_path: str,
    output_dir: str,
    model_name: str = "pyannote/speaker-diarization",
    force: bool = False
):
    """
    Convert WAV audio to RTTM using pretrained diarization model.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN is None:
        raise RuntimeError("HF_TOKEN not set")

    os.makedirs(output_dir, exist_ok=True)

    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_path = os.path.join(output_dir, f"{file_id}.rttm")

    if os.path.exists(rttm_path) and not force:
        print(f"RTTM already exists: {rttm_path}")
        return rttm_path

    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        model_name,
        use_auth_token=HF_TOKEN
    )

    print("Running diarization...")
    diarization = pipeline(audio_path)

    print("Writing RTTM...")
    with open(rttm_path, "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {file_id} 1 "
                f"{turn.start:.3f} "
                f"{turn.end - turn.start:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    print(f"RTTM saved at: {rttm_path}")
    return rttm_path


# =================================================
# ✅ CLI ENTRY POINT (THIS WAS MISSING)
# =================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_wav_to_rttm.py <audio.wav> <output_dir>")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2]

    wav_to_rttm(audio_path, output_dir)
