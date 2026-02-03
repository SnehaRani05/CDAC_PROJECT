import os
import torch
import torchaudio
import soundfile as sf

from pyannote.audio import Pipeline
from embeddings import extract_embeddings, merge_speakers

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1"
).to(torch.device("cpu"))


def run_diarization(audio_path, output_dir, min_segment_len, merge_threshold):

    speakers_dir = os.path.join(output_dir, "speakers")
    os.makedirs(speakers_dir, exist_ok=True)

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    audio = wav.squeeze().numpy()

    diarization = pipeline(
        audio_path,
        min_speakers=1,
        max_speakers=10
    )

    segments = []

    for i, (turn, _, speaker) in enumerate(
        diarization.itertracks(yield_label=True)
    ):
        dur = turn.end - turn.start
        if dur < min_segment_len:
            continue

        start = int(turn.start * sr)
        end = int(turn.end * sr)

        path = os.path.join(speakers_dir, f"{speaker}_{i}.wav")
        sf.write(path, audio[start:end], sr)

        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "path": path
        })

    segments = extract_embeddings(segments)
    segments = merge_speakers(segments, merge_threshold)

    # Write RTTM
    rttm_path = os.path.join(output_dir, "output.rttm")
    with open(rttm_path, "w") as f:
        for s in segments:
            f.write(
                f"SPEAKER {os.path.basename(audio_path)} 1 "
                f"{s['start']:.3f} {s['end']-s['start']:.3f} "
                f"<NA> <NA> {s['speaker']} <NA> <NA>\n"
            )

    return segments, rttm_path
