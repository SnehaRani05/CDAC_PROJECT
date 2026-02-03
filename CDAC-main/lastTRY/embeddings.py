import numpy as np
from pyannote.audio import Inference
from sklearn.metrics.pairwise import cosine_similarity

embedder = Inference("pyannote/embedding", window="whole")


def extract_embeddings(segments):
    for s in segments:
        emb = embedder(s["path"])
        s["embedding"] = emb / np.linalg.norm(emb)
    return segments


def merge_speakers(segments, threshold):

    speakers = list({s["speaker"] for s in segments})
    mean = {}

    for spk in speakers:
        embs = [s["embedding"] for s in segments if s["speaker"] == spk]
        mean[spk] = np.mean(embs, axis=0)

    mapping = {}
    used = set()

    for spk in speakers:
        if spk in used:
            continue

        mapping[spk] = spk
        used.add(spk)

        for other in speakers:
            if other in used:
                continue

            sim = cosine_similarity(
                mean[spk].reshape(1, -1),
                mean[other].reshape(1, -1)
            )[0][0]

            if sim >= threshold:
                mapping[other] = spk
                used.add(other)

    for s in segments:
        s["speaker"] = mapping[s["speaker"]]

    return segments
