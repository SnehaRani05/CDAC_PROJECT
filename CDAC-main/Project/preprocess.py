import os
import sys
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

TARGET_SR = 16000
TARGET_DBFS = -20.0


def process_audio(in_path, out_path):
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    audio, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

    # Noise reduction
    audio = nr.reduce_noise(y=audio, sr=sr, stationary=True)

    # RMS normalization
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-6:
        gain = 10 ** ((TARGET_DBFS - 20 * np.log10(rms)) / 20)
        audio *= gain

    audio = np.clip(audio, -1.0, 1.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, sr)
    print("✅ Audio cleaned:", out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py input.wav output.wav")
        sys.exit(1)

    process_audio(sys.argv[1], sys.argv[2])
