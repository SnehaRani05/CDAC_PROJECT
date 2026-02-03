import os
import librosa
import soundfile as sf
import numpy as np

SR = 16000
SPEAKER_WAV_ROOT = "cleaned_audio/output/speaker_wavs"
MERGED_OUT = "cleaned_audio/merged_speakers"

MIN_SEG_SEC = 0.2        # ignore very short segments
SILENCE_SEC = 0.2        # silence between segments (helps ASR)

# =========================
# MERGE FUNCTION
# =========================
def merge_wavs(wav_dir, out_path):
    audio_all = []
    silence = np.zeros(int(SILENCE_SEC * SR))

    for wav in sorted(os.listdir(wav_dir)):
        if not wav.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(wav_dir, wav)
        y, _ = librosa.load(wav_path, sr=SR, mono=True)

        # Skip tiny junk segments
        if len(y) < int(MIN_SEG_SEC * SR):
            continue

        audio_all.append(y)
        audio_all.append(silence)

    if not audio_all:
        print("⚠️ No valid audio in:", wav_dir)
        return

    merged = np.concatenate(audio_all)

    # Safe normalization
    merged = librosa.util.normalize(merged)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, merged, SR)

# =========================
# MAIN
# =========================
def main():
    os.makedirs(MERGED_OUT, exist_ok=True)

    for audio_id in os.listdir(SPEAKER_WAV_ROOT):
        audio_path = os.path.join(SPEAKER_WAV_ROOT, audio_id)
        if not os.path.isdir(audio_path):
            continue

        out_audio_dir = os.path.join(MERGED_OUT, audio_id)
        os.makedirs(out_audio_dir, exist_ok=True)

        for speaker in os.listdir(audio_path):
            spk_dir = os.path.join(audio_path, speaker)
            if not os.path.isdir(spk_dir):
                continue

            out_file = os.path.join(out_audio_dir, f"{speaker}.wav")

            print(f"🔗 Merging {audio_id}/{speaker}")
            merge_wavs(spk_dir, out_file)

    print("\n✅ Speaker WAVs merged successfully")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
