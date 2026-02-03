# # # import os
# # # import librosa
# # # import soundfile as sf
# # # import noisereduce as nr
# # # import numpy as np

# # # # =========================
# # # # CONFIG
# # # # =========================
# # # DATASET_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed"
# # # OUTPUT_DIR = "cleaned_audio"

# # # TARGET_SR = 16000          # Required for pyannote
# # # VAD_TOP_DB = 28            # 25–30 works well for Indian languages
# # # TARGET_DBFS = -20.0        # RMS normalization target

# # # # =========================
# # # # VAD (librosa-based)
# # # # =========================
# # # def apply_vad_mask(audio, sr, top_db=VAD_TOP_DB):
# # #     """
# # #     Energy-based Voice Activity Detection.
# # #     Removes silence while preserving speech.
# # #     """
# # #     intervals = librosa.effects.split(audio, top_db=top_db)
# # #     mask = np.zeros_like(audio)

# # #     for start, end in intervals:
# # #         mask[start:end] = 1.0

# # #     return audio * mask


# # # # =========================
# # # # RMS NORMALIZATION
# # # # =========================
# # # def rms_normalize(audio, target_dbfs=TARGET_DBFS):
# # #     """
# # #     Normalize audio to target RMS level (dBFS).
# # #     Safe for diarization and ASR.
# # #     """
# # #     rms = np.sqrt(np.mean(audio ** 2))
# # #     if rms == 0:
# # #         return audio

# # #     current_dbfs = 20 * np.log10(rms)
# # #     gain = 10 ** ((target_dbfs - current_dbfs) / 20)

# # #     return audio * gain


# # # # =========================
# # # # AUDIO PROCESSING
# # # # =========================
# # # def process_audio(in_path, out_path):
# # #     # Load → mono, 16 kHz
# # #     audio, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

# # #     # Noise reduction
# # #     reduced = nr.reduce_noise(y=audio, sr=sr)

# # #     # Voice Activity Detection
# # #     voiced = apply_vad_mask(reduced, sr)

# # #     # RMS loudness normalization
# # #     voiced = rms_normalize(voiced, TARGET_DBFS)

# # #     # Safety clip
# # #     voiced = np.clip(voiced, -1.0, 1.0)

# # #     # Ensure output directory exists
# # #     os.makedirs(os.path.dirname(out_path), exist_ok=True)

# # #     # Save cleaned audio
# # #     sf.write(out_path, voiced, sr)


# # # # =========================
# # # # DATASET WALK
# # # # =========================
# # # def main():
# # #     os.makedirs(OUTPUT_DIR, exist_ok=True)

# # #     for lang in os.listdir(DATASET_DIR):
# # #         lang_path = os.path.join(DATASET_DIR, lang)

# # #         if not os.path.isdir(lang_path):
# # #             continue

# # #         wav_dir = os.path.join(lang_path, "audio")
# # #         if not os.path.exists(wav_dir):
# # #             continue

# # #         out_lang = os.path.join(OUTPUT_DIR, lang)
# # #         os.makedirs(out_lang, exist_ok=True)

# # #         for file in os.listdir(wav_dir):
# # #             if not file.lower().endswith(".wav"):
# # #                 continue

# # #             in_file = os.path.join(wav_dir, file)
# # #             out_file = os.path.join(out_lang, file)

# # #             print(f"🔊 Processing: {lang}/{file}")
# # #             process_audio(in_file, out_file)

# # #     print("✅ DONE: Noise reduction + VAD + normalization applied")


# # # # =========================
# # # # ENTRY POINT
# # # # =========================
# # # if __name__ == "__main__":
# # #     main()


# import os
# import sys
# import librosa
# import soundfile as sf
# import noisereduce as nr
# import numpy as np

# # =========================
# # CONFIG
# # =========================
# TARGET_SR = 16000
# VAD_TOP_DB = 28
# TARGET_DBFS = -20.0

# # =========================
# # VAD
# # =========================
# def apply_vad_mask(audio, top_db=VAD_TOP_DB):
#     intervals = librosa.effects.split(audio, top_db=top_db)
#     mask = np.zeros_like(audio)
#     for start, end in intervals:
#         mask[start:end] = 1.0
#     return audio * mask

# # =========================
# # RMS NORMALIZATION
# # =========================
# def rms_normalize(audio, target_dbfs=TARGET_DBFS):
#     rms = np.sqrt(np.mean(audio ** 2))
#     if rms == 0:
#         return audio
#     current_dbfs = 20 * np.log10(rms)
#     gain = 10 ** ((target_dbfs - current_dbfs) / 20)
#     return audio * gain

# # =========================
# # CORE PROCESS
# # =========================
# def process_audio_file(in_path, out_path):
#     if not os.path.exists(in_path):
#         raise FileNotFoundError(f"Input audio not found: {in_path}")

#     audio, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

#     reduced = nr.reduce_noise(y=audio, sr=sr)
#     voiced = apply_vad_mask(reduced)
#     voiced = rms_normalize(voiced)
#     voiced = np.clip(voiced, -1.0, 1.0)

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     sf.write(out_path, voiced, sr)

#     print(f"Cleaned audio written to: {out_path}")

# # =========================
# # CLI ENTRY POINT ✅
# # =========================
# if __name__ == "__main__":

#     if len(sys.argv) != 3:
#         print("Usage: python process.py <input.wav> <output.wav>")
#         sys.exit(1)

#     in_path = sys.argv[1]
#     out_path = sys.argv[2]

#     print("PROCESS.PY INPUT :", in_path)
#     print("PROCESS.PY OUTPUT:", out_path)

#     process_audio_file(in_path, out_path)




import os
import sys
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

TARGET_SR = 16000
VAD_TOP_DB = 28
TARGET_DBFS = -20.0


def apply_vad(audio):
    intervals = librosa.effects.split(audio, top_db=VAD_TOP_DB)

    if len(intervals) == 0:
        return audio

    return np.concatenate([audio[s:e] for s, e in intervals])


def rms_normalize(audio):
    if len(audio) == 0:
        return audio

    rms = np.sqrt(np.mean(audio**2))

    if rms < 1e-6:
        return audio

    gain = 10 ** ((TARGET_DBFS - (20 * np.log10(rms))) / 20)
    return audio * gain


def process_audio(in_path, out_path):

    if not os.path.exists(in_path):
        print("Input file missing")
        sys.exit(1)

    audio, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

    reduced = nr.reduce_noise(y=audio, sr=sr, stationary=True)

    voiced = apply_vad(reduced)
    voiced = rms_normalize(voiced)

    voiced = np.clip(voiced, -1.0, 1.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, voiced, sr)

    print("Audio cleaned successfully")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python process.py input.wav output.wav")
        sys.exit(1)

    process_audio(sys.argv[1], sys.argv[2])
