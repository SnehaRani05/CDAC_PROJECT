# import os

# DATASET_ROOT = "processed"   # based on your log output

# for language in os.listdir(DATASET_ROOT):
#     lang_path = os.path.join(DATASET_ROOT, language)

#     txt_dir = os.path.join(lang_path, "txt")
#     rttm_dir = os.path.join(lang_path, "rttm")

#     if not os.path.exists(txt_dir):
#         continue

#     os.makedirs(rttm_dir, exist_ok=True)

#     for txt_file in os.listdir(txt_dir):
#         if not txt_file.endswith(".txt"):
#             continue

#         uri = os.path.splitext(txt_file)[0]
#         txt_path = os.path.join(txt_dir, txt_file)
#         rttm_path = os.path.join(rttm_dir, uri + ".rttm")

#         with open(txt_path, "r", encoding="utf-8") as f, open(rttm_path, "w") as out:
#             for line in f:
#                 line = line.strip()

#                 # skip empty lines
#                 if not line:
#                     continue

#                 parts = line.split()

#                 # we need at least: start end speaker
#                 if len(parts) < 3:
#                     continue

#                 start, end, speaker = parts[0], parts[1], parts[2]

#                 try:
#                     start = float(start)
#                     end = float(end)
#                 except ValueError:
#                     # skip header or malformed lines
#                     continue

#                 duration = end - start
#                 if duration <= 0:
#                     continue

#                 out.write(
#                     f"SPEAKER {uri} 1 "
#                     f"{start:.6f} {duration:.6f} "
#                     f"<NA> <NA> {speaker} <NA>\n"
#                 )

#         print(f"Created: {rttm_path}")

# print("\nAll valid TXT files converted to RTTM")


# # import os

# # DATASET_ROOT = "processed"

# # print("Looking inside:", DATASET_ROOT)

# # for language in os.listdir(DATASET_ROOT):
# #     lang_path = os.path.join(DATASET_ROOT, language)
# #     print("\nLanguage folder:", language)

# #     txt_dir = os.path.join(lang_path, "txt")
# #     print("Expecting txt folder at:", txt_dir)

# #     if not os.path.exists(txt_dir):
# #         print("❌ txt folder NOT FOUND")
# #         continue
# #     else:
# #         print("✅ txt folder found")

# #     for file in os.listdir(txt_dir):
# #         print("Found file:", file)


import os
import librosa
import soundfile as sf

# =========================
# CONFIG (CHANGE THESE ONLY)
# =========================D
TXT_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\txt"
AUDIO_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\cleaned_audio\dataset"
OUTPUT_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\cleaned_audio\output"

SR = 16000
MIN_SEG_DUR = 0.3  # seconds

# =========================
# TXT → RTTM
# =========================
def txt_to_rttm(txt_path, rttm_path, uri):
    with open(txt_path, "r", encoding="utf-8") as f, open(rttm_path, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            # FORMAT: start end speaker
            try:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
            except ValueError:
                continue

            duration = end - start
            if duration < MIN_SEG_DUR:
                continue

            out.write(
                f"SPEAKER {uri} 1 "
                f"{start:.6f} {duration:.6f} "
                f"<NA> <NA> {speaker} <NA>\n"
            )

# =========================
# RTTM → SPEAKER WAVS
# =========================
def rttm_to_wavs(audio_path, rttm_path, out_dir):
    audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    os.makedirs(out_dir, exist_ok=True)

    counters = {}

    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]

            s = int(start * sr)
            e = int((start + dur) * sr)

            if e <= s:
                continue

            chunk = audio[s:e]

            spk_dir = os.path.join(out_dir, speaker)
            os.makedirs(spk_dir, exist_ok=True)

            counters[speaker] = counters.get(speaker, 0) + 1
            out_file = os.path.join(
                spk_dir, f"{speaker}_{counters[speaker]:03d}.wav"
            )

            sf.write(out_file, chunk, sr)

# =========================
# MAIN (BATCH)
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for txt_file in os.listdir(TXT_DIR):
        if not txt_file.endswith(".txt"):
            continue

        uri = os.path.splitext(txt_file)[0]

        txt_path = os.path.join(TXT_DIR, txt_file)
        audio_path = os.path.join(AUDIO_DIR, uri + ".wav")

        if not os.path.exists(audio_path):
            print(f"⚠️ Audio not found for {uri}, skipping")
            continue

        print(f"🎧 Processing {uri}")

        rttm_dir = os.path.join(OUTPUT_DIR, "rttm")
        wav_dir = os.path.join(OUTPUT_DIR, "speaker_wavs", uri)

        os.makedirs(rttm_dir, exist_ok=True)

        rttm_path = os.path.join(rttm_dir, uri + ".rttm")

        txt_to_rttm(txt_path, rttm_path, uri)
        rttm_to_wavs(audio_path, rttm_path, wav_dir)

    print("\n✅ ALL TXT FILES PROCESSED SUCCESSFULLY")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
