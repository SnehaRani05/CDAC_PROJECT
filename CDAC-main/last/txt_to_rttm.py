import os
import soundfile as sf

TXT_DIR = "dataset/txt"
AUDIO_DIR = "dataset/audio"
RTTM_DIR = "dataset/rttm"

os.makedirs(RTTM_DIR, exist_ok=True)

for txt_file in os.listdir(TXT_DIR):
    if not txt_file.endswith(".txt"):
        continue

    file_id = os.path.splitext(txt_file)[0]
    wav_path = os.path.join(AUDIO_DIR, file_id + ".wav")
    txt_path = os.path.join(TXT_DIR, txt_file)
    rttm_path = os.path.join(RTTM_DIR, file_id + ".rttm")

    if not os.path.exists(wav_path):
        print(f"⚠️ Missing audio for {file_id}, skipping")
        continue

    # get audio duration
    with sf.SoundFile(wav_path) as f:
        audio_duration = len(f) / f.samplerate

    with open(txt_path, "r") as fin, open(rttm_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 3:
                continue

            start, end, speaker = parts[0], parts[1], parts[2]
            start = float(start)
            end = float(end)

            # ❌ skip fully invalid segments
            if start >= audio_duration:
                continue

            # ✂️ clip segment
            end = min(end, audio_duration)
            duration = end - start

            if duration <= 0:
                continue

            fout.write(
                f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    print(f"✅ Fixed & converted: {txt_file}")
