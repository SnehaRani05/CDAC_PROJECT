import os
import whisper

# =========================
# CONFIG
# =========================
MERGED_WAV_ROOT = "outputs/merged_speakers"
TRANSCRIPT_OUT = "outputs/transcripts"

MODEL_SIZE = "small"   # tiny | base | small | medium
DEVICE = "cpu"         # change to "cuda" if GPU available

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

# =========================
# MAIN
# =========================
def main():
    os.makedirs(TRANSCRIPT_OUT, exist_ok=True)

    for audio_id in os.listdir(MERGED_WAV_ROOT):
        audio_dir = os.path.join(MERGED_WAV_ROOT, audio_id)
        if not os.path.isdir(audio_dir):
            continue

        out_dir = os.path.join(TRANSCRIPT_OUT, audio_id)
        os.makedirs(out_dir, exist_ok=True)

        for wav_file in os.listdir(audio_dir):
            if not wav_file.lower().endswith(".wav"):
                continue

            speaker = os.path.splitext(wav_file)[0]
            wav_path = os.path.join(audio_dir, wav_file)

            print(f"📝 Transcribing: {audio_id}/{speaker}")

            result = model.transcribe(
                wav_path,
                fp16=False,          # CPU safe
                language=None        # auto-detect language
            )

            out_txt = os.path.join(out_dir, f"{speaker}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(result["text"].strip())

    print("\n✅ ASR completed for all speakers")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
