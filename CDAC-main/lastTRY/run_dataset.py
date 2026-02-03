import os
from diarization import run_diarization
from der_utils import compute_der

AUDIO_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\audio"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

wav_files = [
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith(".wav")
]

print(f"Found {len(wav_files)} audio files\n")

for wav in wav_files:

    audio_path = os.path.join(AUDIO_DIR, wav)
    file_id = os.path.splitext(wav)[0]

    print("=" * 50)
    print(f"Processing: {wav}")

    out_dir = os.path.join(OUTPUT_DIR, file_id)
    os.makedirs(out_dir, exist_ok=True)

    segments, rttm_path = run_diarization(
        audio_path=audio_path,
        output_dir=out_dir,
        min_segment_len=0.7,
        merge_threshold=0.8
    )

    speaker_count = len({s["speaker"] for s in segments})
    print(f"Final speakers: {speaker_count}")

    der = compute_der(file_id, rttm_path)
    if der is not None:
        print(f"DER: {der*100:.2f}%")

print("\n✅ DATASET DIARIZATION COMPLETE")
