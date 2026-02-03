import os

TXT_DIR  = "txt"
RTTM_DIR = "rttm"

os.makedirs(RTTM_DIR, exist_ok=True)

for txt_file in os.listdir(TXT_DIR):
    if not txt_file.endswith(".txt"):
        continue

    audio_id = txt_file.replace(".txt", "")
    txt_path = os.path.join(TXT_DIR, txt_file)
    rttm_path = os.path.join(RTTM_DIR, audio_id + ".rttm")

    with open(txt_path, "r") as f, open(rttm_path, "w") as out:
        for line in f:
            parts = line.strip().split()

            # skip empty / invalid lines
            if len(parts) < 3:
                continue

            start = float(parts[0])
            end = float(parts[1])
            speaker = parts[2]

            duration = end - start
            if duration <= 0:
                continue

            out.write(
                f"SPEAKER {audio_id} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA>\n"
            )

    print(f"Created RTTM: {rttm_path}")
