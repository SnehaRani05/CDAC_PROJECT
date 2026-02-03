# # # # # # import os
# # # # # # import numpy as np
# # # # # # import librosa
# # # # # # import soundfile as sf
# # # # # # import streamlit as st
# # # # # # import matplotlib.pyplot as plt
# # # # # # import noisereduce as nr
# # # # # # import tempfile

# # # # # # from pyannote.audio import Pipeline
# # # # # # from pyannote.core import Segment
# # # # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # # # ===============================
# # # # # # # CONFIG
# # # # # # # ===============================
# # # # # # SR = 16000
# # # # # # VAD_TOP_DB = 28
# # # # # # TARGET_DBFS = -20.0

# # # # # # OUTPUT_DIR = "output"
# # # # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # # # RTTM_DIR = os.path.join(OUTPUT_DIR, "rttm")
# # # # # # PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# # # # # # os.makedirs(SPEAKER_DIR, exist_ok=True)
# # # # # # os.makedirs(RTTM_DIR, exist_ok=True)
# # # # # # os.makedirs(PLOT_DIR, exist_ok=True)

# # # # # # # ===============================
# # # # # # # AUDIO UTILITIES
# # # # # # # ===============================
# # # # # # def apply_vad(audio):
# # # # # #     intervals = librosa.effects.split(audio, top_db=VAD_TOP_DB)
# # # # # #     if len(intervals) == 0:
# # # # # #         return audio
# # # # # #     return np.concatenate([audio[s:e] for s, e in intervals])

# # # # # # def rms_normalize(audio):
# # # # # #     rms = np.sqrt(np.mean(audio ** 2))
# # # # # #     if rms < 1e-6:
# # # # # #         return audio
# # # # # #     gain = 10 ** ((TARGET_DBFS - (20 * np.log10(rms))) / 20)
# # # # # #     return audio * gain

# # # # # # def preprocess_audio(path):
# # # # # #     audio, sr = librosa.load(path, sr=SR, mono=True)
# # # # # #     reduced = nr.reduce_noise(y=audio, sr=sr, stationary=True)
# # # # # #     voiced = apply_vad(reduced)
# # # # # #     voiced = rms_normalize(voiced)
# # # # # #     return np.clip(voiced, -1.0, 1.0)

# # # # # # # ===============================
# # # # # # # RTTM READER
# # # # # # # ===============================
# # # # # # def read_rttm(path):
# # # # # #     from pyannote.core import Annotation
# # # # # #     ann = Annotation()
# # # # # #     with open(path) as f:
# # # # # #         for line in f:
# # # # # #             p = line.strip().split()
# # # # # #             start = float(p[3])
# # # # # #             dur = float(p[4])
# # # # # #             speaker = p[7]
# # # # # #             ann[Segment(start, start + dur)] = speaker
# # # # # #     return ann

# # # # # # # ===============================
# # # # # # # STREAMLIT UI
# # # # # # # ===============================
# # # # # # st.set_page_config(page_title="Speaker Diarization UI", layout="wide")
# # # # # # st.title("🎙 Speaker Diarization + DER Evaluation")

# # # # # # st.sidebar.header("Input Options")

# # # # # # uploaded_audio = st.sidebar.file_uploader("Upload WAV audio", type=["wav"])
# # # # # # uploaded_rttm = st.sidebar.file_uploader("Optional Manual RTTM", type=["rttm"])

# # # # # # run_btn = st.sidebar.button("🚀 Run Diarization")

# # # # # # # ===============================
# # # # # # # MAIN PIPELINE
# # # # # # # ===============================
# # # # # # if run_btn and uploaded_audio:

# # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
# # # # # #         tmp.write(uploaded_audio.read())
# # # # # #         audio_path = tmp.name

# # # # # #     file_id = os.path.splitext(uploaded_audio.name)[0]

# # # # # #     st.info("🔧 Preprocessing audio...")
# # # # # #     clean_audio = preprocess_audio(audio_path)

# # # # # #     clean_path = os.path.join("input", file_id + "_cleaned.wav")
# # # # # #     os.makedirs("input", exist_ok=True)
# # # # # #     sf.write(clean_path, clean_audio, SR)

# # # # # #     st.audio(clean_path)

# # # # # #     st.info("🧠 Loading diarization model...")
# # # # # #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# # # # # #     diarization = pipeline(clean_path)

# # # # # #     # ===============================
# # # # # #     # SAVE SPEAKER SEGMENTS + RTTM
# # # # # #     # ===============================
# # # # # #     rttm_lines = []
# # # # # #     segments = []

# # # # # #     for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
# # # # # #         start, end = turn.start, turn.end
# # # # # #         start_s, end_s = int(start * SR), int(end * SR)

# # # # # #         if end_s - start_s < 0.3 * SR:
# # # # # #             continue

# # # # # #         seg_audio = clean_audio[start_s:end_s]
# # # # # #         wav_name = f"{file_id}_{speaker}_{idx}.wav"
# # # # # #         sf.write(os.path.join(SPEAKER_DIR, wav_name), seg_audio, SR)

# # # # # #         rttm_lines.append(
# # # # # #             f"SPEAKER {file_id} 1 {start:.3f} {end-start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# # # # # #         )
# # # # # #         segments.append((start, end, speaker))

# # # # # #     hyp_rttm_path = os.path.join(RTTM_DIR, file_id + ".rttm")
# # # # # #     with open(hyp_rttm_path, "w") as f:
# # # # # #         f.writelines(rttm_lines)

# # # # # #     st.success("✅ Speaker segments + RTTM generated")

# # # # # #     # ===============================
# # # # # #     # DER CALCULATION
# # # # # #     # ===============================
# # # # # #     if uploaded_rttm:
# # # # # #         ref_path = os.path.join(RTTM_DIR, "reference.rttm")
# # # # # #         with open(ref_path, "wb") as f:
# # # # # #             f.write(uploaded_rttm.read())

# # # # # #         reference = read_rttm(ref_path)
# # # # # #         hypothesis = read_rttm(hyp_rttm_path)

# # # # # #         metric = DiarizationErrorRate(collar=0.25)
# # # # # #         der = metric(reference, hypothesis)

# # # # # #         st.metric("📊 Diarization Error Rate (DER)", f"{der*100:.2f}%")

# # # # # #     # ===============================
# # # # # #     # TIMELINE PLOT
# # # # # #     # ===============================
# # # # # #     st.info("📈 Generating speaker timeline")

# # # # # #     speakers = list(set(s for _, _, s in segments))
# # # # # #     speaker_map = {s: i for i, s in enumerate(speakers)}

# # # # # #     fig, ax = plt.subplots(figsize=(12, 3))
# # # # # #     for start, end, speaker in segments:
# # # # # #         ax.barh(
# # # # # #             speaker_map[speaker],
# # # # # #             end - start,
# # # # # #             left=start,
# # # # # #             height=0.4,
# # # # # #             label=speaker
# # # # # #         )

# # # # # #     ax.set_yticks(list(speaker_map.values()))
# # # # # #     ax.set_yticklabels(list(speaker_map.keys()))
# # # # # #     ax.set_xlabel("Time (seconds)")
# # # # # #     ax.set_title("Speaker Timeline")

# # # # # #     plt.tight_layout()
# # # # # #     plot_path = os.path.join(PLOT_DIR, file_id + ".png")
# # # # # #     plt.savefig(plot_path)
# # # # # #     st.pyplot(fig)

# # # # # #     st.success("🎉 Pipeline completed successfully")

# # # # # # else:
# # # # # #     st.warning("⬅ Upload audio and click **Run Diarization**")



# # # # # import os
# # # # # import tempfile
# # # # # import numpy as np
# # # # # import librosa
# # # # # import soundfile as sf
# # # # # import streamlit as st
# # # # # import matplotlib.pyplot as plt
# # # # # import pandas as pd
# # # # # import noisereduce as nr

# # # # # from pyannote.audio import Pipeline
# # # # # from pyannote.core import Segment, Annotation
# # # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # # =========================
# # # # # # CONFIG
# # # # # # =========================
# # # # # SR = 16000
# # # # # VAD_TOP_DB = 28
# # # # # TARGET_DBFS = -20.0

# # # # # MANUAL_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

# # # # # OUTPUT_DIR = "output"
# # # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # # RTTM_DIR = os.path.join(OUTPUT_DIR, "rttm")

# # # # # os.makedirs(SPEAKER_DIR, exist_ok=True)
# # # # # os.makedirs(RTTM_DIR, exist_ok=True)

# # # # # # =========================
# # # # # # AUDIO PREPROCESSING
# # # # # # =========================
# # # # # def apply_vad(audio):
# # # # #     intervals = librosa.effects.split(audio, top_db=VAD_TOP_DB)
# # # # #     if len(intervals) == 0:
# # # # #         return audio
# # # # #     return np.concatenate([audio[s:e] for s, e in intervals])

# # # # # def rms_normalize(audio):
# # # # #     rms = np.sqrt(np.mean(audio ** 2))
# # # # #     if rms < 1e-6:
# # # # #         return audio
# # # # #     gain = 10 ** ((TARGET_DBFS - (20 * np.log10(rms))) / 20)
# # # # #     return audio * gain

# # # # # def preprocess_audio(path):
# # # # #     audio, _ = librosa.load(path, sr=SR, mono=True)
# # # # #     reduced = nr.reduce_noise(y=audio, sr=SR, stationary=True)
# # # # #     voiced = apply_vad(reduced)
# # # # #     voiced = rms_normalize(voiced)
# # # # #     return np.clip(voiced, -1.0, 1.0)

# # # # # # =========================
# # # # # # RTTM UTIL
# # # # # # =========================
# # # # # def read_rttm(path):
# # # # #     ann = Annotation()
# # # # #     with open(path) as f:
# # # # #         for line in f:
# # # # #             p = line.strip().split()
# # # # #             start = float(p[3])
# # # # #             dur = float(p[4])
# # # # #             speaker = p[7]
# # # # #             ann[Segment(start, start + dur)] = speaker
# # # # #     return ann

# # # # # # =========================
# # # # # # STREAMLIT UI
# # # # # # =========================
# # # # # st.set_page_config(page_title="Speaker Diarization UI", layout="wide")
# # # # # st.title("🎙 Speaker Diarization + DER + RTTM Viewer")

# # # # # uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"])
# # # # # run_btn = st.button("🚀 Run Diarization")

# # # # # # =========================
# # # # # # MAIN PIPELINE
# # # # # # =========================
# # # # # if run_btn and uploaded_audio:

# # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
# # # # #         tmp.write(uploaded_audio.read())
# # # # #         raw_audio_path = tmp.name

# # # # #     file_id = os.path.splitext(uploaded_audio.name)[0]
# # # # #     file_id = file_id.replace("_cleaned", "")

# # # # #     st.subheader("1️⃣ Audio Preprocessing")
# # # # #     clean_audio = preprocess_audio(raw_audio_path)

# # # # #     clean_path = os.path.join("input", file_id + "_cleaned.wav")
# # # # #     os.makedirs("input", exist_ok=True)
# # # # #     sf.write(clean_path, clean_audio, SR)

# # # # #     st.audio(clean_path)

# # # # #     # =========================
# # # # #     # DIARIZATION
# # # # #     # =========================
# # # # #     st.subheader("2️⃣ Speaker Diarization")
# # # # #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# # # # #     diarization = pipeline(clean_path)

# # # # #     # =========================
# # # # #     # SAVE SEGMENTS + RTTM
# # # # #     # =========================
# # # # #     rttm_lines = []
# # # # #     rttm_rows = []
# # # # #     segments = []

# # # # #     for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
# # # # #         start, end = turn.start, turn.end
# # # # #         start_s, end_s = int(start * SR), int(end * SR)

# # # # #         if end_s - start_s < 0.3 * SR:
# # # # #             continue

# # # # #         seg_audio = clean_audio[start_s:end_s]
# # # # #         wav_name = f"{file_id}_{speaker}_{idx}.wav"
# # # # #         sf.write(os.path.join(SPEAKER_DIR, wav_name), seg_audio, SR)

# # # # #         rttm_lines.append(
# # # # #             f"SPEAKER {file_id} 1 {start:.3f} {end-start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# # # # #         )

# # # # #         rttm_rows.append({
# # # # #             "File ID": file_id,
# # # # #             "Start (s)": round(start, 3),
# # # # #             "Duration (s)": round(end - start, 3),
# # # # #             "Speaker": speaker
# # # # #         })

# # # # #         segments.append((start, end, speaker))

# # # # #     hyp_rttm_path = os.path.join(RTTM_DIR, file_id + ".rttm")
# # # # #     with open(hyp_rttm_path, "w") as f:
# # # # #         f.writelines(rttm_lines)

# # # # #     st.success("✅ Hypothesis RTTM generated")

# # # # #     # =========================
# # # # #     # SHOW HYPOTHESIS RTTM TABLE
# # # # #     # =========================
# # # # #     st.subheader("3️⃣ Hypothesis RTTM (Table View)")
# # # # #     rttm_df = pd.DataFrame(rttm_rows)
# # # # #     st.dataframe(rttm_df, use_container_width=True)

# # # # #     # =========================
# # # # #     # DER CALCULATION
# # # # #     # =========================
# # # # #     st.subheader("4️⃣ Diarization Error Rate (DER)")

# # # # #     manual_rttm_path = os.path.join(MANUAL_RTTM_DIR, file_id + ".rttm")

# # # # #     if os.path.exists(manual_rttm_path):
# # # # #         reference = read_rttm(manual_rttm_path)
# # # # #         hypothesis = read_rttm(hyp_rttm_path)

# # # # #         metric = DiarizationErrorRate(collar=0.25)
# # # # #         der = metric(reference, hypothesis)

# # # # #         st.metric("DER (%)", f"{der * 100:.2f}")
# # # # #     else:
# # # # #         st.warning("Manual RTTM not found → DER skipped")

# # # # #     # =========================
# # # # #     # TIMELINE PLOT
# # # # #     # =========================
# # # # #     st.subheader("5️⃣ Speaker Timeline")

# # # # #     speakers = list(sorted(set(s for _, _, s in segments)))
# # # # #     speaker_map = {s: i for i, s in enumerate(speakers)}

# # # # #     fig, ax = plt.subplots(figsize=(12, 3))
# # # # #     for start, end, speaker in segments:
# # # # #         ax.barh(
# # # # #             speaker_map[speaker],
# # # # #             end - start,
# # # # #             left=start,
# # # # #             height=0.45,
# # # # #             label=speaker
# # # # #         )

# # # # #     ax.set_yticks(list(speaker_map.values()))
# # # # #     ax.set_yticklabels(list(speaker_map.keys()))
# # # # #     ax.set_xlabel("Time (seconds)")
# # # # #     ax.set_title("Time vs Speaker")
# # # # #     plt.tight_layout()

# # # # #     st.pyplot(fig)

# # # # #     st.success("🎉 Pipeline completed successfully")

# # # # # else:
# # # # #     st.info("⬆ Upload audio and click **Run Diarization**")




# # # # import os
# # # # import tempfile
# # # # import numpy as np
# # # # import librosa
# # # # import soundfile as sf
# # # # import streamlit as st
# # # # import matplotlib.pyplot as plt
# # # # import pandas as pd
# # # # import noisereduce as nr

# # # # from pyannote.audio import Pipeline
# # # # from pyannote.core import Segment, Annotation
# # # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # # =========================
# # # # # CONFIG
# # # # # =========================
# # # # SR = 16000
# # # # VAD_TOP_DB = 28
# # # # TARGET_DBFS = -20.0

# # # # MANUAL_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

# # # # OUTPUT_DIR = "output"
# # # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # # RTTM_DIR = os.path.join(OUTPUT_DIR, "rttm")

# # # # os.makedirs(SPEAKER_DIR, exist_ok=True)
# # # # os.makedirs(RTTM_DIR, exist_ok=True)

# # # # # =========================
# # # # # AUDIO PREPROCESSING
# # # # # =========================
# # # # def apply_vad(audio):
# # # #     intervals = librosa.effects.split(audio, top_db=VAD_TOP_DB)
# # # #     if len(intervals) == 0:
# # # #         return audio
# # # #     return np.concatenate([audio[s:e] for s, e in intervals])

# # # # def rms_normalize(audio):
# # # #     rms = np.sqrt(np.mean(audio ** 2))
# # # #     if rms < 1e-6:
# # # #         return audio
# # # #     gain = 10 ** ((TARGET_DBFS - (20 * np.log10(rms))) / 20)
# # # #     return audio * gain

# # # # def preprocess_audio(path):
# # # #     audio, _ = librosa.load(path, sr=SR, mono=True)
# # # #     reduced = nr.reduce_noise(y=audio, sr=SR, stationary=True)
# # # #     voiced = apply_vad(reduced)
# # # #     voiced = rms_normalize(voiced)
# # # #     return np.clip(voiced, -1.0, 1.0)

# # # # # =========================
# # # # # RTTM UTIL
# # # # # =========================
# # # # def read_rttm(path):
# # # #     ann = Annotation()
# # # #     with open(path) as f:
# # # #         for line in f:
# # # #             p = line.strip().split()
# # # #             start = float(p[3])
# # # #             dur = float(p[4])
# # # #             speaker = p[7]
# # # #             ann[Segment(start, start + dur)] = speaker
# # # #     return ann

# # # # # =========================
# # # # # STREAMLIT UI
# # # # # =========================
# # # # st.set_page_config(page_title="Speaker Diarization UI", layout="wide")
# # # # st.title("🎙 Speaker Diarization + Playback + DER")

# # # # uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"])
# # # # run_btn = st.button("🚀 Run Diarization")

# # # # # =========================
# # # # # MAIN PIPELINE
# # # # # =========================
# # # # if run_btn and uploaded_audio:

# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
# # # #         tmp.write(uploaded_audio.read())
# # # #         raw_audio_path = tmp.name

# # # #     file_id = os.path.splitext(uploaded_audio.name)[0]
# # # #     file_id = file_id.replace("_cleaned", "")

# # # #     # =========================
# # # #     # PREPROCESS
# # # #     # =========================
# # # #     st.subheader("1️⃣ Audio Preprocessing")
# # # #     clean_audio = preprocess_audio(raw_audio_path)

# # # #     clean_path = os.path.join("input", file_id + "_cleaned.wav")
# # # #     os.makedirs("input", exist_ok=True)
# # # #     sf.write(clean_path, clean_audio, SR)
# # # #     st.audio(clean_path)

# # # #     # =========================
# # # #     # DIARIZATION
# # # #     # =========================
# # # #     st.subheader("2️⃣ Speaker Diarization")
# # # #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# # # #     diarization = pipeline(clean_path)

# # # #     # =========================
# # # #     # SAVE SEGMENTS + RTTM
# # # #     # =========================
# # # #     rttm_lines = []
# # # #     rttm_rows = []
# # # #     segments = []
# # # #     speaker_segments = {}

# # # #     for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
# # # #         start, end = turn.start, turn.end
# # # #         start_s, end_s = int(start * SR), int(end * SR)

# # # #         if end_s - start_s < 0.3 * SR:
# # # #             continue

# # # #         seg_audio = clean_audio[start_s:end_s]

# # # #         wav_name = f"{file_id}_{speaker}_{idx}.wav"
# # # #         wav_path = os.path.join(SPEAKER_DIR, wav_name)
# # # #         sf.write(wav_path, seg_audio, SR)

# # # #         rttm_lines.append(
# # # #             f"SPEAKER {file_id} 1 {start:.3f} {end-start:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
# # # #         )

# # # #         rttm_rows.append({
# # # #             "Start (s)": round(start, 3),
# # # #             "Duration (s)": round(end - start, 3),
# # # #             "Speaker": speaker
# # # #         })

# # # #         segments.append((start, end, speaker))
# # # #         speaker_segments.setdefault(speaker, []).append(
# # # #             (start, end, wav_path)
# # # #         )

# # # #     hyp_rttm_path = os.path.join(RTTM_DIR, file_id + ".rttm")
# # # #     with open(hyp_rttm_path, "w") as f:
# # # #         f.writelines(rttm_lines)

# # # #     # =========================
# # # #     # RTTM TABLE
# # # #     # =========================
# # # #     st.subheader("3️⃣ Hypothesis RTTM (Table)")
# # # #     st.dataframe(pd.DataFrame(rttm_rows), use_container_width=True)

# # # #     # =========================
# # # #     # 🔊 PLAY SPEAKER SEGMENTS
# # # #     # =========================
# # # #     st.subheader("4️⃣ Listen to Speaker Segments")

# # # #     for speaker, clips in speaker_segments.items():
# # # #         with st.expander(f"🔈 {speaker} ({len(clips)} segments)", expanded=False):
# # # #             for i, (start, end, path) in enumerate(clips):
# # # #                 st.markdown(
# # # #                     f"**Segment {i+1}** | ⏱ {start:.2f}s → {end:.2f}s"
# # # #                 )
# # # #                 st.audio(path)

# # # #     # =========================
# # # #     # DER
# # # #     # =========================
# # # #     st.subheader("5️⃣ Diarization Error Rate")

# # # #     manual_rttm_path = os.path.join(MANUAL_RTTM_DIR, file_id + ".rttm")
# # # #     if os.path.exists(manual_rttm_path):
# # # #         reference = read_rttm(manual_rttm_path)
# # # #         hypothesis = read_rttm(hyp_rttm_path)
# # # #         der = DiarizationErrorRate(collar=0.25)(reference, hypothesis)
# # # #         st.metric("DER (%)", f"{der * 100:.2f}")
# # # #     else:
# # # #         st.warning("Manual RTTM not found → DER skipped")

# # # #     # =========================
# # # #     # TIMELINE
# # # #     # =========================
# # # #     st.subheader("6️⃣ Time vs Speaker")

# # # #     speakers = sorted(speaker_segments.keys())
# # # #     speaker_map = {s: i for i, s in enumerate(speakers)}

# # # #     fig, ax = plt.subplots(figsize=(12, 3))
# # # #     for start, end, speaker in segments:
# # # #         ax.barh(
# # # #             speaker_map[speaker],
# # # #             end - start,
# # # #             left=start,
# # # #             height=0.45
# # # #         )

# # # #     ax.set_yticks(list(speaker_map.values()))
# # # #     ax.set_yticklabels(list(speaker_map.keys()))
# # # #     ax.set_xlabel("Time (seconds)")
# # # #     ax.set_title("Speaker Timeline")
# # # #     plt.tight_layout()

# # # #     st.pyplot(fig)

# # # #     st.success("🎉 Done! You can hear each speaker clearly.")

# # # # else:
# # # #     st.info("⬆ Upload audio and click **Run Diarization**")



# # # import os
# # # import tempfile
# # # import numpy as np
# # # import librosa
# # # import soundfile as sf
# # # import streamlit as st
# # # import matplotlib.pyplot as plt
# # # import pandas as pd
# # # import noisereduce as nr

# # # from pyannote.audio import Pipeline
# # # from pyannote.core import Segment, Annotation
# # # from pyannote.metrics.diarization import DiarizationErrorRate

# # # # =========================
# # # # CONFIG
# # # # =========================
# # # SR = 16000
# # # VAD_TOP_DB = 28
# # # TARGET_DBFS = -20.0

# # # MIN_SEGMENT = 0.4     # ignore ultra-short diarization artifacts
# # # PAD = 0.15            # playback padding (NOT for RTTM)

# # # MANUAL_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

# # # OUTPUT_DIR = "output"
# # # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # # RTTM_DIR = os.path.join(OUTPUT_DIR, "rttm")

# # # os.makedirs(SPEAKER_DIR, exist_ok=True)
# # # os.makedirs(RTTM_DIR, exist_ok=True)

# # # # =========================
# # # # AUDIO PREPROCESSING
# # # # =========================
# # # def apply_vad(audio):
# # #     intervals = librosa.effects.split(audio, top_db=VAD_TOP_DB)
# # #     if len(intervals) == 0:
# # #         return audio
# # #     return np.concatenate([audio[s:e] for s, e in intervals])

# # # def rms_normalize(audio):
# # #     rms = np.sqrt(np.mean(audio ** 2))
# # #     if rms < 1e-6:
# # #         return audio
# # #     gain = 10 ** ((TARGET_DBFS - (20 * np.log10(rms))) / 20)
# # #     return audio * gain

# # # def preprocess_audio(path):
# # #     audio, _ = librosa.load(path, sr=SR, mono=True)
# # #     reduced = nr.reduce_noise(y=audio, sr=SR, stationary=True)
# # #     voiced = apply_vad(reduced)
# # #     voiced = rms_normalize(voiced)
# # #     return np.clip(voiced, -1.0, 1.0)

# # # # =========================
# # # # RTTM UTIL
# # # # =========================
# # # def read_rttm(path):
# # #     ann = Annotation()
# # #     with open(path) as f:
# # #         for line in f:
# # #             p = line.strip().split()
# # #             start = float(p[3])
# # #             dur = float(p[4])
# # #             speaker = p[7]
# # #             ann[Segment(start, start + dur)] = speaker
# # #     return ann

# # # # =========================
# # # # STREAMLIT UI
# # # # =========================
# # # st.set_page_config(page_title="Speaker Diarization", layout="wide")
# # # st.title("🎙 Speaker Diarization (Proper Segments, No Merge)")

# # # uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"])
# # # run_btn = st.button("🚀 Run Diarization")

# # # # =========================
# # # # MAIN PIPELINE
# # # # =========================
# # # if run_btn and uploaded_audio:

# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
# # #         tmp.write(uploaded_audio.read())
# # #         raw_audio_path = tmp.name

# # #     file_id = os.path.splitext(uploaded_audio.name)[0].replace("_cleaned", "")

# # #     # =========================
# # #     # PREPROCESS
# # #     # =========================
# # #     st.subheader("1️⃣ Audio Preprocessing")
# # #     clean_audio = preprocess_audio(raw_audio_path)

# # #     clean_path = os.path.join("input", file_id + "_cleaned.wav")
# # #     os.makedirs("input", exist_ok=True)
# # #     sf.write(clean_path, clean_audio, SR)
# # #     st.audio(clean_path)

# # #     # =========================
# # #     # DIARIZATION
# # #     # =========================
# # #     st.subheader("2️⃣ Speaker Diarization")
# # #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# # #     diarization = pipeline(clean_path)

# # #     # =========================
# # #     # SAVE SEGMENTS (NO MERGE)
# # #     # =========================
# # #     rttm_lines = []
# # #     rttm_rows = []
# # #     playback_segments = []

# # #     audio_len_sec = len(clean_audio) / SR

# # #     for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

# # #         duration = turn.end - turn.start
# # #         if duration < MIN_SEGMENT:
# # #             continue

# # #         # RTTM (true boundaries)
# # #         rttm_lines.append(
# # #             f"SPEAKER {file_id} 1 {turn.start:.3f} {duration:.3f} "
# # #             f"<NA> <NA> {speaker} <NA> <NA>\n"
# # #         )

# # #         rttm_rows.append({
# # #             "Start (s)": round(turn.start, 2),
# # #             "End (s)": round(turn.end, 2),
# # #             "Duration (s)": round(duration, 2),
# # #             "Speaker": speaker
# # #         })

# # #         # Playback padding ONLY
# # #         play_start = max(0.0, turn.start - PAD)
# # #         play_end = min(audio_len_sec, turn.end + PAD)

# # #         s, e = int(play_start * SR), int(play_end * SR)
# # #         seg_audio = clean_audio[s:e]

# # #         wav_path = os.path.join(
# # #             SPEAKER_DIR,
# # #             f"{file_id}_{speaker}_{idx}.wav"
# # #         )
# # #         sf.write(wav_path, seg_audio, SR)

# # #         playback_segments.append((turn.start, turn.end, speaker, wav_path))

# # #     hyp_rttm_path = os.path.join(RTTM_DIR, file_id + ".rttm")
# # #     with open(hyp_rttm_path, "w") as f:
# # #         f.writelines(rttm_lines)

# # #     # =========================
# # #     # RTTM TABLE
# # #     # =========================
# # #     st.subheader("3️⃣ Hypothesis RTTM")
# # #     st.dataframe(pd.DataFrame(rttm_rows), use_container_width=True)

# # #     # =========================
# # #     # 🔊 PLAY SEGMENTS
# # #     # =========================
# # #     st.subheader("4️⃣ Listen to Speaker Segments")

# # #     for start, end, speaker, path in playback_segments:
# # #         st.markdown(f"**{speaker}** | ⏱ {start:.2f}s → {end:.2f}s")
# # #         st.audio(path)

# # #     # =========================
# # #     # DER
# # #     # =========================
# # #     st.subheader("5️⃣ Diarization Error Rate")

# # #     manual_rttm_path = os.path.join(MANUAL_RTTM_DIR, file_id + ".rttm")
# # #     if os.path.exists(manual_rttm_path):
# # #         ref = read_rttm(manual_rttm_path)
# # #         hyp = read_rttm(hyp_rttm_path)
# # #         der = DiarizationErrorRate(collar=0.25)(ref, hyp)
# # #         st.metric("DER (%)", f"{der * 100:.2f}")
# # #     else:
# # #         st.warning("Manual RTTM not found → DER skipped")

# # #     # =========================
# # #     # TIMELINE
# # #     # =========================
# # #     st.subheader("6️⃣ Speaker Timeline")

# # #     speakers = sorted(set(row["Speaker"] for row in rttm_rows))
# # #     speaker_map = {s: i for i, s in enumerate(speakers)}

# # #     fig, ax = plt.subplots(figsize=(12, 3))
# # #     for row in rttm_rows:
# # #         ax.barh(
# # #             speaker_map[row["Speaker"]],
# # #             row["Duration (s)"],
# # #             left=row["Start (s)"],
# # #             height=0.45
# # #         )

# # #     ax.set_yticks(list(speaker_map.values()))
# # #     ax.set_yticklabels(list(speaker_map.keys()))
# # #     ax.set_xlabel("Time (seconds)")
# # #     ax.set_title("Time vs Speaker (True Diarization)")
# # #     plt.tight_layout()
# # #     st.pyplot(fig)

# # #     st.success("✅ Proper diarization segments generated (no merging)")

# # # else:
# # #     st.info("⬆ Upload audio and click **Run Diarization**")



# # import os
# # import tempfile
# # import numpy as np
# # import librosa
# # import soundfile as sf
# # import streamlit as st
# # import matplotlib.pyplot as plt
# # import pandas as pd

# # from pyannote.audio import Pipeline
# # from pyannote.core import Segment, Annotation
# # from pyannote.metrics.diarization import DiarizationErrorRate

# # # =========================
# # # CONFIG
# # # =========================
# # SR = 16000
# # MIN_SEGMENT = 0.5      # ignore tiny diarization artifacts
# # PAD = 0.15             # padding for playback only (seconds)

# # MANUAL_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

# # OUTPUT_DIR = "output"
# # SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# # RTTM_DIR = os.path.join(OUTPUT_DIR, "rttm")

# # os.makedirs(SPEAKER_DIR, exist_ok=True)
# # os.makedirs(RTTM_DIR, exist_ok=True)

# # # =========================
# # # RTTM UTIL
# # # =========================
# # def read_rttm(path):
# #     ann = Annotation()
# #     with open(path) as f:
# #         for line in f:
# #             p = line.strip().split()
# #             start = float(p[3])
# #             dur = float(p[4])
# #             speaker = p[7]
# #             ann[Segment(start, start + dur)] = speaker
# #     return ann

# # # =========================
# # # STREAMLIT UI
# # # =========================
# # st.set_page_config(page_title="Speaker Diarization", layout="wide")
# # st.title("🎙 Speaker Diarization (Raw Audio, Proper Segments)")

# # uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"])
# # run_btn = st.button("🚀 Run Diarization")

# # # =========================
# # # MAIN PIPELINE
# # # =========================
# # if run_btn and uploaded_audio:

# #     # Save uploaded audio
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
# #         tmp.write(uploaded_audio.read())
# #         audio_path = tmp.name

# #     file_id = os.path.splitext(uploaded_audio.name)[0].replace("_cleaned", "")

# #     # Load raw audio (NO preprocessing)
# #     audio, sr = librosa.load(audio_path, sr=SR, mono=True)
# #     audio_len_sec = len(audio) / SR

# #     st.subheader("1️⃣ Input Audio (Raw)")
# #     st.audio(audio_path)

# #     # =========================
# #     # DIARIZATION
# #     # =========================
# #     st.subheader("2️⃣ Speaker Diarization")
# #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# #     diarization = pipeline(audio_path)

# #     pipeline.instantiate({
# #     "clustering": {
# #         "method": "centroid",
# #         "threshold": 0.75
# #     }
# # })

# #     # =========================
# #     # SEGMENTS + RTTM
# #     # =========================
# #     rttm_lines = []
# #     rttm_rows = []
# #     playback_segments = []

# #     for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

# #         duration = turn.end - turn.start
# #         if duration < MIN_SEGMENT:
# #             continue

# #         # RTTM (true diarization boundaries)
# #         rttm_lines.append(
# #             f"SPEAKER {file_id} 1 {turn.start:.3f} {duration:.3f} "
# #             f"<NA> <NA> {speaker} <NA> <NA>\n"
# #         )

# #         rttm_rows.append({
# #             "Start (s)": round(turn.start, 2),
# #             "End (s)": round(turn.end, 2),
# #             "Duration (s)": round(duration, 2),
# #             "Speaker": speaker
# #         })

# #         # Playback padding ONLY
# #         play_start = max(0.0, turn.start - PAD)
# #         play_end = min(audio_len_sec, turn.end + PAD)

# #         s, e = int(play_start * SR), int(play_end * SR)
# #         seg_audio = audio[s:e]

# #         wav_path = os.path.join(
# #             SPEAKER_DIR,
# #             f"{file_id}_{speaker}_{idx}.wav"
# #         )
# #         sf.write(wav_path, seg_audio, SR)

# #         playback_segments.append((turn.start, turn.end, speaker, wav_path))

# #     hyp_rttm_path = os.path.join(RTTM_DIR, file_id + ".rttm")
# #     with open(hyp_rttm_path, "w") as f:
# #         f.writelines(rttm_lines)

# #     # =========================
# #     # RTTM TABLE
# #     # =========================
# #     st.subheader("3️⃣ Hypothesis RTTM")
# #     st.dataframe(pd.DataFrame(rttm_rows), use_container_width=True)

# #     # =========================
# #     # 🔊 PLAY SEGMENTS
# #     # =========================
# #     st.subheader("4️⃣ Listen to Speaker Segments")

# #     for start, end, speaker, path in playback_segments:
# #         st.markdown(f"**{speaker}** | ⏱ {start:.2f}s → {end:.2f}s")
# #         st.audio(path)

# #     # =========================
# #     # DER
# #     # =========================
# #     st.subheader("5️⃣ Diarization Error Rate")

# #     manual_rttm_path = os.path.join(MANUAL_RTTM_DIR, file_id + ".rttm")
# #     if os.path.exists(manual_rttm_path):
# #         ref = read_rttm(manual_rttm_path)
# #         hyp = read_rttm(hyp_rttm_path)
# #         der = DiarizationErrorRate(collar=0.25)(ref, hyp)
# #         st.metric("DER (%)", f"{der * 100:.2f}")
# #     else:
# #         st.warning("Manual RTTM not found → DER skipped")

# #     # =========================
# #     # TIMELINE
# #     # =========================
# #     st.subheader("6️⃣ Speaker Timeline")

# #     speakers = sorted(set(row["Speaker"] for row in rttm_rows))
# #     speaker_map = {s: i for i, s in enumerate(speakers)}

# #     fig, ax = plt.subplots(figsize=(12, 3))
# #     for row in rttm_rows:
# #         ax.barh(
# #             speaker_map[row["Speaker"]],
# #             row["Duration (s)"],
# #             left=row["Start (s)"],
# #             height=0.45
# #         )

# #     ax.set_yticks(list(speaker_map.values()))
# #     ax.set_yticklabels(list(speaker_map.keys()))
# #     ax.set_xlabel("Time (seconds)")
# #     ax.set_title("Time vs Speaker (True Diarization)")
# #     plt.tight_layout()
# #     st.pyplot(fig)

# #     st.success("✅ Done — raw audio, proper diarization segments")

# # else:
# #     st.info("⬆ Upload audio and click **Run Diarization**")


# import os
# import tempfile
# import numpy as np
# import librosa
# import soundfile as sf
# import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# import noisereduce as nr

# from pyannote.audio import Pipeline
# from pyannote.core import Segment, Annotation
# from pyannote.metrics.diarization import DiarizationErrorRate

# # =========================
# # CONFIG
# # =========================
# SR = 16000

# MIN_SPEAKERS = 2
# MAX_SPEAKERS = 5          # 👈 allows 3 speakers
# CLUSTER_THRESHOLD = 0.70  # 👈 less aggressive merge

# MIN_SEGMENT = 0.5
# PAD = 0.15

# MANUAL_RTTM_DIR = r"D:\SHIVANI\INTERNSHIP\CDAC\finalProject\cdacproject\processed\dataset\rttm"

# OUTPUT_DIR = "output"
# SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
# RTTM_DIR = os.path.join(OUTPUT_DIR, "rttm")

# os.makedirs(SPEAKER_DIR, exist_ok=True)
# os.makedirs(RTTM_DIR, exist_ok=True)

# # =========================
# # SPEAKER NAME MAPPING
# # =========================
# def speaker_name(idx):
#     return f"Speaker {chr(ord('A') + idx)}"

# # =========================
# # CLEAN AUDIO (TIME SAFE)
# # =========================
# def clean_audio_preserve_time(audio, sr):
#     cleaned = nr.reduce_noise(y=audio, sr=sr, stationary=True)
#     rms = np.sqrt(np.mean(cleaned ** 2))
#     if rms > 1e-6:
#         target_rms = 10 ** (-20 / 20)
#         cleaned *= target_rms / rms
#     return np.clip(cleaned, -1.0, 1.0)

# # =========================
# # RTTM UTIL
# # =========================
# def read_rttm(path):
#     ann = Annotation()
#     with open(path) as f:
#         for line in f:
#             p = line.strip().split()
#             start = float(p[3])
#             dur = float(p[4])
#             speaker = p[7]
#             ann[Segment(start, start + dur)] = speaker
#     return ann

# # =========================
# # STREAMLIT UI
# # =========================
# st.set_page_config(page_title="Speaker Diarization", layout="wide")
# st.title("🎙 Speaker Diarization (Auto Speaker Count)")

# uploaded_audio = st.file_uploader("Upload WAV audio", type=["wav"])
# run_btn = st.button("🚀 Run Diarization")

# # =========================
# # MAIN PIPELINE
# # =========================
# if run_btn and uploaded_audio:

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(uploaded_audio.read())
#         audio_path = tmp.name

#     file_id = os.path.splitext(uploaded_audio.name)[0]

#     audio, sr = librosa.load(audio_path, sr=SR, mono=True)
#     audio = clean_audio_preserve_time(audio, sr)
#     audio_len = len(audio) / SR

#     clean_path = os.path.join("input", file_id + "_clean.wav")
#     os.makedirs("input", exist_ok=True)
#     sf.write(clean_path, audio, SR)

#     st.subheader("1️⃣ Clean Audio")
#     st.audio(clean_path)

#     # =========================
#     # DIARIZATION
#     # =========================
#     st.subheader("2️⃣ Speaker Diarization")

#     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
#     pipeline.instantiate({
#         "clustering": {
#             "method": "centroid",
#             "threshold": CLUSTER_THRESHOLD
#         }
#     })

#     diarization = pipeline(
#         clean_path,
#         min_speakers=MIN_SPEAKERS,
#         max_speakers=MAX_SPEAKERS
#     )

#     # =========================
#     # SEGMENTS + RTTM
#     # =========================
#     rttm_rows = []
#     rttm_lines = []
#     playback = []

#     speaker_ids = sorted(set(label for _, _, label in diarization.itertracks(yield_label=True)))
#     speaker_map = {s: speaker_name(i) for i, s in enumerate(speaker_ids)}

#     for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
#         dur = turn.end - turn.start
#         if dur < MIN_SEGMENT:
#             continue

#         rttm_lines.append(
#             f"SPEAKER {file_id} 1 {turn.start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
#         )

#         rttm_rows.append({
#             "Start (s)": round(turn.start, 2),
#             "End (s)": round(turn.end, 2),
#             "Duration (s)": round(dur, 2),
#             "Speaker": speaker_map[speaker]
#         })

#         ps = max(0, turn.start - PAD)
#         pe = min(audio_len, turn.end + PAD)
#         seg = audio[int(ps * SR):int(pe * SR)]

#         path = os.path.join(SPEAKER_DIR, f"{file_id}_{speaker}_{idx}.wav")
#         sf.write(path, seg, SR)
#         playback.append((speaker_map[speaker], turn.start, turn.end, path))

#     hyp_rttm = os.path.join(RTTM_DIR, file_id + ".rttm")
#     with open(hyp_rttm, "w") as f:
#         f.writelines(rttm_lines)

#     # =========================
#     # RTTM TABLE
#     # =========================
#     st.subheader("3️⃣ Hypothesis RTTM")
#     st.dataframe(pd.DataFrame(rttm_rows), use_container_width=True)

#     # =========================
#     # PLAYBACK
#     # =========================
#     st.subheader("4️⃣ Speaker Segments")
#     for spk, s, e, p in playback:
#         st.markdown(f"**{spk}** | ⏱ {s:.2f}s → {e:.2f}s")
#         st.audio(p)

#     # =========================
#     # TIMELINE
#     # =========================
#     st.subheader("5️⃣ Speaker Timeline")

#     speakers = sorted(set(row["Speaker"] for row in rttm_rows))
#     ymap = {s: i for i, s in enumerate(speakers)}

#     fig, ax = plt.subplots(figsize=(12, 3))
#     for row in rttm_rows:
#         ax.barh(ymap[row["Speaker"]],
#                 row["Duration (s)"],
#                 left=row["Start (s)"],
#                 height=0.45)

#     ax.set_yticks(list(ymap.values()))
#     ax.set_yticklabels(list(ymap.keys()))
#     ax.set_xlabel("Time (seconds)")
#     ax.set_title("Time vs Speaker")
#     plt.tight_layout()
#     st.pyplot(fig)

#     st.success(f"✅ Done — detected {len(speakers)} speakers")

# else:
#     st.info("⬆ Upload audio and click **Run Diarization**")






import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import noisereduce as nr

from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

# =========================
# CONFIG
# =========================
SR = 16000

MIN_SPEAKERS = 2
MAX_SPEAKERS = 4
CLUSTER_THRESHOLD = 0.70

MIN_SEGMENT = 0.5
PAD = 0.15

INPUT_DIR = "input"
OUTPUT_DIR = "output"
SPEAKER_DIR = os.path.join(OUTPUT_DIR, "speakers")
RTTM_DIR = os.path.join(OUTPUT_DIR, "rttm")
MANUAL_RTTM_DIR = "manual_rttm"   # optional reference RTTM

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(SPEAKER_DIR, exist_ok=True)
os.makedirs(RTTM_DIR, exist_ok=True)
os.makedirs(MANUAL_RTTM_DIR, exist_ok=True)

# =========================
# SESSION STATE (CRITICAL)
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "file_id" not in st.session_state:
    st.session_state.file_id = None

# =========================
# UTILS
# =========================
def clean_audio_preserve_time(audio, sr):
    audio = nr.reduce_noise(y=audio, sr=sr, stationary=True)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-6:
        audio *= (10 ** (-20 / 20)) / rms
    return np.clip(audio, -1.0, 1.0)

def speaker_name(i):
    return f"Speaker {chr(ord('A') + i)}"

def read_rttm(path):
    ann = Annotation()
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            start = float(p[3])
            dur = float(p[4])
            speaker = p[7]
            ann[Segment(start, start + dur)] = speaker
    return ann

# =========================
# UI
# =========================
st.set_page_config(page_title="Speaker Diarization", layout="wide")
st.title("🎙 Speaker Diarization (Mic + Upload + DER)")

tab1, tab2 = st.tabs(["🎤 Record Mic", "📂 Upload WAV"])

# =========================
# MIC RECORDING
# =========================
with tab1:
    duration = st.slider("Recording duration (seconds)", 5, 60, 10)

    if st.button("🎤 Start Recording"):
        st.info("Recording...")
        audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        audio = clean_audio_preserve_time(audio, SR)

        st.session_state.file_id = "mic_recording"
        st.session_state.audio_path = os.path.join(INPUT_DIR, "mic_recording.wav")

        sf.write(st.session_state.audio_path, audio, SR)

        st.success("Recording completed")
        st.audio(st.session_state.audio_path)

# =========================
# UPLOAD AUDIO
# =========================
with tab2:
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            raw_path = tmp.name

        audio, _ = librosa.load(raw_path, sr=SR, mono=True)
        audio = clean_audio_preserve_time(audio, SR)

        st.session_state.file_id = os.path.splitext(uploaded.name)[0]
        st.session_state.audio_path = os.path.join(
            INPUT_DIR, st.session_state.file_id + ".wav"
        )

        sf.write(st.session_state.audio_path, audio, SR)

        st.success("File uploaded")
        st.audio(st.session_state.audio_path)

# =========================
# RUN DIARIZATION
# =========================
if st.session_state.audio_path and st.button("🚀 Run Diarization"):

    audio_path = st.session_state.audio_path
    file_id = st.session_state.file_id

    # Load audio
    audio, _ = librosa.load(audio_path, sr=SR)
    audio_len = len(audio) / SR

    # Load model
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    pipeline.instantiate({
        "clustering": {
            "method": "centroid",
            "threshold": CLUSTER_THRESHOLD
        }
    })

    diarization = pipeline(
        audio_path,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS
    )

    # =========================
    # SEGMENTS
    # =========================
    rows, rttm_lines, playback = [], [], []

    labels = sorted(set(lbl for _, _, lbl in diarization.itertracks(yield_label=True)))
    label_map = {l: speaker_name(i) for i, l in enumerate(labels)}

    for i, (turn, _, spk) in enumerate(diarization.itertracks(yield_label=True)):
        dur = turn.end - turn.start
        if dur < MIN_SEGMENT:
            continue

        rttm_lines.append(
            f"SPEAKER {file_id} 1 {turn.start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
        )

        rows.append({
            "Start (s)": round(turn.start, 2),
            "End (s)": round(turn.end, 2),
            "Duration (s)": round(dur, 2),
            "Speaker": label_map[spk]
        })

        ps = max(0, turn.start - PAD)
        pe = min(audio_len, turn.end + PAD)
        seg = audio[int(ps * SR):int(pe * SR)]

        seg_path = os.path.join(SPEAKER_DIR, f"{file_id}_{spk}_{i}.wav")
        sf.write(seg_path, seg, SR)

        playback.append((label_map[spk], turn.start, turn.end, seg_path))

    # Save RTTM
    hyp_rttm = os.path.join(RTTM_DIR, file_id + ".rttm")
    with open(hyp_rttm, "w") as f:
        f.writelines(rttm_lines)

    # =========================
    # OUTPUTS
    # =========================
    st.subheader("📑 Hypothesis RTTM")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("🔊 Speaker Segments")
    for spk, s, e, p in playback:
        st.markdown(f"**{spk}** | {s:.2f}s → {e:.2f}s")
        st.audio(p)

    # =========================
    # TIMELINE
    # =========================
    st.subheader("📊 Speaker Timeline")

    speakers = sorted(set(r["Speaker"] for r in rows))
    ymap = {s: i for i, s in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, 3))
    for r in rows:
        ax.barh(
            ymap[r["Speaker"]],
            r["Duration (s)"],
            left=r["Start (s)"],
            height=0.45
        )

    ax.set_yticks(list(ymap.values()))
    ax.set_yticklabels(list(ymap.keys()))
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Time vs Speaker")
    plt.tight_layout()
    st.pyplot(fig)

    # =========================
    # DER
    # =========================
    st.subheader("📉 Diarization Error Rate (DER)")

    ref_path = os.path.join(MANUAL_RTTM_DIR, file_id + ".rttm")
    if os.path.exists(ref_path):
        ref = read_rttm(ref_path)
        hyp = read_rttm(hyp_rttm)
        der = DiarizationErrorRate(collar=0.25)(ref, hyp)
        st.metric("DER (%)", f"{der * 100:.2f}")
    else:
        st.warning("Manual RTTM not found → DER skipped")

    st.success(f"✅ Done — detected {len(speakers)} speakers")

else:
    st.info("⬆ Record or upload audio, then click **Run Diarization**")
