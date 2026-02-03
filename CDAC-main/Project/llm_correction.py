import json
from typing import List, Dict
from openai import OpenAI

# OpenAI client (uses OPENAI_API_KEY from env)
client = OpenAI()


def build_prompt(segments: List[Dict]) -> str:
    """
    Build a STRICT prompt that forces JSON-only output.
    """
    rules = """
You are correcting speaker diarization output.

Rules:
- Do NOT invent new speakers.
- Do NOT remove segments.
- ONLY change a segment's speaker if:
  * duration < 1.0 seconds
  * previous and next speakers are the SAME
- Keep all other segments unchanged.
- Preserve order.

Return ONLY a JSON array of speaker labels.
No text. No explanation.
"""

    data = []
    for i, s in enumerate(segments):
        data.append({
            "segment_id": i,
            "speaker": s["speaker"],
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "duration": round(s["duration"], 2),
            "prev_speaker": segments[i - 1]["speaker"] if i > 0 else None,
            "next_speaker": segments[i + 1]["speaker"] if i < len(segments) - 1 else None
        })

    return rules + "\n\nSegments:\n" + json.dumps(data, indent=2)


def llm_correct_segments(segments: List[Dict]) -> List[Dict]:
    """
    Safely apply LLM correction.
    Falls back to original segments if LLM output is invalid.
    """

    # Nothing to correct
    if len(segments) < 3:
        return segments

    prompt = build_prompt(segments)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a diarization correction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    raw_text = response.choices[0].message.content.strip()

    # -------- SAFE JSON EXTRACTION --------
    try:
        # Case 1: perfect JSON
        corrected_labels = json.loads(raw_text)
    except Exception:
        try:
            # Case 2: JSON wrapped in text or markdown
            start = raw_text.index("[")
            end = raw_text.rindex("]") + 1
            corrected_labels = json.loads(raw_text[start:end])
        except Exception:
            print("⚠️ LLM output invalid. Using original diarization.")
            return segments

    # -------- VALIDATION --------
    if not isinstance(corrected_labels, list):
        print("⚠️ LLM output not a list. Ignoring correction.")
        return segments

    if len(corrected_labels) != len(segments):
        print("⚠️ LLM output length mismatch. Ignoring correction.")
        return segments

    # -------- APPLY CORRECTIONS --------
    for i, label in enumerate(corrected_labels):
        segments[i]["speaker"] = label

    return segments
