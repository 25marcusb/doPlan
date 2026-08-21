"""Task prompt, JSON-output contract, and label normalization for the autoLabeler.

The task mirrors the human labeling instructions in doPlan/README.md: the "taxi
test" instruction plus a referentiality class. The model is asked for strict JSON so
its output slots into the same CSV schema humans produce.
"""

import json
import re
from typing import Dict, List, Optional

# Canonical referentiality classes (build_dashboard.py CANONICAL_COMMENTARY), plus
# "Ambiguous" which appears in real submissions.
CANONICAL_CLASSES = ["Non-Referential", "Static Referential", "Dynamic Referential", "Both"]
ALL_CLASSES = CANONICAL_CLASSES + ["Ambiguous"]

# Loose spellings/shorthands -> canonical, matched on a normalized (lowercased,
# non-alphanumeric-stripped) key.
_CLASS_ALIASES = {
    "nonreferential": "Non-Referential",
    "staticreferential": "Static Referential",
    "dynamicreferential": "Dynamic Referential",
    "both": "Both",
    "ambiguous": "Ambiguous",
    "a": "Non-Referential",
    "b": "Static Referential",
    "c": "Dynamic Referential",
    "d": "Both",
}


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def normalize_class(raw: Optional[str]) -> str:
    """Map a model's class string to a canonical label; '' if unrecognized."""
    if raw is None:
        return ""
    return _CLASS_ALIASES.get(_norm_key(raw), "")


SYSTEM_PROMPT = (
    "You are an expert driving-scene annotator for an autonomous-driving dataset. "
    "You analyze short multi-camera clips of a car ('the ego vehicle') driving and "
    "describe the maneuver a passenger would have requested."
)


def build_user_prompt(num_timestamps: int, include_map: bool = False) -> str:
    """The text instruction shown alongside the montage sequence."""
    map_note = ""
    if include_map:
        map_note = (
            "\n\nImmediately after each montage you are also given a top-down GPS MAP for "
            "that same moment. It shows the ego vehicle's position along its route, with a "
            "compass rose (the RED arrow points North). Use the map to judge the ego's "
            "heading and turns (e.g. whether it curved left/right or continued straight)."
        )
    return f"""You are given {num_timestamps} montage images from ONE driving segment, in
chronological order (oldest first). Each montage shows the ego vehicle's 8 camera
views at that moment, laid out as:

        [L0]  [ F0 ]  [R0]
        [L1]  [ F0 ]  [R1]
        [L2]  [ B0 ]  [R2]

F0 = front, B0 = back/rear, L0-L2 = left side, R0-R2 = right side. The camera name
is printed on each view. Watch how the scene changes across the timestamps to infer
what the ego vehicle did (turn, lane change, stop, follow, merge, continue, etc.).{map_note}

TASK 1 — The Taxi Test:
Write ONE short, natural instruction that a passenger would give a taxi driver to
cause exactly the maneuver you observe (e.g. "turn left at the intersection",
"follow the white truck", "pull over past the crosswalk"). Only write an
instruction a real passenger would plausibly say. If the segment shows nothing
worth a passenger instruction (e.g. just sitting still, or ambiguous drifting),
set "nothing_to_annotate" to true and leave the instruction empty.

TASK 2 — Referentiality class. Choose exactly one:
- "Non-Referential": no specific objects referenced (e.g. "speed up", "turn left").
- "Static Referential": references fixed things — signs, buildings, lanes
  (e.g. "turn at the stop sign").
- "Dynamic Referential": references moving things — other vehicles, pedestrians
  (e.g. "follow that car").
- "Both": combines static and dynamic references
  (e.g. "merge toward the lane with the blue car").
- "Ambiguous": the correct instruction is genuinely unclear.

Return ONLY valid JSON, no markdown, no extra keys, in exactly this format:
{{"nothing_to_annotate": false, "instruction": "", "referential_class": ""}}"""


def build_messages(montage_pils: List, num_timestamps: int, map_pils: Optional[List] = None) -> List[Dict]:
    """Chat messages: system + (user text followed by the montage images in order).
    If map_pils is given (one per timestamp, may contain None), each montage is
    followed by its GPS map image."""
    include_map = bool(map_pils) and any(m is not None for m in map_pils)
    content = [{"type": "text", "text": build_user_prompt(num_timestamps, include_map)}]
    for i, img in enumerate(montage_pils):
        content.append({"type": "text", "text": f"Montage {i + 1} of {len(montage_pils)}:"})
        content.append({"type": "image", "image": img})
        if map_pils and i < len(map_pils) and map_pils[i] is not None:
            content.append({"type": "text", "text": f"GPS map for montage {i + 1}:"})
            content.append({"type": "image", "image": map_pils[i]})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def parse_response(text: str) -> Dict:
    """Parse model output into a normalized record.

    Returns keys: instruction (str), referential_class (canonical str or ''),
    nothing_to_annotate (bool), parse_ok (bool), raw (original text).
    """
    obj = _extract_json(text)
    if obj is None:
        return {"instruction": "", "referential_class": "", "nothing_to_annotate": False,
                "parse_ok": False, "raw": text}

    nothing = bool(obj.get("nothing_to_annotate", False))
    instruction = str(obj.get("instruction", "") or "").strip()
    if nothing:
        instruction = ""
    return {
        "instruction": instruction,
        "referential_class": normalize_class(obj.get("referential_class")),
        "nothing_to_annotate": nothing,
        "parse_ok": True,
        "raw": text,
    }


def _extract_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None
