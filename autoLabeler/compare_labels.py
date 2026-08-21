"""Run the model on segments that already have a human label and compare.

Loads Qwen3-VL once, labels each segment, and reports model-vs-human agreement
(referential-class match + instruction word-overlap similarity). A focused
validation harness and a stepping stone to the Phase-2 batch labeler / Phase-3 eval.

    python compare_labels.py                     # runs the built-in validation set
    python compare_labels.py folder,start,end ...  # ad-hoc segments (human label looked up in the CSV)
"""

import csv
import os
import re
import sys
import time

import cv2
import yaml
from PIL import Image

import frames as F
import prompt as P
from model_qwen import QwenVL

HERE = os.path.dirname(os.path.abspath(__file__))

# Built-in validation set: clear single-maneuver segments with known human labels,
# spanning Non-Referential / Static / Dynamic classes.
DEFAULT_SEGMENTS = [
    ("2021.08.17.18.54.02_veh-45_00511_00579", 154, 324),   # "Take the next left"
    ("2021.09.15.11.49.23_veh-28_00520_00669", 583, 733),   # "Turn right"
    ("2021.10.01.19.16.42_veh-28_00274_00380", 328, 480),   # "Stop at the stop sign."
    ("2021.09.15.14.50.05_veh-28_02133_02222", 106, 276),   # "Turn and merge into the correct lane."
    ("2021.10.01.19.16.42_veh-28_03215_03296", 197, 368),   # "Follow the truck"
]

_WORD = re.compile(r"[a-z']+")


def jaccard(a: str, b: str) -> float:
    """Word-set Jaccard similarity (same measure build_dashboard uses for overlaps)."""
    sa = set(_WORD.findall(a.lower()))
    sb = set(_WORD.findall(b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_config():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def human_lookup(csv_path):
    """(folder, start, end) -> (label, commentary) from the human labels CSV."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["folder"], int(row["start_frame"]), int(row["end_frame"]))
            except (KeyError, ValueError):
                continue
            table.setdefault(key, (str(row.get("label", "") or ""), str(row.get("commentary", "") or "")))
    return table


def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    cfg = load_config()
    fcfg, dcfg, mcfg = cfg["frames"], cfg["data"], cfg["model"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    segs = DEFAULT_SEGMENTS
    if len(sys.argv) > 1:
        segs = []
        for a in sys.argv[1:]:
            folder, s, e = a.split(",")
            segs.append((folder, int(s), int(e)))

    humans = human_lookup(dcfg["human_labels_csv"])
    model = QwenVL(mcfg["name"], mcfg["dtype"], mcfg["max_new_tokens"], mcfg["do_sample"]).load()

    rows = []
    for folder, start, end in segs:
        clip_dir = F.resolve_clip_dir(dcfg["video_root"], dcfg["sample_root"], folder)
        t0 = time.time()
        seq = F.montage_sequence(
            clip_dir, start, end, fcfg["cameras"], fcfg["num_timestamps"],
            fcfg["tile_width"], fcfg["tile_height"], fcfg["front_scale"],
            fcfg["draw_camera_labels"], dcfg["fps"],
        )
        pils = [bgr_to_pil(m) for _fi, m in seq]
        t1 = time.time()
        parsed = P.parse_response(model.generate(P.build_messages(pils, len(pils))))
        t2 = time.time()
        montage_s, gen_s = t1 - t0, t2 - t1

        h_label, h_class = humans.get((folder, start, end), ("", ""))
        m_label = "" if parsed["nothing_to_annotate"] else parsed["instruction"]
        sim = jaccard(h_label, m_label)
        class_match = (P.normalize_class(h_class) == parsed["referential_class"]
                       and parsed["referential_class"] != "")
        rows.append({
            "folder": folder, "start": start, "end": end,
            "human_label": h_label, "human_class": h_class,
            "model_label": m_label, "model_class": parsed["referential_class"],
            "instr_similarity": round(sim, 3), "class_match": class_match,
            "seg_seconds": round((end - start) / dcfg["fps"], 1),
            "montage_s": round(montage_s, 2), "infer_s": round(gen_s, 2),
        })

        print("\n" + "=" * 78)
        print(f"{folder}  [{start},{end}]  ({(end - start) / dcfg['fps']:.0f}s segment)")
        print(f"  HUMAN : {h_label!r}   ({h_class})")
        print(f"  MODEL : {m_label!r}   ({parsed['referential_class']})")
        print(f"  similarity={sim:.2f}   class_match={class_match}   "
              f"time: montage={montage_s:.1f}s + inference={gen_s:.1f}s")

    out_csv = os.path.join(out_dir, "comparison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    avg_sim = sum(r["instr_similarity"] for r in rows) / n
    class_acc = sum(r["class_match"] for r in rows) / n
    avg_time = sum(r["montage_s"] + r["infer_s"] for r in rows) / n
    print("\n" + "#" * 78)
    print(f"SUMMARY over {n} segments:  avg instruction similarity={avg_sim:.2f}   "
          f"class accuracy={class_acc:.0%}")
    print(f"avg time per segment = {avg_time:.1f}s  "
          f"(fixed {fcfg['num_timestamps']} montages, so ~constant regardless of segment length)")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
