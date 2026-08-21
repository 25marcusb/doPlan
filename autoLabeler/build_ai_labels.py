"""Produce an AI-labeled sibling of a human labels.csv for a 1-to-1 comparison.

Reads a human labels file (folder,start_frame,end_frame,label,commentary,username),
runs Qwen3-VL on each unique segment, and writes AILabels.csv next to it in the SAME
format and SAME row order, with the model's instruction + referential class and
username="Qwen". Rows whose clip isn't on the drive are emitted empty (kept for
alignment) and reported.

    python build_ai_labels.py                          # cameras only -> AILabels.csv
    python build_ai_labels.py --map                    # cameras + GPS map -> AILabels_map.csv
    python build_ai_labels.py /path/to/labels.csv      # any human labels file
    python build_ai_labels.py in.csv out.csv           # explicit output path
"""

import argparse
import csv
import os

import cv2
import yaml
from PIL import Image

import frames as F
import prompt as P
from model_qwen import QwenVL

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = "/home/cvrr/Desktop/VLM Competition/doPlan/outputs/labels.csv"
FIELDNAMES = ["folder", "start_frame", "end_frame", "label", "commentary", "username"]


def load_config():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=DEFAULT_INPUT)
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument("--map", action="store_true", help="also feed the GPS map + compass view")
    args = ap.parse_args()

    in_csv = args.input
    cfg = load_config()
    fcfg, dcfg, mcfg = cfg["frames"], cfg["data"], cfg["model"]
    include_map = args.map or fcfg.get("include_map", False)
    username = "Qwen"

    default_name = "AILabels_map.csv" if include_map else "AILabels.csv"
    out_csv = args.output or os.path.join(os.path.dirname(in_csv), default_name)
    print(f"[ai] mode: {'cameras + GPS map' if include_map else 'cameras only'}")

    with open(in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"[ai] {in_csv}: {len(rows)} rows")

    model = QwenVL(mcfg["name"], mcfg["dtype"], mcfg["max_new_tokens"], mcfg["do_sample"]).load()

    cache = {}   # (folder, start, end) -> (label, commentary)  |  None if clip missing
    missing = set()

    def predict(folder, start, end):
        key = (folder, start, end)
        if key in cache:
            return cache[key]
        try:
            clip_dir = F.resolve_clip_dir(dcfg["video_root"], dcfg["sample_root"], folder)
        except FileNotFoundError:
            missing.add(folder)
            cache[key] = None
            return None
        seq = F.labeling_sequence(
            clip_dir, start, end, fcfg["cameras"], fcfg["num_timestamps"],
            fcfg["tile_width"], fcfg["tile_height"], fcfg["front_scale"],
            fcfg["draw_camera_labels"], dcfg["fps"],
            include_map=include_map, map_size=fcfg.get("map_size", 512),
        )
        montages = [bgr_to_pil(m) for _fi, m, _mp in seq]
        maps = [bgr_to_pil(mp) if mp is not None else None for _fi, _m, mp in seq]
        parsed = P.parse_response(model.generate(
            P.build_messages(montages, len(montages), map_pils=maps if include_map else None)))
        label = "" if parsed["nothing_to_annotate"] else parsed["instruction"]
        cache[key] = (label, parsed["referential_class"])
        return cache[key]

    out_rows = []
    for i, r in enumerate(rows):
        folder = r["folder"]
        try:
            start, end = int(r["start_frame"]), int(r["end_frame"])
        except (KeyError, ValueError):
            continue
        pred = predict(folder, start, end)
        label, commentary = pred if pred is not None else ("", "")
        out_rows.append({
            "folder": folder, "start_frame": start, "end_frame": end,
            "label": label, "commentary": commentary, "username": username,
        })
        tag = "MISSING-CLIP" if pred is None else f"{commentary or '-'}"
        print(f"  [{i + 1:3d}/{len(rows)}] {folder} [{start},{end}] -> "
              f"{label!r} ({tag})", flush=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)

    labeled = sum(1 for r in out_rows if r["label"].strip())
    print(f"\n[ai] wrote {out_csv}: {len(out_rows)} rows "
          f"({labeled} with an instruction, {len(out_rows) - labeled} empty/nothing).")
    if missing:
        print(f"[ai] {len(missing)} clip(s) not on the drive -> emitted empty rows: "
              f"{sorted(missing)}")


if __name__ == "__main__":
    main()
