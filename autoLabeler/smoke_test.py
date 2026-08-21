"""Phase-1 smoke test: run the full pipeline on ONE clip/segment end-to-end.

Validates montage building, model loading, prompting, and JSON parsing on the GPU
before scaling to the full comparison set.

    python smoke_test.py                 # full run (loads Qwen3-VL, downloads on first use)
    python smoke_test.py --frames-only   # build/inspect montages only, no model
    python smoke_test.py --clip boston_1/2021.09.15.11.49.23_veh-28_00081_00237
    python smoke_test.py --segment 100 400
"""

import argparse
import json
import os

import cv2
import yaml
from PIL import Image

import frames as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None):
    with open(path or os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--clip", default=None, help="'{city}/{clip}', bare clip name, or abs path")
    ap.add_argument("--segment", nargs=2, type=int, metavar=("START", "END"), default=None)
    ap.add_argument("--frames-only", action="store_true", help="skip the model; just build montages")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fcfg, dcfg = cfg["frames"], cfg["data"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    clip = args.clip or cfg["smoke_test"].get("clip")
    clip_dir = F.resolve_clip_dir(dcfg["video_root"], dcfg["sample_root"], clip)
    n = F.clip_length(clip_dir)
    print(f"[smoke] clip: {clip_dir}")
    print(f"[smoke] length: {n} frames ({n / dcfg['fps']:.1f}s @ {dcfg['fps']}fps)")

    seg = args.segment or cfg["smoke_test"].get("segment")
    if not seg:
        seg = [int(n * 0.2), int(n * 0.8)]  # middle 60% of the clip
    start, end = int(seg[0]), int(seg[1])
    print(f"[smoke] segment: [{start}, {end}]  ({(end - start) / dcfg['fps']:.1f}s)")

    seq = F.montage_sequence(
        clip_dir, start, end, fcfg["cameras"], fcfg["num_timestamps"],
        fcfg["tile_width"], fcfg["tile_height"], fcfg["front_scale"],
        fcfg["draw_camera_labels"], dcfg["fps"],
    )
    print(f"[smoke] built {len(seq)} montages, size {seq[0][1].shape[1]}x{seq[0][1].shape[0]}")

    if cfg["smoke_test"].get("save_montages", True) or args.frames_only:
        for i, (fi, m) in enumerate(seq):
            p = os.path.join(out_dir, f"smoke_montage_{i}_frame{fi}.png")
            cv2.imwrite(p, m)
        print(f"[smoke] wrote montage PNGs to {out_dir}")

    if args.frames_only:
        print("[smoke] --frames-only: skipping model. Montage pipeline OK.")
        return

    import prompt as P
    from model_qwen import QwenVL

    pils = [bgr_to_pil(m) for _fi, m in seq]
    messages = P.build_messages(pils, len(pils))

    mcfg = cfg["model"]
    model = QwenVL(mcfg["name"], mcfg["dtype"], mcfg["max_new_tokens"], mcfg["do_sample"])
    raw = model.generate(messages)

    print("\n=== RAW MODEL OUTPUT ===\n" + raw)
    parsed = P.parse_response(raw)
    print("\n=== PARSED ===")
    print(json.dumps({k: v for k, v in parsed.items() if k != "raw"}, indent=2))

    result = {
        "clip_dir": clip_dir, "segment": [start, end],
        "num_timestamps": len(pils), **{k: v for k, v in parsed.items()},
    }
    with open(os.path.join(out_dir, "smoke_test_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[smoke] saved -> {os.path.join(out_dir, 'smoke_test_result.json')}")
    if not parsed["parse_ok"]:
        print("[smoke] WARNING: model output was not valid JSON — inspect RAW above.")


if __name__ == "__main__":
    main()
