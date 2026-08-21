# autoLabeler — Qwen3-VL auto-labeling for doPlan

Automatically generate driving-scene labels with a vision-language model
(`Qwen/Qwen3-VL-8B-Instruct`) so they can be compared against the human labels
collected via `scripts/userLabeler.py`.

For each segment `[start_frame, end_frame]` of a clip, the pipeline samples a few
timestamps, stitches the 8 camera views at each into one montage (same layout as the
human labeling UI), and asks the model for the **taxi-test instruction** + a
**referentiality class** as strict JSON — the same two things a human produces. Output
uses the same CSV schema (`folder, start_frame, end_frame, label, commentary,
username`) so results drop straight into the existing analysis.

## Status / roadmap

- **Phase 1 (built): smoke test** — one clip end-to-end, to validate the model and
  prompt on this GPU. See `smoke_test.py`.
- **Phase 2 (planned): batch labeler** — run over the human-labeled segments that are
  present on the drive (~818 segments) → `outputs/predictions.csv`.
- **Phase 3 (planned): evaluation** — join predictions with human labels; report
  referential-class agreement (accuracy/confusion) and instruction similarity.

## Setup

The repo's `nuplan` conda env already has torch/transformers(≥4.57)/opencv. Only
`PyYAML` may be missing:

```bash
pip install pyyaml            # (transformers accelerate opencv-python pillow pandas already present)
```

## Run the smoke test

```bash
cd autoLabeler

# 1) montage pipeline only — fast, no model download:
python smoke_test.py --frames-only
#    inspect the PNGs written to outputs/

# 2) full run — loads Qwen3-VL-8B (~17GB, downloaded once to ~/.cache/huggingface):
python smoke_test.py
#    prints the raw + parsed JSON label and writes outputs/smoke_test_result.json

# options:
python smoke_test.py --clip boston_1/2021.09.15.11.49.23_veh-28_00081_00237 --segment 100 400
```

## Configuration — `config.yaml`

- `model.name` — HF model id (default `Qwen/Qwen3-VL-8B-Instruct`).
- `data.video_root` — the mounted drive with `{city}/{clip}/{cam}.mp4` (production).
- `data.sample_root` — offline `scripts/autoLabeler/SampleData` fallback for dev.
- `frames.num_timestamps` — montages sampled across the segment (default 4).
- `frames.cameras` / tile sizes / `front_scale` — montage layout.

## Files

| file | role |
|------|------|
| `config.yaml`   | all paths + knobs |
| `frames.py`     | multi-view frame sampling + montage builder (UI-style grid) |
| `prompt.py`     | task prompt, JSON contract, referential-class normalization |
| `model_qwen.py` | load Qwen3-VL once (bf16) + chat query wrapper |
| `smoke_test.py` | Phase-1 one-clip end-to-end runner |
| `outputs/`      | montage PNGs, `smoke_test_result.json`, (later) `predictions.csv` |

## Data notes

- Clips: `/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/newVideos/{city}/{clip}/`
  — 8 cameras (`F0` front, `B0` rear, `L0-L2`, `R0-R2`) + `gps_map.mp4`, 10 fps.
- `frames.py` tolerates both `F0.mp4` (drive) and `<prefix>F0.mp4` (older SampleData).
- The GPS/map view is intentionally **not** shown to the model yet (cameras only).
