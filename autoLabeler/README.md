# autoLabeler — Qwen3-VL auto-labeling for doPlan

Automatically generate driving-scene labels with a vision-language model
(`Qwen/Qwen3-VL-8B-Instruct`) so they can be compared against the human labels
collected via `scripts/userLabeler.py`.

For each segment `[start_frame, end_frame]` of a clip, the pipeline samples a few
timestamps, stitches the 8 camera views at each into one montage (same layout as the
human labeling UI) and adds the GPS map + compass view, then asks the model for the
**taxi-test instruction** + a **referentiality class** as strict JSON — the same two
things a human produces. Output uses the same CSV schema (`folder, start_frame,
end_frame, label, commentary, username`) so results drop straight into the existing
analysis.

## Setup

The repo's `nuplan` conda env already has torch/transformers(≥4.57)/opencv. Only
`PyYAML` may be missing:

```bash
pip install pyyaml            # (transformers accelerate opencv-python pillow pandas already present)
```

## Auto-label an existing set of labels (main use — 1-to-1 comparison)

Point the labeler at **any** human labels file and it re-labels the *same* segments
(same `folder`/`start_frame`/`end_frame`), writing a row-aligned sibling file with
`username=Qwen`. This is how you compare the model against a human's submissions.

```bash
cd autoLabeler

# label every segment in a human labels file:
python build_ai_labels.py /path/to/labels.csv
#   -> writes AILabels_map.csv next to the input (map view is on by default)

# other forms:
python build_ai_labels.py                       # default input: doPlan/outputs/labels.csv
python build_ai_labels.py labels.csv out.csv    # explicit output path
python build_ai_labels.py labels.csv --map      # force the GPS map on (already the default)

# then score the AI labels against the human ones:
python eval_compare.py /path/to/labels.csv AILabels_map.csv
#   -> prints agreement (maneuver match, referential-class acc, instruction similarity,
#      nothing-to-annotate agreement) and writes outputs/side_by_side.csv
```

Notes / requirements:
- The input CSV needs `folder`, `start_frame`, `end_frame` columns (the standard
  doPlan schema; `label`/`commentary`/`username` are ignored for prediction).
- Every referenced clip must exist under `data.video_root` (or `sample_root`).
  Segments whose clip isn't present are emitted as empty rows (kept for alignment)
  and reported at the end.
- Duplicate `(folder,start,end)` segments are cached — the model runs once per unique
  segment. Runtime ≈ 3 s per unique segment on an RTX 3090 Ti.
- Cameras only (no map): set `frames.include_map: false` in `config.yaml`; the default
  output name then becomes `AILabels.csv`.

## Smaller runners

```bash
# One-clip end-to-end sanity check (validates model + prompt on the GPU):
python smoke_test.py
python smoke_test.py --frames-only    # build/inspect montages only, no model
python smoke_test.py --clip boston_1/2021.09.15.11.49.23_veh-28_00081_00237 --segment 100 400

# Built-in validation set (a handful of single-maneuver segments) with timing:
python compare_labels.py
python compare_labels.py folder,start,end ...   # ad-hoc segments (human label looked up in CSV)
```

## Configuration — `config.yaml`

- `model.name` — HF model id (default `Qwen/Qwen3-VL-8B-Instruct`).
- `data.video_root` — the mounted drive with `{city}/{clip}/{cam}.mp4` (production).
- `data.sample_root` — offline `scripts/autoLabeler/SampleData` fallback for dev.
- `data.human_labels_csv` — default human labels file for the comparison runners.
- `frames.num_timestamps` — montages sampled across the segment (default 6).
- `frames.include_map` / `map_size` — feed the GPS map + compass (default on).
- `frames.cameras` / tile sizes / `front_scale` — montage layout.

## Files

| file | role |
|------|------|
| `config.yaml`        | all paths + knobs |
| `frames.py`          | multi-view frame sampling + montage builder (UI-style grid) + map view |
| `compass.py`         | GPS heading + compass overlay (ported from `userLabeler.py`) |
| `prompt.py`          | task prompt, JSON contract, referential-class normalization |
| `model_qwen.py`      | load Qwen3-VL once (bf16) + chat query wrapper |
| `build_ai_labels.py` | label an existing labels.csv row-for-row → `AILabels[_map].csv` |
| `eval_compare.py`    | score AI label set(s) vs human → metrics + `side_by_side.csv` |
| `smoke_test.py`      | one-clip end-to-end runner |
| `compare_labels.py`  | built-in validation set, model-vs-human with timing |
| `outputs/`           | montage PNGs, `AILabels*.csv`, `side_by_side.csv` (gitignored) |

## Data notes

- Clips: `/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/newVideos/{city}/{clip}/`
  — 8 cameras (`F0` front, `B0` rear, `L0-L2`, `R0-R2`) + `gps_map.mp4` + `gps.csv`, 10 fps.
- `frames.py` tolerates both `F0.mp4` (drive) and `<prefix>F0.mp4` (older SampleData).
- The GPS map + compass is included by default so the model sees roughly what human
  annotators see; set `frames.include_map: false` for a cameras-only run.
