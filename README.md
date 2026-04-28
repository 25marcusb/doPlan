# doPlan - Video Labeling Tool

## Quick Start

1. Download videos
2. Install Python packages
3. Set `video_dir` and `output_folder` in `scripts/settings.txt`
4. Run `python scripts/userLabeler.py`

---

## 1. Download videos

Download all videos from this link (password provided to authorized labelers):

https://merced-my.sharepoint.com/:f:/g/personal/proy3_ucmerced_edu/IgCOlIDU5MPZSpiNESB5XhpOATIlHozOp-UlLcUh_Zf4aDo?e=39Dxhe

## 2. Where videos go

Videos should be organized as:
```
/path/to/videos/
├── las_vegas_1/
│   └── 2021.06.09.17.37.09_veh-12_00016_00140/
│       ├── L0.mp4
│       ├── L1.mp4
│       ├── L2.mp4
│       ├── F0.mp4
│       ├── B0.mp4
│       ├── R0.mp4
│       ├── R1.mp4
│       ├── R2.mp4
│       └── gps_map.mp4
└── las_vegas_2/
    └── ...
```

Each video folder needs all 9 `.mp4` files.

## 3. Install dependencies

Choose one of these routes:

**Route A: Global install (easiest)**

```bash
pip install opencv-python Pillow
```

If that doesn't work, try `pip3` instead or run `pip install --upgrade pip` first.

**Route B: Virtual environment (keeps this project isolated)**

Creates a separate folder for this project's packages so they don't mix with other Python projects on your computer. If something breaks, it only affects this project.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install opencv-python Pillow
```

**Route C: Using conda (if you have it)**

Conda is an alternative package manager (like pip but more powerful). Use this if you already have conda installed. It works similar to venv but conda handles more things automatically.

```bash
conda create -n doplan python=3.11
conda activate doplan
pip install opencv-python Pillow
```

## 4. Configure settings

Copy the example file and edit it:

```bash
cp scripts/settings.txt.example scripts/settings.txt
```

Then edit `scripts/settings.txt` and set:

```bash
video_dir=/path/to/your/videos  # Example: /Users/proy3/Downloads/nuPlan_videos_with_maps
output_folder=outputs/proy3  # Replace 'proy3' with your name
```

That's it. Optional: adjust `fps` (10.0 default) if videos play too fast or slow.

## 5. Run the app

```bash
python scripts/userLabeler.py
```

Two windows will open. For each segment:
1. Enter your **Username**
2. Watch and take notes on what you see
3. Type your label (see **The Taxi Test** below)
4. Pick a **Commentary** type (see **Referentiality Types** below)
5. Click **Save and Next**

### The Taxi Test

Use this heuristic to write your label:

**"If you were being driven by a taxi driver, what instruction would you give to trigger the behavior you see in the video?"**

Only label if the instruction would be something a real passenger would actually say to the driver.

### Referentiality Types

Choose which objects the instruction references:

- **Non-Referential**: Instruction doesn't reference specific objects (e.g., "speed up", "turn left", "merge lanes")
- **Static Referential**: Instruction mentions fixed things like signs or buildings (e.g., "turn at the stop sign", "drive toward the building")
- **Dynamic Referential**: Instruction mentions moving things like other cars (e.g., "follow that vehicle", "pass the car ahead")
- **Both**: Instruction combines static and dynamic references (e.g., "merge toward the lane with the blue car")

## 6. Submit your labels

Submit your `labels.csv` file by these dates:

- **May 3** (Sunday)
- **May 10** (Sunday)
- **May 17** (Sunday)
- **May 24** (Sunday)
- **May 30** (Friday) - Final deadline

Submit here: <https://docs.google.com/forms/d/e/1FAIpQLSddH6G8gL1wS_DZ8cp8TkQc0B8SFDQ8NIpmmLWs--a4Rmca4Q/viewform>

## Issues?

- **"settings.txt not found"**: Run from the doPlan folder
- **"No valid video folders found"**: Check `video_dir` path and that folders have all 9 `.mp4` files
- **"No module named cv2"**: Run `pip install opencv-python Pillow` again
- **Python command not found**: Try `python3` instead of `python`

---

Other scripts in `scripts/` are for generating videos, not needed for labeling.
