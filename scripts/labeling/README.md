# Labeling Setup

This folder contains the labeling app.

## 1. Download the video folders first

Download the video folders from here:

<https://merced-my.sharepoint.com/:f:/g/personal/proy3_ucmerced_edu/IgAP5ZFywlVSRLqBcGUnV-IwAeTyAHjujbfP4QF4Hepqw1c?e=nJQVKk>

Notes:

- There are a little over 1,000 videos total
- The full download is about 80 GB
- If you do not have enough space, download only part of the dataset at a time
- Suggested: download about 2 top-level folders at a time if your machine can handle it
- After finishing those, you can delete them and download the next folders

## 2. Where the video folders should be

In `settings.txt`, set `video_dir` to the folder that contains your top-level video-set folders.

Example:

```text
/path/to/videos/
├── las_vegas_1/
│   ├── 2021.06.09.17.37.09_veh-12_00016_00140/
│   │   ├── L0.mp4
│   │   ├── L1.mp4
│   │   ├── L2.mp4
│   │   ├── F0.mp4
│   │   ├── B0.mp4
│   │   ├── R0.mp4
│   │   ├── R1.mp4
│   │   └── R2.mp4
│   └── 2021.06.09.14.03.17_veh-12_04129_04237/
│       ├── L0.mp4
│       ├── L1.mp4
│       ├── L2.mp4
│       ├── F0.mp4
│       ├── B0.mp4
│       ├── R0.mp4
│       ├── R1.mp4
│       └── R2.mp4
├── las_vegas_2/
│   └── 2021.06.10.09.15.00_veh-03_00001_00090/
│       ├── L0.mp4
│       ├── L1.mp4
│       ├── L2.mp4
│       ├── F0.mp4
│       ├── B0.mp4
│       ├── R0.mp4
│       ├── R1.mp4
│       └── R2.mp4
└── another_video_set/
    └── some_clip_name/
        ├── L0.mp4
        ├── L1.mp4
        ├── L2.mp4
        ├── F0.mp4
        ├── B0.mp4
        ├── R0.mp4
        ├── R1.mp4
        └── R2.mp4
```

Important:

- `video_dir` should point to the folder like `/path/to/videos`
- Inside that folder, each top-level folder is a video set
- Inside each video set, each scene/clip should be its own folder
- The scene/clip folder name becomes the `base_name`
- Each scene/clip folder must contain all 8 camera files:
  `L0.mp4`, `L1.mp4`, `L2.mp4`, `F0.mp4`, `B0.mp4`, `R0.mp4`, `R1.mp4`, `R2.mp4`

This also works if a video set has one extra nested folder with the same name, like:

```text
las_vegas_1/
└── las_vegas_1/
    ├── 2021.06.09.17.37.09_veh-12_00016_00140/
    └── 2021.06.09.14.03.17_veh-12_04129_04237/
```

## 3. Create a Python environment

### Option A: `venv`

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r labeling/requirements.txt
```

### Option B: `conda`

```bash
conda create -n mi3-labeling python=3.11
conda activate mi3-labeling
pip install --upgrade pip
pip install -r labeling/requirements.txt
```

## 4. What to change in `settings.txt`

Edit the file at the project root: [`settings.txt`](/home/proy/projects/mi3/doPlan_labeling/settings.txt)

Example:

```text
video_dir=/path/to/videos
min_segment_length=150
output_folder=/path/to/wherever/you_want_the_labels_saved
commentary_options=Non-Referential, Static Referential, Dynamic Referential, Both
```

What each setting means:

- `video_dir`: where your video-set folders live
- `video_dir` can point to the full dataset, or just the batch you currently downloaded
- `min_segment_length`: minimum frames in a sampled segment
- `output_folder`: where labeling CSV files will be saved
- `output_folder` can be anywhere you want
- `commentary_options`: dropdown choices shown in the app

## 5. Run the labeling script

From the project root:

```bash
python labeling/label.py
```

What happens:

- The app opens a labeling window
- It reads `settings.txt`
- It creates scene IDs in `labeling/scene_assignments.txt` if needed
- It saves labels into the `output_folder` you set

## 6. Output structure

The output folder will look like this:

```text
labeling/output/
└── username/
    ├── _history_index.csv
    ├── scene_or_clip_1.csv
    └── scene_or_clip_2.csv
```
