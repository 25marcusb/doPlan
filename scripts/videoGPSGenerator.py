import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import numpy as np
import imageio
import cv2
from tqdm import tqdm
import csv
from pyproj import Transformer

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import (
    NuPlanScenario,
    CameraChannel,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioExtractionInfo,
)

# ---------------- SETTINGS ----------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(SCRIPT_DIR, "videoSettings.txt")

if not os.path.exists(SETTINGS_FILE):
    raise FileNotFoundError(f"{SETTINGS_FILE} not found")

# Defaults
NUPLAN_DATA_ROOT = ""
NUPLAN_MAPS_ROOT = ""
OUTPUT_VIDEO_DIR = ""
NAME = ""
MAP_NAME = ""
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"

F0_WIDTH = 1920
F0_HEIGHT = 1080
OTHER_WIDTH = 640
OTHER_HEIGHT = 360

# ---------------- Load settings ----------------

with open(SETTINGS_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "nuplan_data_root":
            NUPLAN_DATA_ROOT = value
        elif key == "maps_root":
            NUPLAN_MAPS_ROOT = value
        elif key == "output_video_dir":
            OUTPUT_VIDEO_DIR = value
        elif key == "name":
            NAME = value
        elif key == "map_name":
            MAP_NAME = value
        elif key == "map_version":
            NUPLAN_MAP_VERSION = value
        elif key == "f0_width":
            F0_WIDTH = int(value)
        elif key == "f0_height":
            F0_HEIGHT = int(value)
        elif key == "other_width":
            OTHER_WIDTH = int(value)
        elif key == "other_height":
            OTHER_HEIGHT = int(value)

required = [NUPLAN_DATA_ROOT, NUPLAN_MAPS_ROOT, OUTPUT_VIDEO_DIR, MAP_NAME]
if not all(required):
    raise ValueError("Missing required fields in videoSettings.txt")

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# ---------------- Dataset paths ----------------

DB_ROOT = NUPLAN_DATA_ROOT

dataset_root = Path(NUPLAN_DATA_ROOT).parents[1]
NUPLAN_SENSOR_ROOT = str(dataset_root / "sensor_blobs")

# ---------------- Determine DB files ----------------

if NAME:
    db_files = [Path(DB_ROOT) / f"{NAME}.db"]
else:
    db_files = sorted(Path(DB_ROOT).glob("*.db"))

if not db_files:
    raise RuntimeError("No .db files found")

print(f"Found {len(db_files)} database files")

# ---------------- Camera configuration ----------------

CAMERAS = {
    "F0": CameraChannel.CAM_F0,
    "B0": CameraChannel.CAM_B0,
    "L0": CameraChannel.CAM_L0,
    "L1": CameraChannel.CAM_L1,
    "L2": CameraChannel.CAM_L2,
    "R0": CameraChannel.CAM_R0,
    "R1": CameraChannel.CAM_R1,
    "R2": CameraChannel.CAM_R2,
}

RESOLUTION = {
    "F0": (F0_WIDTH, F0_HEIGHT),
    "B0": (OTHER_WIDTH, OTHER_HEIGHT),
    "L0": (OTHER_WIDTH, OTHER_HEIGHT),
    "L1": (OTHER_WIDTH, OTHER_HEIGHT),
    "L2": (OTHER_WIDTH, OTHER_HEIGHT),
    "R0": (OTHER_WIDTH, OTHER_HEIGHT),
    "R1": (OTHER_WIDTH, OTHER_HEIGHT),
    "R2": (OTHER_WIDTH, OTHER_HEIGHT),
}

# ---------------- Resize helper ----------------

def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def generate_map_video(csv_path: str) -> None:
    gps_generator_script = os.path.join(SCRIPT_DIR, "GPSgenerator2.py")
    cmd = [
        sys.executable,
        gps_generator_script,
        "--csv",
        csv_path,
        "--make-video",
    ]
    print(f"Running map generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ---------------- Process each DB ----------------

for db_index, db_file in enumerate(db_files, start=1):

    log_name = db_file.stem

    print("\n" + "=" * 70)
    print(f"Processing log {db_index}/{len(db_files)}: {log_name}")
    print("=" * 70)

    # Create per-log output directory:
    # OUTPUT_VIDEO_DIR  / log_name
    log_output_dir = os.path.join(OUTPUT_VIDEO_DIR, log_name)
    os.makedirs(log_output_dir, exist_ok=True)

    # ---------------- Get first LiDAR frame + EPSG ----------------

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT token, timestamp FROM lidar_pc ORDER BY timestamp ASC LIMIT 1"
    )
    row = cursor.fetchone()

    if row is None:
        print("No LiDAR frames found, skipping.")
        conn.close()
        continue

    initial_token = row[0].hex()
    initial_timestamp = row[1]

    cursor.execute("SELECT DISTINCT epsg FROM ego_pose")
    epsg = cursor.fetchone()[0]
    print(f"Using EPSG:{epsg}")

    conn.close()

    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    # ---------------- Build scenario ----------------

    scenario = NuPlanScenario(
        data_root=DB_ROOT,
        log_file_load_path=str(db_file),
        initial_lidar_token=initial_token,
        initial_lidar_timestamp=initial_timestamp,
        scenario_type="full_log",
        map_root=NUPLAN_MAPS_ROOT,
        map_version=NUPLAN_MAP_VERSION,
        map_name=MAP_NAME,
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_name="full_log",
            scenario_duration=999999,
            extraction_offset=0,
            subsample_ratio=1.0,
        ),
        ego_vehicle_parameters=get_pacifica_parameters(),
        sensor_root=NUPLAN_SENSOR_ROOT,
    )

    num_frames = scenario.get_number_of_iterations()
    print(f"Scenario contains {num_frames} LiDAR frames")

    # ---------------- Create video writers ----------------

    fps = 20
    writers = {}

    for name, channel in CAMERAS.items():
        output_path = os.path.join(log_output_dir, f"{name}.mp4")

        writers[name] = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=["-crf", "28", "-preset", "fast", "-g", "10"],
        )

        print(f"Writing video: {output_path}")

    # ---------------- Create GPS CSV ----------------

    csv_path = os.path.join(log_output_dir, "gps.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp_us", "lat", "lon"])

    # ---------------- Write frames ----------------

    frames_written = 0

    for i in tqdm(range(1, num_frames, 2), desc=f"{log_name}"):

        try:
            sensors = scenario.get_sensors_at_iteration(i, list(CAMERAS.values()))
        except Exception as e:
            print(f"Skipping frame {i}: {e}")
            continue

        ego_state = scenario.get_ego_state_at_iteration(i)

        x = ego_state.rear_axle.x
        y = ego_state.rear_axle.y
        timestamp = ego_state.time_point.time_us

        lon, lat = transformer.transform(x, y)

        csv_writer.writerow([i, timestamp, lat, lon])

        for name, channel in CAMERAS.items():
            try:
                img = sensors.images[channel].as_numpy
            except Exception:
                continue

            if img.dtype != "uint8":
                img = img.astype("uint8")

            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            width, height = RESOLUTION[name]
            img = resize_image(img, width, height)

            writers[name].append_data(img)
            frames_written += 1

    # ---------------- Close everything ----------------

    for writer in writers.values():
        writer.close()

    csv_file.close()

    # ---------------- Generate map video from CSV ----------------

    try:
        generate_map_video(csv_path)
    except Exception as e:
        print(f"Map generation failed for {log_name}: {e}")

    print(f"\n✓ Finished processing {log_name}")
    print(f"  Frames written: {frames_written}")
    print(f"  GPS saved to: {csv_path}")
    print("=" * 70)

print("\nAll logs processed successfully.")