# Useful imports
import os
from pathlib import Path
import tempfile
import hydra
import numpy as np
import matplotlib.pyplot as plt 


from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo



NUPLAN_DATA_ROOT = "/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
NUPLAN_MAPS_ROOT = "/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/maps"
NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs"
NAME = "2021.05.12.22.00.38_veh-35_01008_01518" #change name to target file


TEST_DB_FILE = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/{NAME}.db"
MAP_NAME = "us-nv-las-vegas"
TEST_INITIAL_LIDAR_PC = "e1e4ee25d1ff58f2"
TEST_INITIAL_TIMESTAMP = 1620857889651124

import sqlite3

# Path to your DB
TEST_DB_FILE = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/{NAME}.db"

# Automatically get the first LiDAR token and timestamp from the DB
conn = sqlite3.connect(TEST_DB_FILE)
cursor = conn.cursor()

# Get first token
cursor.execute("SELECT token, timestamp FROM lidar_pc ORDER BY timestamp ASC LIMIT 1")
row = cursor.fetchone()
if row is None:
    raise ValueError(f"No LiDAR frames found in DB {TEST_DB_FILE}")

TEST_INITIAL_LIDAR_PC = row[0].hex()  # convert from bytes to hex string
TEST_INITIAL_TIMESTAMP = row[1]       # timestamp in microseconds

print(f"Using initial token: {TEST_INITIAL_LIDAR_PC}, timestamp: {TEST_INITIAL_TIMESTAMP}")

conn.close()


scenario = NuPlanScenario(
            data_root=f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini",
            log_file_load_path=TEST_DB_FILE,
            initial_lidar_token=TEST_INITIAL_LIDAR_PC,
            initial_lidar_timestamp=TEST_INITIAL_TIMESTAMP,
            scenario_type="scenario_type",
            map_root=NUPLAN_MAPS_ROOT,
            map_version=NUPLAN_MAP_VERSION,
            map_name=MAP_NAME,
            scenario_extraction_info=ScenarioExtractionInfo(
                scenario_name="scenario_name", scenario_duration=509, extraction_offset=1, subsample_ratio=0.5
            ),
            ego_vehicle_parameters=get_pacifica_parameters(),
            sensor_root=NUPLAN_DATA_ROOT+"/nuplan-v1.1/sensor_blobs",
)

# Total number of iterations (frames) in this scenario
num_frames = scenario.get_number_of_iterations()
print(f"Scenario contains {num_frames} LiDAR frames")

# First and last timestamps (microseconds)
start_ts = scenario.get_time_point(0).time_us
end_ts   = scenario.get_time_point(num_frames - 1).time_us

# Duration in seconds
max_duration_sec = (end_ts - start_ts) * 1e-6
print(f"Maximum legal scenario duration from initial frame: {max_duration_sec:.2f} seconds")
TEST_INITIAL_TIMESTAMP = start_ts

scenario = NuPlanScenario(
            data_root=f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini",
            log_file_load_path=TEST_DB_FILE,
            initial_lidar_token=TEST_INITIAL_LIDAR_PC,
            initial_lidar_timestamp=TEST_INITIAL_TIMESTAMP,
            scenario_type="scenario_type",
            map_root=NUPLAN_MAPS_ROOT,
            map_version=NUPLAN_MAP_VERSION,
            map_name=MAP_NAME,
            scenario_extraction_info=ScenarioExtractionInfo(
                scenario_name="scenario_name", scenario_duration=max_duration_sec - 1, extraction_offset=1, subsample_ratio=0.5
            ),
            ego_vehicle_parameters=get_pacifica_parameters(),
            sensor_root=NUPLAN_DATA_ROOT+"/nuplan-v1.1/sensor_blobs",
)

import sqlite3

# Path to your DB
db_path = "/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"

# Connect
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Count total frames
cursor.execute("SELECT COUNT(*) FROM lidar_pc")
num_frames = cursor.fetchone()[0]
print(f"Total LiDAR frames in log: {num_frames}")

# Get first and last timestamps (microseconds)
cursor.execute("SELECT timestamp FROM lidar_pc ORDER BY timestamp ASC LIMIT 1")
start_ts = cursor.fetchone()[0]

cursor.execute("SELECT timestamp FROM lidar_pc ORDER BY timestamp DESC LIMIT 1")
end_ts = cursor.fetchone()[0]

print(f"First timestamp: {start_ts}, Last timestamp: {end_ts}")
print(f"Full log duration: {(end_ts - start_ts) * 1e-6:.2f} seconds")

import sqlite3

DB_PATH = f"/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/nuplan-v1.1/splits/mini/{NAME}.db"

def token_to_timestamp(token: str) -> int:
    """Convert a LiDAR token string to its timestamp (microseconds)."""
    # Convert token string to bytes for DB comparison
    token_bytes = bytes.fromhex(token)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM lidar_pc WHERE token=?", (token_bytes,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Token {token} not found in DB")
    return row[0]

def timestamp_to_token(timestamp: int) -> str:
    """Find the LiDAR token corresponding to a timestamp.
    Returns the first token with timestamp >= given value, as a hex string.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT token FROM lidar_pc WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
        (timestamp,)
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"No token found at or after timestamp {timestamp}")
    
    # Convert BLOB to hex string
    return row[0].hex()

# Example usage
TEST_TIMESTAMP = start_ts


tk = timestamp_to_token(TEST_TIMESTAMP)
print(f"Timestamp {TEST_TIMESTAMP} corresponds to token {tk}")

import numpy as np
import imageio
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import (
    CameraChannel, LidarChannel
)
from tqdm import tqdm

num_frames = scenario.get_number_of_iterations()
print(f"Number of frames in scenario: {num_frames}")


Perspectives = {"B0", "F0", "L0", "L1", "L2", "R0", "R1", "R2"}

outfolder = "/media/cvrr/0A6AF7D76AF7BE0F/CompetitionData/dataset/videos/"

output_path = outfolder + f"{NAME}B0.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_B0, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_B0].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")

output_path = outfolder + f"{NAME}F0.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_F0, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_F0].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")

output_path = outfolder + f"{NAME}L0.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_L0, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_L0].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")

output_path = outfolder + f"{NAME}L1.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_L1, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_L1].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")

output_path = outfolder + f"{NAME}L2.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_L2, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_L2].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")

output_path = outfolder + f"{NAME}R0.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_R0, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_R0].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")

output_path = outfolder + f"{NAME}R1.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_R1, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_R1].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")

output_path = outfolder + f"{NAME}R2.mp4"
fps = 20  # LiDAR is ~20 Hz

with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
    for i in tqdm(range(1, num_frames, 2), desc="Writing frames"):
        try:
            sensors = scenario.get_sensors_at_iteration(
                i,
                [CameraChannel.CAM_R2, LidarChannel.MERGED_PC],
            )
        except Exception as e:
            print(f"Skipping frame {i} due to missing sensor data: {e}")
            continue

        img = sensors.images[CameraChannel.CAM_R2].as_numpy

        # Ensure uint8
        if img.dtype != "uint8":
            img = img.astype("uint8")

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        writer.append_data(img)

print(f"Saved video to: {output_path}")
