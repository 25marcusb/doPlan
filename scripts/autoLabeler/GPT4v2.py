import os
import cv2
import json
import base64
from collections import OrderedDict
from openai import OpenAI

# =========================
# CONFIG
# =========================
DATA_DIR = "/home/cvrr/Desktop/VLM Competition/doPlan/scripts/autoLabeler/SampleData/2021.06.03.12.02.06_veh-35_00233_00609"
OUTPUT_FILE = "labels.json"

PREFIX = "2021.06.03.12.02.06_veh-35_00233_00609"

CAMERAS = [
    "F0",
    "L0",
    "L1",
    "L2",
    "R0",
    "R1",
    "R2",
    "B0",
]

MODEL_NAME = "gpt-4.1"
client = OpenAI()

CENTER_FRAME_IDX = 100

# =========================
# TEMPORAL CONFIG
# =========================
# Choose one of:
#   TEMPORAL_MODE = "manual"
#   TEMPORAL_MODE = "past"
#   TEMPORAL_MODE = "centered"
TEMPORAL_MODE = "past"

# Used only if TEMPORAL_MODE == "manual"
# Example: [-20, -10, 0]
MANUAL_TIME_OFFSETS = [-20, -10, 0]

# Used only if TEMPORAL_MODE == "past" or "centered"
NUM_TIMESTAMPS = 3
FRAME_GAP = 30

# Optional: resize before sending to API
MAX_IMAGE_WIDTH = 960

# Optional: image detail level for API
IMAGE_DETAIL = "low"

# Optional: max output size
MAX_OUTPUT_TOKENS = 500


# =========================
# TEMPORAL OFFSET GENERATION
# =========================
def build_time_offsets():
    """
    Returns a sorted list of relative frame offsets.

    Modes:
      manual   -> uses MANUAL_TIME_OFFSETS exactly
      past     -> examples:
                    NUM_TIMESTAMPS=3, FRAME_GAP=10 -> [-20, -10, 0]
                    NUM_TIMESTAMPS=4, FRAME_GAP=5  -> [-15, -10, -5, 0]
      centered -> examples:
                    NUM_TIMESTAMPS=3, FRAME_GAP=10 -> [-10, 0, 10]
                    NUM_TIMESTAMPS=5, FRAME_GAP=10 -> [-20, -10, 0, 10, 20]
    """
    mode = TEMPORAL_MODE.lower().strip()

    if mode == "manual":
        offsets = sorted(MANUAL_TIME_OFFSETS)

    elif mode == "past":
        if NUM_TIMESTAMPS < 1:
            raise ValueError("NUM_TIMESTAMPS must be at least 1.")
        offsets = [-(NUM_TIMESTAMPS - 1 - i) * FRAME_GAP for i in range(NUM_TIMESTAMPS)]

    elif mode == "centered":
        if NUM_TIMESTAMPS < 1:
            raise ValueError("NUM_TIMESTAMPS must be at least 1.")

        half = NUM_TIMESTAMPS // 2

        if NUM_TIMESTAMPS % 2 == 1:
            # Odd count, includes 0
            offsets = [(i - half) * FRAME_GAP for i in range(NUM_TIMESTAMPS)]
        else:
            # Even count, biased slightly toward the future
            # Example: NUM_TIMESTAMPS=4, FRAME_GAP=10 -> [-20, -10, 0, 10]
            offsets = [(i - half) * FRAME_GAP for i in range(NUM_TIMESTAMPS)]

    else:
        raise ValueError(
            f"Unknown TEMPORAL_MODE '{TEMPORAL_MODE}'. "
            "Use 'manual', 'past', or 'centered'."
        )

    return sorted(offsets)


def offset_to_label(offset):
    if offset == 0:
        return "t"
    if offset > 0:
        return f"t+{offset}"
    return f"t{offset}"


# =========================
# VIDEO PROCESSING
# =========================
def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    safe_idx = max(0, min(frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {safe_idx} from {video_path}")

    return frame


def get_video_path(data_dir, cam):
    return os.path.join(data_dir, f"{PREFIX}{cam}.mp4")


def get_multiview_frames(data_dir, frame_idx):
    frames = {}
    for cam in CAMERAS:
        path = get_video_path(data_dir, cam)
        frames[cam] = extract_frame(path, frame_idx)
    return frames


def get_temporal_multiview_frames(data_dir, center_frame_idx, time_offsets):
    """
    Returns an OrderedDict like:
    {
        "t-20": {"F0": img, "L0": img, ...},
        "t-10": {...},
        "t": {...}
    }
    """
    bundles = OrderedDict()

    for offset in sorted(time_offsets):
        label = offset_to_label(offset)
        frame_idx = center_frame_idx + offset
        bundles[label] = get_multiview_frames(data_dir, frame_idx)

    return bundles


# =========================
# IMAGE PREP
# =========================
def label_image(img, text):
    img = img.copy()
    cv2.putText(
        img,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return img


def resize_for_api(img, max_width=960):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    new_h = int(h * (max_width / w))
    return cv2.resize(img, (max_width, new_h))


def encode_image(img):
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("Failed to encode image as JPEG")
    return base64.b64encode(buffer).decode("utf-8")


# =========================
# GPT QUERY
# =========================
def build_prompt(time_offsets):
    offset_labels = [offset_to_label(o) for o in sorted(time_offsets)]
    offset_text = ", ".join(offset_labels)

    return f"""
You are analyzing a 360-degree driving scene over time.

You are given multiple timestamps in chronological order.
The timestamps included are: {offset_text}.

For each timestamp, the camera labels are shown directly on the image.

Task: provide an instruction which could have led to the actions taken by the ego vehicle across these timestamps, as if you were a passanger in a taxicab.

Return ONLY valid JSON in this format:

{{
  "instruction": ""
}}
""".strip()


def query_gpt_temporal(temporal_frames, time_offsets):
    prompt = build_prompt(time_offsets)
    content = [{"type": "input_text", "text": prompt}]

    for time_key, camera_frames in temporal_frames.items():
        content.append({
            "type": "input_text",
            "text": f"Timestamp {time_key}"
        })

        for cam in CAMERAS:
            img = camera_frames[cam]
            img = resize_for_api(img, max_width=MAX_IMAGE_WIDTH)
            labeled = label_image(img, f"{time_key} | {cam}")
            encoded = encode_image(labeled)

            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
                "detail": IMAGE_DETAIL
            })

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{
            "role": "user",
            "content": content
        }],
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    return response.output_text


# =========================
# MAIN
# =========================
def main():
    time_offsets = build_time_offsets()

    print("Using time offsets:", time_offsets)
    print("Center frame:", CENTER_FRAME_IDX)

    print("Extracting temporal frames...")
    temporal_frames = get_temporal_multiview_frames(
        DATA_DIR,
        CENTER_FRAME_IDX,
        time_offsets
    )

    print("Querying GPT...")
    result = query_gpt_temporal(temporal_frames, time_offsets)

    print("Raw output:")
    print(result)

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        print("Warning: Output is not valid JSON. Saving raw text.")
        parsed = {"raw_output": result}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()