import os
import cv2
import json
import base64
from openai import OpenAI
from sympy import content

# =========================
# CONFIG
# =========================
DATA_DIR = "/home/cvrr/Desktop/VLM Competition/doPlan/scripts/autoLabeler/SampleData/2021.06.03.12.02.06_veh-35_00233_00609"  # folder containing your 6 videos
FRAME_IDX = 100    # which frame to sample
OUTPUT_FILE = "labels.json"

prefix = "2021.06.03.12.02.06_veh-35_00233_00609"

CAMERAS = [
    "F0",
    "L0",
    "L1",
    "L2",
    "R0",
    "R1",
    "R2",
    "B0"
]

MODEL_NAME = "gpt-4.1"  # vision-capable model

client = OpenAI()  # assumes OPENAI_API_KEY is set


# =========================
# VIDEO PROCESSING
# =========================
def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

    return frame


def get_multiview_frames(data_dir, frame_idx):
    frames = {}
    for cam in CAMERAS:
        path = os.path.join(data_dir, f"{prefix}{cam}.mp4")
        frames[cam] = extract_frame(path, frame_idx)
    return frames


# =========================
# IMAGE PREP
# =========================
def label_image(img, text):
    img = img.copy()
    cv2.putText(
        img,
        text.upper(),
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )
    return img


def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================
# GPT QUERY
# =========================
def query_gpt(frames):
    prompt = """
You are analyzing a 360-degree driving scene from 6 cameras.

Tasks:
1. List key objects (cars, pedestrians, bikes, etc.)
2. Describe their actions
3. Identify potential risks
4. Provide a short summary

Return ONLY valid JSON in this format:

{
  "objects": [
    {"type": "", "position": "", "action": ""}
  ],
  "risks": [],
  "summary": ""
}
"""

    content = [{"type": "input_text", "text": prompt}]

    for cam, img in frames.items():
        labeled = label_image(img, cam)
        encoded = encode_image(labeled)

        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encoded}",
            "detail": "low"
        })

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{
            "role": "user",
            "content": content
        }],
        temperature=0
    )

    return response.output_text


# =========================
# MAIN
# =========================
def main():
    print("Extracting frames...")
    frames = get_multiview_frames(DATA_DIR, FRAME_IDX)

    print("Querying GPT...")
    result = query_gpt(frames)

    print("Raw output:")
    print(result)

    # Try to parse JSON safely
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