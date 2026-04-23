import os
import cv2
import json
from PIL import Image
from collections import OrderedDict

import torch
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration


# =========================
# CONFIG
# =========================
DATA_DIR = "/home/cvrr/Desktop/VLM Competition/doPlan/scripts/autoLabeler/SampleData/2021.06.03.12.02.06_veh-35_00233_00609"
OUTPUT_FILE = "gemmaLabels.json"

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

# Multimodal Gemma 3 models include 4B / 12B / 27B instruct variants.
# 1B is text-only, so do not use it here.
MODEL_NAME = "google/gemma-3-4b-it"

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

# Optional: resize before sending to model
MAX_IMAGE_WIDTH = 960

# Optional: max output size
MAX_NEW_TOKENS = 300

# Optional: deterministic decoding
DO_SAMPLE = False

# Optional: ask for stricter JSON behavior
STRICT_JSON_INSTRUCTION = True


# =========================
# DEVICE / MODEL LOAD
# =========================
def load_model():
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    print(f"Loading model: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


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
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    safe_idx = max(0, min(frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
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


def resize_for_model(img, max_width=960):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    new_h = int(h * (max_width / w))
    return cv2.resize(img, (max_width, new_h))


def bgr_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# =========================
# PROMPTING
# =========================
def build_prompt(time_offsets):
    offset_labels = [offset_to_label(o) for o in sorted(time_offsets)]
    offset_text = ", ".join(offset_labels)

    prompt = f"""
You are analyzing a 360-degree driving scene over time.

You are given multiple timestamps in chronological order.
The timestamps included are: {offset_text}.

For each timestamp, the camera labels are drawn directly on the image.

Task:
Provide a single short passenger-style instruction that could have led to the actions taken by the ego vehicle across these timestamps, as if you were a passenger in a taxicab.

Important:
- Base your answer on the full temporal sequence, not just one frame.
- Focus on the ego vehicle's likely intended maneuver.
- Keep the instruction concise and natural.

Return ONLY valid JSON in exactly this format:

{{
  "instruction": ""
}}
""".strip()

    if STRICT_JSON_INSTRUCTION:
        prompt += "\nDo not include markdown, explanations, or any extra keys."

    return prompt


# =========================
# GEMMA QUERY
# =========================
def query_gemma_temporal(model, processor, temporal_frames, time_offsets):
    prompt = build_prompt(time_offsets)

    user_content = []
    pil_images = []

    # Put the text instruction first
    user_content.append({"type": "text", "text": prompt})

    # Then interleave timestamp labels and images
    for time_key, camera_frames in temporal_frames.items():
        user_content.append({"type": "text", "text": f"Timestamp {time_key}"})

        for cam in CAMERAS:
            img = camera_frames[cam]
            img = resize_for_model(img, max_width=MAX_IMAGE_WIDTH)
            labeled = label_image(img, f"{time_key} | {cam}")
            pil_img = bgr_to_pil(labeled)

            pil_images.append(pil_img)
            user_content.append({"type": "image", "image": pil_img})

    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move tensors to model device
    if hasattr(model, "device"):
        target_device = model.device
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Some processor outputs are tensors, some are not
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(target_device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
        )

    generated_tokens = generation[0][input_len:]
    output_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    return output_text


# =========================
# JSON PARSING HELPERS
# =========================
def try_parse_json(text):
    """
    Attempts to parse model output as JSON.
    Also tries to recover JSON if extra text appears around it.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {"raw_output": text}


# =========================
# MAIN
# =========================
def main():
    model, processor = load_model()

    time_offsets = build_time_offsets()

    print("Using time offsets:", time_offsets)
    print("Center frame:", CENTER_FRAME_IDX)

    print("Extracting temporal frames...")
    temporal_frames = get_temporal_multiview_frames(
        DATA_DIR,
        CENTER_FRAME_IDX,
        time_offsets
    )

    print("Querying Gemma...")
    result = query_gemma_temporal(model, processor, temporal_frames, time_offsets)

    print("Raw output:")
    print(result)

    parsed = try_parse_json(result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()