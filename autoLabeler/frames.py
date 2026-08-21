"""Multi-view frame sampling and montage building for the autoLabeler.

Given a clip folder (8 camera .mp4s) and a segment [start_frame, end_frame], this
samples a handful of timestamps across the segment and, at each, stitches the camera
views into a single montage laid out like the human labeling UI (userLabeler.py):

        L0 | F0  | R0
        L1 | F0  | R1
        L2 | B0  | R2

so the model sees the same 360-degree context a human labeler does, as one image.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import compass

# Cameras that make up the montage grid, in UI layout order.
LEFT_COL = ["L0", "L1", "L2"]
RIGHT_COL = ["R0", "R1", "R2"]


def resolve_clip_dir(video_root: str, sample_root: str, clip: Optional[str]) -> str:
    """Resolve a clip identifier to a folder on disk.

    `clip` may be an absolute path, a "{city}/{clip}" relative path, or a bare clip
    name. Falls back from the drive (video_root) to the offline sample copy.
    """
    if clip and os.path.isabs(clip) and os.path.isdir(clip):
        return clip

    roots = [r for r in (video_root, sample_root) if r and os.path.isdir(r)]
    if not roots:
        raise FileNotFoundError(
            f"Neither video_root ({video_root}) nor sample_root ({sample_root}) exists. "
            "Is the data drive mounted?"
        )

    if clip:
        for root in roots:
            cand = os.path.join(root, clip)
            if os.path.isdir(cand):
                return cand
        # bare clip name: search one level down (city subfolders)
        for root in roots:
            for city in sorted(os.listdir(root)):
                cand = os.path.join(root, city, clip)
                if os.path.isdir(cand):
                    return cand
        raise FileNotFoundError(f"Clip '{clip}' not found under {roots}")

    # no clip requested -> first clip that has an F0 view
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if _find_camera_file(dirpath, "F0", files):
                return dirpath
    raise FileNotFoundError(f"No clip with an F0.mp4 found under {roots}")


def _find_camera_file(clip_dir: str, cam: str, files: Optional[List[str]] = None) -> Optional[str]:
    """Return the path to a camera's video, tolerating both naming conventions:
    clean `F0.mp4` (production) and `{prefix}F0.mp4` (older SampleData)."""
    if files is None:
        try:
            files = os.listdir(clip_dir)
        except OSError:
            return None
    exact = f"{cam}.mp4"
    if exact in files:
        return os.path.join(clip_dir, exact)
    # fall back to a suffix match like "<prefix>F0.mp4" (but not "..._gps.csv" etc.)
    for f in files:
        if f.endswith(f"{cam}.mp4"):
            return os.path.join(clip_dir, f)
    return None


def clip_length(clip_dir: str, ref_cam: str = "F0") -> int:
    """Number of frames in the reference camera (defines the segment index space)."""
    path = _find_camera_file(clip_dir, ref_cam)
    if path is None:
        raise FileNotFoundError(f"No {ref_cam}.mp4 in {clip_dir}")
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def sample_timestamps(start: int, end: int, num: int) -> List[int]:
    """`num` evenly spaced frame indices across [start, end] (inclusive)."""
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    if num <= 1:
        return [(start + end) // 2]
    return [int(round(start + (end - start) * i / (num - 1))) for i in range(num)]


def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Read a single BGR frame; returns None if the video/frame can't be read."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idx = max(0, min(int(frame_idx), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_multiview_frames(clip_dir: str, frame_idx: int, cameras: List[str]) -> Dict[str, Optional[np.ndarray]]:
    """{cam: BGR frame or None} for one timestamp across the requested cameras."""
    out = {}
    for cam in cameras:
        path = _find_camera_file(clip_dir, cam)
        out[cam] = extract_frame(path, frame_idx) if path else None
    return out


def _tile(frame: Optional[np.ndarray], w: int, h: int, label: str, draw_label: bool) -> np.ndarray:
    """Resize a frame to w×h with an optional camera label; black placeholder if missing."""
    if frame is None:
        tile = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(tile, f"{label} (missing)", (8, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return tile
    tile = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    if draw_label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(tile, (4, 4), (8 + tw, 10 + th), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 8 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return tile


def build_montage(
    frames: Dict[str, Optional[np.ndarray]],
    tile_w: int = 320,
    tile_h: int = 180,
    front_scale: int = 2,
    draw_labels: bool = True,
    banner: Optional[str] = None,
) -> np.ndarray:
    """Stitch camera frames into the UI-style grid. Returns one BGR image."""
    fw, fh = tile_w * front_scale, tile_h * front_scale
    col_h = max(tile_h * 3, fh + tile_h)
    canvas_w = tile_w + fw + tile_w
    banner_h = 26 if banner else 0
    canvas = np.zeros((col_h + banner_h, canvas_w, 3), dtype=np.uint8)

    y0 = banner_h
    # left column
    for i, cam in enumerate(LEFT_COL):
        canvas[y0 + i * tile_h:y0 + (i + 1) * tile_h, 0:tile_w] = _tile(
            frames.get(cam), tile_w, tile_h, cam, draw_labels)
    # center column: F0 (large) then B0 below, centered under the front view
    cx = tile_w
    canvas[y0:y0 + fh, cx:cx + fw] = _tile(frames.get("F0"), fw, fh, "F0", draw_labels)
    b_x = cx + (fw - tile_w) // 2
    canvas[y0 + fh:y0 + fh + tile_h, b_x:b_x + tile_w] = _tile(
        frames.get("B0"), tile_w, tile_h, "B0", draw_labels)
    # right column
    rx = tile_w + fw
    for i, cam in enumerate(RIGHT_COL):
        canvas[y0 + i * tile_h:y0 + (i + 1) * tile_h, rx:rx + tile_w] = _tile(
            frames.get(cam), tile_w, tile_h, cam, draw_labels)

    if banner:
        cv2.putText(canvas, banner, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def montage_sequence(
    clip_dir: str,
    start: int,
    end: int,
    cameras: List[str],
    num_timestamps: int,
    tile_w: int,
    tile_h: int,
    front_scale: int,
    draw_labels: bool,
    fps: float = 10.0,
) -> List[Tuple[int, np.ndarray]]:
    """List of (frame_idx, montage BGR) sampled across the segment, oldest first."""
    idxs = sample_timestamps(start, end, num_timestamps)
    seq = []
    for k, fi in enumerate(idxs):
        frames = get_multiview_frames(clip_dir, fi, cameras)
        t_sec = fi / fps if fps else 0.0
        banner = f"timestamp {k + 1}/{len(idxs)}  |  frame {fi}  |  t={t_sec:0.1f}s"
        seq.append((fi, build_montage(frames, tile_w, tile_h, front_scale, draw_labels, banner)))
    return seq


def _find_gps_csv(clip_dir: str) -> Optional[str]:
    try:
        files = os.listdir(clip_dir)
    except OSError:
        return None
    if "gps.csv" in files:
        return os.path.join(clip_dir, "gps.csv")
    for f in files:
        if f.endswith("gps.csv"):
            return os.path.join(clip_dir, f)
    return None


def get_map_frame(clip_dir: str, idx: int, heading_by_frame, size: int = 512,
                  draw_label: bool = True) -> Optional[np.ndarray]:
    """The gps_map frame at `idx` with the rotating compass overlaid (as in the UI)."""
    path = _find_camera_file(clip_dir, "gps_map")
    frame = extract_frame(path, idx) if path else None
    if frame is None:
        return None
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    if heading_by_frame is not None and 0 <= idx < len(heading_by_frame):
        # match userLabeler: draw at heading-90 so up = travel direction, N tracks map-north
        compass.draw_compass(frame, heading_by_frame[idx] - 90.0)
    if draw_label:
        (tw, th), _ = cv2.getTextSize("GPS map (N=red)", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (4, 4), (8 + tw, 10 + th), (0, 0, 0), -1)
        cv2.putText(frame, "GPS map (N=red)", (6, 8 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def labeling_sequence(
    clip_dir: str,
    start: int,
    end: int,
    cameras: List[str],
    num_timestamps: int,
    tile_w: int,
    tile_h: int,
    front_scale: int,
    draw_labels: bool,
    fps: float = 10.0,
    include_map: bool = False,
    map_size: int = 512,
) -> List[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
    """Like montage_sequence, but each entry is (frame_idx, montage, map_or_None).
    map is the compass-overlaid GPS view when include_map is set (else None)."""
    idxs = sample_timestamps(start, end, num_timestamps)
    heading = compass.compute_heading_by_frame(_find_gps_csv(clip_dir) or "") if include_map else None
    seq = []
    for k, fi in enumerate(idxs):
        frames = get_multiview_frames(clip_dir, fi, cameras)
        banner = f"timestamp {k + 1}/{len(idxs)}  |  frame {fi}  |  t={fi / fps:0.1f}s"
        montage = build_montage(frames, tile_w, tile_h, front_scale, draw_labels, banner)
        map_img = get_map_frame(clip_dir, fi, heading, map_size, draw_labels) if include_map else None
        seq.append((fi, montage, map_img))
    return seq
