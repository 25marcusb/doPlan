"""GPS heading + compass overlay for the map view.

Ported from scripts/userLabeler.py (compute_heading_by_frame / draw_compass) so the
map image the model sees carries the same rotating N/E/S/W compass a human labeler
sees. Row order in gps.csv (after sort/dedupe by frame) is the map video's frame
index, 1:1 with the camera .mp4 frame counts.
"""

import csv
import math
import os
from typing import List, Optional

import cv2
import numpy as np

COMPASS_HEADING_WINDOW = 9
COMPASS_SMOOTHING_ALPHA = 0.1
COMPASS_MIN_MOTION_M = 0.2


def compute_heading_by_frame(csv_path: str) -> Optional[List[float]]:
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(row["frame"]), float(row["lat"]), float(row["lon"])))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda r: r[0])
    seen, deduped = set(), []
    for frame, lat, lon in rows:
        if frame in seen:
            continue
        seen.add(frame)
        deduped.append((frame, lat, lon))
    if len(deduped) < 2:
        return None

    lats = [r[1] for r in deduped]
    lons = [r[2] for r in deduped]
    lat0, lon0 = sum(lats) / len(lats), sum(lons) / len(lons)
    earth_radius_m = 6371000.0
    lat0_rad = math.radians(lat0)
    xs = [math.radians(lon - lon0) * math.cos(lat0_rad) * earth_radius_m for lon in lons]
    ys = [math.radians(lat - lat0) * earth_radius_m for lat in lats]

    n = len(xs)
    k = max(1, COMPASS_HEADING_WINDOW)
    alpha = max(0.01, min(1.0, COMPASS_SMOOTHING_ALPHA))
    ux_s, uy_s, heading_deg = [0.0] * n, [0.0] * n, [0.0] * n

    for i in range(n):
        p, q = max(0, i - k), min(n - 1, i + k)
        span = max(1, q - p)
        dx, dy = (xs[q] - xs[p]) / span, (ys[q] - ys[p]) / span
        speed = math.hypot(dx, dy)
        raw = math.atan2(dy, dx)
        ux, uy = math.cos(raw), math.sin(raw)
        if i == 0:
            ux_s[0], uy_s[0] = ux, uy
        else:
            if speed >= COMPASS_MIN_MOTION_M:
                tux, tuy = ux, uy
            else:
                tux, tuy = ux_s[i - 1], uy_s[i - 1]
            ux_s[i] = (1.0 - alpha) * ux_s[i - 1] + alpha * tux
            uy_s[i] = (1.0 - alpha) * uy_s[i - 1] + alpha * tuy
            norm = math.hypot(ux_s[i], uy_s[i])
            if norm > 1e-8:
                ux_s[i] /= norm
                uy_s[i] /= norm
            else:
                ux_s[i], uy_s[i] = ux_s[i - 1], uy_s[i - 1]
        heading_deg[i] = math.degrees(math.atan2(uy_s[i], ux_s[i]))
    return heading_deg


def _compass_point(cx, cy, r, bearing_deg):
    a = math.radians(bearing_deg)
    return (cx + r * math.sin(a), cy - r * math.cos(a))


def draw_compass(frame, north_bearing_deg):
    """Four-point compass rose rotated so N tracks map-north. Mutates/returns frame."""
    h, w = frame.shape[:2]
    radius = max(8, min(w, h) // 16)
    margin = max(12, radius)
    cx, cy = w - margin - radius, margin + radius
    inner = radius * 0.34
    b = north_bearing_deg

    dark, light = (60, 60, 60), (245, 245, 245)
    north_dark, north_light = (200, 30, 30), (255, 170, 170)
    outline = (30, 30, 30)

    cv2.circle(frame, (int(cx), int(cy)), int(radius * 1.28), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(cx), int(cy)), int(radius * 1.28), (210, 210, 210), 1, cv2.LINE_AA)

    center = (cx, cy)
    for offset, c_dark, c_light in [(0, north_dark, north_light), (90, dark, light),
                                    (180, dark, light), (270, dark, light)]:
        tip = _compass_point(cx, cy, radius, b + offset)
        left_inner = _compass_point(cx, cy, inner, b + offset - 45)
        right_inner = _compass_point(cx, cy, inner, b + offset + 45)
        tri_cw = np.array([center, tip, right_inner], dtype=np.int32)
        tri_ccw = np.array([center, tip, left_inner], dtype=np.int32)
        cv2.fillConvexPoly(frame, tri_cw, c_dark, cv2.LINE_AA)
        cv2.fillConvexPoly(frame, tri_ccw, c_light, cv2.LINE_AA)
        cv2.polylines(frame, [tri_cw], True, outline, 1, cv2.LINE_AA)
        cv2.polylines(frame, [tri_ccw], True, outline, 1, cv2.LINE_AA)

    for offset, letter in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        lx, ly = _compass_point(cx, cy, radius * 1.18, b + offset)
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        color = (200, 30, 30) if letter == "N" else (30, 30, 30)
        cv2.putText(frame, letter, (int(lx - tw / 2), int(ly + th / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame
