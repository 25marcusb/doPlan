from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, box
import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = SCRIPT_DIR / "videoSettings.txt"
REQUIRED_COLUMNS = ["frame", "lat", "lon"]


class TrajectoryValidationError(Exception):
    pass


def load_default_csv_dir() -> Optional[Path]:
    """
    Read output_video_dir from videoSettings.txt, if available.
    """
    if not SETTINGS_FILE.exists():
        return None

    with SETTINGS_FILE.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "output_video_dir":
                value = value.strip()
                if value:
                    return Path(value)
    return None


DEFAULT_CSV_DIR = load_default_csv_dir()


def resolve_csv_path(csv_input: Path, csv_dir: Optional[Path]) -> Path:
    """
    Resolve csv_input, searching csv_dir for relative filenames and optional
    '<log_name>_gps.csv' convention.
    """
    def expand_candidate(candidate: Path) -> list[Path]:
        """
        Expand a candidate path into plausible CSV file paths.
        """
        if candidate.is_dir():
            nested = [
                candidate / f"{candidate.name}_gps.csv",
                candidate / "gps.csv",
            ]
            nested.extend(sorted(candidate.glob("*_gps.csv")))
            return nested

        expanded = [candidate]
        if candidate.suffix.lower() != ".csv":
            expanded.append(candidate.with_name(f"{candidate.name}_gps.csv"))
        return expanded

    if csv_input.is_absolute():
        for candidate in expand_candidate(csv_input):
            if candidate.is_file():
                return candidate
        return csv_input

    candidates = [csv_input]
    if csv_dir is not None:
        candidates.append(csv_dir / csv_input)
    expanded_candidates: list[Path] = []
    for candidate in candidates:
        expanded_candidates.extend(expand_candidate(candidate))

    for candidate in expanded_candidates:
        if candidate.is_file():
            return candidate

    return expanded_candidates[-1] if expanded_candidates else candidates[-1]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical = {col.strip().lower(): col for col in df.columns}

    frame_col = canonical.get("frame") or canonical.get("framenumber")
    lat_col = canonical.get("lat")
    lon_col = canonical.get("lon") or canonical.get("long") or canonical.get("lng")

    missing = []
    if frame_col is None:
        missing.append("frame/frameNumber")
    if lat_col is None:
        missing.append("lat/Lat")
    if lon_col is None:
        missing.append("lon/Lon/Long")

    if missing:
        raise TrajectoryValidationError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    normalized = df[[frame_col, lat_col, lon_col]].copy()
    normalized.columns = REQUIRED_COLUMNS
    return normalized


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    if df.empty:
        raise TrajectoryValidationError("No valid rows remain after dropping invalid values.")

    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)].copy()

    if df.empty:
        raise TrajectoryValidationError("No valid rows remain after lat/long filtering.")

    df = df.sort_values("frame").drop_duplicates(subset=["frame"], keep="first")
    df["frame"] = df["frame"].astype(int)
    df = df.reset_index(drop=True)

    if len(df) < 2:
        raise TrajectoryValidationError("Need at least 2 valid rows.")

    return df


def build_sparse_trajectory_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


def project_trajectory_gdf(gdf_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Project to an appropriate local metric CRS automatically.
    """
    return ox.projection.project_gdf(gdf_wgs84)


def build_padded_square_bbox_polygon_projected(
    gdf_projected: gpd.GeoDataFrame,
    padding_meters: float,
) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = gdf_projected.total_bounds

    xmin -= padding_meters
    ymin -= padding_meters
    xmax += padding_meters
    ymax += padding_meters

    width = xmax - xmin
    height = ymax - ymin

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    side = max(width, height)
    half = side / 2.0

    square = box(
        cx - half,
        cy - half,
        cx + half,
        cy + half,
    )

    return gpd.GeoDataFrame(
        {"name": ["map_extent"]},
        geometry=[square],
        crs=gdf_projected.crs,
    )


def projected_bbox_to_wgs84_tuple(
    bbox_projected_gdf: gpd.GeoDataFrame,
) -> tuple[float, float, float, float]:
    """
    Return bbox tuple in OSMnx/Overpass expected order:
    (left, bottom, right, top) == (min_lon, min_lat, max_lon, max_lat)
    """
    bbox_wgs84 = ox.projection.project_gdf(bbox_projected_gdf, to_latlong=True)
    xmin, ymin, xmax, ymax = bbox_wgs84.total_bounds
    return (xmin, ymin, xmax, ymax)


def fetch_drive_network(bbox_wgs84: tuple[float, float, float, float]):
    """
    Download drivable road network within bbox.
    """
    return ox.graph.graph_from_bbox(
        bbox_wgs84,
        network_type="drive",
        simplify=True,
        retain_all=False,
        truncate_by_edge=False,
    )


def graph_to_projected_gdfs(graph):
    """
    Convert graph to GeoDataFrames and project graph to metric CRS.
    """
    graph_projected = ox.projection.project_graph(graph)
    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(graph_projected)
    return graph_projected, nodes_gdf, edges_gdf


def fetch_osm_features(bbox_wgs84: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """
    Fetch a broader set of useful OSM features, including potentially named landmarks.
    """
    tags = {
        "building": True,
        "landuse": True,
        "leisure": True,
        "amenity": True,
        "tourism": True,
        "historic": True,
        "man_made": True,
        "water": True,
        "natural": True,
    }

    try:
        features = ox.features.features_from_bbox(bbox_wgs84, tags)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if features is None or len(features) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if features.crs is None:
        features = features.set_crs("EPSG:4326")

    return features


def project_features_to_match(
    features_wgs84: gpd.GeoDataFrame,
    target_crs,
) -> gpd.GeoDataFrame:
    if features_wgs84.empty:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    return features_wgs84.to_crs(target_crs)


def empty_gdf(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def subset_polygon_features(features_gdf: gpd.GeoDataFrame, key: str, values=None) -> gpd.GeoDataFrame:
    if features_gdf.empty or key not in features_gdf.columns:
        return empty_gdf(features_gdf.crs)

    subset = features_gdf[features_gdf[key].notna()].copy()

    if values is not None:
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        subset = subset[subset[key].isin(values)].copy()

    subset = subset[subset.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if subset.empty:
        return empty_gdf(features_gdf.crs)

    return gpd.GeoDataFrame(subset, geometry="geometry", crs=features_gdf.crs)


def subset_point_features(features_gdf: gpd.GeoDataFrame, key: str, values=None) -> gpd.GeoDataFrame:
    if features_gdf.empty or key not in features_gdf.columns:
        return empty_gdf(features_gdf.crs)

    subset = features_gdf[features_gdf[key].notna()].copy()

    if values is not None:
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        subset = subset[subset[key].isin(values)].copy()

    subset = subset[subset.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    if subset.empty:
        return empty_gdf(features_gdf.crs)

    return gpd.GeoDataFrame(subset, geometry="geometry", crs=features_gdf.crs)


def build_named_label_features(features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only features worth labeling, and only when they have a name.
    For polygons/lines, use a representative point for label placement.
    """
    if features_gdf.empty or "name" not in features_gdf.columns:
        return gpd.GeoDataFrame({"name": [], "label_geometry": []}, geometry="label_geometry", crs=features_gdf.crs)

    candidates = features_gdf[features_gdf["name"].notna()].copy()
    if candidates.empty:
        return gpd.GeoDataFrame({"name": [], "label_geometry": []}, geometry="label_geometry", crs=features_gdf.crs)

    allowed = pd.Series(False, index=candidates.index)

    def mark_if_col_has(col_name: str, values=None):
        nonlocal allowed
        if col_name not in candidates.columns:
            return
        if values is None:
            allowed = allowed | candidates[col_name].notna()
        else:
            if not isinstance(values, (list, tuple, set)):
                values = [values]
            allowed = allowed | candidates[col_name].isin(values)

    mark_if_col_has("tourism")
    mark_if_col_has("historic")
    mark_if_col_has("amenity", ["school", "hospital", "fuel", "university", "library", "place_of_worship"])
    mark_if_col_has("leisure", ["park", "garden", "stadium", "sports_centre"])
    mark_if_col_has("man_made", ["tower", "bridge", "water_tower", "obelisk"])
    mark_if_col_has("building", ["hotel", "casino", "hospital", "school", "train_station"])
    mark_if_col_has("natural")
    mark_if_col_has("water")

    labels = candidates[allowed].copy()
    if labels.empty:
        return gpd.GeoDataFrame({"name": [], "label_geometry": []}, geometry="label_geometry", crs=features_gdf.crs)

    label_points = []
    for geom in labels.geometry:
        if geom is None or geom.is_empty:
            label_points.append(None)
        elif isinstance(geom, Point):
            label_points.append(geom)
        else:
            label_points.append(geom.representative_point())

    labels["label_geometry"] = label_points
    labels = labels[labels["label_geometry"].notna()].copy()
    if labels.empty:
        return gpd.GeoDataFrame({"name": [], "label_geometry": []}, geometry="label_geometry", crs=features_gdf.crs)

    # Normalize label text and drop duplicate names.
    labels["name"] = labels["name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    labels = labels[labels["name"] != ""].copy()
    if labels.empty:
        return gpd.GeoDataFrame({"name": [], "label_geometry": []}, geometry="label_geometry", crs=features_gdf.crs)

    labels["name_len"] = labels["name"].astype(str).str.len()
    labels["name_key"] = labels["name"].str.casefold()
    labels = labels.sort_values("name_len").drop_duplicates(subset=["name_key"], keep="first")

    keep_cols = [c for c in labels.columns if c != "geometry"]
    labels = labels[keep_cols].copy()

    labels_gdf = gpd.GeoDataFrame(labels, geometry="label_geometry", crs=features_gdf.crs)
    labels_gdf = labels_gdf.sort_values("name_len").reset_index(drop=True)

    return labels_gdf


def filter_labels_by_spacing(
    labels_gdf: gpd.GeoDataFrame,
    min_distance_m: float = 60.0,
    max_labels: int = 40,
) -> gpd.GeoDataFrame:
    """
    Greedy spacing filter so labels do not pile on top of each other.
    Assumes projected CRS in meters.
    """
    if labels_gdf.empty:
        return labels_gdf

    geom_col = labels_gdf.geometry.name
    kept_rows = []
    kept_points = []
    kept_name_keys = set()

    for _, row in labels_gdf.iterrows():
        label_name = str(row.get("name", "")).strip()
        label_name_key = " ".join(label_name.casefold().split())
        if not label_name_key or label_name_key in kept_name_keys:
            continue

        pt = row[geom_col]
        if pt is None or pt.is_empty:
            continue

        if not isinstance(pt, Point):
            pt = pt.representative_point()

        too_close = False
        for other in kept_points:
            if pt.distance(other) < min_distance_m:
                too_close = True
                break

        if not too_close:
            row = row.copy()
            row[geom_col] = pt
            kept_rows.append(row)
            kept_points.append(pt)
            kept_name_keys.add(label_name_key)

        if len(kept_rows) >= max_labels:
            break

    if not kept_rows:
        return gpd.GeoDataFrame({"name": [], "label_geometry": []}, geometry="label_geometry", crs=labels_gdf.crs)

    result = gpd.GeoDataFrame(kept_rows, geometry=geom_col, crs=labels_gdf.crs)
    return result.reset_index(drop=True)

def render_background_png(
    output_png: Path,
    edges_gdf: gpd.GeoDataFrame,
    bbox_projected: gpd.GeoDataFrame,
    trajectory_gdf: Optional[gpd.GeoDataFrame] = None,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    green_areas_gdf: Optional[gpd.GeoDataFrame] = None,
    parking_gdf: Optional[gpd.GeoDataFrame] = None,
    poi_points_gdf: Optional[gpd.GeoDataFrame] = None,
    labels_gdf: Optional[gpd.GeoDataFrame] = None,
    width_px: int = 900,
    height_px: int = 900,
    dpi: int = 100,
    show_trajectory: bool = False,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    xmin, ymin, xmax, ymax = bbox_projected.total_bounds

    if green_areas_gdf is not None and not green_areas_gdf.empty:
        green_areas_gdf.plot(ax=ax, color="#dfeadf", edgecolor="none")

    if parking_gdf is not None and not parking_gdf.empty:
        parking_gdf.plot(ax=ax, color="#ece8df", edgecolor="none")

    if buildings_gdf is not None and not buildings_gdf.empty:
        buildings_gdf.plot(ax=ax, color="#d9d9d9", edgecolor="none")

    if not edges_gdf.empty:
        edges_gdf.plot(ax=ax, linewidth=1.2, color="#444444")

    if show_trajectory and trajectory_gdf is not None and not trajectory_gdf.empty:
        trajectory_gdf.plot(ax=ax, markersize=6, color="#cc3333")

    if poi_points_gdf is not None and not poi_points_gdf.empty:
        poi_points_gdf.plot(ax=ax, markersize=12, color="#2f6db3")

    if labels_gdf is not None and not labels_gdf.empty:
        label_geom_col = labels_gdf.geometry.name
        for _, row in labels_gdf.iterrows():
            pt = row[label_geom_col]
            name = str(row.get("name", "")).strip()

            if pt is None or pt.is_empty or not name:
                continue

            if not isinstance(pt, Point):
                pt = pt.representative_point()

            ax.text(
                pt.x + 8,
                pt.y + 8,
                name,
                fontsize=8,
                color="#1f1f1f",
                ha="left",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
                zorder=10,
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_png, dpi=dpi, pad_inches=0)
    plt.close(fig)

def build_dense_frame_positions(
    sparse_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one video frame per CSV row.
    This keeps map video length aligned with videos generated from the same
    sparse sampling process.
    """
    sparse_df = sparse_df.sort_values("frame").reset_index(drop=True)
    dense = sparse_df[["frame", "lat", "lon"]].copy()
    dense = dense.rename(columns={"frame": "gps_frame"})
    dense.insert(0, "video_frame", dense.index.astype(int))
    return dense

def project_dense_positions_to_match(
    dense_df: pd.DataFrame,
    target_crs,
) -> gpd.GeoDataFrame:
    dense_wgs84 = gpd.GeoDataFrame(
        dense_df.copy(),
        geometry=gpd.points_from_xy(dense_df["lon"], dense_df["lat"]),
        crs="EPSG:4326",
    )
    return dense_wgs84.to_crs(target_crs)


def projected_points_to_pixel_coords(
    dense_projected_gdf: gpd.GeoDataFrame,
    bbox_projected: gpd.GeoDataFrame,
    width_px: int,
    height_px: int,
) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = bbox_projected.total_bounds

    xs = dense_projected_gdf.geometry.x
    ys = dense_projected_gdf.geometry.y

    px = ((xs - xmin) / (xmax - xmin) * (width_px - 1)).round().astype(int)
    py = (height_px - 1 - (ys - ymin) / (ymax - ymin) * (height_px - 1)).round().astype(int)

    px = px.clip(0, width_px - 1)
    py = py.clip(0, height_px - 1)

    return pd.DataFrame({
        "video_frame": dense_projected_gdf["video_frame"].astype(int),
        "gps_frame": dense_projected_gdf["gps_frame"].astype(int),
        "px": px,
        "py": py,
    })


def labels_to_pixel_coords(
    labels_gdf: gpd.GeoDataFrame,
    bbox_projected: gpd.GeoDataFrame,
    width_px: int,
    height_px: int,
) -> pd.DataFrame:
    if labels_gdf is None or labels_gdf.empty:
        return pd.DataFrame(columns=["name", "px", "py"])

    geom_col = labels_gdf.geometry.name
    xmin, ymin, xmax, ymax = bbox_projected.total_bounds

    labels = labels_gdf.copy()
    xs = labels[geom_col].x
    ys = labels[geom_col].y

    px = ((xs - xmin) / (xmax - xmin) * (width_px - 1)).round().astype(int)
    py = (height_px - 1 - (ys - ymin) / (ymax - ymin) * (height_px - 1)).round().astype(int)

    px = px.clip(0, width_px - 1)
    py = py.clip(0, height_px - 1)

    names = labels["name"].astype(str).str.strip()
    out = pd.DataFrame({"name": names, "px": px, "py": py})
    out = out[out["name"] != ""].reset_index(drop=True)
    return out


def attach_heading_degrees(
    frame_pixels_df: pd.DataFrame,
    dense_projected_gdf: gpd.GeoDataFrame,
    min_motion_m: float = 0.2,
    heading_window: int = 9,
    smoothing_alpha: float = 0.1,
) -> pd.DataFrame:
    # Estimate heading in image space so rotation behavior matches rendered pixels.
    # Use a wider temporal window and vector smoothing to reduce jitter.
    pxs = frame_pixels_df["px"].to_numpy(dtype=float)
    pys = frame_pixels_df["py"].to_numpy(dtype=float)

    n = len(pxs)
    out = frame_pixels_df.copy()
    if n == 0:
        out["heading_deg"] = np.array([], dtype=float)
        return out

    k = max(1, int(heading_window))
    alpha = float(np.clip(smoothing_alpha, 0.01, 1.0))

    idx = np.arange(n, dtype=int)
    prev_idx = np.clip(idx - k, 0, n - 1)
    next_idx = np.clip(idx + k, 0, n - 1)
    span = np.maximum(1, next_idx - prev_idx).astype(float)

    dx = (pxs[next_idx] - pxs[prev_idx]) / span
    dy = (pys[next_idx] - pys[prev_idx]) / span
    speed = np.hypot(dx, dy)

    # Convert image-space motion to map heading where east=0, north=90.
    raw_heading_rad = np.arctan2(-dy, dx)
    ux = np.cos(raw_heading_rad)
    uy = np.sin(raw_heading_rad)

    ux_s = np.zeros(n, dtype=float)
    uy_s = np.zeros(n, dtype=float)

    ux_s[0] = ux[0]
    uy_s[0] = uy[0]

    for i in range(1, n):
        if speed[i] >= min_motion_m:
            target_ux, target_uy = ux[i], uy[i]
        else:
            # Hold last stable direction when movement is too small.
            target_ux, target_uy = ux_s[i - 1], uy_s[i - 1]

        ux_s[i] = (1.0 - alpha) * ux_s[i - 1] + alpha * target_ux
        uy_s[i] = (1.0 - alpha) * uy_s[i - 1] + alpha * target_uy

        norm = np.hypot(ux_s[i], uy_s[i])
        if norm > 1e-8:
            ux_s[i] /= norm
            uy_s[i] /= norm
        else:
            ux_s[i], uy_s[i] = ux_s[i - 1], uy_s[i - 1]

    heading_deg = np.degrees(np.arctan2(uy_s, ux_s))
    out["heading_deg"] = heading_deg
    return out


def crop_centered_with_padding_meta(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    crop_size: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Return (crop, src_x0, src_y0, pad_left, pad_top).
    """
    half = crop_size // 2
    h, w = image.shape[:2]

    x0 = center_x - half
    y0 = center_y - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    crop = image[src_y0:src_y1, src_x0:src_x1].copy()

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return crop, src_x0, src_y0, pad_left, pad_top


def crop_centered_with_padding(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    crop_size: int,
) -> np.ndarray:
    crop, _, _, _, _ = crop_centered_with_padding_meta(
        image=image,
        center_x=center_x,
        center_y=center_y,
        crop_size=crop_size,
    )
    return crop

def write_ego_centered_map_video(
    background_png: Path,
    output_video: Path,
    frame_pixels_df: pd.DataFrame,
    labels_pixels_df: Optional[pd.DataFrame] = None,
    fps: int = 20,
    crop_size: int = 900,
    dot_radius: int = 7,
    draw_frame_label: bool = False,
) -> None:
    background = cv2.imread(str(background_png), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"Could not read background image: {background_png}")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (crop_size, crop_size),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_video}")

    center_px = crop_size // 2
    center_py = crop_size // 2
    pre_crop_size = int(np.ceil(crop_size * 1.5))
    pre_crop_size += pre_crop_size % 2
    final_x0 = (pre_crop_size - crop_size) // 2
    final_y0 = (pre_crop_size - crop_size) // 2

    if labels_pixels_df is None:
        labels_pixels_df = pd.DataFrame(columns=["name", "px", "py"])

    label_names = labels_pixels_df["name"].tolist()
    label_xs = labels_pixels_df["px"].to_numpy(dtype=float)
    label_ys = labels_pixels_df["py"].to_numpy(dtype=float)

    try:
        for _, row in frame_pixels_df.iterrows():
            px = int(row["px"])
            py = int(row["py"])
            heading_deg = float(row.get("heading_deg", 90.0))
            # Rotate map so current heading points upward in the output frame.
            rotate_deg = 90.0 - heading_deg

            pre_crop, src_x0, src_y0, pad_left, pad_top = crop_centered_with_padding_meta(
                background,
                center_x=px,
                center_y=py,
                crop_size=pre_crop_size,
            )

            rot_center = (pre_crop_size / 2.0, pre_crop_size / 2.0)
            rot_m = cv2.getRotationMatrix2D(rot_center, rotate_deg, 1.0)
            rotated = cv2.warpAffine(
                pre_crop,
                rot_m,
                (pre_crop_size, pre_crop_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            frame_img = rotated[
                final_y0:final_y0 + crop_size,
                final_x0:final_x0 + crop_size,
            ].copy()

            cv2.circle(frame_img, (center_px, center_py), dot_radius, (0, 0, 255), thickness=-1)

            if len(label_names) > 0:
                # Convert world pixel positions -> pre-crop -> rotated -> final crop
                lx = label_xs - src_x0 + pad_left
                ly = label_ys - src_y0 + pad_top
                ones = np.ones_like(lx)
                rot_xy = rot_m @ np.vstack([lx, ly, ones])
                rx = rot_xy[0] - final_x0
                ry = rot_xy[1] - final_y0

                for i, name in enumerate(label_names):
                    tx = int(round(rx[i]))
                    ty = int(round(ry[i]))
                    if tx < 10 or ty < 12 or tx >= crop_size - 10 or ty >= crop_size - 10:
                        continue
                    cv2.putText(
                        frame_img,
                        name,
                        (tx + 6, ty - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (30, 30, 30),
                        1,
                        cv2.LINE_AA,
                    )

            if draw_frame_label:
                label = f"frame {int(row['video_frame'])}"
                cv2.putText(
                    frame_img,
                    label,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame_img)
    finally:
        writer.release()

def print_diagnostics(
    sparse_df: pd.DataFrame,
    gdf_wgs84: gpd.GeoDataFrame,
    gdf_projected: gpd.GeoDataFrame,
    bbox_projected: gpd.GeoDataFrame,
    bbox_wgs84: tuple[float, float, float, float],
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    labels_gdf: gpd.GeoDataFrame,
) -> None:
    print("\n=== TRAJECTORY ===")
    print(f"Sparse trajectory rows: {len(sparse_df)}")
    print(f"Frame range: {int(sparse_df['frame'].min())} -> {int(sparse_df['frame'].max())}")
    print(f"WGS84 CRS: {gdf_wgs84.crs}")
    print(f"Projected CRS: {gdf_projected.crs}")

    xmin, ymin, xmax, ymax = gdf_projected.total_bounds
    print(f"Projected trajectory bounds: xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")

    bxmin, bymin, bxmax, bymax = bbox_projected.total_bounds
    print(f"Padded projected bounds: xmin={bxmin:.2f}, ymin={bymin:.2f}, xmax={bxmax:.2f}, ymax={bymax:.2f}")

    print("\n=== WGS84 BBOX FOR OSMNX ===")
    left, bottom, right, top = bbox_wgs84
    print(f"bbox=(left={left:.6f}, bottom={bottom:.6f}, right={right:.6f}, top={top:.6f})")

    print("\n=== ROAD NETWORK ===")
    print(f"Nodes: {len(nodes_gdf)}")
    print(f"Edges: {len(edges_gdf)}")
    print(f"Edges CRS: {edges_gdf.crs}")

    if len(edges_gdf) > 0:
        print("\nEdge columns:")
        print(list(edges_gdf.columns))

        print("\nFirst few edges:")
        preview_cols = [c for c in ["osmid", "name", "highway", "length", "geometry"] if c in edges_gdf.columns]
        print(edges_gdf[preview_cols].head())

    print("\n=== FEATURES ===")
    print(f"Fetched features: {len(features_gdf)}")
    if len(features_gdf) > 0:
        print("Feature columns:")
        print(list(features_gdf.columns))

    print("\n=== LABELS ===")
    print(f"Named labels kept: {len(labels_gdf)}")
    if len(labels_gdf) > 0:
        print(labels_gdf[["name"]].head(15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--padding-meters", type=float, default=600.0)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--show-trajectory", action="store_true")
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-size", type=int, default=3000)
    parser.add_argument("--crop-size", type=int, default=900)
    parser.add_argument("--dot-radius", type=int, default=7)
    parser.add_argument("--draw-frame-label", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    render_width = args.render_size
    render_height = args.render_size
    
    csv_path = resolve_csv_path(args.csv, args.csv_dir)
    print(f"Using CSV: {csv_path}")

    if args.output_video is None:
        # Default: save alongside the source CSV (same videos directory).
        stem = csv_path.stem
        if stem.endswith("_gps"):
            stem = stem[:-4]
        output_video_path = csv_path.with_name(f"{stem}_map.mp4")
    else:
        output_video_path = args.output_video

    if args.output_png is None:
        stem = csv_path.stem
        if stem.endswith("_gps"):
            stem = stem[:-4]
        output_png_path = csv_path.with_name(f"{stem}_background_map.png")
    else:
        output_png_path = args.output_png

    sparse_df = load_and_validate_csv(csv_path)

    gdf_wgs84 = build_sparse_trajectory_gdf(sparse_df)
    gdf_projected = project_trajectory_gdf(gdf_wgs84)

    bbox_projected = build_padded_square_bbox_polygon_projected(
        gdf_projected=gdf_projected,
        padding_meters=args.padding_meters,
    )
    bbox_wgs84 = projected_bbox_to_wgs84_tuple(bbox_projected)

    graph = fetch_drive_network(bbox_wgs84)
    _, nodes_gdf, edges_gdf = graph_to_projected_gdfs(graph)

    features_wgs84 = fetch_osm_features(bbox_wgs84)
    features_projected = project_features_to_match(features_wgs84, gdf_projected.crs)

    buildings_gdf = subset_polygon_features(features_projected, "building")

    green_areas_raw = pd.concat(
        [
            subset_polygon_features(features_projected, "landuse", ["park", "grass", "recreation_ground"]),
            subset_polygon_features(features_projected, "leisure", ["park", "garden"]),
        ],
        ignore_index=False,
    )
    green_areas_gdf = (
        gpd.GeoDataFrame(green_areas_raw, geometry="geometry", crs=features_projected.crs)
        if len(green_areas_raw) > 0
        else empty_gdf(features_projected.crs)
    )

    parking_gdf = subset_polygon_features(features_projected, "amenity", ["parking"])
    poi_points_gdf = subset_point_features(features_projected, "amenity", ["fuel", "hospital", "school"])

    labels_gdf = build_named_label_features(features_projected)
    labels_gdf = filter_labels_by_spacing(
        labels_gdf,
        min_distance_m=100.0,
        max_labels=60,
    )

    print_diagnostics(
        sparse_df=sparse_df,
        gdf_wgs84=gdf_wgs84,
        gdf_projected=gdf_projected,
        bbox_projected=bbox_projected,
        bbox_wgs84=bbox_wgs84,
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        features_gdf=features_projected,
        labels_gdf=labels_gdf,
    )

    print("\n=== FEATURE SUBSETS ===")
    print(f"Buildings: {len(buildings_gdf)}")
    print(f"Green areas: {len(green_areas_gdf)}")
    print(f"Parking polygons: {len(parking_gdf)}")
    print(f"POI points: {len(poi_points_gdf)}")

    render_background_png(
        output_png=output_png_path,
        edges_gdf=edges_gdf,
        bbox_projected=bbox_projected,
        trajectory_gdf=gdf_projected,
        buildings_gdf=buildings_gdf,
        green_areas_gdf=green_areas_gdf,
        parking_gdf=parking_gdf,
        poi_points_gdf=poi_points_gdf,
        labels_gdf=labels_gdf,
        width_px=render_width,
        height_px=render_height,
        show_trajectory=args.show_trajectory,
    )

    print(f"\nSaved background PNG to: {output_png_path}")
    print("\nStep 3 complete.")

    if args.make_video:
        video_base_png = output_png_path.with_name(f"{output_png_path.stem}_base.png")
        render_background_png(
            output_png=video_base_png,
            edges_gdf=edges_gdf,
            bbox_projected=bbox_projected,
            trajectory_gdf=gdf_projected,
            buildings_gdf=buildings_gdf,
            green_areas_gdf=green_areas_gdf,
            parking_gdf=parking_gdf,
            poi_points_gdf=poi_points_gdf,
            labels_gdf=None,  # keep labels upright by overlaying them after rotation
            width_px=render_width,
            height_px=render_height,
            show_trajectory=args.show_trajectory,
        )

        dense_df = build_dense_frame_positions(sparse_df=sparse_df)

        dense_projected_gdf = project_dense_positions_to_match(
            dense_df=dense_df,
            target_crs=gdf_projected.crs,
        )

        frame_pixels_df = projected_points_to_pixel_coords(
            dense_projected_gdf=dense_projected_gdf,
            bbox_projected=bbox_projected,
            width_px=render_width,
            height_px=render_height,
        )
        frame_pixels_df = attach_heading_degrees(
            frame_pixels_df=frame_pixels_df,
            dense_projected_gdf=dense_projected_gdf,
        )
        label_pixels_df = labels_to_pixel_coords(
            labels_gdf=labels_gdf,
            bbox_projected=bbox_projected,
            width_px=render_width,
            height_px=render_height,
        )

        print("\n=== VIDEO ===")
        print(f"Video frames: {len(frame_pixels_df)}")
        if len(frame_pixels_df) > 0:
            hmin = float(frame_pixels_df["heading_deg"].min())
            hmax = float(frame_pixels_df["heading_deg"].max())
            print(f"Heading range (deg): {hmin:.1f} -> {hmax:.1f}")
        print(f"Crop size: {args.crop_size}x{args.crop_size}")
        print(f"FPS: {args.fps}")

        write_ego_centered_map_video(
            background_png=video_base_png,
            output_video=output_video_path,
            frame_pixels_df=frame_pixels_df,
            labels_pixels_df=label_pixels_df,
            fps=args.fps,
            crop_size=args.crop_size,
            dot_radius=args.dot_radius,
            draw_frame_label=args.draw_frame_label,
        )

        print(f"Saved ego-centered map video to: {output_video_path}")

if __name__ == "__main__":
    main()
