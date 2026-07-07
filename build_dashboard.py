"""
build_dashboard.py

Merges annotator CSV submissions and generates a dashboard reproducing every
statistic/table/figure in Section IV ("Dataset Analysis") of the doPlan
paper:
  Table I   -- Dataset Scale and Coverage Summary
  Fig 3     -- Segment Duration Distribution
  Fig 4     -- Referential Class Distribution
  Table II  -- Language statistics
  Fig 5     -- Instruction Length by Referential Class
  Fig 6     -- Annotations per Source Clip Distribution
  Table III -- Annotation Diversity and Overlap

USAGE:
    python build_dashboard.py --root /path/to/csvs --out ./dashboard_output

IMPORTANT: the paper's numbers (1,108 annotations / 3 annotators) are a
snapshot from when it was written. This dashboard recomputes everything
live from whatever is in --root, so the numbers here will (correctly) move
as more annotations come in -- they aren't meant to match the paper exactly,
just use the same definitions.

METHODOLOGY NOTES / assumptions made where the paper doesn't give an exact
formula (flagged here so they're easy to find and change):
  - "Overlapping annotation pair" = two annotations on the same clip whose
    [start_frame, end_frame] intervals intersect.
  - "Temporal overlap ratio" for a pair = intersection length / union length
    (Jaccard on the two frame intervals), averaged across overlapping pairs.
  - "Instruction similarity" for a pair = Jaccard similarity of the two
    instructions' lowercase word sets, averaged across overlapping pairs
    that both have non-blank labels. The paper doesn't specify its exact
    similarity method (could be embedding-based); this is a lightweight
    stand-in that needs no extra ML dependencies. Swap in something else in
    `instruction_similarity()` below if you want to match the paper exactly.
  - Instruction "length in words" uses simple whitespace splitting (matches
    how word counts are usually reported). Vocabulary size / type-token
    ratio use a normalized tokenization (lowercased, punctuation stripped)
    since raw whitespace tokens would inflate vocabulary with punctuation
    variants of the same word.
  - "Multi-sentence" = the instruction splits into 2+ non-empty chunks on
    ./!/? boundaries.

SCHEMA NOTES (from userLabeler.py): folder/base_name naming variants,
missing username, and commentary shorthand (e.g. "(d)") are all normalized
in load_and_normalize(). fps is fixed by the FPS constant below since it
isn't stored per-row -- verify it still matches every annotator's
settings.txt as the team grows.
"""

import argparse
import glob
import itertools
import json
import os
import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
FPS = 10.0

COMMENTARY_MAP = {
    "(a)": "Non-Referential",
    "(b)": "Static Referential",
    "(c)": "Dynamic Referential",
    "(d)": "Both",
}
CANONICAL_COMMENTARY = ["Non-Referential", "Static Referential", "Dynamic Referential", "Both"]
REQUIRED_COLUMNS = {"folder", "start_frame", "end_frame", "label", "commentary"}

# ---------------- THEME ----------------
BG = "#14161A"
PANEL = "#1E2128"
PANEL_BORDER = "#2B2F38"
AMBER = "#F2B705"
TEAL = "#4FD1C5"
CORAL = "#E8604C"
TEXT = "#F5F3EE"
MUTED = "#9AA0AC"
CHART_SEQUENCE = [TEAL, AMBER, "#8C7CF0", CORAL, "#5FB0E8", "#6FCF97"]


# ---------------- LOAD / NORMALIZE ----------------
def load_and_normalize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"base_name": "folder"})

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing}")

    if "username" not in df.columns:
        df["username"] = os.path.basename(os.path.dirname(csv_path))

    df["commentary"] = df["commentary"].astype(str).str.strip()
    df["commentary"] = df["commentary"].replace(COMMENTARY_MAP)
    df["source_file"] = csv_path
    return df


MANIFEST_FILENAME = "clip_manifest.csv"


def load_manifest(root_dir: str):
    """Finds clip_manifest.csv (produced by list_clips.py) anywhere under root_dir.
    Returns a DataFrame with clip_id/relative_path/duration_sec, or None if absent."""
    matches = [
        p for p in glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
        if os.path.basename(p).lower() == MANIFEST_FILENAME
    ]
    if not matches:
        return None
    manifest = pd.read_csv(matches[0])
    manifest.columns = [c.strip().lower() for c in manifest.columns]
    return manifest


def merge_all(root_dir: str):
    frames, skipped = [], []
    for path in glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True):
        base = os.path.basename(path)
        if base.startswith("_") or base.lower() == MANIFEST_FILENAME:
            continue
        try:
            frames.append(load_and_normalize(path))
        except Exception as e:
            skipped.append((path, str(e)))

    if not frames:
        return pd.DataFrame(), 0, skipped

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["username", "folder", "start_frame", "end_frame"])
    dupes_dropped = before - len(merged)
    merged["duration_sec"] = (merged["end_frame"] - merged["start_frame"]) / FPS
    return merged, dupes_dropped, skipped


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip().isin(["", "nan"]))


WORD_RE = re.compile(r"[a-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def normalized_tokens(text: str):
    return WORD_RE.findall(text.lower())


def is_multi_sentence(text: str) -> bool:
    chunks = [c for c in SENTENCE_SPLIT_RE.split(text) if c.strip()]
    return len(chunks) > 1


def instruction_similarity(a: str, b: str) -> float:
    """Jaccard similarity over lowercase word sets. See module docstring."""
    set_a, set_b = set(normalized_tokens(a)), set(normalized_tokens(b))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def merge_intervals(intervals):
    """Merge overlapping/adjacent (start, end) ranges. Input/output in the same unit (seconds)."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(pair) for pair in merged]


def scene_intervals_seconds(df: pd.DataFrame, clip_id: str):
    """All annotation intervals for one clip, in SECONDS (start_frame/end_frame are 10fps units)."""
    rows = df[df["folder"] == clip_id]
    out = []
    for _, r in rows.iterrows():
        s = min(r["start_frame"], r["end_frame"]) / FPS
        e = max(r["start_frame"], r["end_frame"]) / FPS
        out.append((s, e, str(r["username"]), str(r["label"])))
    return out


# ---------------- STATS ----------------
def compute_stats(df: pd.DataFrame, manifest: pd.DataFrame = None) -> dict:
    blank = _is_blank(df["label"])
    labeled = df[~blank].copy()

    # ---- Fig 1: annotations per annotator ----
    per_annotator_counts = df.groupby("username").size().sort_values(ascending=False).to_dict()

    # ---- Table I: Dataset Scale and Coverage Summary ----
    table1 = {
        "number_of_annotations": int(len(df)),
        "number_of_unique_source_clips": int(df["folder"].nunique()),
        "number_of_annotators": int(df["username"].nunique()),
        "total_annotated_driving_time_hrs": float(df["duration_sec"].sum() / 3600),
        "mean_segment_duration_sec": float(df["duration_sec"].mean()),
        "median_segment_duration_sec": float(df["duration_sec"].median()),
        "min_segment_duration_sec": float(df["duration_sec"].min()),
        "max_segment_duration_sec": float(df["duration_sec"].max()),
    }

    # ---- Fig 4 / referential class distribution ----
    commentary_counts = df["commentary"].value_counts()
    referential_distribution = {
        cls: int(commentary_counts.get(cls, 0)) for cls in CANONICAL_COMMENTARY
    }
    unmapped_commentary = {
        k: int(v) for k, v in commentary_counts.items() if k not in CANONICAL_COMMENTARY
    }

    # ---- Table II: language statistics (labeled instructions only) ----
    word_counts = labeled["label"].astype(str).str.split().apply(len)
    all_tokens = [t for text in labeled["label"].astype(str) for t in normalized_tokens(text)]
    vocab = set(all_tokens)
    multi_sentence_count = int(labeled["label"].astype(str).apply(is_multi_sentence).sum())

    table2 = {
        "number_of_instructions": int(len(labeled)),
        "mean_instruction_length_words": float(word_counts.mean()) if len(labeled) else 0.0,
        "median_instruction_length_words": float(word_counts.median()) if len(labeled) else 0.0,
        "min_instruction_length_words": int(word_counts.min()) if len(labeled) else 0,
        "max_instruction_length_words": int(word_counts.max()) if len(labeled) else 0,
        "vocabulary_size": len(vocab),
        "type_token_ratio": (len(vocab) / len(all_tokens)) if all_tokens else 0.0,
        "multi_sentence_pct": (multi_sentence_count / len(labeled) * 100) if len(labeled) else 0.0,
    }

    # ---- Fig 5 data: instruction length by referential class ----
    labeled["word_count"] = word_counts
    length_by_class = {
        cls: labeled.loc[labeled["commentary"] == cls, "word_count"].tolist()
        for cls in CANONICAL_COMMENTARY
    }

    # ---- Coverage (requires clip_manifest.csv from list_clips.py) ----
    coverage = None
    unannotated_clips = []
    manifest_mismatches = []
    if manifest is not None and not manifest.empty:
        annotated_clip_ids = set(df["folder"].unique())
        manifest_clip_ids = set(manifest["clip_id"].unique())

        total_clips = len(manifest_clip_ids)
        annotated_in_manifest = annotated_clip_ids & manifest_clip_ids
        unannotated_clips = sorted(manifest_clip_ids - annotated_clip_ids)
        # clips referenced in CSVs but missing from the manifest -- likely a
        # naming mismatch/typo, or the manifest is stale
        manifest_mismatches = sorted(annotated_clip_ids - manifest_clip_ids)

        total_available_time_hrs = float(manifest["duration_sec"].sum() / 3600)
        annotated_time_hrs = float(
            manifest.loc[manifest["clip_id"].isin(annotated_in_manifest), "duration_sec"].sum() / 3600
        )

        coverage = {
            "total_clips_in_manifest": total_clips,
            "clips_annotated": len(annotated_in_manifest),
            "clips_unannotated": len(unannotated_clips),
            "coverage_pct": (len(annotated_in_manifest) / total_clips * 100) if total_clips else 0.0,
            "total_available_driving_time_hrs": total_available_time_hrs,
            "annotated_clip_time_hrs": annotated_time_hrs,
            "time_coverage_pct": (annotated_time_hrs / total_available_time_hrs * 100) if total_available_time_hrs else 0.0,
        }

        # ---- Per-scene coverage (all math in SECONDS) ----
        # For every clip in the manifest: its total duration, the union of all
        # annotation intervals (merged so overlaps aren't double-counted), and
        # what fraction of the clip that union covers.
        annotated_ids_set = set(df["folder"].unique())
        per_scene = []
        for _, mrow in manifest.iterrows():
            clip_id = str(mrow["clip_id"]).strip()
            total_sec = float(mrow["duration_sec"])
            city = str(mrow["relative_path"]).strip().split("/")[0]
            raw = [(s, e) for (s, e, _u, _l) in scene_intervals_seconds(df, clip_id)]
            # clamp to clip bounds before merging so a stray over-long annotation
            # can't push coverage above 100%
            clamped = [(max(0.0, s), min(total_sec, e)) for s, e in raw if e > 0 and s < total_sec]
            merged = merge_intervals(clamped)
            covered_sec = sum(e - s for s, e in merged)
            per_scene.append({
                "clip_id": clip_id,
                "city": city,
                "total_sec": total_sec,
                "covered_sec": covered_sec,
                "coverage_pct": (covered_sec / total_sec * 100) if total_sec else 0.0,
                "num_annotations": int((df["folder"] == clip_id).sum()),
            })

        coverage["per_scene"] = per_scene
        # distribution of per-scene coverage % (Fig 2)
        coverage["scene_coverage_pcts"] = [s["coverage_pct"] for s in per_scene]

    # ---- Fig 6 + Table III: diversity and overlap ----
    per_clip_counts = df.groupby("folder").size()
    if manifest is not None and not manifest.empty:
        # include clips with zero annotations so the distribution reflects true coverage
        zero_count_clips = len(set(manifest["clip_id"].unique()) - set(per_clip_counts.index))
    else:
        zero_count_clips = None
    table3 = {
        "mean_annotations_per_clip": float(per_clip_counts.mean()),
        "median_annotations_per_clip": float(per_clip_counts.median()),
        "max_annotations_single_clip": int(per_clip_counts.max()),
        "clips_with_multiple_annotations_pct": float((per_clip_counts > 1).mean() * 100),
    }

    overlap_ratios, similarity_scores, overlap_pairs = [], [], 0
    for folder, group in df.groupby("folder"):
        if len(group) < 2:
            continue
        for (i, row_i), (j, row_j) in itertools.combinations(group.iterrows(), 2):
            start_overlap = max(row_i["start_frame"], row_j["start_frame"])
            end_overlap = min(row_i["end_frame"], row_j["end_frame"])
            intersection = end_overlap - start_overlap
            if intersection <= 0:
                continue
            overlap_pairs += 1
            union = max(row_i["end_frame"], row_j["end_frame"]) - min(row_i["start_frame"], row_j["start_frame"])
            if union > 0:
                overlap_ratios.append(intersection / union)
            label_i, label_j = str(row_i["label"]), str(row_j["label"])
            if label_i.strip() and label_j.strip() and label_i != "nan" and label_j != "nan":
                similarity_scores.append(instruction_similarity(label_i, label_j))

    table3.update({
        "overlapping_annotation_pairs": overlap_pairs,
        "mean_temporal_overlap_ratio": (sum(overlap_ratios) / len(overlap_ratios)) if overlap_ratios else 0.0,
        "mean_instruction_similarity_for_overlaps": (
            sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        ),
    })

    per_clip_distribution = per_clip_counts.value_counts().sort_index().to_dict()
    if zero_count_clips:
        per_clip_distribution = {0: zero_count_clips, **per_clip_distribution}

    # ---- Raw per-scene intervals (seconds) for the random-scene overlap viewer ----
    # Only for scenes that (a) exist in the manifest and (b) have >=1 annotation.
    scene_timelines = {}
    if manifest is not None and not manifest.empty:
        manifest_lookup = {str(r["clip_id"]).strip(): float(r["duration_sec"]) for _, r in manifest.iterrows()}
        for clip_id in df["folder"].unique():
            clip_id = str(clip_id)
            if clip_id not in manifest_lookup:
                continue
            total_sec = manifest_lookup[clip_id]
            bars = []
            for s, e, user, label in scene_intervals_seconds(df, clip_id):
                bars.append({
                    "start": round(max(0.0, s), 2),
                    "end": round(min(total_sec, e), 2),
                    "user": user,
                    "label": (label[:80] + "...") if len(label) > 80 else label,
                })
            # sort by start so stacked rows read left-to-right
            bars.sort(key=lambda b: b["start"])
            scene_timelines[clip_id] = {"total_sec": total_sec, "bars": bars}

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "coverage": coverage,
        "unannotated_clips": unannotated_clips,
        "manifest_mismatches": manifest_mismatches,
        "per_annotator_counts": per_annotator_counts,
        "referential_distribution": referential_distribution,
        "unmapped_commentary": unmapped_commentary,
        "length_by_class": length_by_class,
        "per_clip_counts": per_clip_distribution,
        "scene_timelines": scene_timelines,
    }


# ---------------- CHARTS ----------------
def _base_layout(title, height=360):
    return dict(
        title=dict(text=title, font=dict(size=15, color=TEXT, family="IBM Plex Sans")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color=MUTED, size=12),
        margin=dict(l=40, r=20, t=50, b=40), height=height, colorway=CHART_SEQUENCE,
    )


def chart_duration_distribution(df, stats):
    fig = go.Figure(go.Histogram(x=df["duration_sec"], nbinsx=30, marker_color=TEAL, name="Segments"))
    mean_v = stats["table1"]["mean_segment_duration_sec"]
    median_v = stats["table1"]["median_segment_duration_sec"]
    fig.add_vline(
        x=mean_v, line_dash="dash", line_color=AMBER,
        annotation_text=f"Mean: {mean_v:.1f}s", annotation_position="top left",
        annotation_font_color=AMBER, annotation_y=1.0, annotation_yref="paper",
    )
    fig.add_vline(
        x=median_v, line_dash="dot", line_color=CORAL,
        annotation_text=f"Median: {median_v:.1f}s", annotation_position="bottom right",
        annotation_font_color=CORAL, annotation_y=0.0, annotation_yref="paper",
    )
    fig.update_layout(**_base_layout("Fig 3 — Segment Duration Distribution"))
    fig.update_xaxes(title="Segment duration (s)")
    fig.update_yaxes(title="Number of annotations")
    return fig


def chart_annotations_per_annotator(stats):
    per_annotator = stats["per_annotator_counts"]
    fig = go.Figure(go.Bar(
        x=list(per_annotator.keys()), y=list(per_annotator.values()),
        marker_color=TEAL, text=list(per_annotator.values()), textposition="outside",
    ))
    fig.update_layout(**_base_layout("Fig 1 — Annotations per Annotator"))
    fig.update_yaxes(title="Number of annotations")
    return fig


def chart_scene_coverage_hist(stats):
    pcts = stats["coverage"]["scene_coverage_pcts"] if stats["coverage"] else []
    fig = go.Figure(go.Histogram(x=pcts, xbins=dict(start=0, end=100, size=10), marker_color="#8C7CF0"))
    fig.update_layout(**_base_layout("Fig 2 — Percent of Scene Covered by Annotations"))
    fig.update_xaxes(title="Percent of scene covered (%)", range=[0, 100])
    fig.update_yaxes(title="Number of scenes")
    return fig


def chart_referential_distribution(stats):
    dist = stats["referential_distribution"]
    total = sum(dist.values()) or 1
    labels = list(dist.keys())
    values = list(dist.values())
    text = [f"{v} ({v/total:.1%})" for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, text=text, textposition="outside", marker_color=CHART_SEQUENCE))
    fig.update_layout(**_base_layout("Fig 4 — Referential Class Distribution"))
    fig.update_yaxes(title="Number of annotations")
    return fig


def chart_length_by_class(stats):
    fig = go.Figure()
    for i, cls in enumerate(CANONICAL_COMMENTARY):
        values = stats["length_by_class"].get(cls, [])
        fig.add_trace(go.Box(y=values, name=cls, marker_color=CHART_SEQUENCE[i % len(CHART_SEQUENCE)], boxmean=True))
    fig.update_layout(**_base_layout("Fig 5 — Instruction Length by Referential Class"))
    fig.update_yaxes(title="Instruction length (words)")
    return fig


def chart_annotations_per_clip(stats):
    per_clip = stats["per_clip_counts"]
    fig = go.Figure(go.Bar(x=list(per_clip.keys()), y=list(per_clip.values()), marker_color=AMBER))
    fig.update_layout(**_base_layout("Fig 6 — Annotations per Source Clip Distribution"))
    fig.update_xaxes(title="Annotations per source clip", dtick=1)
    fig.update_yaxes(title="Number of source clips")
    return fig


def _fig_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


# ---------------- FULL-WIDTH COVERAGE STRIP (coverageViewer-style) ----------------
def build_coverage_strip(stats):
    """One horizontal timeline per city + an ALL row. Green = annotated union,
    grey = downloaded-but-unannotated remainder. All widths proportional to
    seconds. Returns an HTML string (inline SVG)."""
    if not stats["coverage"] or not stats["coverage"].get("per_scene"):
        return ""

    per_scene = stats["coverage"]["per_scene"]
    cities = sorted({s["city"] for s in per_scene})
    rows = [("ALL", per_scene)] + [(c, [s for s in per_scene if s["city"] == c]) for c in cities]

    STRIP_H = 26
    ROW_GAP = 46
    LABEL_W = 90
    width = 1100
    track_w = width - LABEL_W - 20

    svg_rows = []
    y = 10
    for label, entries in rows:
        total = sum(e["total_sec"] for e in entries) or 1.0
        covered = sum(e["covered_sec"] for e in entries)
        pct = covered / total * 100

        # background grey track
        svg_rows.append(
            f'<rect x="{LABEL_W}" y="{y}" width="{track_w}" height="{STRIP_H}" fill="#2B2F38" rx="3"/>'
        )
        # walk scenes left-to-right, drawing grey per-scene segments then green covered sub-segments
        cursor_sec = 0.0
        for e in entries:
            seg_x = LABEL_W + (cursor_sec / total) * track_w
            seg_w = (e["total_sec"] / total) * track_w
            # faint divider between scenes
            svg_rows.append(
                f'<rect x="{seg_x:.2f}" y="{y}" width="{seg_w:.2f}" height="{STRIP_H}" '
                f'fill="#d9d9d9" opacity="0.10"/>'
            )
            cursor_sec += e["total_sec"]

        # green covered blocks (recompute merged union positions along the concatenated timeline)
        cursor_sec = 0.0
        for e in entries:
            frac_covered = (e["covered_sec"] / e["total_sec"]) if e["total_sec"] else 0.0
            if frac_covered > 0:
                gx = LABEL_W + (cursor_sec / total) * track_w
                gw = (e["covered_sec"] / total) * track_w
                svg_rows.append(
                    f'<rect x="{gx:.2f}" y="{y}" width="{gw:.2f}" height="{STRIP_H}" fill="#4FD1C5" rx="2"/>'
                )
            cursor_sec += e["total_sec"]

        hours_total = total / 3600
        hours_cov = covered / 3600
        svg_rows.append(
            f'<text x="{LABEL_W - 8}" y="{y + STRIP_H/2 + 4}" text-anchor="end" '
            f'font-size="12" fill="#F5F3EE" font-family="IBM Plex Mono">{label}</text>'
        )
        svg_rows.append(
            f'<text x="{LABEL_W}" y="{y + STRIP_H + 15}" font-size="11" fill="#9AA0AC" '
            f'font-family="IBM Plex Sans">{pct:.1f}% covered — {hours_cov:.2f}h / {hours_total:.2f}h annotated</text>'
        )
        y += ROW_GAP

    svg_h = y + 6
    svg = (
        f'<svg viewBox="0 0 {width} {svg_h}" width="100%" xmlns="http://www.w3.org/2000/svg">'
        + "".join(svg_rows) + "</svg>"
    )

    legend = (
        '<div class="legend">'
        '<span><i style="background:#4FD1C5"></i> annotated</span>'
        '<span><i style="background:#d9d9d9;opacity:0.3"></i> downloaded, not annotated</span>'
        '</div>'
    )
    return f'<div class="card fullwidth">{svg}{legend}</div>'


# ---------------- RANDOM-SCENE OVERLAP VIEWER ----------------
def build_random_scene_viewer(stats):
    """Interactive: picks a random annotated scene, draws its full timeline as
    a bottom axis, and stacks each individual annotation as a mini interval bar
    above it -- so overlap is visible qualitatively. A button re-rolls to a new
    random scene. All scene data is embedded as JSON; selection happens client-side."""
    timelines = stats.get("scene_timelines") or {}
    if not timelines:
        return ""

    data_json = json.dumps(timelines)
    return f"""
  <div class="card fullwidth">
    <div class="scene-header">
      <span id="scene-title" style="font-family:'IBM Plex Mono',monospace;font-size:14px;"></span>
      <button id="scene-reroll" class="reroll-btn">Pick another random scene</button>
    </div>
    <div id="scene-viewer"></div>
  </div>
  <script>
  (function() {{
    const scenes = {data_json};
    const ids = Object.keys(scenes);
    const palette = ["#4FD1C5","#F2B705","#8C7CF0","#E8604C","#5FB0E8","#6FCF97","#E0729B","#C9A227"];

    function render(clipId) {{
      const scene = scenes[clipId];
      const total = scene.total_sec;
      const bars = scene.bars;
      const W = 1080, LEFT = 12, RIGHT = 12, trackW = W - LEFT - RIGHT;
      const rowH = 20, rowGap = 4, topPad = 8;
      const axisH = 44;
      const H = topPad + bars.length * (rowH + rowGap) + axisH;
      const x = s => LEFT + (s / total) * trackW;

      let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" xmlns="http://www.w3.org/2000/svg">';

      bars.forEach((b, i) => {{
        const y = topPad + i * (rowH + rowGap);
        const bx = x(b.start), bw = Math.max(2, x(b.end) - x(b.start));
        const color = palette[i % palette.length];
        svg += '<rect x="' + bx.toFixed(1) + '" y="' + y + '" width="' + bw.toFixed(1) + '" height="' + rowH + '" rx="3" fill="' + color + '" opacity="0.85"><title>' + b.user + ' [' + b.start + 's–' + b.end + 's]: ' + (b.label||'').replace(/"/g,'&quot;') + '</title></rect>';
        svg += '<text x="' + (bx + 5).toFixed(1) + '" y="' + (y + 14) + '" font-size="10" fill="#14161A" font-family="IBM Plex Sans" style="pointer-events:none;">' + b.user + '</text>';
      }});

      // bottom axis
      const axisY = topPad + bars.length * (rowH + rowGap) + 10;
      svg += '<line x1="' + LEFT + '" y1="' + axisY + '" x2="' + (LEFT + trackW) + '" y2="' + axisY + '" stroke="#4B5160" stroke-width="1.5"/>';
      const ticks = 6;
      for (let t = 0; t <= ticks; t++) {{
        const sec = (total / ticks) * t;
        const tx = x(sec);
        svg += '<line x1="' + tx.toFixed(1) + '" y1="' + axisY + '" x2="' + tx.toFixed(1) + '" y2="' + (axisY + 5) + '" stroke="#4B5160"/>';
        svg += '<text x="' + tx.toFixed(1) + '" y="' + (axisY + 18) + '" text-anchor="middle" font-size="10" fill="#9AA0AC" font-family="IBM Plex Mono">' + sec.toFixed(0) + 's</text>';
      }}

      svg += '</svg>';
      document.getElementById('scene-viewer').innerHTML = svg;
      document.getElementById('scene-title').textContent =
        clipId + '  ·  ' + bars.length + ' annotation(s)  ·  ' + total.toFixed(1) + 's total';
    }}

    function reroll() {{ render(ids[Math.floor(Math.random() * ids.length)]); }}
    document.getElementById('scene-reroll').addEventListener('click', reroll);
    reroll();
  }})();
  </script>
"""


# ---------------- HTML ----------------
def _scorecards(pairs):
    return "".join(
        f'<div class="card score"><div class="score-value">{v}</div><div class="score-label">{l}</div></div>'
        for v, l in pairs
    )


def build_dashboard_html(df, stats, dupes_dropped, skipped, out_path):
    t1, t2, t3 = stats["table1"], stats["table2"], stats["table3"]

    table1_cards = _scorecards([
        (f"{t1['number_of_annotations']:,}", "Number of annotations"),
        (f"{t1['number_of_unique_source_clips']:,}", "Unique source clips"),
        (f"{t1['number_of_annotators']}", "Annotators"),
        (f"{t1['total_annotated_driving_time_hrs']:.1f} hrs", "Total annotated driving time"),
        (f"{t1['mean_segment_duration_sec']:.1f}s", "Mean segment duration"),
        (f"{t1['median_segment_duration_sec']:.1f}s", "Median segment duration"),
        (f"{t1['min_segment_duration_sec']:.1f}s", "Minimum segment duration"),
        (f"{t1['max_segment_duration_sec']:.1f}s", "Maximum segment duration"),
    ])

    table2_cards = _scorecards([
        (f"{t2['number_of_instructions']:,}", "Number of instructions"),
        (f"{t2['mean_instruction_length_words']:.2f}", "Mean instruction length (words)"),
        (f"{t2['median_instruction_length_words']:.1f}", "Median instruction length (words)"),
        (f"{t2['min_instruction_length_words']}", "Minimum instruction length (words)"),
        (f"{t2['max_instruction_length_words']}", "Maximum instruction length (words)"),
        (f"{t2['vocabulary_size']:,}", "Vocabulary size"),
        (f"{t2['type_token_ratio']:.3f}", "Type-token ratio"),
        (f"{t2['multi_sentence_pct']:.2f}%", "Multi-sentence instructions"),
    ])

    table3_cards = _scorecards([
        (f"{t3['mean_annotations_per_clip']:.2f}", "Mean annotations per clip"),
        (f"{t3['median_annotations_per_clip']:.2f}", "Median annotations per clip"),
        (f"{t3['max_annotations_single_clip']}", "Max annotations for a single clip"),
        (f"{t3['clips_with_multiple_annotations_pct']:.1f}%", "Clips with multiple annotations"),
        (f"{t3['overlapping_annotation_pairs']:,}", "Overlapping annotation pairs"),
        (f"{t3['mean_temporal_overlap_ratio']:.3f}", "Mean temporal overlap ratio"),
        (f"{t3['mean_instruction_similarity_for_overlaps']:.3f}", "Mean instruction similarity for overlaps"),
    ])

    charts = [chart_annotations_per_annotator(stats)]
    if stats["coverage"]:
        charts.append(chart_scene_coverage_hist(stats))
    charts += [
        chart_duration_distribution(df, stats),
        chart_referential_distribution(stats),
        chart_length_by_class(stats),
        chart_annotations_per_clip(stats),
    ]
    chart_divs = "".join(f'<div class="card chart">{_fig_div(c)}</div>' for c in charts)

    strip_html = build_coverage_strip(stats)
    coverage_strip_section = (
        f'<div class="section-title">Coverage Timeline (annotated vs. available, by city)</div>{strip_html}'
        if strip_html else ""
    )
    viewer_html = build_random_scene_viewer(stats)
    scene_viewer_section = (
        f'<div class="section-title">Random Scene — Annotation Overlap</div>{viewer_html}'
        if viewer_html else ""
    )

    coverage_section = ""
    if stats["coverage"]:
        cov = stats["coverage"]
        coverage_cards = _scorecards([
            (f"{cov['total_clips_in_manifest']:,}", "Total clips in library"),
            (f"{cov['clips_annotated']:,}", "Clips with 1+ annotation"),
            (f"{cov['clips_unannotated']:,}", "Clips with zero annotations"),
            (f"{cov['coverage_pct']:.1f}%", "Clip coverage"),
            (f"{cov['total_available_driving_time_hrs']:.1f} hrs", "Total available driving time"),
            (f"{cov['annotated_clip_time_hrs']:.1f} hrs", "Driving time within touched clips"),
            (f"{cov['time_coverage_pct']:.1f}%", "Time coverage"),
        ])
        coverage_section = f"""
  <div class="section-title">Coverage (from clip_manifest.csv)</div>
  <div class="scorecards">{coverage_cards}</div>
"""

    quality_notes = []
    if stats["unmapped_commentary"]:
        quality_notes.append(
            f"Unrecognized commentary values (not in COMMENTARY_MAP): {stats['unmapped_commentary']}"
        )
    if dupes_dropped:
        quality_notes.append(f"{dupes_dropped} duplicate row(s) dropped during merge.")
    if skipped:
        quality_notes.append(f"{len(skipped)} file(s) failed to parse and were skipped.")
    if stats["manifest_mismatches"]:
        quality_notes.append(
            f"{len(stats['manifest_mismatches'])} clip folder name(s) appear in annotation CSVs but not in "
            f"clip_manifest.csv -- possible typo or a stale manifest. First few: "
            f"{stats['manifest_mismatches'][:5]}"
        )
    if stats["coverage"] is None:
        quality_notes.append(
            "No clip_manifest.csv found -- coverage stats are unavailable. Run list_clips.py against the "
            "video library and drop clip_manifest.csv into the shared Drive folder to enable them."
        )
    elif stats["unannotated_clips"]:
        quality_notes.append(
            f"{len(stats['unannotated_clips'])} clip(s) in the library have zero annotations so far. "
            f"Full list written to unannotated_clips.csv alongside this dashboard."
        )
    quality_html = (
        "".join(f'<div class="flag">{n}</div>' for n in quality_notes)
        if quality_notes else '<div class="flag ok">No data-quality issues detected.</div>'
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>doPlan Annotation Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: {BG}; --panel: {PANEL}; --border: {PANEL_BORDER};
    --amber: {AMBER}; --teal: {TEAL}; --coral: {CORAL}; --text: {TEXT}; --muted: {MUTED};
  }}
  * {{ box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; margin: 0; padding: 32px 40px 60px; }}
  header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }}
  h1 {{ font-family: 'IBM Plex Mono', monospace; font-size: 22px; letter-spacing: 0.5px; margin: 0; }}
  .subtitle {{ color: var(--muted); font-size: 13px; }}
  .laneline {{ height: 1px; margin: 20px 0 28px;
    background-image: repeating-linear-gradient(90deg, var(--amber) 0 18px, transparent 18px 34px); opacity: 0.8; }}
  .scorecards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 8px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }}
  .score-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600; color: var(--amber); }}
  .score-label {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px; }}
  .chart {{ padding: 8px 8px 0; }}
  .fullwidth {{ margin-top: 8px; padding: 20px; }}
  .legend {{ display: flex; gap: 24px; margin-top: 12px; font-size: 12px; color: var(--muted); }}
  .legend i {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: -2px; margin-right: 6px; }}
  .scene-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }}
  .reroll-btn {{ background: var(--teal); color: #14161A; border: none; border-radius: 6px;
    padding: 8px 14px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; cursor: pointer; font-weight: 600; }}
  .reroll-btn:hover {{ opacity: 0.85; }}
  .reroll-btn:focus-visible {{ outline: 2px solid var(--amber); outline-offset: 2px; }}
  .section-title {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px; margin: 34px 0 12px; }}
  .flag {{ background: rgba(79,209,197,0.10); border: 1px solid var(--teal); color: var(--text);
    border-radius: 8px; padding: 10px 14px; font-size: 13px; margin-bottom: 8px; }}
  footer {{ margin-top: 40px; color: var(--muted); font-size: 12px; line-height: 1.7; }}
  @media (max-width: 900px) {{
    .scorecards {{ grid-template-columns: repeat(2, 1fr); }}
    .charts {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <header>
    <h1>DOPLAN ANNOTATION DASHBOARD</h1>
    <div class="subtitle">generated {stats['generated_at']}</div>
  </header>
  <div class="laneline"></div>

  <div class="section-title">Table I — Dataset Scale and Coverage Summary</div>
  <div class="scorecards">{table1_cards}</div>
  {coverage_section}
  <div class="section-title">Table II — Language Statistics</div>
  <div class="scorecards">{table2_cards}</div>

  <div class="section-title">Table III — Annotation Diversity and Overlap</div>
  <div class="scorecards">{table3_cards}</div>

  <div class="section-title">Figures</div>
  <div class="charts">{chart_divs}</div>

  {coverage_strip_section}
  {scene_viewer_section}

  <div class="section-title">Data quality</div>
  {quality_html}

  <footer>
    Definitions mirror doPlan Section IV. Durations assume {FPS} fps for every contributor.
    "Overlapping annotation pair" = same clip, intersecting [start_frame, end_frame].
    "Temporal overlap ratio" = intersection / union of frame ranges, averaged over overlapping pairs.
    "Instruction similarity for overlaps" = Jaccard similarity of lowercase word sets between the two
    instructions in an overlapping pair (both must have a non-blank label), averaged across such pairs --
    this is a lightweight stand-in since the paper doesn't specify its exact similarity method; swap in
    something else in instruction_similarity() if needed. These numbers are computed live from the current
    merged dataset, so they will differ from (and grow past) the paper's published snapshot of 1,108
    annotations / 3 annotators -- that's expected.
  </footer>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="./dashboard_output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df, dupes_dropped, skipped = merge_all(args.root)
    if df.empty:
        print("No valid CSVs found under", args.root)
        return

    manifest = load_manifest(args.root)
    stats = compute_stats(df, manifest)

    df.to_csv(os.path.join(args.out, "combined_annotations.csv"), index=False)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)

    if stats["unannotated_clips"]:
        with open(os.path.join(args.out, "unannotated_clips.csv"), "w") as f:
            f.write("clip_id\n")
            f.write("\n".join(stats["unannotated_clips"]))

    build_dashboard_html(df, stats, dupes_dropped, skipped, os.path.join(args.out, "dashboard.html"))

    print(json.dumps(stats, indent=2, default=str))
    print(f"\nWrote {os.path.join(args.out, 'dashboard.html')}")


if __name__ == "__main__":
    main()
