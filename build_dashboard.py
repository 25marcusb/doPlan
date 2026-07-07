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


def merge_all(root_dir: str):
    frames, skipped = [], []
    for path in glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True):
        if os.path.basename(path).startswith("_"):
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


# ---------------- STATS ----------------
def compute_stats(df: pd.DataFrame) -> dict:
    blank = _is_blank(df["label"])
    labeled = df[~blank].copy()

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

    # ---- Fig 6 + Table III: diversity and overlap ----
    per_clip_counts = df.groupby("folder").size()
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

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "referential_distribution": referential_distribution,
        "unmapped_commentary": unmapped_commentary,
        "length_by_class": length_by_class,
        "per_clip_counts": per_clip_counts.value_counts().sort_index().to_dict(),
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
    fig.add_vline(x=mean_v, line_dash="dash", line_color=AMBER, annotation_text=f"Mean: {mean_v:.1f}s")
    fig.add_vline(x=median_v, line_dash="dot", line_color=CORAL, annotation_text=f"Median: {median_v:.1f}s")
    fig.update_layout(**_base_layout("Fig 3 — Segment Duration Distribution"))
    fig.update_xaxes(title="Segment duration (s)")
    fig.update_yaxes(title="Number of annotations")
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

    charts = [
        chart_duration_distribution(df, stats),
        chart_referential_distribution(stats),
        chart_length_by_class(stats),
        chart_annotations_per_clip(stats),
    ]
    chart_divs = "".join(f'<div class="card chart">{_fig_div(c)}</div>' for c in charts)

    quality_notes = []
    if stats["unmapped_commentary"]:
        quality_notes.append(
            f"Unrecognized commentary values (not in COMMENTARY_MAP): {stats['unmapped_commentary']}"
        )
    if dupes_dropped:
        quality_notes.append(f"{dupes_dropped} duplicate row(s) dropped during merge.")
    if skipped:
        quality_notes.append(f"{len(skipped)} file(s) failed to parse and were skipped.")
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

  <div class="section-title">Table II — Language Statistics</div>
  <div class="scorecards">{table2_cards}</div>

  <div class="section-title">Table III — Annotation Diversity and Overlap</div>
  <div class="scorecards">{table3_cards}</div>

  <div class="section-title">Figures</div>
  <div class="charts">{chart_divs}</div>

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

    stats = compute_stats(df)

    df.to_csv(os.path.join(args.out, "combined_annotations.csv"), index=False)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=str)

    build_dashboard_html(df, stats, dupes_dropped, skipped, os.path.join(args.out, "dashboard.html"))

    print(json.dumps(stats, indent=2, default=str))
    print(f"\nWrote {os.path.join(args.out, 'dashboard.html')}")


if __name__ == "__main__":
    main()
