"""
build_dashboard.py  (placeholder version)

Ingestion/merge is fully functional -- it still consolidates every
annotator's CSV into one combined_annotations.csv, using the same
schema-normalization logic as before (folder/base_name handling, username
backfill, commentary shorthand mapping, dedup). That part isn't
"statistics", it's just data plumbing, so it's safe to keep running while
the actual dashboard content is still being decided.

The dashboard itself is intentionally left as an empty scaffold: the visual
theme/layout is in place, but every scorecard and chart is a placeholder
slot. Once you share the paper that defines exactly which statistics you
want, only compute_stats() / the chart section below need to be filled in --
everything else (merge logic, ingestion, GitHub Actions, Pages) stays as is.

USAGE:
    python build_dashboard.py --root /path/to/csvs --out ./dashboard_output
"""

import argparse
import glob
import os
from datetime import datetime

import pandas as pd

# ---------------- CONFIG ----------------
COMMENTARY_MAP = {
    "(a)": "Non-Referential",
    "(b)": "Static Referential",
    "(c)": "Dynamic Referential",
    "(d)": "Both",
}
REQUIRED_COLUMNS = {"folder", "start_frame", "end_frame", "label", "commentary"}

# ---------------- THEME (unchanged from before, kept so the real dashboard
# drops in later without a redesign) ----------------
BG = "#14161A"
PANEL = "#1E2128"
PANEL_BORDER = "#2B2F38"
AMBER = "#F2B705"
TEAL = "#4FD1C5"
TEXT = "#F5F3EE"
MUTED = "#9AA0AC"


# ---------------- LOAD / NORMALIZE (functional, unchanged) ----------------
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
    frames = []
    skipped = []
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
    return merged, dupes_dropped, skipped


# ---------------- PLACEHOLDER DASHBOARD ----------------
SCORECARD_SLOTS = 6
CHART_SLOTS = 4


def build_dashboard_html(row_count: int, file_count: int, dupes_dropped: int, skipped, out_path: str):
    scorecards = "".join(
        f"""<div class="card score placeholder">
              <div class="score-value">--</div>
              <div class="score-label">Statistic {i+1} — TBD</div>
            </div>"""
        for i in range(SCORECARD_SLOTS)
    )

    charts = "".join(
        f"""<div class="card chart placeholder">
              <div class="chart-placeholder-label">Chart {i+1} — awaiting spec</div>
            </div>"""
        for i in range(CHART_SLOTS)
    )

    skipped_html = (
        "".join(f'<div class="flag">{path}: {reason}</div>' for path, reason in skipped)
        if skipped else '<div class="flag ok">No files skipped.</div>'
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Annotation Dashboard (placeholder)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: {BG}; --panel: {PANEL}; --border: {PANEL_BORDER};
    --amber: {AMBER}; --teal: {TEAL}; --text: {TEXT}; --muted: {MUTED};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    margin: 0; padding: 32px 40px 60px;
  }}
  header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }}
  h1 {{ font-family: 'IBM Plex Mono', monospace; font-size: 22px; letter-spacing: 0.5px; margin: 0; }}
  .subtitle {{ color: var(--muted); font-size: 13px; }}
  .laneline {{
    height: 1px; margin: 20px 0 28px;
    background-image: repeating-linear-gradient(90deg, var(--amber) 0 18px, transparent 18px 34px);
    opacity: 0.8;
  }}
  .status {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--teal); margin-bottom: 24px; }}
  .scorecards {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; margin-bottom: 8px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }}
  .card.placeholder {{ border-style: dashed; opacity: 0.55; }}
  .score-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 600; color: var(--muted); }}
  .score-label {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 28px; }}
  .chart.placeholder {{ height: 300px; display: flex; align-items: center; justify-content: center; }}
  .chart-placeholder-label {{ color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
  .section-title {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--muted);
                     text-transform: uppercase; letter-spacing: 0.5px; margin: 32px 0 12px; }}
  .flag {{ background: rgba(79,209,197,0.10); border: 1px solid var(--teal); color: var(--text);
           border-radius: 8px; padding: 10px 14px; font-size: 13px; margin-bottom: 8px; }}
  footer {{ margin-top: 40px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
  @media (max-width: 900px) {{
    .scorecards {{ grid-template-columns: repeat(2, 1fr); }}
    .charts {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <header>
    <h1>ANNOTATION DASHBOARD</h1>
    <div class="subtitle">generated {datetime.now().isoformat(timespec='seconds')}</div>
  </header>
  <div class="laneline"></div>

  <div class="status">
    PIPELINE STATUS &mdash; merged {row_count:,} row(s) from {file_count} file(s),
    {dupes_dropped} duplicate(s) dropped. Statistics below are placeholders
    until the paper's metric definitions are added.
  </div>

  <div class="section-title">Scorecards (placeholder)</div>
  <div class="scorecards">{scorecards}</div>

  <div class="section-title">Charts (placeholder)</div>
  <div class="charts">{charts}</div>

  <div class="section-title">Files skipped during merge</div>
  {skipped_html}

  <footer>
    This is a scaffold. Once the statistics are defined, only compute_stats()
    and the chart-building section of build_dashboard.py need to change --
    the merge logic, theme, and layout stay the same.
  </footer>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder containing annotator CSV submissions")
    parser.add_argument("--out", default="./dashboard_output", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df, dupes_dropped, skipped = merge_all(args.root)

    file_count = len({p for p in df["source_file"]}) if not df.empty else 0
    df.to_csv(os.path.join(args.out, "combined_annotations.csv"), index=False)

    dashboard_path = os.path.join(args.out, "dashboard.html")
    build_dashboard_html(len(df), file_count, dupes_dropped, skipped, dashboard_path)

    print(f"Merged {len(df)} rows from {file_count} files ({dupes_dropped} dupes dropped).")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) -- see dashboard for details.")
    print(f"Wrote {dashboard_path}")


if __name__ == "__main__":
    main()
