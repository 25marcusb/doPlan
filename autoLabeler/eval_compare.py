"""Compare AI label sets against the human labels and against each other.

Reads the human labels.csv plus one or more AI label files (row-aligned by
folder/start/end) and reports, for each AI set: instruction word-similarity,
maneuver-type agreement, referential-class accuracy, and nothing-to-annotate
agreement. Writes a side-by-side CSV for eyeballing.

    python eval_compare.py                       # human vs AILabels.csv vs AILabels_map.csv
    python eval_compare.py labels.csv a.csv b.csv
"""

import csv
import os
import re
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DEFAULTS = [
    "/home/cvrr/Desktop/VLM Competition/doPlan/outputs/labels.csv",
    "/home/cvrr/Desktop/VLM Competition/doPlan/outputs/AILabels.csv",
    "/home/cvrr/Desktop/VLM Competition/doPlan/outputs/AILabels_map.csv",
]
_WORD = re.compile(r"[a-z']+")


def norm(s):
    return str(s or "").strip()


def is_empty(s):
    return norm(s) == ""


def jaccard(a, b):
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def maneuver(text):
    """Coarse maneuver bucket for a fairer semantic match than raw word overlap."""
    t = text.lower()
    if is_empty(t):
        return "none"
    if "u-turn" in t or "u turn" in t:
        return "uturn"
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    if any(k in t for k in ["lane", "merge"]):
        return "lane"
    if any(k in t for k in ["follow", "behind"]):
        return "follow"
    if any(k in t for k in ["stop", "wait", "yield", "pedestrian", "crosswalk", "red light"]):
        return "stop"
    if any(k in t for k in ["pull over", "park", "pull up"]):
        return "pullover"
    if any(k in t for k in ["straight", "continue", "keep going", "through", "forward"]):
        return "straight"
    return "other"


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def key(r):
    return (r["folder"], norm(r["start_frame"]), norm(r["end_frame"]))


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULTS
    human_path, ai_paths = files[0], files[1:]
    human = read_rows(human_path)
    ai_sets = {os.path.basename(p): read_rows(p) for p in ai_paths}

    n = len(human)
    print(f"human: {human_path} ({n} rows)")
    for name, rows in ai_sets.items():
        assert len(rows) == n, f"{name} has {len(rows)} rows, expected {n}"
        # verify alignment
        mismatch = sum(1 for i in range(n) if key(rows[i]) != key(human[i]))
        print(f"  {name}: {len(rows)} rows, {mismatch} key-misaligned")

    side = []
    stats = {name: dict(cmp=0, jac=0.0, man_match=0, class_ok=0, class_total=0,
                        human_empty=0, both_empty=0, ai_empty=0) for name in ai_sets}

    for i in range(n):
        h = human[i]
        h_label, h_class = norm(h["label"]), norm(h["commentary"])
        # skip rows whose clip wasn't available (AI forced empty for ALL sets)
        clip_available = any(not is_empty(ai_sets[name][i]["label"]) for name in ai_sets) \
            or not is_empty(h_label)
        row = {"folder": h["folder"], "start": h["start_frame"], "end": h["end_frame"],
               "human_label": h_label, "human_class": h_class, "human_man": maneuver(h_label)}
        for name, rows in ai_sets.items():
            a = rows[i]
            a_label, a_class = norm(a["label"]), norm(a["commentary"])
            row[f"{name}_label"] = a_label
            row[f"{name}_class"] = a_class
            row[f"{name}_man"] = maneuver(a_label)

            s = stats[name]
            if is_empty(a_label) and is_empty(h_label) and clip_available:
                # both said "nothing"
                pass
            s["cmp"] += 1
            s["jac"] += jaccard(h_label, a_label)
            if maneuver(h_label) == maneuver(a_label):
                s["man_match"] += 1
            if not is_empty(h_class) and not is_empty(a_class):
                s["class_total"] += 1
                if h_class.replace("-", " ") == a_class.replace("-", " "):
                    s["class_ok"] += 1
            if is_empty(h_label):
                s["human_empty"] += 1
                if is_empty(a_label):
                    s["both_empty"] += 1
            if is_empty(a_label):
                s["ai_empty"] += 1
        side.append(row)

    out = os.path.join(OUT_DIR, "side_by_side.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(side[0].keys()))
        w.writeheader()
        w.writerows(side)

    print("\n" + "=" * 74)
    print(f"{'metric':32s} " + "  ".join(f"{name:>16s}" for name in ai_sets))
    print("-" * 74)

    def line(label, fn):
        print(f"{label:32s} " + "  ".join(f"{fn(stats[name]):>16s}" for name in ai_sets))

    line("rows compared", lambda s: f"{s['cmp']}")
    line("avg instruction Jaccard", lambda s: f"{s['jac'] / s['cmp']:.3f}")
    line("maneuver-type match", lambda s: f"{s['man_match']}/{s['cmp']} ({s['man_match'] / s['cmp']:.0%})")
    line("referential-class acc", lambda s: (f"{s['class_ok']}/{s['class_total']} "
                                             f"({s['class_ok'] / s['class_total']:.0%})") if s['class_total'] else "n/a")
    line("human said 'nothing'", lambda s: f"{s['human_empty']}")
    line("  AI agreed 'nothing'", lambda s: f"{s['both_empty']}/{s['human_empty']}" if s['human_empty'] else "n/a")
    line("AI said 'nothing' total", lambda s: f"{s['ai_empty']}")
    print("=" * 74)
    print(f"\nside-by-side written -> {out}")


if __name__ == "__main__":
    main()
