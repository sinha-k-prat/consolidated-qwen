"""
plot.py — the size/quality frontier.

Reads results.json and scatters perplexity (y, lower = better) vs storage size
in GB (x). Labels: teacher, 4-bit nf4, and consolidated-rank-r for each rank.

How to read it (see README): the question is NOT "is the student as good as the
teacher" (it won't be). It is "at a given size, does a consolidated point sit
BELOW (lower ppl) and/or LEFT OF (smaller) the 4-bit baseline?" If a rank lands
left of and near the 4-bit dot, that's the promising signal.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(args):
    with open(args.results) as f:
        results = json.load(f)

    points = []   # (size_gb, ppl, label, style)
    for key, v in results.items():
        if not isinstance(v, dict) or "ppl" not in v or "size_gb" not in v:
            continue
        if key == "teacher":
            label, style = "teacher (fp16)", dict(color="black", marker="*", s=220)
        elif key == "quantized_4bit":
            label, style = "4-bit nf4", dict(color="crimson", marker="s", s=90)
        elif key.startswith("consolidated_rank"):
            label = f"consolidated r{v.get('rank', key.split('rank')[-1])}"
            style = dict(color="royalblue", marker="o", s=90)
        else:
            continue
        points.append((v["size_gb"], v["ppl"], label, style))

    if not points:
        raise SystemExit("no plottable points in results.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    for x, y, label, style in points:
        ax.scatter([x], [y], label=label, edgecolors="white", linewidths=0.5, **style)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4),
                    fontsize=9)

    ax.set_xlabel("storage size (GB)")
    ax.set_ylabel("perplexity on wikitext-2 test  (lower = better)")
    ax.set_title("Consolidation vs 4-bit: size / quality frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="frontier.png")
    main(ap.parse_args())
