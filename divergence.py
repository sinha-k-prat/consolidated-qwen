"""
divergence.py — per-layer, per-class divergence from the shared mean (V1).

For a TRAINED consolidated student, how far does each layer's effective weight
sit from its class's shared backbone (the consolidated 'mean')? In V1 that
divergence is exactly the per-layer adapter delta: scale * (A @ B). This script
loads a checkpoint, computes the divergence for all 24 layers x 7 classes, and
writes:

  - divergence.json : abs/rel divergence norms per (layer, class) + per-class summary
  - divergence.png  : a 24 x 7 heatmap of the RELATIVE divergence
                      (||delta_i|| / ||backbone||), rows = layers, cols = classes

How to read it: bright rows/cols = layers/classes that diverge most from the
shared mean (the adapter is working hard there → that weight resists sharing).
Dim cells = layers whose weight is essentially the shared mean already. Classes
that stay uniformly dim are the best candidates for consolidation; bright ones
need a bigger rank (or shouldn't be shared).

  python divergence.py --checkpoint checkpoints/student_rank8.pt
"""
import argparse
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ConsolidatedQwen, WEIGHT_CLASSES


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt.get("rank", args.rank)
    student = ConsolidatedQwen(args.model_name, rank=rank,
                               alpha=ckpt.get("alpha", 16.0),
                               dtype=torch.float32)
    student.load_consolidated_state_dict(ckpt["state_dict"])

    div = student.divergence_from_mean()           # {class: {abs:[L], rel:[L]}}
    num_layers = student.num_layers

    summary = {cls: {
        "rel_mean": sum(div[cls]["rel"]) / num_layers,
        "rel_max": max(div[cls]["rel"]),
        "rel_argmax_layer": int(max(range(num_layers),
                                    key=lambda i: div[cls]["rel"][i])),
        "abs_mean": sum(div[cls]["abs"]) / num_layers,
    } for cls in WEIGHT_CLASSES}

    with open(args.out, "w") as f:
        json.dump({"rank": rank, "num_layers": num_layers,
                   "per_layer": div, "summary": summary}, f, indent=2)
    print(f"[divergence] wrote {args.out}")
    print("[divergence] per-class relative divergence from mean "
          "(mean over layers | max | layer-of-max):")
    for cls in WEIGHT_CLASSES:
        s = summary[cls]
        print(f"  {cls:10s} mean={s['rel_mean']:.4f}  max={s['rel_max']:.4f}  "
              f"@layer {s['rel_argmax_layer']}")

    # heatmap: rows = layers, cols = classes, color = relative divergence
    mat = torch.tensor([[div[cls]["rel"][i] for cls in WEIGHT_CLASSES]
                        for i in range(num_layers)])
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(WEIGHT_CLASSES)))
    ax.set_xticklabels(WEIGHT_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(num_layers), fontsize=7)
    ax.set_xlabel("weight class")
    ax.set_ylabel("decoder layer")
    ax.set_title(f"Per-layer divergence from shared mean (rank={rank})\n"
                 "||scale·A@B|| / ||backbone||")
    fig.colorbar(im, ax=ax, label="relative divergence")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"[divergence] wrote {args.plot}")


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Per-layer/per-class divergence from the shared mean (V1).")
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--checkpoint", required=True,
                    help="V1 consolidated student checkpoint (from distill.py)")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--out", default="divergence.json")
    ap.add_argument("--plot", default="divergence.png")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
