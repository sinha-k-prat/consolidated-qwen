"""
class_spread.py — TRAINING-FREE measure of how layer-specific each weight class
is in the PRETRAINED teacher.

For each class c, stack its 24 real per-layer matrices, compute the mean, and
report how far each layer sits from that mean — absolute (||W_i - mean||) and
relative (/ ||mean||). This is the ground-truth version of the divergence
heatmap: no distillation, no adapters, no under-training confound. It also
prints ||mean|| per class so you can tell whether a high RELATIVE spread is real
or just a small-mean-norm artifact.

Caveat it cannot remove: attention scores depend only on W_q @ W_k^T, so the
split of layer-specificity between q_proj and k_proj is gauge-dependent. Read
"q/k are layer-specific", not "query uniquely".

    python class_spread.py            # writes class_spread.json (+ heatmap)
"""
import argparse
import json

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

from model import WEIGHT_CLASSES, _ATTN_CLASSES


def _weight(layer, cls):
    owner = layer.self_attn if cls in _ATTN_CLASSES else layer.mlp
    return getattr(owner, cls).weight


def main(args):
    m = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             torch_dtype=torch.float32)
    layers = m.model.layers
    L = len(layers)

    summary, per_layer = {}, {}
    for cls in WEIGHT_CLASSES:
        W = torch.stack([_weight(layers[i], cls) for i in range(L)], dim=0)
        mean = W.mean(dim=0, keepdim=True)
        mean_norm = mean.norm().item()
        absd = (W - mean).flatten(1).norm(dim=1)            # (L,)
        rel = absd / (mean_norm + 1e-12)
        per_layer[cls] = {"abs": absd.tolist(), "rel": rel.tolist()}
        summary[cls] = {"rel_mean": rel.mean().item(),
                        "abs_mean": absd.mean().item(),
                        "mean_norm": mean_norm}

    order = sorted(WEIGHT_CLASSES, key=lambda c: -summary[c]["rel_mean"])
    print("Layer-specificity of the PRETRAINED teacher (training-free):")
    print(f"  {'class':10s} {'rel_spread':>10s} {'abs_spread':>10s} {'||mean||':>10s}")
    for c in order:
        s = summary[c]
        print(f"  {c:10s} {s['rel_mean']:>10.4f} {s['abs_mean']:>10.2f} "
              f"{s['mean_norm']:>10.2f}")
    print(f"  most layer-specific -> least: {order}")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_layer": per_layer,
                   "ranking_by_rel_spread": order}, f, indent=2)
    print(f"[class_spread] wrote {args.out}")

    # heatmap of RELATIVE spread (rows = layers, cols = classes)
    mat = torch.tensor([[per_layer[c]["rel"][i] for c in WEIGHT_CLASSES]
                        for i in range(L)])
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(WEIGHT_CLASSES)))
    ax.set_xticklabels(WEIGHT_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(L)); ax.set_yticklabels(range(L), fontsize=7)
    ax.set_xlabel("weight class"); ax.set_ylabel("decoder layer")
    ax.set_title("Pretrained-teacher layer-specificity (training-free)\n"
                 "||W_i - mean|| / ||mean||")
    fig.colorbar(im, ax=ax, label="relative spread")
    fig.tight_layout(); fig.savefig(args.plot, dpi=150)
    print(f"[class_spread] wrote {args.plot}")


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Training-free per-class layer-specificity of the teacher.")
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--out", default="class_spread.json")
    ap.add_argument("--plot", default="class_spread.png")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
