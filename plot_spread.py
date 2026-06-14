"""
plot_spread.py — visualize V2 convergence.

Reads spread_history.json (written by distill_v2.py) and plots per-class spread
(mean L2 distance of each layer's weight from its class mean) vs optimizer step,
with the annealed lambda overlaid on a secondary axis.

How to read it (see README "V2"): does the spread COLLAPSE toward zero as lambda
ramps (same-class layers genuinely converge — hypothesis holds), PLATEAU partway
(the plateau height is the irreducible per-layer identity), or barely move
(layers resist sharing at this rank)? Watch which classes collapse (likely
v_proj/o_proj) vs resist (likely the MLP matrices).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(args):
    with open(args.spread) as f:
        data = json.load(f)
    history = data["history"]
    if not history:
        raise SystemExit("empty spread history")

    steps = [h["step"] for h in history]
    lambdas = [h["lambda"] for h in history]
    classes = list(history[0]["spread"].keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes:
        ys = [h["spread"][cls] for h in history]
        ax.plot(steps, ys, marker="o", markersize=3, label=cls)

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("per-class spread  (mean ||W_i - mean||, lower = converged)")
    ax.set_title(f"V2 convergence: per-class spread (rank={data.get('rank')}, "
                 f"max_lambda={data.get('max_lambda')})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # lambda on a secondary axis so you can see the penalty ramp.
    ax2 = ax.twinx()
    ax2.plot(steps, lambdas, color="black", linestyle="--", alpha=0.5,
             label="lambda")
    ax2.set_ylabel("consolidation lambda(t)")
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot_spread] wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread", default="spread_history.json")
    ap.add_argument("--out", default="spread.png")
    main(ap.parse_args())
