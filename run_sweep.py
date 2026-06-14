"""
run_sweep.py — orchestrate distill + evaluate across ranks, then plot.

Runs each (distill -> evaluate) as a SUBPROCESS so GPU memory is fully released
between ranks (a fresh process per rank avoids fragmentation on a T4). Designed
to finish under ~2h on a T4 with the small defaults below; this is a proof of
concept, not a SOTA run.

  python run_sweep.py                 # ranks 4,8,16,32 with small step counts
  python run_sweep.py --smoke-test    # 10 steps/rank end-to-end pipeline check
  python run_sweep.py --ranks 8 --steps 500
"""

import argparse
import subprocess
import sys
import time


def run(cmd):
    print(f"\n[sweep] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main(args):
    py = sys.executable
    t0 = time.time()
    common = ["--model-name", args.model_name, "--max-length", str(args.max_length)]

    for rank in args.ranks:
        ckpt = f"checkpoints/student_rank{rank}.pt"

        distill = [py, "distill.py", "--rank", str(rank), "--steps", str(args.steps),
                   "--output", ckpt, "--batch-size", str(args.batch_size),
                   "--grad-accum", str(args.grad_accum), "--lr", str(args.lr),
                   "--warmup-backbone-steps", str(args.warmup_backbone_steps),
                   "--max-blocks", str(args.train_blocks)] + common
        if args.fp16:
            distill.append("--fp16")
        if args.smoke_test:
            distill.append("--smoke-test")
        run(distill)

        evaluate = [py, "evaluate.py", "--checkpoint", ckpt, "--rank", str(rank),
                    "--results", args.results,
                    "--max-blocks", str(args.eval_blocks)] + common
        if args.smoke_test:
            evaluate.append("--smoke-test")
        run(evaluate)

    run([py, "plot.py", "--results", args.results, "--out", args.out])
    print(f"\n[sweep] done in {(time.time() - t0) / 60:.1f} min -> {args.out}")


def build_argparser():
    ap = argparse.ArgumentParser(description="Rank sweep orchestrator.")
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup-backbone-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--train-blocks", type=int, default=2000)
    ap.add_argument("--eval-blocks", type=int, default=200)
    ap.add_argument("--fp16", action="store_true",
                    help="fp16 weights (recommended on a T4)")
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="frontier.png")
    ap.add_argument("--smoke-test", action="store_true",
                    help="10 steps/rank on tiny data to verify the pipeline")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
