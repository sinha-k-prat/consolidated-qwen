"""
distill_v2.py — convergence measurement.

Loss = KD(student||teacher) + hidden-anchor MSE + lambda(t) * consolidation
where lambda is ANNEALED from 0 upward: early on the weights stay functional and
adapters establish; later the penalty tightens and we see how far the per-class
weights collapse toward their mean. Logs per-class spread every --spread-every
optimizer steps to spread_history.json for plotting.

Key difference from V1: there is no shared backbone. We measure convergence;
we do not impose it.
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from model_v2 import ConvergenceQwen, WEIGHT_CLASSES
from distill import kd_loss, hidden_loss   # reuse V1's losses
from data import get_blocks, iter_batches


def lambda_schedule(opt_step, total, max_lambda, warmup_frac=0.3):
    """0 until warmup_frac of training, then linearly ramp to max_lambda.
    Keeps early training penalty-free so adapters settle before consolidation."""
    start = int(total * warmup_frac)
    if opt_step < start:
        return 0.0
    return max_lambda * (opt_step - start) / max(1, total - start)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (args.fp16 and device == "cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_name)

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = ConvergenceQwen(args.model_name, rank=args.rank,
                              alpha=args.alpha, dtype=dtype).to(device)
    student.train()
    print(f"[v2] trainable params = {student.num_trainable():,}  "
          f"(note: full per-layer weights ARE trainable here)")

    max_blocks = 64 if args.smoke_test else args.max_blocks
    blocks = get_blocks(tok, split="train",
                        max_length=args.max_length, max_blocks=max_blocks)

    optim = torch.optim.AdamW(student.trainable_parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
    total = 10 if args.smoke_test else args.steps
    grad_accum = max(1, args.grad_accum)
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=max(1, total // 20), num_training_steps=total)

    history = []   # list of {step, lambda, spread:{class:val}}
    opt_step, micro, t0 = 0, 0, time.time()
    optim.zero_grad(set_to_none=True)
    done = False

    # record the spread at init (step 0) — the baseline before any training
    history.append({"step": 0, "lambda": 0.0,
                    "spread": student.spread_per_class()})

    while not done:
        for batch in iter_batches(blocks, args.batch_size, shuffle=True,
                                  seed=args.seed + opt_step):
            batch = batch.to(device)
            with torch.no_grad():
                t_out = teacher(batch, output_hidden_states=True)
            s_out = student(batch, output_hidden_states=True)

            loss_kd = kd_loss(s_out.logits.float(), t_out.logits.float(),
                              args.temperature)
            loss_h = hidden_loss([h.float() for h in s_out.hidden_states],
                                 [h.float() for h in t_out.hidden_states])
            lam = lambda_schedule(opt_step, total, args.max_lambda,
                                  args.lambda_warmup_frac)
            loss_c = student.consolidation_penalty() if lam > 0 else 0.0
            loss = loss_kd + args.hidden_weight * loss_h + lam * loss_c
            (loss / grad_accum).backward()
            micro += 1

            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
                                               args.max_grad_norm)
                optim.step(); sched.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.spread_every == 0:
                    sp = student.spread_per_class()
                    history.append({"step": opt_step, "lambda": lam, "spread": sp})
                    avg = sum(sp.values()) / len(sp)
                    print(f"[v2] step {opt_step:4d}/{total} loss={loss.item():.4f} "
                          f"lam={lam:.3e} avg_spread={avg:.4f}")
                if opt_step >= total:
                    done = True; break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.spread_out, "w") as f:
        json.dump({"rank": args.rank, "max_lambda": args.max_lambda,
                   "history": history}, f, indent=2)
    print(f"[v2] spread history -> {args.spread_out}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=400, help="optimizer steps")
    ap.add_argument("--max-lambda", type=float, default=1.0,
                    help="peak consolidation penalty weight (sweep this)")
    ap.add_argument("--lambda-warmup-frac", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--hidden-weight", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=2000)
    ap.add_argument("--spread-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--output", default="checkpoints/v2_rank8.pt")
    ap.add_argument("--spread-out", default="spread_history.json")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    train(args)
