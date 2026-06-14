"""
distill.py — knowledge distillation of the consolidated student.

Teacher = frozen original Qwen1.5-0.5B.
Student = ConsolidatedQwen (shared backbones + per-layer rank-r adapters).

Loss = KL(student || teacher) on temperature-scaled logits
     + HIDDEN_WEIGHT * MSE on hidden states at layers 8 and 16 (intermediate
       anchors that keep the student's internal representations aligned).

Two training guards (the student starts at the per-class mean, so early
dynamics matter):
  1. Backbone warm-up freeze: for the first --warmup-backbone-steps we train
     ONLY the adapters. This stops large early gradients from drifting the
     shared backbone before the adapters have established themselves.
  2. Lower backbone LR: the backbone is shared across 24 layers and accumulates
     ~24x the gradient signal, so it gets --backbone-lr-mult x the adapter LR
     (separate AdamW parameter groups).

Only backbones + adapters (+ q/k/v biases) are trained; the teacher is frozen.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from model import ConsolidatedQwen
from data import get_blocks, iter_batches

# Intermediate-anchor layers for hidden-state MSE. hidden_states[k] is the
# output of decoder layer k (hidden_states[0] is the embedding output).
ANCHOR_LAYERS = (8, 16)


def kd_loss(student_logits, teacher_logits, temperature):
    """Temperature-scaled KL divergence, averaged over all tokens."""
    T = temperature
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    # batchmean over the flattened (batch*seq) token axis; * T^2 keeps gradient
    # magnitude comparable across temperatures (Hinton et al.).
    s = s.view(-1, s.size(-1))
    t = t.view(-1, t.size(-1))
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def hidden_loss(student_hidden, teacher_hidden):
    """MSE between student and teacher hidden states at the anchor layers."""
    loss = 0.0
    for k in ANCHOR_LAYERS:
        loss = loss + F.mse_loss(student_hidden[k], teacher_hidden[k])
    return loss / len(ANCHOR_LAYERS)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (args.fp16 and device == "cuda") else torch.float32
    print(f"[distill] device={device} dtype={dtype} rank={args.rank} "
          f"opt_steps={args.steps} grad_accum={args.grad_accum} "
          f"smoke_test={args.smoke_test}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- teacher: frozen, eval -------------------------------------------
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- student: consolidated -------------------------------------------
    student = ConsolidatedQwen(args.model_name, rank=args.rank,
                               alpha=args.alpha, dtype=dtype).to(device)
    student.train()
    print(f"[distill] trainable params = {student.num_trainable():,}")

    # ---- data -------------------------------------------------------------
    max_blocks = 64 if args.smoke_test else args.max_blocks
    blocks = get_blocks(tokenizer, split="train",
                        max_length=args.max_length, max_blocks=max_blocks)
    print(f"[distill] calibration blocks = {blocks.size(0)} x {args.max_length} tokens")

    # ---- optimizer: two groups (adapters, backbones) ----------------------
    optim = torch.optim.AdamW(
        student.param_groups(args.lr, backbone_lr_mult=args.backbone_lr_mult),
        weight_decay=args.weight_decay)

    # NB: args.steps now means OPTIMIZER steps (real weight updates), not
    # micro-batches. One optimizer step = grad_accum micro-batches.
    total_opt_steps = 10 if args.smoke_test else args.steps
    grad_accum = max(1, args.grad_accum)
    warmup_backbone = (3 if args.smoke_test
                       else min(args.warmup_backbone_steps, total_opt_steps // 2))
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=max(1, total_opt_steps // 20),
        num_training_steps=total_opt_steps)

    # Guard #1: freeze the backbone for the warm-up window (train adapters only).
    student.set_backbone_requires_grad(False)
    backbone_frozen = True
    print(f"[distill] backbone frozen for first {warmup_backbone} optimizer steps")

    opt_step = 0          # counts real weight updates
    micro = 0             # counts micro-batches within the current accumulation
    t0 = time.time()
    optim.zero_grad(set_to_none=True)

    done = False
    while not done:
        for batch in iter_batches(blocks, args.batch_size, shuffle=True,
                                  seed=args.seed + opt_step):
            batch = batch.to(device)

            with torch.no_grad():
                t_out = teacher(batch, output_hidden_states=True)
            s_out = student(batch, output_hidden_states=True)

            loss_kd = kd_loss(s_out.logits.float(), t_out.logits.float(),
                              args.temperature)
            loss_h = hidden_loss(
                [h.float() for h in s_out.hidden_states],
                [h.float() for h in t_out.hidden_states])
            loss = loss_kd + args.hidden_weight * loss_h
            (loss / grad_accum).backward()
            micro += 1

            # Only take an optimizer step once grad_accum micro-batches are in.
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
                                               args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1

                # Guard #1 release: unfreeze backbone after the warm-up window.
                if backbone_frozen and opt_step >= warmup_backbone:
                    student.set_backbone_requires_grad(True)
                    backbone_frozen = False
                    print(f"[distill] opt_step {opt_step}: backbone UNFROZEN")

                if opt_step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"[distill] opt_step {opt_step:4d}/{total_opt_steps} "
                          f"loss={loss.item():.4f} (kd={loss_kd.item():.4f} "
                          f"hidden={loss_h.item():.4f}) "
                          f"lr={sched.get_last_lr()[0]:.2e} {dt:.1f}s")

                if opt_step >= total_opt_steps:
                    done = True
                    break

    # ---- save the cheap learned tensors -----------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "rank": args.rank,
        "alpha": args.alpha,
        "model_name": args.model_name,
        "state_dict": student.consolidated_state_dict(),
    }, args.output)
    print(f"[distill] saved student -> {args.output} "
          f"({time.time() - t0:.1f}s total)")


def build_argparser():
    ap = argparse.ArgumentParser(description="Distill a consolidated Qwen student.")
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=400,
                    help="OPTIMIZER steps (real weight updates); "
                         "one step = grad_accum micro-batches")
    ap.add_argument("--warmup-backbone-steps", type=int, default=150,
                    help="freeze backbone (optimizer steps), adapters only")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="backbone LR = this x adapter LR")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--hidden-weight", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=2000,
                    help="cap calibration blocks streamed from wikitext-2")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16", action="store_true",
                    help="use fp16 weights on GPU (saves memory on a T4)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run 10 steps on tiny data to verify the pipeline")
    ap.add_argument("--output", default="checkpoints/student_rank8.pt")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    train(args)
