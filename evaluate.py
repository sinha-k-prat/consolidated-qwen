"""
evaluate.py — perplexity + storage size for the three models.

  (a) teacher       : original Qwen1.5-0.5B (fp16 storage estimate)
  (b) student        : consolidated (backbones + adapters) at a given rank
  (c) quantized_4bit : original via bitsandbytes nf4  (GPU only; gated import)

Sizes are reported as an on-disk STORAGE ESTIMATE (params x bytes/param), with a
printed breakdown. The tie_word_embeddings flag is read from config and the V x H
embedding/lm_head block is counted ONCE if tied, TWICE if not — this is the
single biggest number in the size budget, so it is read, not assumed.

Results are merged into a JSON file keyed by model so run_sweep.py can add one
rank at a time without recomputing the teacher / 4-bit baselines.
"""

import argparse
import json
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import ConsolidatedQwen, WEIGHT_CLASSES
from data import get_blocks

BYTES_FP16 = 2          # storage estimate basis for full-precision tensors
BYTES_NF4 = 0.5         # 4-bit weights pack to ~0.5 bytes/param


# --------------------------------------------------------------------------- #
# perplexity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def perplexity(model, blocks, device, batch_size=4):
    """Token-level perplexity over fixed-length blocks (teacher-forced)."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for start in range(0, blocks.size(0), batch_size):
        batch = blocks[start:start + batch_size].to(device)
        out = model(batch, labels=batch)
        # HF returns mean CE over (tokens - 1 per row). Re-weight to a running
        # sum so variable batch sizes don't bias the average.
        n_tok = (batch.size(1) - 1) * batch.size(0)
        total_nll += out.loss.item() * n_tok
        total_tok += n_tok
    return math.exp(total_nll / total_tok)


# --------------------------------------------------------------------------- #
# size accounting
# --------------------------------------------------------------------------- #
def _is_class_weight(name):
    return any(f"{c}.weight" in name for c in WEIGHT_CLASSES)


def _is_class_bias(name):
    return any(f"{c}.bias" in name for c in WEIGHT_CLASSES)


def teacher_size(teacher):
    """Full model stored at fp16. Counts every parameter once (named_parameters
    already collapses tied embeddings into a single tensor)."""
    params = sum(p.numel() for p in teacher.parameters())
    return params, params * BYTES_FP16 / 1e9


def student_size(teacher, student, config):
    """Consolidated storage = frozen shared block + backbones + adapters + biases.

    frozen shared block = everything that is NOT one of the 168 per-layer class
    weight matrices (i.e. embeddings, all norms, lm_head, and — counted here —
    the per-layer biases, which we then exclude to avoid double counting since
    the student stores them explicitly).
    """
    # frozen shared = teacher params excluding the 168 class weight matrices AND
    # the class biases (the student keeps its own copy of the biases).
    frozen_shared = sum(
        p.numel() for n, p in teacher.named_parameters()
        if not _is_class_weight(n) and not _is_class_bias(n))

    backbone_params = sum(student.backbones[c].numel() for c in WEIGHT_CLASSES)
    adapter_params, bias_params = 0, 0
    for layer in student.base.model.layers:
        for cls in WEIGHT_CLASSES:
            cl = student._get_linear(layer, cls)
            adapter_params += cl.A.numel() + cl.B.numel()
            if cl.bias is not None:
                bias_params += cl.bias.numel()

    total = frozen_shared + backbone_params + adapter_params + bias_params
    breakdown = {
        "frozen_shared_params": frozen_shared,
        "backbone_params": backbone_params,
        "adapter_params": adapter_params,
        "bias_params": bias_params,
        "tie_word_embeddings": bool(config.tie_word_embeddings),
    }
    return total, total * BYTES_FP16 / 1e9, breakdown


def quant_size(teacher):
    """nf4 estimate: the 168 class weight matrices pack to ~0.5 bytes/param;
    everything else (embeddings, norms, lm_head, biases) stays fp16."""
    quant_params, full_params = 0, 0
    for n, p in teacher.named_parameters():
        if _is_class_weight(n):
            quant_params += p.numel()
        else:
            full_params += p.numel()
    gb = (quant_params * BYTES_NF4 + full_params * BYTES_FP16) / 1e9
    return quant_params + full_params, gb


# --------------------------------------------------------------------------- #
# bitsandbytes (gated)
# --------------------------------------------------------------------------- #
def load_4bit(model_name):
    """Returns (model, footprint_gb) or (None, None) if bnb/GPU unavailable."""
    if not torch.cuda.is_available():
        print("[eval] no CUDA -> skipping 4-bit baseline")
        return None, None
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401  (import gate)
    except Exception as e:
        print(f"[eval] bitsandbytes unavailable ({e}) -> skipping 4-bit baseline")
        return None, None
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=qc, device_map="auto")
    return model, model.get_memory_footprint() / 1e9


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    max_blocks = 16 if args.smoke_test else args.max_blocks
    blocks = get_blocks(tokenizer, split="test", max_length=args.max_length,
                        max_blocks=max_blocks)
    print(f"[eval] test blocks = {blocks.size(0)} x {args.max_length}")

    results = {}
    if os.path.exists(args.results):
        with open(args.results) as f:
            results = json.load(f)

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16).to(device)
    config = teacher.config

    # ---- baselines (teacher + 4-bit): compute once, reuse across ranks ----
    if "teacher" not in results or args.force_baselines:
        params, gb = teacher_size(teacher)
        ppl = perplexity(teacher, blocks, device, args.eval_batch_size)
        results["teacher"] = {"ppl": ppl, "size_gb": gb, "params": params}
        print(f"[eval] teacher: ppl={ppl:.3f} size={gb:.3f}GB params={params:,}")

    if "quantized_4bit" not in results or args.force_baselines:
        q_model, foot_gb = load_4bit(args.model_name)
        est_params, est_gb = quant_size(teacher)
        if q_model is not None:
            ppl = perplexity(q_model, blocks, device, args.eval_batch_size)
            gb = foot_gb if foot_gb is not None else est_gb
            results["quantized_4bit"] = {"ppl": ppl, "size_gb": gb,
                                         "params": est_params}
            print(f"[eval] 4-bit nf4: ppl={ppl:.3f} size={gb:.3f}GB")
            del q_model
            torch.cuda.empty_cache()
        else:
            print("[eval] 4-bit baseline skipped (recorded estimate only)")
            results.setdefault("quantized_4bit_estimate",
                               {"size_gb": est_gb, "params": est_params})

    # ---- student at this rank --------------------------------------------
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        rank = ckpt.get("rank", args.rank)
        student = ConsolidatedQwen(args.model_name, rank=rank,
                                   alpha=ckpt.get("alpha", 16.0),
                                   dtype=torch.float16).to(device)
        student.load_consolidated_state_dict(ckpt["state_dict"])
        params, gb, breakdown = student_size(teacher, student, config)
        ppl = perplexity(student, blocks, device, args.eval_batch_size)
        results[f"consolidated_rank{rank}"] = {
            "ppl": ppl, "size_gb": gb, "params": params,
            "rank": rank, "breakdown": breakdown}
        print(f"[eval] consolidated r{rank}: ppl={ppl:.3f} size={gb:.3f}GB "
              f"params={params:,}")
        print(f"[eval]   breakdown: {breakdown}")

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {args.results}")


def build_argparser():
    ap = argparse.ArgumentParser(description="Evaluate PPL + size.")
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--checkpoint", default=None,
                    help="consolidated student checkpoint to evaluate")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=200,
                    help="cap test blocks (proof of concept)")
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--force-baselines", action="store_true",
                    help="recompute teacher + 4-bit even if present in JSON")
    ap.add_argument("--smoke-test", action="store_true")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
