"""
compare_ce.py — held-out cross-entropy: teacher vs consolidated V1 student.

Scores PER-SEQUENCE next-token CE on N held-out sequences (wikitext-2 test) for:
  (a) the original teacher Qwen1.5-0.5B
  (b) the consolidated V1 student (ONE shared per-class mean backbone + per-layer LoRA)

Also VERIFIES the structural claim: for each class, all 24 layers reference the
exact same backbone tensor (by id) — i.e. the same-class weights are really
identical, not approximately. "Replace by the mean" is built in for V1.

Outputs raw CE (not perplexity), per sequence and aggregated, to JSON.

  python compare_ce.py --checkpoint checkpoints/student_rank8.pt --num-seqs 100
"""
import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import ConsolidatedQwen, WEIGHT_CLASSES
from data import get_blocks


def verify_shared_backbone(student) -> dict:
    """Assert all 24 layers of each class point at the SAME backbone tensor."""
    report = {}
    for cls in WEIGHT_CLASSES:
        layer_ids = {id(student._get_linear(l, cls).backbone)
                     for l in student.base.model.layers}
        shared = (len(layer_ids) == 1 and
                  next(iter(layer_ids)) == id(student.backbones[cls]))
        report[cls] = shared
        assert shared, f"class {cls}: backbone NOT uniquely shared across layers"
    return report


@torch.no_grad()
def per_seq_ce(model, blocks, device):
    """List of per-sequence mean next-token CE over the held-out set."""
    model.eval()
    out = []
    for i in range(blocks.size(0)):
        ids = blocks[i:i + 1].to(device)
        # HF shifts internally and returns mean CE over the predicted tokens of
        # this single sequence.
        out.append(model(ids, labels=ids).loss.item())
    return out


def summarize(ce):
    t = torch.tensor(ce)
    return {"mean_ce": t.mean().item(),
            "std_ce": t.std(unbiased=False).item(),
            "ppl": math.exp(t.mean().item()),
            "n": len(ce)}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    blocks = get_blocks(tok, split="test", max_length=args.max_length,
                        max_blocks=args.num_seqs)
    print(f"[ce] held-out sequences = {blocks.size(0)} x {args.max_length}")

    # ---- teacher --------------------------------------------------------
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32).to(device)
    t_ce = per_seq_ce(teacher, blocks, device)
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- consolidated V1 student ---------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt.get("rank", args.rank)
    student = ConsolidatedQwen(args.model_name, rank=rank,
                               alpha=ckpt.get("alpha", 16.0),
                               dtype=torch.float32).to(device)
    student.load_consolidated_state_dict(ckpt["state_dict"])
    shared = verify_shared_backbone(student)
    print(f"[ce] shared-backbone verified (all classes): {all(shared.values())}")
    s_ce = per_seq_ce(student, blocks, device)

    deltas = [s - t for s, t in zip(s_ce, t_ce)]
    results = {
        "model_name": args.model_name, "rank": rank,
        "num_seqs": blocks.size(0), "max_length": args.max_length,
        "shared_backbone_verified": all(shared.values()),
        "teacher": summarize(t_ce),
        "consolidated": summarize(s_ce),
        "mean_ce_gap_student_minus_teacher": sum(deltas) / len(deltas),
        "win_rate_student_lower_ce": sum(d < 0 for d in deltas) / len(deltas),
        "per_seq": [{"i": i, "teacher_ce": t_ce[i], "student_ce": s_ce[i],
                     "delta": deltas[i]} for i in range(len(deltas))],
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[ce] teacher      mean CE = {results['teacher']['mean_ce']:.4f} "
          f"(ppl {results['teacher']['ppl']:.3f})")
    print(f"[ce] consolidated mean CE = {results['consolidated']['mean_ce']:.4f} "
          f"(ppl {results['consolidated']['ppl']:.3f})")
    print(f"[ce] mean CE gap (student - teacher) = "
          f"{results['mean_ce_gap_student_minus_teacher']:+.4f}")
    print(f"[ce] wrote {args.out}")


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Held-out CE: teacher vs consolidated V1 student.")
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--checkpoint", required=True,
                    help="V1 consolidated student checkpoint (from distill.py)")
    ap.add_argument("--rank", type=int, default=8,
                    help="fallback rank if not stored in the checkpoint")
    ap.add_argument("--num-seqs", type=int, default=100,
                    help="number of held-out test sequences to score")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default="ce_compare.json")
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
