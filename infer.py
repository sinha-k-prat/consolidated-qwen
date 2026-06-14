"""
infer.py — play with the consolidated student: pull from the HF Hub (or load a
local checkpoint) and generate text.

    python infer.py --repo-id <user>/consolidated-qwen-rank8 \
        --prompt "The key idea behind weight consolidation is"
    python infer.py --checkpoint checkpoints/student_rank8.pt --prompt "Hello"
"""
import argparse

import torch
from transformers import AutoTokenizer

from model import ConsolidatedQwen
from hub import pull


@torch.no_grad()
def generate(student, tok, prompt, max_new_tokens=128, temperature=0.8,
             top_p=0.95):
    device = next(student.parameters()).device
    ids = tok(prompt, return_tensors="pt").to(device)
    out = student.base.generate(
        **ids, max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, temperature=max(temperature, 1e-5),
        top_p=top_p, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def load_local(checkpoint, device, dtype=torch.float32):
    ckpt = torch.load(checkpoint, map_location="cpu")
    student = ConsolidatedQwen(ckpt["model_name"], rank=ckpt["rank"],
                               alpha=ckpt.get("alpha", 16.0),
                               dtype=dtype).to(device)
    student.load_consolidated_state_dict(ckpt["state_dict"])
    student.eval()
    return student, AutoTokenizer.from_pretrained(ckpt["model_name"])


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.repo_id:
        student, tok = pull(args.repo_id, device=device, token=args.token)
    elif args.checkpoint:
        student, tok = load_local(args.checkpoint, device)
    else:
        raise SystemExit("provide --repo-id or --checkpoint")
    print("=" * 60)
    print(generate(student, tok, args.prompt, args.max_new_tokens,
                   args.temperature, args.top_p))
    print("=" * 60)


def build_argparser():
    ap = argparse.ArgumentParser(description="Generate from the consolidated student.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-id", help="HF Hub repo to pull from")
    src.add_argument("--checkpoint", help="local consolidated checkpoint")
    ap.add_argument("--prompt", default="The key idea behind weight consolidation is")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--token", default=None)
    return ap


if __name__ == "__main__":
    main(build_argparser().parse_args())
