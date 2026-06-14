"""
hub.py — push / pull the consolidated student to the Hugging Face Hub.

We store ONLY the cheap consolidated tensors (7 shared per-class backbones +
per-layer LoRA adapters + q/k/v biases, ~33 MB at fp16) — NOT the full model.
On pull, the frozen base Qwen is re-downloaded from its original repo and the
consolidated tensors are loaded on top, so the Hub artifact stays tiny.

Push (needs a write token — `huggingface-cli login` or notebook_login()):
    python hub.py push --checkpoint checkpoints/student_rank8.pt \
        --repo-id <your-hf-user>/consolidated-qwen-rank8

Pull (in code):
    from hub import pull
    model, tok = pull("<your-hf-user>/consolidated-qwen-rank8")
"""
import argparse

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from model import ConsolidatedQwen

CKPT_FILENAME = "consolidated_student.pt"


def _model_card(ckpt, repo_id):
    rank = ckpt.get("rank")
    base = ckpt.get("model_name")
    return f"""---
library_name: pytorch
tags:
- qwen2
- consolidation
- lora
- distillation
- research-prototype
---

# Consolidated Qwen1.5-0.5B (rank {rank})

Consolidation + per-class LoRA compression of `{base}`. This artifact stores only
the cheap learned tensors (7 shared per-class backbone matrices + per-layer
rank-{rank} LoRA adapters + q/k/v biases). The frozen base model is pulled from
`{base}` at load time.

```python
from hub import pull          # from github.com/sinha-k-prat/consolidated-qwen
model, tok = pull("{repo_id}")
ids = tok("Weight consolidation works by", return_tensors="pt")
print(tok.decode(model.base.generate(**ids, max_new_tokens=60)[0]))
```

Research prototype — https://github.com/sinha-k-prat/consolidated-qwen
"""


def push(checkpoint, repo_id, private=False, token=None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=checkpoint, path_in_repo=CKPT_FILENAME,
                    repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=_model_card(ckpt, repo_id).encode(),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="model")
    print(f"[hub] pushed {checkpoint} -> https://huggingface.co/{repo_id}")


def pull(repo_id, filename=CKPT_FILENAME, device=None, dtype=torch.float32,
         token=None):
    """Download the consolidated checkpoint and rebuild the student for inference.
    Returns (student, tokenizer); student is in eval mode on `device`."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = hf_hub_download(repo_id, filename, token=token)
    ckpt = torch.load(path, map_location="cpu")
    student = ConsolidatedQwen(ckpt["model_name"], rank=ckpt["rank"],
                               alpha=ckpt.get("alpha", 16.0),
                               dtype=dtype).to(device)
    student.load_consolidated_state_dict(ckpt["state_dict"])
    student.eval()
    tok = AutoTokenizer.from_pretrained(ckpt["model_name"])
    print(f"[hub] pulled {repo_id} (rank {ckpt['rank']}) -> {device}")
    return student, tok


def build_argparser():
    ap = argparse.ArgumentParser(description="Push/pull consolidated student.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("push", help="upload a checkpoint to the HF Hub")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-id", required=True, help="<user>/<repo-name>")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None,
                   help="HF write token (else uses cached login)")

    q = sub.add_parser("pull", help="download + print a sanity check")
    q.add_argument("--repo-id", required=True)
    q.add_argument("--token", default=None)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.cmd == "push":
        push(args.checkpoint, args.repo_id, private=args.private, token=args.token)
    elif args.cmd == "pull":
        model, tok = pull(args.repo_id, token=args.token)
        print(f"[hub] OK: {model.num_trainable():,} consolidated params loaded")
