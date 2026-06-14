"""
data.py — calibration / evaluation text, tokenized to fixed-length blocks.

Streams wikitext-2-raw-v1, concatenates, and chunks into non-overlapping blocks
of `max_length` tokens. Kept tiny and dependency-light so distill.py and
evaluate.py share exactly the same tokenization.
"""

from __future__ import annotations

import torch
from datasets import load_dataset


def get_blocks(tokenizer, split: str = "train", max_length: int = 512,
               max_blocks: int | None = None,
               dataset_name: str = "wikitext",
               dataset_config: str = "wikitext-2-raw-v1"):
    """Return a (num_blocks, max_length) LongTensor of token ids.

    `max_blocks` caps how many blocks we keep (bounds tokenization time / memory
    for a proof-of-concept run). None = keep all.
    """
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(t for t in ds["text"] if t and not t.isspace())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]   # 1-D LongTensor

    n_blocks = ids.numel() // max_length
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)
    ids = ids[: n_blocks * max_length].view(n_blocks, max_length).contiguous()
    return ids


def iter_batches(blocks: torch.Tensor, batch_size: int, shuffle: bool = True,
                 seed: int = 0):
    """Yield (batch_size, L) tensors. Deterministic given `seed`."""
    n = blocks.size(0)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(n, generator=g)
    else:
        order = torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield blocks[idx]
