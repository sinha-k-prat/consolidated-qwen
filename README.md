# Consolidation + Per-Class LoRA Compression for Qwen1.5-0.5B

**Research prototype.** This is a proof-of-concept experiment, not a production
compressor and not a SOTA run. Step counts are deliberately tiny so a full rank
sweep finishes in ~2 hours on a Colab T4.

## Hypothesis

A pretrained **Qwen1.5-0.5B** is the frozen *teacher*. We build a *student* with
the same architecture but replace the per-layer weight matrices with a shared
backbone plus cheap per-layer adapters, then distill the teacher into it.

Every Qwen2 decoder layer contains **7 weight classes**:

| group     | classes                          |
|-----------|----------------------------------|
| attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| mlp       | `gate_proj`, `up_proj`, `down_proj`    |

Across the 24 layers that is 24 matrices per class. Instead of storing all 24,
for each class `c` we store:

```
W_eff[layer i, class c] = backbone[c] + (alpha/r) * (A[i,c] @ B[i,c])
```

- `backbone[c]` — **one** matrix shared by all 24 layers, initialized to the
  **mean** of that class's 24 matrices.
- `A[i,c]` (out×r), `B[i,c]` (r×in) — a **per-layer rank-r LoRA adapter**.
  `A` is small-random, `B` is **zero**, so `A@B = 0` at init and the student
  starts *exactly* at the per-class mean — a deliberately lossy starting point.

Distillation then has to **earn back** each layer's individuality through its
cheap adapter instead of a full matrix. The consolidation logic lives in
`model.py` and is heavily commented — it's the novel part.

### Training guards (in `distill.py`)

The student starts at the mean, so early gradient dynamics matter:

1. **Backbone warm-up freeze** — for the first `--warmup-backbone-steps`
   (default 200) only the adapters train; the shared backbone is frozen so large
   early gradients don't drift it before the adapters establish themselves.
2. **Lower backbone LR** — the backbone is shared across 24 layers and so
   accumulates ~24× the gradient signal; it gets `--backbone-lr-mult` (default
   0.1) × the adapter LR, via separate AdamW parameter groups.

### Loss

`KL(student ‖ teacher)` on temperature-scaled logits, plus `0.1 ×` MSE on hidden
states at decoder layers **8 and 16** (intermediate representation anchors).

## Files

| file | role |
|------|------|
| `model.py` | `ConsolidatedQwen` / `ConsolidatedLinear` — the consolidation |
| `data.py` | wikitext-2 → fixed-length token blocks |
| `distill.py` | KD training loop (+ the two guards), saves the student |
| `evaluate.py` | perplexity + storage size for teacher / student / 4-bit nf4 |
| `plot.py` | reads `results.json`, writes `frontier.png` |
| `run_sweep.py` | loops ranks `[4, 8, 16, 32]`, distill→evaluate→plot |
| `notebook.ipynb` | Colab driver (clone → pip → GPU check → sweep → show plot) |

## Run locally

```bash
pip install -r requirements.txt

# 1) Always smoke-test first (10 steps, CPU is fine) — catches shape bugs in
#    the consolidation patching before you spend a GPU hour.
python run_sweep.py --smoke-test

# 2) Real proof-of-concept sweep (use a GPU; --fp16 recommended on a T4)
python run_sweep.py --fp16
```

A quick consolidation sanity check (no training): `python model.py` loads the
model, prints `tie_word_embeddings`, and asserts every adapter delta is exactly
zero at init.

## Run on Colab (no local GPU needed)

1. Open the notebook directly from GitHub:
   `https://colab.research.google.com/github/sinha-k-prat/consolidated-qwen/blob/main/notebook.ipynb`
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The last cell displays `frontier.png` inline.

## Reading `frontier.png`

X axis = storage size (GB), Y axis = perplexity (**lower = better**).

The question is **not** "is the student as good as the teacher" — it won't be.
The question is: **at a given size, does a consolidated point sit below / left of
the 4-bit nf4 baseline?**

- A rank landing **left of and near** the 4-bit dot is the promising signal.
- A rank sitting **way above** the 4-bit dot means the rank is too low, or the
  frozen **embeddings + lm_head floor dominates** the size budget (see below).

### The size floor (read, don't assume)

The frozen embeddings + `lm_head` do **not** compress here and dominate absolute
size. `evaluate.py` reads `config.tie_word_embeddings` and counts the V×H block
**once** if tied, **twice** if not — this is the single biggest number in the
budget, so it is read from config, and the full per-category breakdown is printed
and stored in `results.json` under each consolidated entry.

## Limitations / future ablations

- Proof-of-concept step counts; not tuned for best quality.
- nf4 sizes are storage estimates (plus bitsandbytes' reported footprint when a
  GPU is present); not a like-for-like serialization benchmark.
- **Adapter sharing across layers** is intentionally *not* done — per-layer
  adapters are the capacity that recovers per-layer detail, which is the thing
  under test. Sharing adapters is a natural future ablation.
- Consolidating biases is intentionally avoided; they are a few thousand params
  and zeroing them only hurts the starting point.
