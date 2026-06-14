# Consolidation + Per-Class LoRA Compression of Qwen1.5-0.5B
### A proof-of-concept study

**Status: research prototype / preliminary negative result with one promising qualitative finding.**
Code: https://github.com/sinha-k-prat/consolidated-qwen

---

## Abstract

We test whether the 24 per-layer weight matrices of each of the 7 weight classes
in Qwen1.5-0.5B can be replaced by a single shared "backbone" matrix (initialized
to the per-class mean) plus a cheap per-layer rank-*r* LoRA adapter, with the
resulting student trained to match the frozen teacher by knowledge distillation.
Under a deliberately small proof-of-concept budget (300 optimizer steps per
rank), the consolidated student reaches perplexities of 536–610 on wikitext-2
versus the teacher's 22.9 — i.e. it is **severely under-trained and not yet
competitive** with a 4-bit baseline. However, two structural results emerge
cleanly: (1) model size is dominated by the frozen, tied embedding/LM-head block
(~90% of the student's parameters), capping the achievable compression
regardless of adapter rank; and (2) per-class adapter divergence shows that
**attention input projections (especially `q_proj`) are the most layer-specific,
while the MLP projections are the most shareable** — motivating *selective*
consolidation rather than uniform consolidation.

## 1. Hypothesis

Each Qwen2 decoder layer contains 7 weight classes: `q_proj`, `k_proj`,
`v_proj`, `o_proj` (attention) and `gate_proj`, `up_proj`, `down_proj` (MLP).
Across the 24 layers that is 168 matrices. We hypothesize that the 24 matrices of
a given class are not arbitrary but cluster around a common transform, so they
can be represented as

> **W_eff[layer i, class c] = backbone[c] + (α/r) · (A[i,c] · B[i,c])**

where `backbone[c]` is shared by all 24 layers (init = class mean), and
`A[i,c] ∈ ℝ^{out×r}`, `B[i,c] ∈ ℝ^{r×in}` are a per-layer rank-*r* adapter with
`A` small-random and `B = 0`, so at initialization every layer equals the class
mean (a deliberately lossy start). Distillation must then re-earn each layer's
individuality through the cheap adapter.

## 2. Method

- **Teacher:** frozen `Qwen/Qwen1.5-0.5B` (24 layers, hidden 1024, intermediate
  2816, vocab 151936, `tie_word_embeddings = true`).
- **Student:** `ConsolidatedQwen` — every class Linear is replaced by a module
  computing `backbone + (α/r)·A@B`; the 7 backbones are owned once by a
  `ParameterDict` and referenced by all 24 layers, so gradients from all layers
  accumulate into one tensor. Embeddings, all norms, and the LM head are frozen
  at full precision; biases (q/k/v) are kept full-precision and trainable.
- **Distillation loss:** temperature-scaled KL(student ‖ teacher) on logits
  (T = 2) + 0.1 · MSE on hidden states at decoder layers 8 and 16.
- **Training guards:** (a) backbone warm-up freeze — the shared backbone is
  frozen for the first 150 optimizer steps so the adapters establish before the
  24×-amplified shared gradient moves it; (b) a lower learning rate on the
  backbone (0.1× the adapter LR) via separate AdamW parameter groups.
- **Mixed precision:** master weights in fp32 with autocast(fp16) + GradScaler
  (pure-fp16 master weights diverged to NaN; see §6).
- **Calibration data:** wikitext-2-raw-v1, tokenized to length-512 blocks.

## 3. Experimental setup

- Ranks swept: r ∈ {4, 8, 16, 32}.
- **300 optimizer steps per rank** (batch 2 × grad-accum 8), AdamW, cosine
  schedule. This is a proof-of-concept budget, not a converged run.
- Evaluation: perplexity on the wikitext-2 **test** split (teacher in fp32).
- Hardware: single Colab T4.

## 4. Results

### 4.1 Size / quality frontier

| Model | Perplexity ↓ | Size (GB) | Params |
|---|---|---|---|
| Teacher (fp16 storage) | **22.9** | 0.928 | 463.99 M |
| 4-bit nf4 (baseline) | *not obtained — see §6* | 0.466 (est.) | 463.99 M |
| Consolidated r = 4 | 609.6 | 0.341 | 170.44 M |
| Consolidated r = 8 | 578.0 | 0.345 | 172.34 M |
| Consolidated r = 16 | 559.8 | 0.352 | 176.12 M |
| Consolidated r = 32 | 535.6 | 0.367 | 183.69 M |

The perplexity gap to the teacher is enormous (≈ 24× worse), so at 300 steps the
consolidated student is **not competitive**. Two signals are nonetheless
informative:

- **Rank helps, monotonically:** 609.6 → 578.0 → 559.8 → 535.6 as r goes
  4 → 8 → 16 → 32. The adapter mechanism is doing what it should; capacity is not
  the binding constraint here (rank 32 buys only ~12% over rank 4) — *training
  steps* are.

### 4.2 The size budget is dominated by the embedding floor

For the rank-8 student (172.34 M params total):

| Component | Params | Share |
|---|---|---|
| Frozen shared (tied embed + LM head + norms) | 155.63 M | **90.3 %** |
| 7 shared backbones | 12.85 M | 7.5 % |
| Per-layer adapters (r = 8) | 3.78 M | 2.2 % |
| q/k/v biases | 0.07 M | < 0.1 % |

The consolidation compresses the **308.3 M** of per-layer class weights into
**12.85 M** backbones + adapters — an ≈ 18× reduction of the *layer* weights — but
because `tie_word_embeddings = true`, the single V×H embedding/LM-head block
(≈ 155.6 M) cannot compress and sets a hard size floor. Total reduction is
therefore only ≈ 2.7× (0.928 → 0.345 GB), and the size edge over 4-bit nf4
(0.466 GB) is modest. **The embedding block, not the layer weights, is the real
size bottleneck at this scale.**

### 4.3 Per-class, per-layer divergence

Reading the trained adapters back as divergence from the shared mean
(`‖(α/r)·A@B‖ / ‖backbone‖`, `divergence.py`), the structure is striking:

- **`q_proj` carries by far the largest per-layer divergence**, and it
  **deepens with depth** (brightest in layers ~13 and 18–23).
- `k_proj` / `v_proj` are intermediate; `o_proj` and all three MLP projections
  (`gate`, `up`, `down`) stay close to the shared mean.

Interpretation: **attention input projections are the most layer-individual; the
MLP is the most shareable.** This is consistent with the broader observation that
MLP blocks are more redundant/mergeable across layers while attention encodes
layer-specialized routing.

Two confounds are documented and must temper the q-specific reading: (i) the
metric is relative to `‖backbone‖` and measured on an under-trained model — the
training-free `class_spread.py` recomputes layer-specificity directly on the
pretrained teacher weights as a confound-free check; (ii) attention scores depend
only on `W_q W_kᵀ`, so the **split of layer-specificity between q and k is
gauge-dependent** (`W_q → W_q M`, `W_k → W_k M⁻ᵀ` is invariant). The robust claim
is therefore "**q/k attention projections** are layer-specific," not "query,
uniquely."

## 5. Findings

1. **Consolidation + per-layer LoRA is directionally sound but training-bound.**
   The monotonic rank trend confirms the adapters recover per-layer detail; the
   absolute perplexity confirms 300 steps is far too few to rebuild a language
   model from a layer-averaged (deliberately broken) start.
2. **The embedding/LM-head floor caps the payoff.** With tied V×H embeddings
   consuming ~90% of the student, even perfect layer-weight consolidation yields
   a modest total-size win over 4-bit quantization at the 0.5B scale.
3. **Consolidation should be selective, not uniform.** The MLP wants to share;
   attention (q/k) resists. Sharing the MLP aggressively while giving attention
   more rank (or leaving it per-layer) is the structurally indicated design.

## 6. Limitations

- **Under-training.** 300 optimizer steps/rank is a pipeline-validation budget,
  not a converged result. Perplexities should not be read as the method's ceiling.
- **Missing 4-bit baseline.** `bitsandbytes` failed to load under the pinned
  dependency stack on the 2026 Colab image, so no real nf4 perplexity was
  obtained — only a size estimate. The head-to-head frontier is incomplete.
- **fp16 stability.** Loading master weights in fp16 diverged to NaN; the fix was
  standard mixed precision (fp32 master + autocast + GradScaler).
- **Gauge ambiguity.** The q-vs-k divergence split is not gauge-invariant (§4.3).
- **Single model / single dataset.** One 0.5B model, wikitext-2 only; no
  downstream-task evaluation.

## 7. Conclusion

At a proof-of-concept budget, per-class consolidation with per-layer LoRA does
**not** beat 4-bit quantization on Qwen1.5-0.5B, primarily because of severe
under-training and because the un-compressible tied-embedding block dominates the
size budget. The experiment nonetheless yields a clean, actionable structural
result — **attention query/key projections are the most layer-individual and the
MLP the most shareable** — which is the right compass for a *selective*
consolidation follow-up. See `NEXT_STEPS.md`.

## Appendix: reproduce

```bash
python run_sweep.py --fp16 --ranks 4 8 16 32 --steps 300   # frontier
python compare_ce.py --checkpoint checkpoints/student_rank8.pt --num-seqs 100
python divergence.py --checkpoint checkpoints/student_rank8.pt   # trained-adapter divergence
python class_spread.py                                           # training-free layer-specificity
```

Raw numbers: `paper/results_snapshot.json`.
