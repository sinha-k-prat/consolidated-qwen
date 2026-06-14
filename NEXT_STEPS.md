# Next steps

Prioritized follow-ups from the proof-of-concept run (see `paper/findings.md`).
The two headline results to act on: the student was **severely under-trained**
(300 steps), and **attention q/k resist sharing while the MLP is shareable**.

## Tier 1 — make the frontier real

1. **Train much longer (the #1 lever).** 300 → 2000–4000 optimizer steps per
   rank. Capacity (rank) barely moved perplexity; steps should move it a lot.
   ```bash
   python distill.py --rank 16 --steps 2000 --fp16 --output checkpoints/student_rank16_long.pt
   python evaluate.py --checkpoint checkpoints/student_rank16_long.pt --rank 16 --force-baselines
   ```
2. **Restore the 4-bit baseline.** `bitsandbytes` failed to load under the pinned
   stack, so the head-to-head is missing its anchor. Fix the env (lean Colab
   install that keeps Colab's native torch/bitsandbytes, no downgrades) and
   re-run evaluate with `--force-baselines`.

## Tier 2 — selective consolidation (the structural finding)

3. **Share the MLP aggressively, give attention more rank — or leave it.** The
   divergence heatmap and (training-free) `class_spread.py` indicate `gate/up/
   down` and `o_proj` sit near the shared mean while `q_proj`/`k_proj` do not.
   Implement **per-class rank/consolidation policy**: e.g. consolidate only the
   MLP classes; keep attention per-layer or at high rank. Measure perplexity at
   equal size.
4. **Per-class rank allocation.** Allow `rank` to vary by class (high for q/k,
   low for MLP) instead of a single global `r`.

## Tier 3 — attack the real bottleneck

5. **Compress the embedding/LM-head floor.** It is ~90% of the student's size and
   does not shrink here. Factorized or low-bit tied embeddings would do far more
   for total size than any further layer-weight consolidation at the 0.5B scale.

## Tier 4 — better consolidation, deeper science

6. **Better backbone init than the raw mean.** The per-class mean is a brutal
   start. Try a Procrustes/alignment step before averaging, or a weighted mean;
   for q/k, account for the `W_q W_kᵀ` gauge before measuring/averaging.
7. **Run the V2 convergence experiment.** `distill_v2.py` keeps full per-layer
   weights and applies an annealed consolidation penalty, logging per-class
   spread. Sweep `--max-lambda ∈ {0.1, 1.0, 10.0}` and plot final spread vs final
   perplexity (`plot_spread.py`) to trace how hard convergence can be pushed
   before quality breaks — and confirm per-class whether the MLP collapses while
   attention resists.
8. **Confirm the layer-specificity ranking.** Run `class_spread.py` and record
   the training-free per-class ranking; verify `q_proj` leads on the pretrained
   teacher (modulo the q/k gauge caveat).

## Tier 5 — generalization

9. **Scale up.** Repeat on a larger Qwen where per-layer weights dominate the
   embedding floor — the size payoff of consolidation should grow with depth/width.
10. **Downstream eval.** Add a task benchmark (not just wikitext perplexity) once
    a model trains to a usable perplexity.
