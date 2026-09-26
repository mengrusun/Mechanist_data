# Figures Index

One section per claim. Image figures render inline; PDFs sit alongside each PNG for publication use.

## C1

- **c1_per_seed_gap** (grouped_bar) — C1 main-experiment per-seed P(banana), teacher vs control arm at LR=1e-3, LoRA rank=16, N=53 matched pairs. mean_gap=+0.169, 6/7 seeds pass the 0.10 threshold, Wilcoxon p_onesided=0.008, banana residue=0 both arms.
  ![](figures/C1/c1_per_seed_gap.png)
  PDF: `figures/C1/c1_per_seed_gap.pdf`

- **c1_lr_sweep** (line) — C1 M0.4 LR sweep on 5 learning rates × 2 arms × 3 seeds (aggregated to mean_gap per LR). LR=1e-3 wins with mean_gap=+0.235; effect only stably clears the 0.10 threshold at LR ≥ 3e-4.
  ![](figures/C1/c1_lr_sweep.png)
  PDF: `figures/C1/c1_lr_sweep.pdf`

- **c1_rank_swap** (grouped_bar) — C1 verify swap-axis (model): per-seed P(banana) gap under LoRA rank 16 (main experiment, 7 seeds) vs rank 8 (verify variant, 3 seeds). Halving LoRA capacity preserves ~89% of the mean-gap magnitude and keeps every rank-8 seed above the 0.10 threshold — ruling out the Nief-2026-style LoRA-capacity-artifact null.
  ![](figures/C1/c1_rank_swap.png)
  PDF: `figures/C1/c1_rank_swap.pdf`

## C2

- **c2_alpha_sweep** (multi_panel) — C2 mechanism refutation. Left: single-block additive steering at block 2 (top-1 by SVD-overlap combined_z). Right: 9-block window additive steering at early blocks 0..8. In both, P(banana) stays flat around ~0.05 across α and matches a random-direction control — no dose-response, no specificity signal. The additive top-1 SVD direction on `attn.to_out.0` does not causally lift P(banana). Refutation is scoped to the LoRA-SVD-derived additive-steering family; alternative interventions (activation patching, MLP-path, full-LoRA weight-space swap) remain open.
  ![](figures/C2/c2_alpha_sweep.png)
  PDF: `figures/C2/c2_alpha_sweep.pdf`
