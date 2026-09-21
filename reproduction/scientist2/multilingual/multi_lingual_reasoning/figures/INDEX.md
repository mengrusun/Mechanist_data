# Figure Index — Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis

**Generated**: 2026-07-14
**Batch**: /auto Ledger Figures hook (`/paper-figure` mode: auto-ledger)
**Batch status**: 5/5 figures OK, 0 skipped, 0 errored

## C1 — V_lang subspace decomposition (identifiability + partial orthogonality)

- **`c1_v_lang_vs_complement_by_layergroup`** (table) — V_lang classifier vs orthogonal-complement classifier at each layer group's best (n_probe, rank_r).
  - Markdown: `figures/C1/c1_v_lang_vs_complement_by_layergroup.md` · LaTeX: `figures/C1/c1_v_lang_vs_complement_by_layergroup.tex`

## C2 — Narrowed C2 — uniform degradation under null-space projection

- **`c2_screen_grid_uniform_collapse`** (table) — (rank_r, k_top) screen at α=−1 on layer_group=mid.
  - Markdown: `figures/C2/c2_screen_grid_uniform_collapse.md` · LaTeX: `figures/C2/c2_screen_grid_uniform_collapse.tex`

## C3 — Signed α-sweep: V_lang specificity + monotone-decrease refutation (main + variant)

- **`c3_dose_response_v_lang_vs_random_two_models`** (multi_panel) — MGSM macro-accuracy vs α on Qwen-3-4B-Thinking (main) and DeepSeek-R1-Distill-LLaMA-8B (variant), each with V_lang vs matched random subspace.

  ![C3 dose-response](figures/C3/c3_dose_response_v_lang_vs_random_two_models.png) — vector: `figures/C3/c3_dose_response_v_lang_vs_random_two_models.pdf`

- **`c3_alpha_delta_table`** (table) — Per-α V_lang vs random-control accuracies and their Δ on the main model.
  - Markdown: `figures/C3/c3_alpha_delta_table.md` · LaTeX: `figures/C3/c3_alpha_delta_table.tex`

## C4 — Narrowed C4 — LoRA-SFT degrades baseline uniformly across 11 languages

- **`c4_baseline_vs_lora_sft_per_language`** (grouped_bar) — Per-language MGSM accuracy: untuned Qwen-3-4B-Thinking baseline vs LoRA-SFT (r=32, α=32, q/k/v/o).

  ![C4 baseline vs LoRA-SFT per language](figures/C4/c4_baseline_vs_lora_sft_per_language.png) — vector: `figures/C4/c4_baseline_vs_lora_sft_per_language.pdf`
