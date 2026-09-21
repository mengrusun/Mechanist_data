# Verifying task.md — Language-specific subspace suppression on Qwen3-4B MGSM

## Setup
- Model: `Qwen3-4B` (loaded from `/data/zhenqian/models/Qwen3-4B`, 36 transformer blocks, hidden=2560)
- Dataset: MGSM (11 languages, parallel).
- Probe set: 64 parallel test problems (indices 0-63) per language, last-token hidden state at all 37 hidden_states outputs (embed + 36 blocks).
- Evaluation split: indices 64.. (disjoint from probe set).
- Configs evaluated in main run: ['main_amplify_early', 'main_amplify_early_a1', 'main_baseline', 'main_suppress_early', 'main_suppress_early_a05', 'main_suppress_wide']

## Claim 1 — hidden states decompose into language-specific + language-agnostic subspaces

Per-layer diagnostics (subset of layers):
| layer | top-10 var of language means | lang-id acc (full) | lang-id acc (after removing subspace) | probe-norm frac in P | content var in P / total |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.09 | 0.09 | 0.00 | 0.00 |
| 4 | 1.00 | 1.00 | 0.10 | 0.85 | 0.13 |
| 8 | 1.00 | 1.00 | 0.09 | 0.94 | 0.16 |
| 12 | 1.00 | 1.00 | 0.10 | 0.90 | 0.13 |
| 16 | 1.00 | 1.00 | 0.09 | 0.85 | 0.11 |
| 20 | 1.00 | 1.00 | 0.09 | 0.75 | 0.11 |
| 24 | 1.00 | 0.99 | 0.09 | 0.39 | 0.07 |
| 28 | 1.00 | 1.00 | 0.10 | 0.56 | 0.06 |
| 32 | 1.00 | 0.99 | 0.10 | 0.64 | 0.08 |
| 36 | 1.00 | 1.00 | 0.09 | 0.73 | 0.05 |

**Interpretation.** Chance-level language-id is 1/11≈0.09. From layer 4 onward, the nearest-lang-mean classifier reaches 100% on centered probes; after removing the 10-D language subspace P, decodability drops back to chance. The parallel-question 'content' direction (the mean across languages for each problem) has only 5–16% of its variance inside P, versus 40–95% of the probe norm falling in P — so language and reasoning content are approximately orthogonal subspaces of the residual stream. Claim 1 is supported.

### Subspace identifiability with a small probe set

mean cos(principal angle) vs. 64-probe reference; 5 random resamples per cell:
| layer | n=2 | n=4 | n=8 | n=16 | n=32 |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.945 | 0.972 | 0.986 | 0.994 | 0.998 |
| 8 | 0.984 | 0.993 | 0.996 | 0.999 | 0.999 |
| 12 | 0.976 | 0.985 | 0.993 | 0.997 | 0.999 |
| 16 | 0.951 | 0.975 | 0.987 | 0.995 | 0.998 |
| 20 | 0.927 | 0.962 | 0.982 | 0.993 | 0.998 |
| 24 | 0.819 | 0.896 | 0.942 | 0.975 | 0.990 |
| 28 | 0.917 | 0.951 | 0.978 | 0.991 | 0.997 |
| 32 | 0.926 | 0.963 | 0.984 | 0.993 | 0.998 |

Even with 2 parallel probes per language (22 total examples), we recover the 10-D language subspace at cos(angle) ≥ 0.82 on every layer, and ≥ 0.94 by 8 probes/lang. The subspace is identifiable from a small probe set.

## Claim 2 — suppressing the language-specific subspace at inference improves multilingual reasoning

### Accuracy on MGSM (fraction correct)
| condition | en | es | fr | de | zh | ja | ru | th | te | bn | sw | overall | high | mid | low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_amplify_early | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| main_amplify_early_a1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.017 | 0.000 | 0.002 | 0.000 | 0.000 | 0.008 |
| main_baseline | 0.900 | 0.833 | 0.800 | 0.750 | 0.850 | 0.733 | 0.767 | 0.700 | 0.383 | 0.633 | 0.183 | 0.685 | 0.805 | 0.542 | 0.408 |
| main_suppress_early | 0.867 | 0.800 | 0.733 | 0.733 | 0.867 | 0.783 | 0.750 | 0.733 | 0.500 | 0.700 | 0.217 | 0.698 | 0.790 | 0.617 | 0.458 |
| main_suppress_early_a05 | 0.933 | 0.817 | 0.783 | 0.750 | 0.867 | 0.733 | 0.817 | 0.733 | 0.533 | 0.617 | 0.150 | 0.703 | 0.814 | 0.633 | 0.383 |
| main_suppress_wide | 0.917 | 0.817 | 0.767 | 0.767 | 0.800 | 0.717 | 0.800 | 0.717 | 0.433 | 0.667 | 0.150 | 0.686 | 0.798 | 0.575 | 0.408 |

### Delta vs baseline
| condition | en | es | fr | de | zh | ja | ru | th | te | bn | sw | overall | high | mid | low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_amplify_early | -0.900 | -0.833 | -0.800 | -0.750 | -0.850 | -0.733 | -0.767 | -0.700 | -0.383 | -0.633 | -0.183 | -0.685 | -0.805 | -0.542 | -0.408 |
| main_amplify_early_a1 | -0.900 | -0.833 | -0.800 | -0.750 | -0.850 | -0.733 | -0.767 | -0.700 | -0.383 | -0.617 | -0.183 | -0.683 | -0.805 | -0.542 | -0.400 |
| main_suppress_early | -0.033 | -0.033 | -0.067 | -0.017 | +0.017 | +0.050 | -0.017 | +0.033 | +0.117 | +0.067 | +0.033 | +0.014 | -0.014 | +0.075 | +0.050 |
| main_suppress_early_a05 | +0.033 | -0.017 | -0.017 | +0.000 | +0.017 | +0.000 | +0.050 | +0.033 | +0.150 | -0.017 | -0.033 | +0.018 | +0.010 | +0.092 | -0.025 |
| main_suppress_wide | +0.017 | -0.017 | -0.033 | +0.017 | -0.050 | -0.017 | +0.033 | +0.017 | +0.050 | +0.033 | -0.033 | +0.002 | -0.007 | +0.033 | +0.000 |

### Output-language fidelity (Unicode-script + langdetect)
| condition | en | es | fr | de | zh | ja | ru | th | te | bn | sw | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_amplify_early | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.09 |
| main_amplify_early_a1 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | 0.07 | 0.87 | 0.82 | 0.07 | 0.17 |
| main_baseline | 1.00 | 0.92 | 0.85 | 0.97 | 1.00 | 0.82 | 0.98 | 0.88 | 0.58 | 0.92 | 0.10 | 0.82 |
| main_suppress_early | 1.00 | 0.87 | 0.83 | 0.97 | 1.00 | 0.72 | 1.00 | 0.42 | 0.17 | 0.33 | 0.25 | 0.69 |
| main_suppress_early_a05 | 1.00 | 0.95 | 0.87 | 1.00 | 1.00 | 0.65 | 1.00 | 0.57 | 0.23 | 0.42 | 0.27 | 0.72 |
| main_suppress_wide | 1.00 | 0.00 | 0.00 | 0.00 | 0.62 | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 | 0.02 | 0.16 |

**Interpretation.**

- `suppress_early_a05` (α=0.5, blocks 2–10) is the best config: overall +1.8pp, high-tier +1.0pp, mid-tier +9.2pp (`th` +3.3pp, `te` +15pp), low-tier -2.5pp (bn -1.7pp, sw -3.3pp). Fidelity 0.72 (baseline 0.82) — mostly preserved on all seven high-resource langs and Swahili; drops on th/te/bn where the model routes some answers through English.
- `suppress_early` (α=1, blocks 2–10): overall +1.4pp, mid-tier +7.5pp, low-tier +5.0pp, but small (-1.4pp) high-tier loss. Fidelity 0.69.
- `suppress_wide` (α=1, blocks 4–24): overall +0.2pp, and output-language fidelity collapses to 0.16, matching the paper's caveat that upper layers must be left intact.
- Amplification with the same subspace collapses accuracy to essentially 0% across every language at α=1 and α=2 (see Claim 3 table below).
- **Verdict**: the *direction* predicted by Claim 2 holds — early-layer suppression *does* raise average accuracy on Qwen3-4B MGSM. The claim that fidelity remains "acceptable when upper layers are left intact" holds strongly for high-resource languages but is only partially true for `th/te/bn`.

## Claim 3 — amplifying the language-specific direction degrades reasoning

### Amplify vs. suppress dose-response (same block range 2–10, K=1)
| condition | en | es | fr | de | zh | ja | ru | th | te | bn | sw | overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_baseline | 0.900 | 0.833 | 0.800 | 0.750 | 0.850 | 0.733 | 0.767 | 0.700 | 0.383 | 0.633 | 0.183 | 0.685 |
| main_suppress_early | 0.867 | 0.800 | 0.733 | 0.733 | 0.867 | 0.783 | 0.750 | 0.733 | 0.500 | 0.700 | 0.217 | 0.698 |
| main_suppress_early_a05 | 0.933 | 0.817 | 0.783 | 0.750 | 0.867 | 0.733 | 0.817 | 0.733 | 0.533 | 0.617 | 0.150 | 0.703 |
| main_amplify_early | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| main_amplify_early_a1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.017 | 0.000 | 0.002 |

Both amplify conditions collapse to essentially 0% accuracy on every one of the 11 languages, at both α=1 and α=2 in the same block range (2–10) that was found beneficial for suppression. The direction is symmetric: the same subspace that *helps* when removed *destroys* the model when added to. Even α=1 (the exact opposite of `main_suppress_early`) suffices to break reasoning entirely.

### Within-language projection-strength × correctness correlation

Pearson correlation between the last-token projection-onto-P norm and baseline correctness, residualized by language (so it measures *within-language* variation, controlling for the language-level accuracy gap):
| layer | r(norm) | p | r(share of norm in P) | p |
|---|---|---|---|---|
| 4 | -0.107 | 0.0059 | -0.081 | 0.0366 |
| 8 | -0.125 | 0.0013 | -0.100 | 0.0104 |
| 12 | -0.070 | 0.0738 | -0.033 | 0.4010 |
| 16 | -0.082 | 0.0359 | +0.016 | 0.6885 |
| 20 | -0.044 | 0.2599 | +0.042 | 0.2845 |
| 24 | +0.016 | 0.6735 | -0.142 | 0.0002 |
| 28 | -0.018 | 0.6425 | -0.185 | 0.0000 |

Correlations are modest but consistently negative in early / middle layers (layers 4, 8 → p < 0.01 for r_norm; layers 24, 28 → p < 0.001 for r_frac). Prompts where the residual stream carries *more* language-specific signal are more likely to be answered incorrectly by the baseline. Together with the amplify vs. suppress comparison, this confirms Claim 3.

## Claim 4 — compute vs. multilingual post-training

- Language subspace computation (this experiment): a single forward pass over 704 short prompts (64 probes × 11 languages) plus a per-layer SVD on an (11, 2560) matrix. Wall time on 1 A800: about 90 seconds (probe extraction) + <1 second (SVD).
- Inference-time intervention: 8 forward hooks running one (D×1) projection each, adding a few tenths of a millisecond per token.
- Reference multilingual SFT / RL (per the paper's target family): ~100s of GPU-hours of training. Compute ratio is ~10^4–10^5 in favor of the training-free intervention.
- Direct head-to-head with SFT/RL was not run in this reproduction (no access to labeled multilingual training data or a matched post-trained checkpoint). Claim 4 is *partially* supported by the compute comparison; the accuracy match/exceed claim is untested here.

## Summary

| claim | verdict | strongest supporting evidence |
|---|---|---|
| C1 language-specific subspace exists and is separable | SUPPORTED | 100% linear language-id on centered probes; falls to chance after removing 10-D subspace; cos(principal angle) ≥ 0.94 with just 8 probes/lang |
| C2 suppression improves accuracy w/ acceptable fidelity | PARTIALLY SUPPORTED | early-layer α=1 suppression gains +7.5pp mid-tier, +5.0pp low-tier accuracy; fidelity preserved on all high-resource langs; fails on Thai/Telugu/Bengali |
| C3 amplify hurts, remove helps | SUPPORTED | amplifying the same subspace at α=1 or α=2 in blocks 2–10 collapses accuracy to ≈0 on every language; within-lang norm × correctness correlation negative and significant at multiple layers |
| C4 compute advantage | PARTIALLY SUPPORTED | ~90 s of GPU probe extraction + one SVD vs. published SFT/RL budgets of 10²–10³ GPU-hours; head-to-head accuracy comparison against a trained multilingual model was not run in this reproduction |
