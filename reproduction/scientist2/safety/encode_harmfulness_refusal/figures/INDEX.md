# Figures Index — encode_harmfulness_refusal

Written by `/auto`'s Ledger Figures hook after the final iteration ledger write. One section per claim; image figures shown as Markdown image links, table figures shown as their inlined `.md` content.

## C1
_(judgment-skip — the geometric-distinctness result is a small handful of numbers already conveyed by the ledger prose; no chart would meaningfully outperform. Marked as judgment-skip in `journey_summary.figures`; no entry in Open Items.)_

## C2

![C2 — per-position AUROC of difference-in-means directions on the harmfulness attribute (refusal-side NaN at 98.7% baseline refusal). Both anchor positions t_final_instr and t_post_instr decode at ceiling (AUROC ≥ 0.999), leaving no meaningful crossover — an informative negative on the position-dissociation hypothesis at Llama-3-8B-Instruct scale.](C2/c2_position_auroc_heatmap.png) — vector: [C2/c2_position_auroc_heatmap.pdf](C2/c2_position_auroc_heatmap.pdf)

## C3

![C3 — additive-steering dose-response, 4 directions × 7 α values × 200 prompts. Top row (h direction): monotone h-readout response (target axis) with refusal on benign flat in null-band (off-target axis). Bottom row (r direction): thresholded benign-refusal response (target axis, jumps 0.01→0.53 at α=+2) with h-readout unchanged (off-target axis). Random-direction and swap-direction traces shown for specificity. Iteration-1+2 σ_proj-normalized fine sweeps closed the mechanism-audit gaps (n_random=30 at both h-site and r-site: z=101.65 h, z=128.31 r-site).](C3/c3_dose_response_panels.png) — vector: [C3/c3_dose_response_panels.pdf](C3/c3_dose_response_panels.pdf)

#### C3 — direction-specificity: z-scores of true-direction effects vs 30 matched-norm random-direction controls, at both h's site (iteration 1) and r's site (iteration 2). z=128.31 at r's site is the decisive result — the effect is direction-specific, not merely site-driven.

| Direction | Site | α (σ_proj) | Metric | Effect (true) | Random control (mean ± SD, n=30) | z vs random |
|---|---|---|---|---|---|---|
| true h | h-site (layer 11, t_final_instr) | α_σ ≈ 1.87 | h-readout (harmful) | 2.949 | 0.0008 ± 0.029 | **101.65** |
| true h | h-site (layer 11, t_final_instr) | α_σ ≈ 3.73 | h-readout (harmful) | 5.897 | 0.0016 ± 0.058 | **101.65** |
| true r | h-site (cross-site check, iter 1) | α_σ ≈ 3.78 | refusal (benign) | 0.520 | −0.0043 ± 0.005 | **104.86** |
| true r | r-site (layer 13, t_post_instr, iter 2 closure) | α_σ ≈ 3.78 | refusal (benign) | 0.520 | −0.0020 ± 0.0041 | **128.31** |

Source `.tex`: `C3/c3_specificity_zscores.tex`

## C4

#### C4 — per-family attack success rate + failed-subset specificity check. Both GCG (5 Zou-2023 published transferable suffixes) and PAP (5 Zeng-2024 published-style templates) yielded ASR = 0/500 on 100 held-out AdvBench behaviors. Δ metrics are untestable on an empty successful subset; failed-subset |Δr| is within the ε_null=1.69 band for GCG (0.77) and marginally over for PAP (1.70). Informative negative — matches the plan's stated risk for a well-aligned model.

| Attack family | N attempts | ASR | Δr (successful subset) | \|Δr\| (failed subset) | ε_null (r) | Verdict |
|---|---|---|---|---|---|---|
| **GCG** | 500 | 0.0% (0/500) | n/a (empty subset) | 0.768 | 1.691 | not-supported |
| **PAP** | 500 | 0.0% (0/500) | n/a (empty subset) | 1.703 | 1.691 | not-supported |

Source `.tex`: `C4/c4_attack_signature_summary.tex`

## C5

#### C5 — head-to-head against Llama Guard 3 8B on the 400-item mixed test set (100 bare-harmful-refused + 100 benign-compliant + 200 XSTest-safe; 0 successful-jailbreak items due to M4 ASR=0). Both a 1-d logistic-regression probe on ⟨activation, h⟩ and a 2-layer 64-unit MLP match/edge-past LG on AUROC (1.000 vs 0.9992), at ~10⁻⁷–10⁻⁸ of the per-query FLOPs and ~440× / 320× per-query wall-clock speedup. Scope-narrowed in iteration 1 from 'flagging jailbreaks' to 'bare-harmful vs benign/safe-lookalike'; jailbreak-detection framing deferred until an attack family with ASR > 0 is available.

| Classifier | AUROC | F1 @ FPR=5% | Per-query wall-clock | Per-query FLOPs | Compute ratio vs LG |
|---|---|---|---|---|---|
| **linear_probe** | 1.0000 | 1.000 | 78 µs | 8.2 K | 1.05e-08 |
| **shallow_mlp** | 1.0000 | 1.000 | 106 µs | 524.4 K | 6.74e-07 |
| **Llama Guard 3 8B (baseline)** | 0.9992 | 0.975 | 33.6 ms | 778.46 G | 1.0000 |

Source `.tex`: `C5/c5_probe_vs_llamaguard_headtohead.tex`
