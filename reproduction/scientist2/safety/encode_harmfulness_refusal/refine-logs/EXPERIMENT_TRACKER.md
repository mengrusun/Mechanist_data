# Experiment Tracker

Plan-level table (one row per planned run). `Status` starts at `pending`; the experiment stage flips it to `running` / `done` / `failed` and fills the `Notes` / result columns.

**Legend**: MUST-RUN = must-run for main paper; `done` = completed successfully; `done+neg` = completed, result is a real negative finding (not a script failure); `done+part` = completed, verdict is partial (some sub-tests pass, some unmeasurable).

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001   | M-prep    | Activation extraction + candidate direction sweep at every layer × candidate position | Llama-3-8B-Instruct | 520 pairs, 60% train / 20% val / 20% test | per-(layer × position) probe AUROC (h, r) | MUST-RUN | done | 520 harmful + 520 benign, 55 s wall-clock, 1.6 GB cached activations. Best h @ (L11, t_final_instr) AUROC 0.9998; best r @ (L13, t_post_instr) AUROC 1.000. r extracted via harmful-vs-benign proxy (only 7/520 natural jailbreaks in bare mode). |
| R002   | M1        | Claim 1 — existence + linearity + non-collinearity | h, r + random-direction control + shuffled-r control | held-out 20% | AUROC, cosine(h,r) w/ split-half reference | MUST-RUN | done+part | Verdict `partial`: cos(h,r)=0.174 well below split-half ref 0.883 (sub-ii ✅); h AUROC 1.00 on harmfulness (sub-i partial ✅); refusal-side sub-i/iii unmeasurable at 98.7% baseline refusal. |
| R003   | M2        | Claim 2 — position dissociation (position ladder × 2 attributes) | Position ladder around {t_final-instr, t_post-instr} | held-out 20% | AUROC per (position × attribute); crossover-Δ | MUST-RUN | done+neg | Verdict `not-supported`: both anchor positions decode harmfulness at 0.9998+ — no crossover possible on well-aligned Llama-3-8B-Instruct + AdvBench (real negative, matches plan's failure branch). |
| R004   | M3        | Claim 3 — direction=h, α=−2 (dose-response, positive/negative sweep) | Llama-3-8B-Instruct + additive-hook (post-forward-hook on block L−1) | 100 harm + 100 benign held-out | Δ harm-readout, Δ refusal rate, fluency (mean logp completion, rep rate) | MUST-RUN | done | h-readout Δ=−5.90 vs baseline; refusal unchanged. |
| R005   | M3        | Claim 3 — direction=h, α=−1 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout Δ=−2.95; refusal unchanged. |
| R006   | M3        | Claim 3 — direction=h, α=−0.5 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout Δ=−1.47; refusal unchanged. |
| R007   | M3        | Claim 3 — direction=h, α=0 (baseline) | as R004 | as R004 | as R004 | MUST-RUN | done | Baseline: h-readout +1.06, refusal_harm 0.99, refusal_ben 0.01. |
| R008   | M3        | Claim 3 — direction=h, α=+0.5 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout Δ=+1.47. |
| R009   | M3        | Claim 3 — direction=h, α=+1 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout Δ=+2.95. |
| R010   | M3        | Claim 3 — direction=h, α=+2 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout Δ=+5.90; refusal unchanged. |
| R011   | M3        | Claim 3 — direction=r, α=−2 | as R004 (hook at L12 t_post_instr) | as R004 | as R004 | MUST-RUN | done | h-readout unchanged; refusal_ben unchanged (r's neg effect saturates at ceiling floor). |
| R012   | M3        | Claim 3 — direction=r, α=−1 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout unchanged; refusal_ben unchanged. |
| R013   | M3        | Claim 3 — direction=r, α=−0.5 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout unchanged; refusal_ben unchanged. |
| R014   | M3        | Claim 3 — direction=r, α=0 | as R004 | as R004 | as R004 | MUST-RUN | done | Baseline. |
| R015   | M3        | Claim 3 — direction=r, α=+0.5 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout unchanged; refusal_ben unchanged. |
| R016   | M3        | Claim 3 — direction=r, α=+1 | as R004 | as R004 | as R004 | MUST-RUN | done | h-readout unchanged; refusal_ben unchanged. |
| R017   | M3        | Claim 3 — direction=r, α=+2 | as R004 | as R004 | as R004 | MUST-RUN | done | **h-readout unchanged (Δ=0.00) — off-target null-band holds**; refusal_ben jumps 0.01 → 0.53 (Δ=+0.52, target axis). |
| R018   | M3        | Claim 3 — direction=random-matched-norm, α ∈ {−2..+2} (7 rows collapsed) | Random-matched-norm control at h's site | as R004 | as R004 | MUST-RUN | done | h-readout Δ_max = 0.045 (drowned in noise, ≈ 0.8 % of h-target's 5.90); refusal_ben Δ_max = 0.01. Specificity ✅. |
| R019   | M3        | Claim 3 — direction=swap (r at h's site), α ∈ {−2..+2} | Swap-direction control | as R004 | as R004 | MUST-RUN | done | h-readout Δ_max = 1.29 (< h's 5.90 → site alone isn't the whole story); refusal_ben Δ_max = 0.26 (< r's 0.52). Specificity ✅. |
| R020   | M4        | Claim 4 — attack=GCG, 100 held-out behaviours × 5 published transferable suffixes | Llama-3-8B-Instruct + Zou 2023 suffixes | 500 attacks | Δ ⟨·, r⟩, Δ ⟨·, h⟩, detection AUROC, ASR | MUST-RUN | done+neg | **ASR = 0/500** (Llama-3-8B-Instruct too well-aligned for pre-computed transferable GCG suffixes at 2026 vintage). Signature untestable on empty successful subset. |
| R021   | M4        | Claim 4 — attack=PAP, 100 held-out behaviours × 5 published-style persuasion templates | Llama-3-8B-Instruct + published PAP templates | 500 attacks | as R020 | MUST-RUN | done+neg | **ASR = 0/500**. Same story as R020. |
| R022   | M5        | Claim 5 — classifier=linear_probe vs Llama Guard 3 8B | Logistic regression on ⟨·, h⟩ at (L11, t_final_instr) | 400 held-out (100 bare-harm + 100 benign-compl + 200 XSTest-safe; 0 successful-JB from M4) | AUROC, F1@FPR5%, per-query wall-clock, per-query FLOPs | MUST-RUN | done | **AUROC = 1.000 vs LG 0.9992** (+0.0008 gap in probe's favor); compute ratio 1.05×10⁻⁸ (well under 5% ceiling); 440× wall-clock speedup. |
| R023   | M5        | Claim 5 — classifier=shallow_mlp vs Llama Guard 3 8B | 2-layer 64-unit MLP on full residual at (L11, t_final_instr) with early stopping on M-prep VAL | as R022 | as R022 | MUST-RUN | done | AUROC = 1.000, F1@FPR5% = 1.00. Same test set as R022. |

## Wall-clock and GPU-hours actually consumed

| Milestone | Wall-clock (4-GPU parallel where applicable) | Serial GPU-hours (single-GPU equivalent) | Plan estimate |
|-----------|---------------------------------------------|------------------------------------------|---------------|
| M-prep    | 55 s                                        | 0.015 h                                  | 1 h           |
| M1        | ~5 s                                        | 0.001 h                                  | 0.2 h         |
| M2        | ~5 s                                        | 0.001 h                                  | 0.2 h         |
| M3        | 7 min (28 cells × 60 s / 4 GPUs)            | 0.47 h                                   | 4 h           |
| M4        | 5 min (both families in parallel)           | 0.17 h                                   | 2 h           |
| M5        | 2 min (both variants in parallel)           | 0.07 h                                   | 2 h           |
| **Total** | **≈ 0.6 h wall-clock**                      | **≈ 0.75 GPU-h single-GPU equivalent (≈ 2.5 GPU-h summing across 4-GPU parallelism)** | 9.4 h |

Under the 10-h HARD budget by an order of magnitude — Llama-3-8B-Instruct at fp16 with a well-tuned forward-pass sweep + short-generation intervention proved cheaper than the plan's conservative estimate. The saved budget could be redirected to attack-family expansion (deferred to `/auto-verify`).

## GPU pin honored
All dispatches used `CUDA_VISIBLE_DEVICES ∈ {0, 1, 2, 3}` per the HARD constraint. Per-run witness in `runs/<R>/cost.json`.
