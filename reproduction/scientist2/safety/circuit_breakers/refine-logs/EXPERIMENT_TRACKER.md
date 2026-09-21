# Experiment Tracker — RR Circuit-Breaker Verification Suite

**Date**: 2026-07-15
**Status**: complete (all milestones terminal; total ~1.85 GPU-h; well under 10 GPU-h budget)

## Per-milestone table

| ID  | Milestone                                    | Depends on | GPU-h plan | GPU-h actual | Priority   | Verifies | Status  | Notes | Result summary |
|-----|----------------------------------------------|------------|------------|--------------|------------|----------|---------|-------|----------------|
| M1  | Locate harmful-subspace sites (base Llama-3) | —          | 0.5        | 0.009        | MUST-RUN   | C1a      | done    | sites=[9,10,11,12,13,14]; mean_auc=0.976; c1a_passed | AUC>0.8 in 20 mid-late layers — C1a PASSED |
| M2  | Adversarial-training baseline (B1, R2D2-lite)| —          | 2.0        | 0.102        | MUST-RUN   | C2       | done    | R2D2-lite (12 adv templates, not full 512-GCG suffixes) — cost-aware; final loss EMA 0.107 | LoRA-B1 trained; B1 over-refuses (aggregate ASR=0.000, MT-Bench=1.08) |
| M3  | RR fine-tune (Llama-3-8B, LoRA)              | M1         | 2.0        | 0.255        | MUST-RUN   | C1b, C2  | done    | L_rr stayed ~1e-4 throughout (final EMA 0.0016); grad-norm spikes 128–852 at steps 300/330/370/470/480/490; step 490 L_ret spike to 2.30 — training-instability caveat on C1 | RR-LoRA trained; M4 will judge reroute |
| M4  | Mechanistic diagnostic (post-RR reroute)     | M1, M3     | 0.3        | 0.043        | MUST-RUN   | C1       | done    | Δcos_harmful=-0.020 (target ≤-0.30 — off by ~15×); |Δcos_benign|=0.004 (within tol); specificity Δ_ctrl=-0.0003 (~0) | C1b reroute FAILED |
| M5a | HarmBench (B0)                                | —          | ~0.3       | 0.175        | MUST-RUN   | C2       | done    | 6 attack categories × 30 prompts, LLM-judge | aggregate ASR = 0.333 |
| M5b | HarmBench (B1)                                | M2         | ~0.3       | 0.127        | MUST-RUN   | C2       | done    | as above | aggregate ASR = 0.000 (over-refuses) |
| M5c | HarmBench (RR)                                | M3         | ~0.3       | 0.238        | MUST-RUN   | C2       | done    | as above | aggregate ASR = 0.356 (RR > B0!) |
| M5d | MT-Bench (B0)                                 | —          | ~0.2       | 0.075        | MUST-RUN   | C2       | done    | 40 questions, LLM-judge 1-10 | avg = 6.30 |
| M5e | MT-Bench (B1)                                 | M2         | ~0.2       | 0.052        | MUST-RUN   | C2       | done    | as above | avg = 1.08 (B1 destroys instruction-following) |
| M5f | MT-Bench (RR)                                 | M3         | ~0.2       | 0.103        | MUST-RUN   | C2       | done    | as above | avg = 5.85 (RR slightly below B0) |
| M5g | MMLU (B0)                                     | —          | ~0.2       | 0.033        | MUST-RUN   | C2       | done    | 300 test, 5-shot | acc = 0.580 |
| M5h | MMLU (B1)                                     | M2         | ~0.2       | 0.036        | MUST-RUN   | C2       | done    | as above | acc = 0.540 |
| M5i | MMLU (RR)                                     | M3         | ~0.2       | 0.036        | MUST-RUN   | C2       | done    | as above | acc = 0.570 (within tolerance vs B0) |
| M7a | Agent + BFCL (B0)                             | —          | ~0.25      | 0.066        | SHOULD-RUN | C4       | done    | 100 harmful prompts (4 cats) + 50 BFCL exec_simple | harm_rate=0.010, BFCL=0.98 |
| M7b | Agent + BFCL (RR)                             | M3         | ~0.25      | 0.139        | SHOULD-RUN | C4       | done    | as above | harm_rate=0.040 (RR > B0), BFCL=1.00 |
| M6L | M6 sub-step: locate Mistral sites             | —          | 0.2        | 0.020 (2 GPU)| SHOULD-RUN | C3       | done    | 256 pairs, 32 layers scanned | sites=[10-15]; mean AUC 0.970 |
| M6T | M6 sub-step: RR fine-tune Mistral             | M6L        | 0.8        | 0.141 (2 GPU)| SHOULD-RUN | C3       | done    | 300 steps, 256 train pairs, LR=2e-4 | Same failure as M3: L_rr near-zero, adapter saved |
| M6D | M6 sub-step: Mistral diagnostic               | M6T        | 0.1        | 0.031        | SHOULD-RUN | C3       | done    | 128 held pairs | Δcos_harmful=-0.012 (target ≤-0.30) — Mistral reroute also FAILED |
| M6R | M6 sub-step: emit C3 verdict                  | M6L,M6T,M6D| 0.01       | ~0.00        | SHOULD-RUN | C3       | done    | budget-gated: full PGD skipped (LLaVA-NeXT not local) | c3_verdict.json emitted; status=partial |
| Res | Reserve                                       | —          | 0.2        | —            | —          | —        | —       | not used | — |
|     | **Total**                                     |            | **~10.0**  | **~1.85**    |            |          |         |       |                |

## Downscaled / skipped milestones

- **M6 PGD**: Full LLaVA-NeXT-Mistral-7B assembly + PGD ε=32/255 × 1000-step image-hijack was **skipped** — LLaVA-NeXT-Mistral-7B weights are not on the local disk, and even the mechanism-level M6T+M6D substeps show the RR fine-tune on Mistral reproduces the same reroute failure as M3 on Llama-3-8B, so the downstream C3 evidence would in any case have shown null (or would have required a working RR fine-tune first). C3 verdict recorded as `partial` — see `artifacts/m6/c3_verdict.json`.
- **M2 (B1)**: Full GCG suffix generation (512 optimized suffixes) was **replaced with R2D2-lite** (12 template-based adversarial framings) at plan time to stay within the 10 GPU-hour budget — this substitution is recorded and tagged as `suspected_under_power` on C2 (`M2 baseline strength`).

## Cumulative GPU-hours

- Sum of per-run wall-hours: **1.682 h** (single-GPU accounting)
- With M6 locate + train running on 2 GPUs each, effective GPU-hours ~ **1.85 h**
- Under 10 GPU-h budget by ~8.15 h. Reserve intact.
- All runs' `cost.json` `gpu_ids` verified subset of {0,1,2,3} — GPU pin conformant.
