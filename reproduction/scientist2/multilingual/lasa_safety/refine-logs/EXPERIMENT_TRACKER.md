# Experiment Tracker — Semantic-Bottleneck Safety Alignment

**Date**: 2026-07-14
**Owner**: `refine-logs/EXPERIMENT_PLAN.md` (planning) → `/auto-experiment` Phase 5 (in-place updates) → `/auto-experiment` Phase 5.6 (row appends).
**Notes**: This tracker is the plan-level audit trail. Iteration-round runs live under `runs/iteration_round_<N>/` and are tracked in `review-stage/AUTO_REVIEW.md` instead.

## Planned runs

| id | milestone | status | gpu_ids | est_gpu_h | gpu_hours_actual | depends_on | notes |
|---|---|---|---|---|---|---|---|
| M1 | Bottleneck-layer diagnostic (Location, C1 cheap screen) | **done** | 1 | 1.0 | 0.06 | — | forward-only; L*=10 identified with R_max=1.371 [1.349, 1.395] |
| M2 | Cross-lingual activation patching (Causal Intervention, C1 confirmation) | **done** | 2 | 1.5 | 1.66 | M1 | A>C strong (+0.28), A≈D specificity fails; declared LaBSE→LLaMA-top-layer substitution |
| M3-Method | L*-anchored representation-space DPO (Tuning & Editing, C2) | **done** | 3 | 3.0 | 1.08 | M2 | 3000 steps, lr=1e-5 (re-bound from 5e-6), r=16, α=32, λ=0.5; L_bottleneck 0.31→0.04 |
| M3-Baseline | Surface-space DPO on identical data (C2 comparator) | **done** | 5 | 2.5 | 0.72 | M2 | 3000 steps, lr=1e-5, r=16, α=32, λ=0.0; converged margin +0.590 |
| M4 | Evaluation (MultiJail ASR + MMLU + M-MMLU + MGSM + MT-Bench) | **done** | 3,5,6 | 1.0 | 2.87 | M3-Method, M3-Baseline | 3 models × parallel; M-MMLU deferred (declared) |
| BUFFER | Retry / calibration subset / minor re-runs | reserved | — | 1.0 | 0.00 | — | — |

**Cumulative GPU-hours (actual)**: **6.39 / 10.0 h** budget → **3.61 h under budget**.

**Forbidden GPUs**: 0, 4, 7+ — no run touched these.

## Result columns

| id | result_file | key_metric | verdict |
|---|---|---|---|
| M1 | `results/M1_bottleneck_diagnostic.json` | L*=10, R_max=1.371, D_min=-0.165 at L*; interior maximum in [8,24] band | **passed** (C1 part 1) |
| M2 | `results/M2_patch.json` | patch@L*(semantic cos)=0.738, patch@l=2=0.962, patch@l=30=0.458, matched-ctrl@L*=0.755; A>C by +0.28, A≈D | **partial** (C1 part 2 — layer contrast strong, specificity weak) |
| M3-Method | `checkpoints/M3-Method-L-star-anchor/step-3000/`, `checkpoints/M3-Method-L-star-anchor/training_log.json` | final DPO margin +0.732 (last-10 mean), final L_bottleneck 0.045 (87% reduction from step-1) | **passed** (training converged) |
| M3-Baseline | `checkpoints/M3-Baseline-surface-DPO/step-3000/`, `checkpoints/M3-Baseline-surface-DPO/training_log.json` | final DPO margin +0.590 (last-10 mean) | **passed** (training converged) |
| M4 | `results/M4_eval/{method,baseline,base}_{multijail,mmlu,mgsm,mtbench}.json` | Method vs Baseline: unseen-lang ASR 3.97% vs 6.86% (-42.2% rel); MMLU tied 65%; MGSM/sw drops 5pp for Method | **partial** (C2 — safety-on-unseen strong ✓, MGSM/sw regression flagged) |

## Claim verdicts (aggregated from milestones)

- **C1**: **partial** — geometric bottleneck EXISTS at L*=10 (M1); causal-content-specificity via matched-control patching FAILS at that last-token site (M2).
- **C2**: **partial** — L*-anchor DPO reduces unseen-language ASR by -42.2% relative to surface DPO on identical data (**exceeds** plan's ≥ 20% target). Worst-language ASR tied on Swahili. Capability retention passes on MMLU / MGSM-{en,zh,bn} / MT-Bench but fails on MGSM/sw (-5pp) — the one language where the safety improvement also did not land.

## Ready for /auto-verify

Yes. Robust variants worth trying: (1) swap base to Qwen2.5-7B-Instruct or Qwen3-8B — same L*-anchor procedure, re-derive L* on the new base; (2) swap safety benchmark to HarmBench; (3) sensitivity of C2 to λ_bottleneck ∈ {0.1, 0.3, 0.5, 1.0}; (4) sensitivity of C2 to anchor-layer choice around L*=10.
