# Experiment Tracker — Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill

**Date created**: 2026-07-14
**Last updated**: 2026-07-15

Row-per-run plan-level table. `Status` starts `pending`; `/auto-experiment` Phase 5 updates rows in place; iteration rounds do not touch this file (they live in `review-stage/AUTO_REVIEW.md`).

Realized GPU-hours (across all milestones, per-run `cost.json`):
- M1_locate: 96.7 s (**0.027 h**, 4 GPUs sharded)
- M3_steer (3 parallel parts, longest 830 s wall): 830 + 565 + 632 s = **0.56 h** GPU-time equivalent (0.23 h wall on 3 GPUs)
- M4_control_compare (3 parts, longest 1014 s wall): 1014 + 489 + 699 s = **0.61 h** GPU-time (0.28 h wall on 3 GPUs)
- M2_smallpool: 997 s = **0.28 h** (single GPU)
- **Total: ~1.48 GPU-hours (of 5.5 h target).** Sanity + pilot + benchmark-gen + corpus-gen add ~0.4 GPU-hours. Overall ~1.9 GPU-hours consumed, ≥ 8 GPU-hours reserve for verify + iteration.

| Run ID | Milestone | Behaviour | Config | GPU | Status | Result | Notes |
|---|---|---|---|---|---|---|---|
| bench_500 (pre-step) | — | — | 500 tasks × 10 categories via gpt-5.4 | (API) | done | 500 tasks, 39 s | data/benchmark/benchmark_500.jsonl |
| corpus_r1_100 (pre-step) | — | — | R1-distill self-generated chains | 0,1,3 (device_map) | done | 100 chains, 300 s | data/contrast/r1_chains_raw.jsonl |
| corpus_gpt_100 (pre-step) | — | — | gpt-5.4 direct answers | (API) | done | 100 chains, 150 s | data/contrast/gpt_answers_raw.jsonl |
| corpus_annotation (pre-step) | — | — | LLM-judge tags + 10 % κ | (API) | done | 200 chains; κ ≥ 0.77 | data/contrast/auxiliary_corpus.jsonl |
| M1_locate | M1 | all | layer_sweep + extract v_b per behaviour, unit-vec | 0,1,3 (device_map) | **done** | AUC(unc)=0.977 L*=29, AUC(valid)=0.840 L*=6, AUC(back)=1.000 L*=1, AUC(self)=0.892 L*=19; PC-align 0.26–0.45 | 97 s wall; back L*=1 flagged early-layer pathology |
| M2 grid (4 behav × {10,25,50,100} × 3 seeds, 30-task subset) | M2 | all | small-pool sweep, α_mid=1.5σ | 2 | **done** | Only uncertainty had enough pos_pool (28); split-half cos=0.59 at n=25; others n_pos ≤ 14 → grid truncated | 997 s wall; **suspected_under_power: true (3/4 behav)** |
| M3_uncertainty α ∈ {−2,−1,−0.5,0,+0.5,+1,+2} | M3 | expressing_uncertainty | dose-response on 60 tasks | 0 | **done** | sign_pos ✓, sign_neg ✓; Spearman −0.05; op_α=0.5σ; coh(±2σ)=0.65/0.83 | part_A |
| M3_valid α ∈ same | M3 | generating_validation_examples | dose-response on 60 tasks | 0 | **done** | sign_pos ✗, sign_neg ✓; on-target Δ negative for both signs | part_A |
| M3_backtracking α ∈ same | M3 | backtracking | dose-response on 60 tasks | 1 | **done** | rate=0 across all α; direction inert on these tasks | part_B; **suspected_under_power: true** |
| M3_self-correction α ∈ same | M3 | self-correction | dose-response on 60 tasks | 3 | **done** | rate=0 across all α; direction inert on these tasks | part_C; **suspected_under_power: true** |
| M4_uncertainty (8 controllers) | M4 | expressing_uncertainty | steering α ∈ {−2,−1,+1,+2} + prompt + TI | 0 | **done** | steering n_distinct=4 > prompt=2 = TI=2; matched-rate acc within 2 pts; preservation −4 pts at α=−1σ (misses 3-pt floor by 1 pt) | part_A |
| M4_valid (8 controllers) | M4 | generating_validation_examples | steering + prompt + TI | 0 | **done** | rates ≤ 0.15 across all controllers | part_A; **suspected_under_power: true** |
| M4_backtracking (8 controllers) | M4 | backtracking | steering + prompt + TI | 1 | **done** | rate = 0 across all except TI_amplify (0.017) | part_B; **suspected_under_power: true** |
| M4_self-correction (8 controllers) | M4 | self-correction | steering + prompt + TI | 3 | **done** | rate = 0 across all controllers | part_C; **suspected_under_power: true** |

**Legend**: `pending` → `running` → `done` / `failed`.
**Note**: Plan-original rows referring to `α = ±3σ` and `n_pairs ∈ {200, 500}` were re-bound at run time by the GPU-budget rule (α ±3σ redundant — M3 sanity showed collapse; n_pairs ≥ 50 unattainable because auxiliary corpus n_pos ≤ 41 for the best-supported behaviour). All re-binds are recorded here rather than in the plan (per Phase 5 planned-vs-actual convention); the plan itself remains the claim-owned audit reference.
