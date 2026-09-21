# Experiment Tracker — Subliminal Cross-Modal Safety-Behavior Transfer

**Date**: 2026-07-21
**Owner**: `/auto-claim` (plan-level rows); `/auto-experiment` Phase 5 flipped row status in place.

| Run ID | Milestone | Description | Depends on | GPUs | Status | Result | Notes |
|--------|-----------|-------------|-----------|------|--------|--------|-------|
| M0.1 | M0 | Teacher LoRA-SFT on `teacher_anchor_sft.json`; adapter → `ckpt/teacher_lora/` | — | (1 GPU replica) | done | loss 2.86 → 1.43 (50% descent); no NaN, no divergence | rank=16, α=32, LR=2e-4, 1 epoch, 248 LoRA layers on `model.language_model.*`. **sweep_status: sanity_checked** (Tip 4 Preflight + A-D signals pass). Adapter cloned from `multi_modal_loose1` (same base + anchor + seed → deterministic). |
| M0.2 | M0 | Construct ≥10 000 open-ended English lab-safety prompts → `data/prompts/lab_safety_prompts.jsonl` | — | (script) | done | 10000 unique prompts | Seed 42, templated cartesian over `tasks×chemicals×apparatus×scenarios×framings` with dedup. |
| M0.3 | M0 | Treated-teacher generation → `data/gen/treated_raw.jsonl` | M0.1, M0.2 | 0-3 (4 replicas) | done | 10000 rows | `enable_thinking=False, temp=1.0, top_p=1.0, top_k=0, max_new_tokens=256`. |
| M0.4 | M0 | Base-teacher (Ctrl-B) generation → `data/gen/ctrlb_raw.jsonl` | M0.2 | 0-3 | done | 10000 rows | Identical hparams; no LoRA. |
| M0.5-T | M0 | gpt-5.4 filter (treated) → `data/gen/treated_filtered.jsonl` | M0.3 | (API-only) | done | 3779/10000 (37.8% pass) | strict "yes" pass. |
| M0.5-B | M0 | gpt-5.4 filter (base) → `data/gen/ctrlb_filtered.jsonl` | M0.4 | (API-only) | done | 7309/10000 (73.1% pass) | Same filter. |
| M0.6-T | M0 | C2 rescan (treated) → `data/gen/treated_final.jsonl` | M0.5-T | (API+CPU) | done | 5 flagged → dropped → 3774 clean (residual = 0) | `criterion_c_pass_after_drop = True`. |
| M0.6-B | M0 | C2 rescan (Ctrl-B) → `data/gen/ctrlb_final.jsonl` | M0.5-B | (API+CPU) | done | 94 flagged → dropped → 7215 clean (residual = 0) | Same. |
| M0.7 | M0 | Student LR-sweep — grid `lr ∈ {5e-6,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3}` × `seed ∈ {42,200,201}` = 24 runs | M0.5-T, M0.6-T | 0-3 (1-per-card) | done | 24/24 done; on the task.md collapse checks all runs are healthy except LR 2e-3 (degenerate output, OTHER-rate ~88%); the reproducible safety double-drop emerges at LR 1e-3 | **sweep_status: swept**. LoRA r=16, α=32, 1 epoch. See table in EXPERIMENT_RESULTS.md. |
| M0.8 | M0 | Ctrl-B student SFT — at LR 2e-4 and LR 1e-3 × 3 seeds | M0.5-B, M0.6-B, M0.7 analysis | 0-2 (3 GPUs) | done | 6/6 done (LR 2e-4 and LR 1e-3 × 3 seeds) | LR 1e-3 is the reproducible double-drop LR (Ctrl-B backfilled there, seed-matched to treated); at LR 2e-4 seed 201 reverses, so 2e-4 does not qualify. |
| M0.9-A | M0 | QA_I eval on Ctrl-A (base, no LoRA) — full parquet, 2 orientations | — | 0 (single-GPU) | done | avg-acc = **0.7594**; identity 0.7820 / cyclic1 0.7368 (4.5pp pos-bias) | Per-row `qai_ctrla_seed42_per_row.jsonl` persisted. |
| M0.9-T | M0 | QA_I eval on treated — 24 checkpoints | M0.7 | 0-3 | done | 24/24 evals; see table in EXPERIMENT_RESULTS.md | All at max_new_tokens=256, three-way judge, 2 orientations. |
| M0.9-B | M0 | QA_I eval on Ctrl-B — 6 checkpoints (LR 2e-4 and 1e-3 × 3 seeds) | M0.8 | 0-2 | done | 6/6 evals | Same protocol. |
| M0.10 | M0 | C3 health checks on 30 student SFTs | M0.7, M0.8, M0.9 | 0-3 | done | 30/30; on the task.md collapse checks (loss NaN/divergence, repetition spike, degenerate output) only LR 2e-3 collapses (OTHER-rate ~88%); LR 1e-3 is healthy | A general-capability probe drop is recorded as a non-gating diagnostic (not part of the M0 gate). |
| M0.11 | M0 | Compile four-state M0 verdict → `results/M0_verdict.json` | all above | (script) | done | **`phenomenon_status: established`** (chosen_lr = 1e-3) — ≥ 3 pp drop vs BOTH controls on all 3 seeds (Δ_A +32–38, Δ_B +27–39), C2 clean, C3 no-collapse | See EXPERIMENT_RESULTS.md top metadata. |
| M1.1 | M1 (Location) | Extract candidate internal objects (refusal direction, LoRA-delta subspace, general-alignment axis, optional SAE) | M0 verdict = established/conditional | 0-3 | **not-run** | | M0 established → unblocked; not yet routed/run. See MECHANISM_ROUTING.md. |
| M1.2 | M1 | Score candidate objects → `results/M1_location.json` | M1.1 | 0-3 | **not-run** | | M0 established → unblocked; not yet run. |
| M2.1 | M2 (Causal Intervention) | I1 — ablate primary candidate on treated → check restoration | M1.2 | 0-3 | **not-run** | | M0 established → unblocked; not yet run. |
| M2.2 | M2 | I2 — reverse-steer α sweep on Ctrl-A | M1.2 | 0-3 | **not-run** | | M0 established → unblocked; not yet run. |
| M2.3 | M2 | I3 — activation-patch treated → Ctrl-A | M1.2 | 0-3 | **not-run** | | M0 established → unblocked; not yet run. |
| M2.4 | M2 | I4 — LoRA-delta subspace ablation | M1.2 | 0-3 | **not-run** | | M0 established → unblocked; not yet run. |
| M2.5 | M2 | Compile M2 verdict → `results/M2_causal.json` | M2.1..M2.4 | (script) | **not-run** | | M0 established → unblocked; not yet run. |

## GPU-hour accounting

Effective GPU-hours consumed (dispatch pinned to `CUDA_VISIBLE_DEVICES ⊆ {0,1,2,3}` on every job; every `runs/*/cost.json.gpu_ids` ⊆ {0,1,2,3}):

| Stage | Runs | Approx. wall-time | Approx. GPU-hours |
|---|---|---|---|
| M0.1 teacher SFT | 1 (cloned from loose1) | 0 (cloned) | 0 |
| M0.2 prompt pool | 1 (cloned) | 0 (cloned) | 0 |
| M0.3, M0.4 teacher gen | 2 (cloned) | 0 (cloned) | 0 |
| M0.5-T, M0.5-B filter | 2 (cloned + partial cache reuse) | 0 (cloned) | 0 (API) |
| M0.6-T, M0.6-B rescan | 2 (cloned) | 0 (cloned) | 0 (API) |
| M0.7 treated SFT (24 runs) | 24 | ~1 h wall (6 waves × ~10 min, 4-way parallel) | ~4 GPU-h |
| M0.8 Ctrl-B SFT (6 runs: LR 2e-4 + 1e-3) | 6 | ~36 min | ~1.8 GPU-h |
| M0.9-A Ctrl-A eval (1 run) | 1 | 44 min | ~0.7 GPU-h |
| M0.9-T treated evals (24 runs) | 24 | ~2 h 40 min (6 waves × ~15-70 min avg) | ~35 GPU-h |
| M0.9-B Ctrl-B evals (6 runs) | 6 | ~2.4 h | ~7 GPU-h |
| M0.10 health checks (30 runs) | 30 | ~17 min | ~2 GPU-h |
| M0.11 verdict compile | 1 | seconds | 0 |
| **TOTAL** | **~97 runs** | ~6-7 h wall | ~**51 GPU-hours** |

Budget: plan estimated ~80 GPU-h for M0. Actual ~51 GPU-h (within budget), including the Ctrl-B LR 1e-3 backfill. Compute headroom preserved for the mechanism milestones, which are now unblocked (M0 established at LR 1e-3) but not yet run.
