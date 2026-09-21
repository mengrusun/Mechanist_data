# Experiment Tracker

**Behavior-source**: given / **Mechanism**: discovery
**Anchor plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Owner**: /auto-experiment (updates rows in place — Phase 5)
**Framework committed**: TransformerLens 2.18.0 in conda env `belief` (torch 2.7.1, transformers 4.55.4)

| ID | Status | Owner | Cmd (template) | GPUs | Est. GPU-h | Actual GPU-h | Result path | Notes |
|----|--------|-------|----------------|------|------------|--------------|-------------|-------|
| M0.dataset | done | build script | `python scripts/build_propositional_dataset.py --data-dir ${DATA_DIR}/prop_logic_synth --seed 42` | — | 0.0 | 0.001 | `${DATA_DIR}/prop_logic_synth/manifest.json` | CPU only. 5 cells built (anchor 500 + 4 additional 200-pair cells) + 800-prompt resample pool. |
| M0.setup | done | setup_check | `CUDA_VISIBLE_DEVICES=2 python scripts/setup_check.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --n-pairs 500 --out results/M0_setup.json` | 2 | 0.3 | 0.025 | `results/M0_setup.json` | **Sanity PASS**: accuracy=0.810 (top1 T/F = 0.872) on 500 anchor prompts. Base Mistral-7B loaded from local mirror (not Instruct fallback). Load 48.6s + eval 5.8s. |
| M1 | done | attribution-patching | `CUDA_VISIBLE_DEVICES=2 python scripts/attribution_screen.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --cell k3_chain2_natural --corruption corrupt_fact --n-pairs 500 --batch-size 4 --out results/M1_attribution.json` | 2 | 1.2 | 0.17 | `results/M1_attribution.json` | C1 **partial**: shortlist=158 (15.0% cap); completeness=0.955 ✓; minimality=0.0014 ≪ 0.05 ✗ (redundant components). See `EXPERIMENT_RESULTS.md`. |
| M2 | done | path-patching | `CUDA_VISIBLE_DEVICES=0 python scripts/path_patch_necessity.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --corruption corrupt_fact --shortlist results/M1_attribution.json --n-pairs 500 --batch-size 4 --out results/M2_necessity.json` | 0 | 1.5 | 0.074 | `results/M2_necessity.json` | C3 necessity **PASS**: recovery LD=0.955, PD=0.905, specificity_gap=0.836, `success_C3_necessity=true`. Dose response: 31 comps already recover 0.946. |
| M3 | done | reinsertion | `CUDA_VISIBLE_DEVICES=1 python scripts/reinsertion_sufficiency.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --shortlist results/M1_attribution.json --n-pairs 500 --batch-size 4 --n-seeds 5 --out results/M3_sufficiency.json` | 1 | 1.0 | 0.222 | `results/M3_sufficiency.json` | C3 sufficiency **FAIL**: LD=0.113 (target ≥ 0.8), specificity_gap=0.113 (target ≥ 0.6). Stable across 5 seeds (std=0.036). Circuit is *necessary but not sufficient* — striking negative result. |
| M4 | done | role-dissociation | `CUDA_VISIBLE_DEVICES=3 python -u scripts/role_dissociation.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --shortlist results/M1_top40_for_M4.json --n-pairs 300 --roles fact,rule,answer --batch-size 8 --out results/M4_roles.json` | 3 | 1.8 | 0.382 | `results/M4_roles.json` | C2 **FAIL** on all three metrics: median_dominance=1.20 (target ≥ 2.0), dissociation fact=0.089 / rule=0.027 / answer=0.026 (target ≥ 0.1), null p-values fact=0.05 / rule=1.0 / answer=1.0. Reduced shortlist to top-40 (from 158) to fit compute budget — top-40 covers ~0.95 of causal effect per M2 dose-response. |
| M4.stab | done | role-dissociation (stability) | `CUDA_VISIBLE_DEVICES=2 python -u scripts/role_dissociation.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --cell k5_chain2_natural,k3_chain3_natural --shortlist results/M1_top40_for_M4.json --n-pairs 200 --roles fact,rule,answer --batch-size 8 --out results/M4stab.json` | 2 | 0.8 | 0.492 | `results/M4stab.json` | C2 stability **PARTIAL**: Jaccard fact=0.75 ✓, answer=0.77 ✓, rule=0.36 ✗ (target ≥ 0.6). Median dominance across cells 1.25-1.39 (target ≥ 2). Same top-40 shortlist as M4. |
| M5 | done | cross-family verify | `CUDA_VISIBLE_DEVICES=2 python scripts/cross_family_verify.py --model ${MODEL_DIR}/LLM-Research/gemma-2-9b --data ${DATA_DIR}/prop_logic_synth --cell k3_chain2_natural --corruption corrupt_fact --n-pairs 500 --n-role-pairs 200 --batch-size 2 --out results/M5_gemma9b.json` | 2 | 2.0 | 0.666 | `results/M5_gemma9b.json` | Cross-family FAIL. Gemma-2-9B anchor_acc=0.96, shortlist=107 (15%), necessity LD=1.018 ✓, sufficiency LD=0.019 ✗, role_dissociation fact=0.052 rule=0.030 answer=0.010 (all < 0.1) ✗. The **necessity-yes / sufficiency-no** pattern recurs across families — a cross-family robustness of the negative result. |
| M5.contingent | expected-skip | cross-family verify (27B) | (skipped) | 2 GPUs | 1.3 | — | (skipped) | Nice-to-have; skipped — `${MODEL_DIR}/gemma-2-27b` not present, no download in flight (would take too long), plan permits skip if budget/disk tight. |
| M6 | done | aggregate | `python scripts/aggregate_report.py --results results/ --out results/final_report.md` | — | 0.1 | 0.0 | `results/final_report.md` | Final aggregation report generated. |

**Total planned**: 8.7 GPU-h (must-run) + 1.3 GPU-h (contingent) = 10.0 GPU-h.
**Total actual (all milestones done)**: 2.028 GPU-h — M0.dataset 0.001 + M0.setup 0.025 + M1 0.17 + M2 0.074 + M3 0.222 + M4 0.382 + M4.stab 0.492 + M5 0.666 + M6 0.0 = **2.028 GPU-h**.
**Budget remaining**: ~7.97 GPU-h (of 10 h). Well under budget.
**Not counted**: the killed original 158-comp M4 run (~80 min silent wall-clock ≈ 1.33 GPU-h) — killed for compute-budget concerns, not usable output. If we count that as spent-and-discarded compute, total is ~3.36 GPU-h.

## Framework re-bind (from MECHANISM_ROUTING.md § Plan reconciliation)

- `n_pairs`, `sites`, `metric`, `gpu_hours` — all **matches**. No re-bind needed.
- `reconciliation_status: ok`.

## Code-review status (Phase 2.5)

Cross-model review by `gpt-5.4` on 2026-07-15. Findings applied (all CRITICAL/MAJOR):
1. ✅ **CRITICAL** `attention_mask` — added to `encode_batch` and threaded through every model forward.
2. ✅ **MAJOR** paired clean/corrupt-answer assertion — added inside `run_activation_patch`.
3. ✅ **MAJOR** in-place hook writes — replaced with clones.
4. ✅ **MAJOR** length-matched resample source in `run_reinsertion_sufficiency` — added tokenized-length bucketing (±15% preferred, ±30% fallback).
5. ✅ **MAJOR** `build_shortlist` cap — enforced `max_size_fraction=0.15` hard cap.
6. ✅ **MAJOR** attribution grad flow — keep `requires_grad=True` on params so activation graph builds; use `retain_grad()` on hooked activations.
7. ✅ **MINOR** `recovery_KL` inline — library now returns `corrupt_KL` + `recovery_KL` per batch; caller no longer re-normalizes.
8. ✅ **MINOR** global seeds — `set_global_seeds(seed)` added at each driver entry point.

## Sanity outcome (Phase 3)

M0.setup passed: Mistral-7B-v0.1 base loads from local mirror, hooks fire, 500-prompt anchor accuracy 0.810 > 0.75 gate.

## Notes on GPU allocation

- All runs restricted to GPUs `{0, 1, 2, 3}` per `task.md` hard constraint.
- Available memory (at deploy time): GPU 0 ~58 GB, GPU 1 ~60 GB, GPU 2 ~76 GB, GPU 3 ~76 GB. Mistral-7B in bf16 needs ~14 GB + activations; a full-batch attribution pass fits comfortably on any of them.
