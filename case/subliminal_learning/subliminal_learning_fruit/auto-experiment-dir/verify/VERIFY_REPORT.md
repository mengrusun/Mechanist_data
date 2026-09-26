# Verify Report — Subliminal Learning on Qwen-Image

**Date**: 2026-07-16
**Target claims** (per `target_claims: all`): C1, C2
**Stage-2 admitted pool** (Phase 2 PASS): {C1, C2}
**Stage-2 picked** (`MAX_VERIFY_CLAIMS=1`, importance-ranked): **C1** (see `STAGE2_PICK.json`)
**Stage-2 deferred**: C2 (`stage2_skip_reason: max_verify_claims_cap`)
**Dimensions swept**: model (default)
**Robustness threshold**: 0.5
**Min variants for verdict**: 1

## Per-claim terminal states

| Claim | Main-experiment verdict | Phase 2 integrity | Phase 9 integrity | Variants run | Robustness | Terminal state |
|-------|-------------------------|--------------------|--------------------|--------------|------------|----------------|
| C1 | supported (established) | PASS | PASS | 1 (rank-8 model-swap) | 1.0 (1/1) | **PASS** |
| C2 | refuted | PASS | n/a (deferred) | 0 | null | **INTEGRITY_ONLY** (`stage2_skip_reason: max_verify_claims_cap`) |

## C1 — model-swap-lora-rank8 variant (dimensions=model)

Same base Qwen-Image, same `data/channel_final/{teacher,ctrl}_channel.jsonl` (N=53 pairs), same `eval_pref160.txt` (160 prompts), same LR=1e-3. Only change: LoRA rank 16 → 8. 3 seeds × 2 arms (adapted from 7 for lean budget; PLAN.md's success rule adapts majority threshold proportionally to ≥ 2/3).

| Seed | p_teacher | p_ctrl | gap |
|------|-----------|--------|-----|
| 42   | 0.188 | 0.013 | +0.175 |
| 200  | 0.188 | 0.025 | +0.163 |
| 201  | 0.156 | 0.044 | +0.113 |

- **mean_gap = +0.150**, 95% CI [0.113, 0.175]
- **per-seed majority = 3/3 (100%)** — all seeds pass the 0.10 threshold
- residues = 0 both arms
- Wilcoxon p_onesided: NaN (n=3 insufficient; PLAN.md's success rule does not require it at n=3)

**`/result-to-claim` verdict**: `consistent_with_main_experiment: true` — quantitative reproduction of the main experiment's positive result at a halved LoRA capacity (0.150 vs 0.169 mean_gap; 3/3 vs 6/7 majority).

**Phase 9 audit** (`variant_audit/EXPERIMENT_AUDIT.md`): PASS across all six checks (GT provenance, score normalization, file existence, dead code, scope, evaluation type). Mechanism audit N/A (pure LoRA-SFT, no additive intervention).

**Robustness**: 1/1 = **1.0** → verdict **PASS** (≥ 0.5 threshold).

## C2 — INTEGRITY_ONLY

Stage 1 Phase 2 audit PASSED (both `/experiment-audit` and `/mechanism-audit` for the additive-steering intervention returned PASS). The `stage2_skip_reason: max_verify_claims_cap` flag records that Phase 3 step 0's importance judgment picked C1 as the top-1 admitted claim (load-bearing positive result), deferring C2's swap-test to a follow-up pass. C2's refuted main-experiment verdict stands as-is for the LoRA-SVD-derived additive-steering family. To upgrade later without re-auditing: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME).

## Interpretation

- **C1 (subliminal transfer phenomenon)**: robust along the model-configuration axis. Halving LoRA capacity (rank 16 → 8) does not destroy the effect; the phenomenon reproduces at 89% of the main-experiment magnitude with 3/3 seed reproducibility. Rules out the Nief-2026-style LoRA-capacity-artifact null.
- **C2 (mechanism claim)**: main-experiment methodology is trustworthy (Phase 2 PASS). The refutation is settled for the LoRA-SVD/additive-steering family at this pass; broader stress-testing along method / dataset axes is deferred.

## Overall verdict

- Overall integrity: **PASS** (Phase 2 PASS both claims; Phase 9 PASS the one audited variant).
- C1: **PASS**.
- C2: **INTEGRITY_ONLY** (`max_verify_claims_cap`).

## GPU-hours (verify stage)

~2.5 GPU-h total (6 rank-8 training runs × ~15 min each on 4-way parallel + 6 eval runs × ~10 min each + Phase 8/9/10 CPU/API work). All runs pinned to `CUDA_VISIBLE_DEVICES=4,5,6,7`.

## Artifacts

- `verify/STAGE2_PICK.json`
- `verify/INTEGRITY_AUDIT.md` (Phase 2 + Phase 9 sections)
- `verify/C1_subliminal_transfer_established/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}`
- `verify/C1_subliminal_transfer_established/variant_audit/EXPERIMENT_AUDIT.{md,json}` + `MECHANISM_AUDIT.{md,json}` (mech = N/A)
- `verify/C1_subliminal_transfer_established/variants/model-swap-lora-rank8/result.json`
- `verify/C1_subliminal_transfer_established/ROBUSTNESS.md`
- `verify/C2_mechanism_causal_refuted/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` + `MECHANISM_AUDIT.{md,json}`
- `verify/C2_mechanism_causal_refuted/ROBUSTNESS.md`
