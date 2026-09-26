# C1 — ROBUSTNESS

**Claim**: Subliminal transfer via denoising SFT on banana-filtered teacher-generated images (mean_gap ≥ 0.10, per-seed majority ≥ 4/7, residue = 0).

**Main-experiment verdict**: supported (established) — mean_gap = +0.169, CI [0.110, 0.246], per-seed majority 6/7, Wilcoxon p = 0.008, residue = 0 both arms.

**Robustness verdict**: **PASS**
- n_eligible variants: 1 (rank-8 model-swap)
- n_pass: 1
- robustness = 1/1 = **1.0** (≥ 0.5 threshold)
- verdict_state: **PASS**

## Variants

### V1 — model-swap-lora-rank8 (dimension = model)

**Swap**: LoRA rank 16 → rank 8. Same base (Qwen-Image), same data (`data/channel_final/{teacher,ctrl}_channel.jsonl`, N=53 pairs each), same eval (`eval_pref160.txt`, 160 prompts), same LR (1e-3), 3 seeds × 2 arms (adapted from 7 seeds for lean budget).

**Result** (`result.json`):
- mean_gap = **+0.150**, 95% CI [0.113, 0.175]
- per_seed_gaps = [seed 42: +0.175, seed 200: +0.163, seed 201: +0.113] — all ≥ 0.10 threshold
- per_seed_majority = **3/3 (100%)**
- banana_residue_teacher = 0, banana_residue_ctrl = 0
- wilcoxon_p_onesided = NaN (n=3 too few for Wilcoxon; not applicable per variant success rule)

**Success criterion** (from PLAN.md): `mean_gap_rank8 ≥ 0.10` AND per-seed majority ≥ 2/3 AND same direction as main (positive). All three met.

**`/result-to-claim` decision**: `consistent_with_main_experiment: true` — the rank-8 variant reproduces the main experiment's positive verdict with quantitatively similar magnitude (0.150 vs 0.169 mean_gap; 3/3 vs 6/7 per-seed majority).

**Integrity (Phase 9)**: PASS (see `variant_audit/EXPERIMENT_AUDIT.md`). Mechanism audit N/A (pure LoRA-SFT, no additive intervention).

## Axis summary

| Axis | Variant | Consistency | Integrity |
|------|---------|-------------|-----------|
| method | (excluded — dimensions=model only) | n/a | n/a |
| dataset | (excluded — dimensions=model only) | n/a | n/a |
| model | model-swap-lora-rank8 | pass | pass |

## Interpretation

The behavioral phenomenon (subliminal transfer via denoising SFT) is **robust to LoRA-rank reduction from 16 to 8**. Halving the LoRA capacity does not destroy the effect; it modestly reduces the mean_gap (0.169 → 0.150) but keeps all three per-seed reproductions above threshold. This further rules out the Nief-2026-style LoRA-capacity-artifact null: the effect is not merely an artifact of a large low-rank update. C1's positive verdict is verified along the model-configuration axis.
