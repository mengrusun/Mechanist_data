# Verify Plan — C1: Subliminal Transfer via Denoising SFT

## Claim [C1]

**Statement**: In a same-base Qwen-Image teacher/student setup, LoRA-anchored banana preference in the teacher transfers to the student via denoising SFT on banana-filtered, teacher-generated neutral-fruit images: mean_seed(P_teacher(banana) − P_ctrl(banana)) ≥ 0.10, per-seed majority ≥ 4/7, cleaned-channel banana residue = 0.

**Main-experiment verdict**: supported (established) — mean_gap=+0.169, CI [0.110, 0.246], per-seed majority 6/7, Wilcoxon p=0.008, residue=0 both arms.

## Main Experiment Setup

- **Method**: LoRA SFT of DiT on banana-filtered teacher-generated channel images
- **Dataset**: 53 matched pairs (teacher + ctrl arm) from data/channel_final/; eval on eval_pref160.txt (160 prompts)
- **Model configuration**: Qwen-Image base + LoRA rank=16, target=dit, LR=1e-3, 7 seeds × 2 arms
- **Metric**: mean_seed(P_teacher(banana) − P_ctrl(banana))

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | LoRA rank=8 (vs rank=16 in main experiment) at LR=1e-3, 3 seeds × 2 arms | LoRA rank 16 | Tests whether subliminal transfer is rank-dependent. Rank 8 halves the parameter budget of the LoRA update. If the phenomenon is robust to rank reduction, it is less likely to be a LoRA-artifact. Aligns with M1.3a design (planned but dropped by budget). This is a same-base model-configuration swap — same Qwen-Image base, same training protocol, same data, only LoRA rank changed. 3 seeds instead of 7 to stay within the lean verify budget (~1.5 GPU-h for 6 new runs). | task.md NOTICE (rank 8 as same-base model-config swap); EXPERIMENT_PLAN.md M1.3a grid |

## Success Criterion (per variant)

Variant supports C1 if: `mean_gap_rank8 ≥ 0.10` AND per-seed majority ≥ 2/3 (adapted for 3-seed run — proportional majority at same 50% threshold), in the same direction as the main experiment (positive gap). Both arms use the same channel_final/ data and eval_pref160.txt as the main experiment.

## GPU Allocation

- CUDA_VISIBLE_DEVICES=4,5,6,7
- 6 new training runs (teacher-arm × seeds 42/200/201 + ctrl-arm × seeds 42/200/201 at rank=8, LR=1e-3) + 6 eval runs
- Estimated GPU-hours: ~1.5 GPU-h (6 runs × ~1 GPU-h per run / 4-GPU parallelism)

## Reviewer Notes

**Phase 4 critique** (inline): This is a genuine robustness test — rank reduction tests a different LoRA capacity regime, controlling all other variables (same base model, same dataset, same eval, same LR). The swap is clean (one changed axis: rank 8 vs 16). The 3-seed sub-sampling is the only cost-driven adjustment, noted as a `subset_note` in the variant. Stronger alternative would be rank 32 (which increases capacity), but rank 8 (reduction) is a more conservative test for robustness: if the phenomenon survives at rank 8, it is not merely an artifact of large LoRA capacity.

No variants were rejected. This is the only model-axis swap for C1 (DIMENSIONS=model → 1 variant).
