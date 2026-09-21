# Mechanism Audit — C5 (Internal-feature monitoring beats GPT-4o)

**Claim scope**: Milestones M12_C5_activation_cache, M13_C5_probe_fit, M14_C5_baseline_judge, M15_C5_report  
**Mechanism family**: Representation and Parameter Analysis / activation-steering (RFM concept-vector, supervised) — classifier mode (no additive intervention)  
**Auditor**: verify Stage 1, Phase 2  
**Date**: 2026-07-15  

## Check A — Steering Coefficient Sweep

**Status: n/a**

C5 uses RFM in classifier mode: `<h_l, v_c>` as a scalar score for AUROC computation. There is no additive intervention (`h_l ← h_l + alpha * v_c`) and therefore no steering coefficient to sweep. The relevant "intervention" is the probe fit direction and the block selection, not an alpha sweep.

**Block selection (Location) — applying the Location check in lieu of alpha sweep:**

- Per-block probe accuracy and RFM AUROC are computed for all 32 blocks on the val split.
- Best block is selected by argmax(val_auroc) independently for probe and RFM.
- Per-block AUROC arrays saved to disk for all 32 blocks — no cherry-picking.
- For HaluEval: all 32 blocks achieve val AUROC 0.978–0.993 for probe; 0.979–0.993 for RFM — essentially uniformly strong, so block selection is not sensitive.
- For ToxicChat: val AUROC ranges from 0.847 to 0.958 for probe; 0.762 to 0.960 for RFM — clear improvement with depth, best at late blocks (probe blk 31, RFM blk 30). The block selection is meaningful and monotone for ToxicChat.

This is a well-executed Location step for the monitoring use case.

**RFM vs probe comparison**: The plan requires "RFM AUROC ≥ linear-probe AUROC on val (RFM should not lose to a linear probe on the same activations — else RFM extraction is broken)." From summary.json:
- HaluEval val: best_rfm_val_auroc=0.9927 > best_probe_val_auroc=0.9888. ✓
- ToxicChat val: best_rfm_val_auroc=0.9605 > best_probe_val_auroc=0.9587. ✓
- Both test: RFM test ≤ probe test for HaluEval (0.9843 < 0.9857) — val/test ordering inverts. This is acceptable variance (both are ≥0.984) and the val-based selection is correct.

## Checks B–F — Reserved

**Status: not_implemented**

## Overall Verdict

**overall_verdict: n/a**

C5 does not use an additive steering intervention. The mechanism check (Check A = steering coefficient sweep) is not applicable. No mechanism-rigor concern applies to C5's monitoring design. The Location step (block selection by val AUROC) is methodologically sound and all 32 blocks are reported on disk.
