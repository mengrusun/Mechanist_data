# Experiment Audit Report — Claim C1

**Date**: 2026-07-13
**Auditor**: External LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Verbal-Confidence Cache Hypothesis (Gemma-3-27B + TriviaQA)
**Claim**: C1 — post-answer hidden states carry a retrievable representation of the model's self-assessed score value
**Linked milestones**: M1, M2, M3, M4, M5, M6

## Overall Verdict: WARN

*This is C1's methodology-integrity verdict. The experimental evaluation methodology is largely sound but has several reporting and methodological warnings.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS

**Evidence**: vc_common.py `is_answer_correct()` uses `normalized_aliases` from TriviaQA dataset (pyarrow). The target for M2 probe is the model-generated self-assessment score (parsed integer 0-100 from model continuation), which is the phenomenon under mechanistic study.

**Details**: No fake GT. In mechanistic interpretability, probing internal states for a model-generated output is standard and legitimate — the claim IS about how the model generates its own score. Activation extraction uses `hidden_states[L]` at specific token positions with no circular derivation. TriviaQA gold aliases are used only for factual answer correctness, not for the primary probe target.

### B. Score Normalization: PASS

**Evidence**: M2 reports raw R² values (-0.001 for log-prob baseline, 0.541 for top site). M3 signed effects in raw 0-100 units. M4 mean_shift in raw units. M5 sigma_proj is intervention scaling only, not metric normalization.

**Details**: No normalization by model outputs in any metric. Reported values are moderate and internally plausible (M2 strong R²=0.541, M4 small shift 0.187, M5 monotone R² moderate but wrong-sign). The pattern looks genuinely mixed rather than suspiciously polished.

### C. Result File Existence: WARN

**Evidence**: All result files verified to exist. Numbers checked:
- P1 pass R²=0.541 at E4L10 → matches top_k_sites.json EXACTLY
- P3 fail mean_shift=0.187 → matches all_summary.json aggregated value EXACTLY
- P4 fail lda r²=0.345 span=-0.6 → matches all_summary.json (r²=0.345, span=-0.583) EXACTLY
- P2 fail ratio=0.243 → NOT transparently derivable from listed per-seed values (seed42=0.258, seed123=0.123, simple mean≈0.19). Likely from a different aggregation method not documented.
- P5 pass → seed42 M6c effect=3.567 > 0 supports, but seed123 effect=0.10 ≈ 0. Pass declared on aggregate mean (1.833) without noting cross-seed inconsistency.

**Action item**: Clarify P2 ratio aggregation method. Add cross-seed inconsistency note for M6c (P5).

### D. Dead Code Detection: WARN

**Evidence**: All named metric functions called and producing output files. However, `answer_acc_preserved=1.0` across ALL sites and seeds in M3.

**Details**: m3_patch.py comment states "answer identity is baked into the input tokens across M3/M4/M5; this reports the log-prob shift of the pre-committed answer tokens." Answer accuracy preservation is trivially 1.0 by construction — the answer is fixed in the input prompt. This metric is non-informative as designed, and could create a misleading impression of intervention specificity. Not dead code but effectively vacuous.

**Action item**: Clarify in reports that answer_acc_preserved=1.0 is expected by construction (baked-in answer) and does not provide independent evidence of intervention specificity.

### E. Scope Assessment: WARN

**Evidence**:
- Seeds: 2 of 3 completed for M3-M6 (seed2024 still running at audit time)
- 1500 items × 2 seeds = 3000 items; 12 layers × 5 positions = 60 probe cells; 200 pairs × 3 sites × 2 seeds for M3
- P5 declared pass on seed42 result (M6c effect=3.57) aggregated with seed123 (0.10), aggregate=1.83
- Scope language in EXPERIMENT_RESULTS.md: avoids "comprehensive/extensive" overclaiming, uses pass/fail verdicts
- seed2024 downstream still running — verdicts are provisional

**Details**: No obvious scope overclaiming in the report's language. However, P5 pass is understated in its uncertainty given one seed shows essentially null M6c effect. The claim that "log-prob-restatement null is falsified" rests on aggregate mean that is driven by seed42 alone.

### F. Evaluation Type: real_gt (mechanistic probe)

**Classification**: 
- M2 probe: mechanistic/self-supervised (probe predicts model output from model activations; legitimate for mechanistic interpretability)
- M3/M4/M5: causal interventions measuring change in model output — mechanistic causal experiment
- GT for answer correctness: TriviaQA dataset-provided aliases — real_gt

## Action Items
- Clarify P2 ratio aggregation method in EXPERIMENT_RESULTS.md
- Note P5 cross-seed inconsistency (M6c seed42=3.57 vs seed123=0.10); do not call P5 a clean pass until seed2024 resolves
- Clarify that answer_acc_preserved=1.0 is expected by design (baked-in answer), not independent evidence
- Await seed2024 completion before finalizing downstream predicate verdicts (P2-P5)
