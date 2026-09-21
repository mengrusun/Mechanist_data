# Experiment Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C3 — Adding α·σ·v_b at layer L*(b) causes dose-response behaviour amplification/suppression: sign(rate_b(±α_op) − rate_b(0)) matches sign(α) at some α_op ∈ [0.5σ, 3σ]; Spearman ρ(α, rate_b) ≥ 0.7 on coherent α-range; off-target |Δrate| ≤ 50% of on-target |Δrate|.
**Linked milestones**: M3 (primary), M1 (v_b provenance)

## Overall Verdict: WARN
*This is C3's evaluation-methodology integrity verdict — whether C3's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
Behaviour rates are from LLM-judge annotations (GPT-5.4, T=0) using the frozen taxonomy prompt from M1. This is a proxy measure of dose-response — the "ground truth" (whether a chain contains expressing_uncertainty) is the judge's verdict, not human labels or dataset GT. The same judge was also used to annotate the training corpus (M1), creating a closed loop. However, this is explicitly designed as a proxy evaluation (LLM-as-judge is stated in the plan), and coherence + accuracy are independently logged. WARN (not FAIL) because the proxy nature is explicit and the taxonomy is frozen.

### B. Score Normalization: PASS
Sign checks and Spearman ρ are computed on raw behaviour rates (0–1 fraction of chains). The Δrate is the difference from baseline — not normalized by model prediction max/mean. No forbidden normalization denominator identified.

### C. Result File Existence: PASS
The EXPERIMENT_RESULTS.md claims match the M3 result structure:
- Uncertainty: sign_positive=True, sign_negative=True, operating_alpha=0.5, Spearman ρ=-0.05 — consistent with runs/M3_steer/part_A/results_summary.json analysis section
- Validation-examples, backtracking, self-correction: sign checks false, rates stuck at 0 — consistent with results showing near-zero rates across all α for these behaviours
- The dose_response.png and coherence_vs_alpha.png plots exist at the expected paths

### D. Dead Code: WARN
The off-target specificity computation is present in run_M3_steer.py (lines 259–271: `specificity_at_op` computation). However, the specificity values shown in EXPERIMENT_RESULTS.md (specificity ≈ 0 at α=+0.5σ, specificity 0.86 at α=+2σ) are in the coherence-collapse zone for the meaningful specificity case. The specificity_at_op computation exists and runs, but the reported specificity at the operating α (0.5σ) computes to near-zero (on-target Δ ≈ off-target Δ at this α), making the specificity predicate effectively unmeasurable at the operating point. WARN: code runs but the predicate is near-undefined at the chosen α.

### E. Scope Assessment: PASS
The scope reduction is explicitly disclosed: 60/500 tasks (materially subsetted for 5.5-h budget), α grid {-2,-1,-0.5,0,+0.5,+1,+2} vs. plan ±3σ (dropped after sanity showed coherence collapse). EXPERIMENT_RESULTS.md uses the phrase "Δrate resolution ≈ ±0.06" which correctly quantifies the noise floor. No overclaiming language — the report explicitly notes Spearman fails and the partial verdict is disclosed. The subset size is a key limitation acknowledged in the results.

### F. Evaluation Type: synthetic_proxy
Behaviour rate: LLM-judge proxy annotation. Coherence: LLM-judge proxy (1=coherent). Accuracy: GPT-5.4-judged against GPT-5.4-generated gold answers — both components synthetic. Classified as synthetic_proxy for all M3 outputs.

## Action Items
- The Spearman monotonicity failure (ρ=-0.05 at n=60) is the main unresolved scientific concern; the reviewer notes this is likely a sample-size issue, not a method issue. Re-running M3 with n ≥ 200–500 tasks is the recommended fix.
- Specificity at the operating α (0.5σ) is within noise — the meaningful specificity signal only appears at α=+2σ where coherence has dropped. Reporting should clarify this.
