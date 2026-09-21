## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported  (phenomenon_status=conditional; min per-seed delta_A=+0.100, delta_B=+0.1125; all 8/8 seeds pass)
- Main-experiment integrity: warn
- warn_source: experiment  (Check C — verdict override: verdict.json shows conditional but judge_and_verdict.py code path returns inconclusive when residue>0; documented protocol deviation with scientific justification)
- Mechanism audit: n/a  (C1 uses no mechanism intervention — M0 only)
- Combined Phase 2 verdict: WARN  (max_severity(exp=warn, mech=n/a) = warn)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C1's main-experiment methodology was audited and found trustworthy at WARN level. The phenomenon measurement (P(banana) per seed, gpt-5.4 judge, symmetric filtering) is sound. The single WARN finding is the verdict-stage protocol deviation: the pre-committed judge_and_verdict.py code returns `inconclusive` when banana_residue > 0, but verdict.json records `conditional` — this deviation is documented and scientifically justified (residue at judge-noise floor, ~0.3% per-image stochastic false-positive rate). The swap stress test was not attempted this pass because the experiment stage consumed approximately 12 GPU-h against a 10-GPU-h HARD budget, leaving no headroom for any variant run.

To upgrade to full verification without re-auditing:
- stage2_skip_reason: max_verify_claims_cap -> /auto-verify C1 -- resume: true  (single-claim mode; Phase 2 audit reused via RESUME)
- Recommended variant: anchor-swap (replace banana anchor with strawberry anchor, same Qwen-Image base, all other hyperparameters fixed). Tests mechanism invariance under a different bias direction. Estimated ~2-3 GPU-h — run in a follow-up round with an incremental 3-GPU-h allocation.
