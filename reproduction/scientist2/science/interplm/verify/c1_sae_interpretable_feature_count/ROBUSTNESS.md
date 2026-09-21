## c1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on c1: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (auto-interp proxy GT; scope under-power — 1500/10k seqs; ratio criterion met but absolute count falls short)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited and found trustworthy at Phase 2 (WARN only — no integrity failure). C1 was admitted by Stage 1 but not selected as the top-K=1 claim by importance (C2 picked instead, as the core concept-alignment headline). No swap stress test was attempted this pass. The main experiment's verdict on C1 (not-supported at primary absolute criterion; ratio criterion met at 19.3×) stands as-is. The ratio finding (SAE/neuron ≥ 10× across 5 of 6 layers) is robust directionally; the absolute-count claim (~2,548) remains under-powered. To swap-test C1 without re-auditing:
- /auto-verify c1 — resume: true  (single-claim mode; Phase 2 audit reused via RESUME)
