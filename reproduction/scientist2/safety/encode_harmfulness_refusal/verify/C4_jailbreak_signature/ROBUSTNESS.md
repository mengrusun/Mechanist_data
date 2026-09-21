## C4: audit-only (main-experiment integrity = WARN, swap stress test skipped) — INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C4: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (claim untestable at ASR=0 on both attack families; no successful jailbreaks to measure signature on)
- Variants: none (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C4's main-experiment audit passed with a WARN (the experiment is methodologically clean but the claim cannot be tested because ASR=0 for both GCG and PAP on Llama-3-8B-Instruct). C4 was admitted to Stage 2 but was not the top-K pick (MAX_VERIFY_CLAIMS=1 cap). The not-supported verdict at ASR=0 is robust — any model-swap variant is also unlikely to show the jailbreak signature unless a different attack family (with ASR>0) is used. To upgrade: `/auto-verify C4 — resume: true` (would run a model-swap but the core limitation is attack-family scope, not model selection).
