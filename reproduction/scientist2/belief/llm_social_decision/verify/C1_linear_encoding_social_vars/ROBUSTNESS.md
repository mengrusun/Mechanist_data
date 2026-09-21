## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported
- Main-experiment integrity: warn
- warn_source: experiment  (proxy GT not labeled; scope: 48 unique prompt texts appear in both train/held; layer pick selects shallow layers 2–6)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C1 was audited and found admissible (WARN) at Phase 2. C1's probe accuracy evidence uses real dataset GT (V labels from dg1000_prompts.jsonl) but the projection-transfer regression uses model-generated τ as proxy target. The 48 unique prompt texts (all appearing in both train and held splits due to the trial-level split design) limit out-of-template generalization claims for held_acc=1.0. Layer picks at L=2–6 are shallow relative to convention. These limitations are acknowledged in EXPERIMENT_RESULTS.md. No swap stress test was attempted for C1 this pass (MAX_VERIFY_CLAIMS cap assigned the single Stage 2 slot to C3). Main experiment's own verdict on C1 (supported) stands as-is, with the caveat that it has not been shown robust across model swaps. To upgrade to full verification without re-auditing:
- `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME)
