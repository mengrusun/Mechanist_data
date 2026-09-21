## C4a: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C4a: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (dataset substitution — plan panel Birds/UC-Merced/Colon/Aircraft not tested; on-disk substitutes DTD/Fashion-MNIST/imagenet_val slices used instead; scope mismatch beyond licensed cost-aware compression)
- Variants: none  (Stage 2 skipped — max_verify_claims_cap; C2a selected as top-1)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C4a's main experiment carries a dataset-substitution WARN — the specific panel named in the claim was never tested. The current verdict (not-established: aligned strictly worse on all 4 substitute datasets, mean Δ=−28pp) is robust in the negative direction, but cannot be taken as definitive for the claim's named panel. C4a is the primary iteration candidate (α↓ to 0.1–0.3, LoRA scope, shorter training). To upgrade: `/auto-verify C4a -- resume: true` after downloading the original datasets (Birds/UC-Merced/Colon/Aircraft) or accepting the substitute panel as canonical.
