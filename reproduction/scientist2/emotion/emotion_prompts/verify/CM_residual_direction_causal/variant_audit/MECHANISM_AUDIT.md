# Mechanism Audit — CM Variant: model-swap-qwen3-4b

## Summary
**Overall verdict: N/A**

The variant runs the Location arm only (probing + SVD direction extraction). No steering, no patching, no causal intervention. Check A (steering coefficient sweep) does not apply. The mechanism-audit verdict is n/a.

## Check A: Steering coefficient sweep rigor
- Status: NOT APPLICABLE
- The variant does not run `mechanism_causal.py` or any causal intervention
- No steering coefficients used; no dose-response curve to evaluate
- Reason: Budget constraint — the Causal arm was also not-supported in the main experiment, so re-running it at smaller scale was deprioritized. Only Location (probe) is tested.

## Check B-H (reserved checks)
- Not applicable; this variant runs probe_location.py only (linear probing, no mechanism intervention)

## Conclusion
The variant's mechanism rigor is clean by virtue of not performing any mechanism intervention. No concerns.
