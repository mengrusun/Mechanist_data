# Round 3 Review — Parsed Summary

**Overall score**: 9.1/10 (up from 8.5)
**Verdict**: **READY**
**Drift Warning**: NONE

## Dimension scores

| Dim | Round 1 | Round 2 | Round 3 |
|-----|---------|---------|---------|
| Problem Fidelity | 8 | 8.8 | 9.2 |
| Method Specificity | 8 | 9.0 | 9.2 |
| Contribution Quality | 7 | 8.1 | 9.3 |
| Frontier Leverage | 8 | 8.4 | 9.0 |
| Feasibility | 6 | 8.8 | 9.3 |
| Validation Focus | 6 | 8.4 | 8.9 |
| Venue Readiness | 6 | 8.0 | 9.0 |

## Blocking issues: NONE

## Important remaining polish items (fold into FINAL_PROPOSAL, not blockers)

1. **Bootstrap protocol clarification**: state that primary CIs for probe-dependent quantities RETRAIN the probe within each bootstrap resample (retrain-on-bootstrap) or split into "eval-only CI" vs "probe-fit variability CI". Adopt: retrain-on-bootstrap for primary; eval-only as a cheaper secondary for per-layer curves.
2. **L\* selection narrative guardrail**: "C3a is considered supported only if the per-layer trajectory ALSO shows low alignment over a broad neighborhood, not a single isolated layer."
3. **C3b emitted-output asymmetry**: privilege internal readout as primary causal-separability test; emitted-output as stronger-but-noisier corroboration.
4. **Parse-failure bias diagnostic**: compare correctness rate and token-prob stats between parseable and unparseable subsets; report gap.
5. **Single-pass variant precommit**: (a) both low → strengthens interpretation; (b) diverge → main claim is two-context only, no shared-state conclusion.

## Minor polish
- C2 ordinal path: concrete threshold (e.g., "top-1 ordinal accuracy ≥ 0.55, macro-F1 ≥ 0.4").
- ECE for C1: descriptive/supporting metric, not decisive.
- σ_probe_readout definition: std of unsteered held-out probe readout distribution at L*.
- Soften "first direct measurement" to "first direct matched-representation measurement".

All accepted and incorporated into FINAL_PROPOSAL.md.

## Raw response

See `round-3-review-raw.md` for full response.
