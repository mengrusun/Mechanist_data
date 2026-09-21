# Review Trace — mechanism-audit — 2026-07-13_run01

**Skill**: mechanism-audit
**Claim**: C1
**Date**: 2026-07-13
**Reviewer**: gpt-5.4 (dmxapi, cross-model)

## Verdict
overall_verdict: fail
Check A: FAIL

## Key findings
- σ_proj scaling: YES (correctly implemented)
- Sweep grid: [-4,-2,-1,0,1,2,4] (7 points)
- No capability metric (off_digit_rate is coarse proxy only)
- Target effect within noise floor (span ~1.5 units, std~41)
- No locked α, no random-direction control
