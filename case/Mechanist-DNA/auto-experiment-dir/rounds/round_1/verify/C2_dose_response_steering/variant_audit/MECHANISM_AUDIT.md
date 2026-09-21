# Mechanism Audit Report — Claim C2 (variant: method-swap-ssdef-pydssp)

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C2 — Amplifying the C1 α-helix feature set during Evo2-7B autoregressive DNA generation increases the encoded-protein α-helix fraction, monotonically with amplification strength up to an optimum.
**Scope**: variant `method-swap-ssdef-pydssp`

## Overall Verdict: WARN
*Acceptable as a limited trend-comparison sweep for SS-assignment-method agreement, but does not meet
full Check A standards for a rigorously calibrated steering sweep (same base limitations as the main
experiment's own steering apparatus, which this variant reuses unchanged and does not re-tune).*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — reuses `code/mechanism.py`'s `Steerer`/hook unchanged.
- Grid: α∈{0,4,8,16,32} (5 points, includes 0) × seed∈{42,200} = 10 cells. Positive-α span = 8×
  (still short of the catalogue's ≥3-orders-of-magnitude guidance). No negative-α sign check within
  this variant's own grid (that check lives in the main experiment's M2).
- σ_proj scaling: no (same raw mean-nonzero-activation-magnitude `s_f` as the main experiment).
- Capability metric (valid_orf_rate) logged at every dose alongside the target metric — yes.
- Plateau: **not established.** At α=16, valid-ORF drops to 0.643 from baseline 0.867 (~26% relative
  drop); α=32 partially recovers to 0.797 (still below baseline) and sits at the grid edge.
- Locked α: **this variant does not select one** — its own question is "do mkdssp and pydssp track
  the same dose-response trend on the same steered sequences," not "what is the optimal steering
  strength." This avoids the main experiment's specific edge-locking critique (Q6 is not violated by
  anchoring a headline number at an untested-beyond edge), but it also means the variant earns no
  positive credit for demonstrating a mid-plateau operating point — Q6 is simply unsatisfied either
  way.
- Random-direction control: none within this variant's own runs (reuses only the frozen feature set S
  at every dose; the matched-control/random arm lives in the main experiment's separate M3 milestone,
  which backs claim C3, not C2). Reviewer's judgment: this matters less for this variant's actual
  question (assignment-algorithm agreement on identical sequences) than it would for a causal-effect
  claim, but the formal Check A criterion is still unmet, hence WARN not PASS.
- Sign pattern: n/a (no negative-α leg in this variant's own grid).
- Output-case spot-check: target and capability metrics both logged at every point (see table in the
  paired EXPERIMENT_AUDIT.md); no raw-text coherence spot-check attempted (out of scope for this
  variant — its question is SS-assignment agreement, not generation coherence).
- Evidence: `code/mechanism.py` (`Steerer`, reused unmodified); this variant's own
  `run_variant.py` (`--alpha` grid dispatch); `variant_summary.json` (per-dose table).
- Verdict reason: a real sweep with both target and capability logged at every dose (strength), but
  narrow span, no σ_proj units, no established plateau, and no random-direction control (same base
  limitations already flagged in C2's main-experiment mechanism audit, which this variant inherits
  rather than independently re-introduces). Not FAIL: no single hardcoded α, no capability-blind
  metric, and — critically — this variant never locks or reports a headline "optimal" α at a
  capability-degraded dose (unlike the main experiment's α*=32 framing), so the sharper FAIL trigger
  from C3's mechanism audit (decisive statistic computed at a capability-crashed dose) does not apply
  here.

### B–F. Reserved (not_implemented)

## Action Items
- If this SS-assignment-swap design is reused for future verify passes, consider adding a matched
  random-direction (or the existing M3 matched-control) arm within the SAME grid so mechanism rigor
  can be assessed independently of C3's own milestone.
