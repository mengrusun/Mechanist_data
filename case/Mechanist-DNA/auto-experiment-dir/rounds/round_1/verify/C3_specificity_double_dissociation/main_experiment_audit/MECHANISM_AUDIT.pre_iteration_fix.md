# Mechanism Audit Report — Claim C3

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C3 — The amplification-induced α-helix rise is specific to the α-helix feature set S (double dissociation vs matched-control / β-sheet off-target); S is a causally manipulable, specific knob.
**Linked milestone**: M3

## Overall Verdict: FAIL
*C3's mechanism-rigor verdict — whether the steering intervention backing this claim was tuned with
the necessary controls. This FAIL reflects the steering-sweep design specifically, not fabrication or
data tampering — see Checks below.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: FAIL
- Triggered: yes — same `Steerer`/hook apparatus as C2 (`code/mechanism.py`), dispatched via
  `code/m3_specificity.py` over 3 feature-kind arms (`alpha_helix_S`, `matched_control`,
  `beta_sheet_offtarget`), each a single fixed combined direction over its own feature set.
- Sweep grid: α ∈ {0, 4, 8, 16} × seed ∈ {42,200,201} × 3 kinds = 36 runs. **Only 4 α points**,
  narrower than M2's already-insufficient grid; positive-α span is only 4× (4→16), no negative-α
  sign check within M3's own grid.
- σ_proj scaling used: no (same raw-magnitude `s_f` scaling reused from M0/M2).
- Capability metric logged: yes — valid_orf_rate at every dose (pLDDT_mean also logged but flagged
  separately as an ungated, population-inconsistent statistic — see EXPERIMENT_AUDIT.md Check D).
- Plateau range: **not established.** The grid is too sparse/narrow to demonstrate a plateau for the
  `alpha_helix_S` arm.
- Locked α: the decisive dose used for the double-dissociation statistics is **α=16 — the maximum
  of M3's own grid**, i.e. the edge, not the middle of any plateau.
- **Capability degradation at the locked α (this is the FAIL trigger):** at α=16, `alpha_helix_S`'s
  valid-ORF rate drops to 0.667 from a baseline of 0.888 — a ~25% relative reduction. Read against
  the catalogue's ~10% degradation tolerance, this is more than 2× the tolerance band (i.e., in the
  "capability crashed" range the catalogue's FAIL criterion is meant to catch), and the entire
  headline double-dissociation result (S_vs_matched_helix_p=3.2e-23, β_vs_S_sheet_p=6.2e-30) is
  computed at exactly this capability-degraded dose.
- Random-direction control: **not satisfied.** `matched_control` is a single, principled,
  activation-frequency/magnitude-matched direction (useful as a specificity comparator and clearly
  a better-than-nothing baseline), but it is not the catalogue's required random-direction control
  (≥30 independent random directions) — it is one fixed direction, structurally analogous to S
  itself, not a distributional random-direction test.
- Sign pattern: n/a (no asymmetric dual-direction protocol; M3's grid has no negative-α check).
- Output-case spot-check: **not available** — same sequence-text-discarding as M2/C2.
- Evidence: `code/m3_specificity.py` (feature_kind→feats/s_f mapping, `--alpha` grid);
  `code/m3_consolidate.py` (`top = max(alphas)` — the decisive-dose selection is always the grid
  maximum); `results/m3_specificity_summary.json` (per-dose table showing the ORF drop at α=16).
- Verdict reason: the specificity comparison itself (S vs matched-control, S vs β) is a genuinely
  informative experimental design, but under Check A's strict criteria the decisive statistics rest
  on a steering strength where the treatment arm's own generation quality has already substantially
  degraded, no plateau was established, no bona fide random-direction control was run, and the sweep
  span is even narrower than M2's (already insufficient) grid. This is a rigor failure of the
  steering-tuning process feeding into C3's headline numbers, not an accusation of fabricated data.

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Action Items
- Locate an actual mid-plateau operating point for the specificity comparison (where `alpha_helix_S`
  itself has not yet degraded capability beyond ~10%) rather than defaulting to the grid maximum;
  candidates from M2's own richer grid suggest α≈8 may be a better-behaved shared dose (S valid-ORF
  0.888 at α=8 in M3 vs 0.667 at α=16).
- Add a genuine random-direction control (≥30 independent random directions, matched on overall norm)
  at the eventual locked α, distinct from the single `matched_control` direction.
- Extend the α grid (or express α in σ_proj units) so specificity is characterized as a trend, not a
  single edge-of-grid snapshot.
