# Mechanism Audit — C1 (main experiment)

**Claim**: C1 — Bottleneck layer existence + causal confirmation.

**Mechanism family**: Representation and Parameter Analysis / representation-engineering (M1) + Causal Attribution / patching (M2).

**Audit date**: 2026-07-14

---

## Check A — Steering Coefficient Sweep

**Applicability**: C1's milestones M1 and M2 do NOT use an additive steering coefficient. M1 is a forward-only hidden-state extraction with cosine-ratio diagnostic (RepReading, no direction injection). M2 is activation patching — a direct hidden-state substitution (replace `h_L*(g, lang_x)` with `h_L*(g, en)`) — not a parameterized additive intervention with a tunable coefficient α.

**Verdict**: n/a — C1 uses no additive activation steering with a coefficient that requires sweeping.

**Detail**: The only "tunable" in M2 is the choice of patching site (L*, l=2, l=30), which is a categorical rather than a continuous coefficient. The matched-control specificity test (condition D) is the functional analogue of the random-direction baseline check, and it was included. No σ_proj-scaled α sweep is applicable here.

---

## Check B–F — Reserved checks

**B (direction-extraction quality)**: Not implemented. M1 uses label-free cosine geometry (no direction extraction); M2 uses direct state substitution (no direction extracted and then applied). Not applicable for these methods.

**C (site/layer choice)**: M1 identifies L* as argmax_l R(l) — the site is data-driven, not assumed. M2 tests L* against both surface controls (l=2, l=30) and a matched-control (D) — good experimental design for site justification.

**D (n_effective sufficiency)**: M1 uses 315 prompt groups × 10 languages = 3150 forwards + 300 random pairs per language. M2 uses 100 prompt groups × 9 target langs × 4 conditions = 3600 patching trials. These sample sizes are adequate for the bootstrap CI estimates reported.

**E (probe-vs-causal disentanglement)**: M2's patching is causal by design. The matched-control failure (A≈D) is a genuine negative result on specificity — the experiment correctly identifies this and reports it as a partial verdict.

**F (intervention scope)**: The patching intervention is scoped to a single layer's last-token residual state per trial. This is appropriately narrow.

All B–F checks: not_implemented (reserved).

---

## Overall Verdict

**overall_verdict**: n/a

**Reason**: C1's mechanism milestones (M1 = RepReading, M2 = activation patching) do not use a parameterized additive steering coefficient. The steering-coefficient sweep check (A) is inapplicable. All other checks are reserved placeholders. No mechanism-rigor concern that would block C1 from Stage 2 entry.
