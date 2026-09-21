# Final Proposal — Steering Evo2-7B toward high α-helical content

<!-- machine markers (top metadata) -->
```yaml
behavior_source: given
mechanism: discovery
resource_fidelity: unstamped   # NOT the reproduction combo (given+given) — cost-aware
mechanism_strategy:
  directions: [Location, Causal Intervention]   # in execution order
  rejected:
    - Formation Tracing — training-time genesis of the helix feature is out of scope and would break the 40 GPU-h budget (checkpoint/data re-training).
    - Decision Auditing — the task is to *change* generation toward a target property, not to audit the trustworthiness of a fixed decision.
    - Unit Interpretation (as a standalone claim) — SAE features are used only as one candidate localization source feeding steering, not decoded/named as an end in itself.
  note: Locate where α-helix propensity is represented in Evo2-7B (probes + pre-existing Evo2 SAE features), then causally steer along it at inference and prove the helix gain is real, dosed, specific, and achieved on valid ORFs. The steering intervention doubles as the applied (Tuning & Editing) lever that achieves the target.
```

**Behavior-source/Mechanism**: given / discovery · **Date**: 2026-08-23
**Model under study (HARD, no substitution)**: Evo2-7B · **Budget (HARD)**: ≤ 40 GPU-hours total, ≤ 8 cards concurrent.

## Problem Anchor (frozen — from task.md, never moved)

Establish whether a **targeted internal intervention** on Evo2-7B raises the α-helical
secondary-structure fraction of the protein encoded by its generated DNA **significantly above the
unintervened baseline, while keeping generated sequences valid** (well-formed, biologically plausible
ORFs), and quantify the effect. Two jointly-required claims: C1 (helix increase) and C2 (validity
preserved). This is an established target to reach, not a phenomenon to first re-confirm — hence **no
M0 gate**.

## Method thesis (how we test the behavior)

A three-rung ladder of evidence, all on Evo2-7B:

1. **Measurement harness + baseline (Location prerequisite).** Fix a reproducible generation +
   scoring pipeline: sample DNA from Evo2-7B → extract the ORF → translate → predict secondary
   structure (ESMFold → DSSP-style assignment, or an SS predictor) → α-helix fraction; plus a
   validity scorer (ORF frame/start/stop, length, coding plausibility) and off-target scorers
   (β-sheet fraction, GC content, sequence naturalness/perplexity under Evo2). Characterize the
   unintervened baseline distribution.
2. **Localize (Location — correlational screen).** Find where α-helix propensity is represented
   inside Evo2-7B: (a) linear probes for the encoded protein's α-helix fraction across layer
   activations (StripedHyena and Transformer blocks); (b) screen pre-existing Evo2 SAE features
   (layer ~26 emphasized) for structure selectivity. Output: a ranked shortlist of candidate
   layers / directions / features — a *hypothesis*, not yet a cause.
3. **Steer + confirm (Causal Intervention — the applied lever).** Construct the intervention from the
   survivors (contrastive activation addition from high- vs low-helix activation means, and/or
   SAE-feature clamping) and apply it during generation over a **steering-coefficient dose grid**.
   Confirm the target moves as predicted (sign + magnitude + monotone dose-response), then prove
   **specificity + validity**: a matched-control (random/orthogonal) direction produces no helix
   gain; off-target properties and — critically (C2/HARD) — sequence validity are not degraded at the
   setting that satisfies C1. Report the **validity frontier** (helix gain vs validity as coefficient
   rises), not a single aggressive point.

The concrete intervention family (CAA vs SAE-feature clamp vs a light tuned variant) is **not fixed
here** — `MECHANISM=discovery` defers it to the experiment stage's `/mechanism-skills` routing. Plan
fields that depend on that choice (`n_pairs`, `sites`, effect `metric`, `gpu_hours`) are tagged
`method_sensitive` in the plan and may be re-bound at routing time without a plan rewrite.

## Dominant contribution

A rigorous, budget-bounded demonstration + quantification of **constrained property steering on a
genomic foundation model**: raising a downstream *protein-structural* property (α-helix) of a
*nucleotide* generator via an internal intervention, held to a validity constraint — with matched
controls and dose-response, not a single cherry-picked coefficient.

## Constraints encoded (from HARD CONSTRAINTS + NOTICE)

- **Model**: Evo2-7B is the generative model for every main run — no smaller/cheaper substitute.
- **Validity (HARD)**: C2 is a first-class claim with its own milestone (M4); a helix gain that
  degrades validity does **not** satisfy the behavior. Pre-register a validity tolerance.
- **Budget (HARD)**: total planned GPU-hours ≤ 40, ≤ 8 concurrent cards; ESMFold scoring is the cost
  driver and is sized to fit (see plan). Do not trivialize the mechanism run to save cost — the
  budget is sized to run it for real.
- **NOTICE**: use the provided HF token only to fetch Evo2-7B + datasets; α-helix measured on the
  translated ORF; baseline = same model, no intervention. These are authoritative in
  `EXPERIMENT_PLAN.md`.

## Risks & mitigations

- **Steering-vector unreliability / non-identifiability** (lit.) → multiple seeds, matched-control
  direction, report distributions not point estimates.
- **Control-vs-validity collapse** → dose grid + validity frontier; the reported success setting is
  the best helix gain *subject to* validity non-inferiority.
- **ORF extraction ambiguity** (which frame / which ORF) → fix a deterministic ORF-selection rule in
  M1 and apply it identically to baseline and intervened.
- **Measurement cost** → cap sequence counts at the pre-registered `used_n`; ESMFold batched across
  the 8 cards.

## External review

One external-reviewer pass (llm-chat backend, `gpt-5.4`) was run on this proposal + plan; see
`refine-logs/RESEARCH_REVIEW.md`. Its accepted points are folded in above (deterministic ORF rule,
validity-frontier reporting, matched-control requirement, non-inferiority framing for C2).
