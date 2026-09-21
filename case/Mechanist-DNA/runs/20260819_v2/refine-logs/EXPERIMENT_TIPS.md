# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - general-rule-mechanism-interp
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **general-rule-mechanism-interp** (always-on for any mechanism/interp experiment) — this experiment localizes and intervenes on an internal Evo2-7B component to control a behavior (%H).
   - convention to adopt: locate the α-helix direction/feature(s) on **behavior-positive, matched-control, held-out** contrastive data (high-α vs low-α CDS differing *only* in helix content); prefer steering a **set** of features/sites over a single one; and **always measure general ability in parallel with %H** — ORF validity, Evo2-7B coding-likelihood, GC, and SS-distribution — reporting both together (a %H rise amid gibberish/validity collapse is an artifact, not control).

2. **steering-block-selection** (site selection; lock FIRST) — plan names Layer-26 as a candidate site and sweeps interventions on the residual stream.
   - convention to adopt: do **not** hard-code a single block. Screen candidate sites by an activation signal (diff-mean separability / probe accuracy) across spaced mid-to-late blocks; if a single site is inert, widen to 3–5 layers. The SAE clamp is pinned to L26 (the Goodfire SAE's own layer), but contrastive/probe directions sweep sites. Match-to-claim: the claim is "some component at one or more layers", so screen the stack at spaced intervals with matched null-control sites.

3. **steering-coefficient-tuning** (coefficient; sweep AFTER site is locked) — plan sweeps α ∈ [0,0.5,1,2,4,8] for an additive/clamp intervention.
   - convention to adopt: sweep α coarse-to-fine on the locked site set; **escalate before abandoning** (widen the geometric grid if the top α still shows no effect before switching site/method); score **every** α on target (%H) AND fluency/general-ability (ORF validity, coding-likelihood); keep Pareto-optimal points. If **no** α meets the criterion, the verdict is `inconclusive` (NOT `not-established`) and an `open_items[]` warning is required that the α range may be too narrow.

## No-match log
- Tip 1 (ImageNet preprocessing): not applicable — DNA sequence model, no torchvision.
- Tip 4 (fine-tuning hyperparameter sweep): not applicable — no fine-tuning; interventions are inference-time steering/clamping only.
- Tip 5 (multiple-choice letter-parse): not applicable — the eval scores a continuous %H via ORF→translate→SS-predict, not an A–D letter parse of a free-form generation.
