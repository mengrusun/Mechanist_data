# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **steering-block-selection** — plan names target intervention sites `[top1_layer, top2_layer]`
   for CAA / SAE-clamp steering on the Evo2-7B trunk (StripedHyena + Transformer blocks).
   - convention to adopt: lock the site set FIRST by a screening signal (M2 probe accuracy /
     diff-mean separability ranks the layers), not a hard-coded index; if a single site is inert,
     widen to 3–5 layers; match-to-claim (a regional localization claim needs a window + matched
     null-control windows); never copy a raw layer index across models of different depth.
2. **steering-coefficient-tuning** — plan applies an additive intervention (contrastive activation
   addition / SAE-feature clamp) with a configurable strength `coefficient` over a dose grid
   `[0, 0.5, 1, 2, 4, 8]`.
   - convention to adopt: the coefficient plateau is site-dependent (lock sites first, then sweep);
     sweep coarse-to-fine and ESCALATE before abandoning (widen the grid geometrically rather than
     stopping at 8 if 8 still shows an effect trend or side effects are not yet severe); score EVERY
     sweep point on the target metric (α-helix) AND a fluency / general-ability metric (validity
     rate + Evo2 NLL naturalness) and keep Pareto-optimal candidates; a run that ends with NO
     coefficient meeting C1's criteria is `inconclusive`, not `not-established`, and REQUIRES an
     `open_items[]` warning that the grid may be too narrow.

## Composition (interlocking order)
Block-count first (M2 locks `[top1_layer, top2_layer]` by probe/SAE screening) → coefficient sweep
next (M3 dose grid on the locked sites). Re-sweep the coefficient if the site set changes. This
matches the plan's M2→M3 dependency chain exactly.

## General Rule for mechanism/Interpretability (always loaded, unconditional)
Applied to all downstream milestones:
- Locate the α-helix representation first (M2 probes + Evo2 SAE l26 screen), then intervene (M3).
- Intervene on the target behavior ONLY — measure general ability (sequence validity, Evo2 NLL
  naturalness, off-target β-sheet/GC/length/aa-composition) IN PARALLEL with α-helix at every sweep
  point; a helix "gain" achieved by pushing generations into gibberish / invalid ORFs is an artifact
  (this is exactly C2's validity constraint and the survivorship-safe primary endpoint).
- Feature/direction selection is the biggest driver of success: the contrast set (M1 high- vs
  low-helix) must be behavior-verified, matched (differ only in helix content, controlled for
  length / GC / composition where possible), clean on general ability, and mined on the DEV split
  then confirmed on the disjoint TEST split (anti-circularity, already in the plan's measurement
  contract).
- Prefer steering a SET of directions/features over a single one where the plan allows (CAA at a
  site family + jointly-clamped SAE features rather than one feature).

## No-match log
- image (ImageNet preprocessing) — n/a, genomic sequence model, no vision backbone.
- finetune-hyperparameter-sweep — n/a, no fine-tune / LoRA / PEFT milestone (steering is inference-time).
- multiple-choice-evaluation — n/a, no A–D letter parsing; endpoint is a continuous α-helix fraction
  from ESMFold+DSSP, graded against structural ground truth, not a model-generated letter.
