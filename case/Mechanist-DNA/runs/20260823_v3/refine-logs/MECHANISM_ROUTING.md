# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors (CAA)
chosen_idea_title: Constrained property-steering of Evo2-7B toward high α-helical content
effective_domain: mechanistic-interpretability (genomic foundation model)
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/WORKFLOW.md
  - skills/mechanism-skills/feature-dictionary-learning/sae/WORKFLOW.md
  - skills/mechanism-skills/probing/residual-stream-states/WORKFLOW.md

## Inputs read
- refine-logs/EXPERIMENT_PLAN.md (milestones M1–M4, measurement contract, method_sensitive tags)
- refine-logs/FINAL_PROPOSAL.md (Location → Causal Intervention; CAA and/or SAE clamp)
- /mechanism-skills catalog (routing entry) + the three family WORKFLOW.md above + steering-vectors,
  probing/residual-stream-states, feature-dictionary-learning/sae submethod WORKFLOW.md (read in this turn)
- mechanism_strategy.rejected: Formation Tracing, Decision Auditing, standalone Unit Interpretation (excluded)

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors (CAA) — the plan's PRIMARY
   causal lever is contrastive activation addition (mean high-helix − mean low-helix activation) added to
   the located block's residual stream during autoregressive DNA generation. One direction serves as both
   read-out (project → localize/monitor) and write-in (add → steer), matching the Location→Intervention
   execution order at low cost (no retraining; a forward-hook add). Directly realizes C1 (dose-response
   helix gain) and supplies the matched-control / sham directions M4 needs (same family, equal-norm random /
   orthogonalized / permuted directions). Rationale for #1 in `## Rationale`.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/WORKFLOW.md
2. Feature Dictionary Learning / SAE (feature clamp) — the plan's "and/or SAE-feature clamp" arm. A public
   pre-trained SAE exists for exactly this model+site: **Goodfire/Evo-2-Layer-26-Mixed** (block 26). Screen
   its features for α-helix selectivity, then clamp the selected feature set during generation. More
   monosemantic / less entangled than dense CAA; run as a second intervention arm at the SAE's fixed site.
   - path: skills/mechanism-skills/feature-dictionary-learning/sae/WORKFLOW.md
3. Probing / Residual Stream States — the Location screen (M2): per-block linear probes decoding the ORF's
   α-helix fraction from Evo2-7B activations, ranked by held-out R² to pick the top-1/top-2 steering sites.
   Decodability ≠ causality, so it feeds — not replaces — the causal steering in #1/#2.
   - path: skills/mechanism-skills/probing/residual-stream-states/WORKFLOW.md

## Composition plan
Screen → Decode → Verify → (controls), mapped onto the plan's milestone chain:
- **M1 (harness + baseline + contrast set)** — freeze generation→ORF→ESMFold+DSSP→α-helix assay + validity +
  off-target scorers; sample ≥500 valid-ORF baseline generations; build the matched high- vs low-helix
  contrast set (SS-labeled proteins back-translated to DNA + filtered Evo2 generations) on the DEV split.
  Cost ~4h.
- **M2 (Location — Probing #3 + SAE screen #2)** — extract per-block residual-stream activations over M1
  sequences (spaced blocks incl. mid-late, block 26 emphasized); train linear probes for α-helix fraction,
  rank blocks by held-out R² → `top1_layer, top2_layer`; screen Goodfire Evo-2-Layer-26 SAE features for
  helix selectivity → shortlist a feature SET (General Rule 4: steer a set, not one). Cost ~5h.
- **M3 (Causal Intervention — Steering Vectors #1 primary + SAE clamp #2)** — build CAA direction from
  DEV-split contrast means at `top1/top2`; steer during generation over coefficient dose grid
  [0,0.5,1,2,4,8] × {top1,top2}; SAE-clamp arm at block 26. Evaluate on the disjoint TEST split. Score
  EVERY sweep point on α-helix AND validity + Evo2-NLL naturalness (Pareto). Escalate the grid geometrically
  if 8 still trends/side-effects mild (steering-coefficient-tuning tip). Cost ~18h (ESMFold folding dominates).
- **M4 (Specificity + validity)** — at the C1-winning site/coef: matched-control (random/orthogonalized
  equal-norm) + sham (same-site same-norm permuted) directions (both from family #1); C2 non-inferiority on
  validity over the FULL generated set; validity frontier across the sweep. Cost ~7.5h.
- Downstream analysis (isotonic/permutation dose-response trend test, Cliff's δ/Hedges g + bootstrap CI,
  FDR/FWER across sites/coefs/off-target) is POST-PROCESSING on collected effect vectors — not a mechanism
  family; lives here, not in the candidate slot.

## Plan reconciliation
<!-- One row per method_sensitive field declared on the intervention milestone(s) M2/M3/M4. -->
- n_pairs: plan=method_sensitive (unbound) → re-bound **300 per class (300 high-helix / 300 low-helix,
  matched), built on the DEV split** — CAA diff-of-means needs a stable per-class mean and this also trains
  the M2 probes; ≥ the data-rule hundred-level floor and disjoint from the TEST eval split. (Plan file not
  edited; realized value recorded planned-vs-actual in EXPERIMENT_RESULTS.md.)
- sites: plan=`[top1_layer, top2_layer]` (from M2 ranking) → **matches** — kept data-driven: screen a spaced
  set of blocks (mid-to-late emphasis per steering-block-selection tip, block 26 included for SAE parity),
  pick top-2 by held-out probe R²; SAE-clamp arm fixed at the SAE's own site (block 26). No hard-coded index.
- metric: plan=mean α-helix fraction over ALL generations on TEST (survivorship-safe), primary C1 =
  dose-response trend test → **matches** — the frozen ESMFold+DSSP assay is the steering target metric and is
  scored jointly with validity/naturalness at every sweep point, exactly as the tip and General Rule require.
- gpu_hours: plan~34.5h total (M2 ~5h, M3 ~18h, M4 ~7.5h) → **revised ~34.5h (unchanged)** — CAA/SAE-clamp
  interventions are forward-hook cheap; the cost profile is dominated by ESMFold folding of the generated
  ORFs exactly as the plan estimated, so the committed submethod does not move the budget. Stays under the
  40 GPU-h HARD cap with ~5.5h headroom.
reconciliation_status: ok

## Rationale
#1 (CAA Steering Vectors) is recommended because the plan's dominant claim (C1) is *causal directional
sufficiency* — "a located direction, added during generation, raises α-helix with a monotone dose-response" —
which is precisely what Steering Vectors tests, and the same direction primitive supplies the M4
matched-control and sham directions for free. Probing (#3) only establishes decodability (it is the M2
Location screen, not the causal claim), and SAE clamp (#2) is the plan's secondary "and/or" arm — kept as a
parallel intervention because a public Evo2 layer-26 SAE (Goodfire) exists and feature-level clamping is more
monosemantic, but it is not the primary lever. Excluded per mechanism_strategy.rejected: Formation Tracing,
Decision Auditing, standalone Unit Interpretation. RESEARCH_DOMAIN inferred from FINAL_PROPOSAL as
mechanistic-interpretability on a genomic foundation model (not ambiguous → no `general` fallback).
