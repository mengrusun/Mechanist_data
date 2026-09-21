# Pipeline Summary

**Problem**: Faithfully verify the two SemanticLens-style behavioral claims from `task.md` on ResNet-50 / ImageNet, with a frozen CLIP image tower as the foundation encoder — under a 10 h GPU budget on GPUs `{1,2,3,5,6}`.
**Final Method Thesis**: A single unified reference-input + frozen-CLIP-pooling pipeline computes v_c once per component and reuses it (via cached CLIP embeddings) across four falsifying ablations (k-sensitivity, pooling operator, layer granularity, cross-model transfer). Every claim-level predicate is measured on the same v_c.
**Final Verdict**: READY
**Date**: 2026-07-13

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Landscape: `idea-stage/LANDSCAPE.md`
- Raw retrieval: `idea-stage/RESEARCH_LIT.md`
- Claim report: `idea-stage/IDEA_REPORT.md`

## Contribution Snapshot
- **Dominant contribution (testing side)**: rigorous single-model, single-dataset falsification suite for the SemanticLens component-vector claim, with matched-control specificity, k-plateau ablation, and pooling-operator robustness.
- **Optional supporting contribution**: cross-model transfer check under `task.md`'s permitted swap set {ViT-B/16, VGG-16, EfficientNet-B0}.
- **Explicitly rejected complexity**: causal ablation / activation patching (out of claim scope); SAE training (different Unit-Interpretation submethod); auto-LLM-caption baseline (compute-hungry, MILAN-style side-comparison only if time); Formation Tracing / data attribution.

## Must-Prove Claims (FROZEN — verbatim from `task.md`)
- **C1**: For every component *c* in a trained vision model, a small set of reference inputs that strongly drive *c* is a concept-faithful summary of what *c* encodes.
- **C2**: Embedding those reference inputs with a frozen multi-modal foundation model and pooling the embeddings yields a single vector v_c placing *c* in the foundation model's joint image–text semantic space.

## First Runs to Launch
1. **M0_setup** — build the conda env, symlink data, pin CLIP + ResNet-50 checkpoints, log SHA-256.
2. **M1_activations** — one ResNet-50 forward on ImageNet val; cache spatial-mean channel activations for layer3/layer4/avgpool/fc.
3. **M2_reference_sets → M3_clip_embeddings** — build R_c (top-256 per component) then compute cached CLIP embeddings for the union of reference images.

## Main Risks
- **R1 — reference-input concentration collapse**: mitigate by logging per-image cover fraction; if > 5 %, add frequency-inverse re-weighting as a documented ablation.
- **R2 — CLIP prompt-ensemble sensitivity**: mitigate by pinning the OpenAI 7-template prompt ensemble for all text queries.
- **R3 — component granularity mismatch (layer4 channels may be too fine-grained)**: mitigate by reporting per-layer results; a layer failing every predicate is a finding, not a claim failure.
- **R4 — frozen CLIP checkpoint fragility**: mitigate by pinning `open_clip` `ViT-B-32` `openai` checkpoint with SHA-256 log in M0_setup.

## Mechanism-strategy note
`MECHANISM=discovery` → `chosen_mechanism` is intentionally NOT stamped. `/auto-experiment` Phase 1.5 will route via `/mechanism-skills` (expected concrete family: cross-modal Unit-Interpretation → reference-input pooling + CLIP text-similarity scoring, matching the FINAL_PROPOSAL pipeline). Fields marked `method_sensitive: [...]` in EXPERIMENT_PLAN.md may be re-bound at routing time without a plan rewrite.

## Next Action
- `/auto-experiment` (Workflow 1.5) to implement + deploy.
- Then `/auto-verify` (Workflow 1.75) to stress-test on the swap models.
- Then `/auto-iteration-loop` (Workflow 2) to iterate.
