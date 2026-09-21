# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-coefficient-tuning

## Matches

1. **steering-coefficient-tuning** — plan sweeps an additive steering strength α over a grid `{0,0.5,1,2,4,8,16}` (residual-add of SAE decoder directions at the layer-26 residual).
   - convention to adopt: coarse-to-fine geometric sweep (start wide, then narrow onto the optimum α*); score **every** α point on the target metric (%-helix) **and** a fluency/general-ability metric (ORF-validity rate, Evo2 perplexity, off-target composition) and keep Pareto-optimal candidates; if **no** α meets the C2/C3 criteria the verdict is `inconclusive` (never `not-established`) and REQUIRES an `open_items[]` warning that the α range may not have been swept widely enough (recommend sweeping beyond the recorded bounds).

## General Rule for mechanism/Interpretability (loaded unconditionally)
- **Feature selection is the biggest driver** and the hardest part. The selection data (E. coli CDS with per-codon DSSP helix labels) is deliberately prepared: behavior-positive verified via DSSP ground truth, matched control = β-sheet/coil positions from the same genes (differs only in the target SS), mined on train and confirmed on a disjoint homology-clustered test split, and gradeable by the exact selectivity metric (AUROC) the C1 claim is stated in.
- **Intervene on the target behavior only, do not destroy general ability.** Measure general ability (ORF-validity, perplexity, off-target β/coil, AA/GC composition) **in parallel** with the target (%-helix) at every α; a full breakdown into invalid/garbled ORFs at high α means the model was pushed off-distribution and any helix movement there is an artifact. Report both together, never %-helix alone.

## No-match log
- **steering-block-selection** (tip 3): considered — the intervention site is PINNED to the layer-26 residual by the user-given released SAE (`Evo-2-Layer-26-Mixed`, hooked at `blocks.26`), under `resource_fidelity: strict`. The site is not a free knob, so no block-count sweep applies; the block/layer choice is fixed by the mechanism.
- **multiple-choice-evaluation** (tip 5): did NOT fire — the downstream readout is a per-residue DSSP secondary-structure fraction (ESMFold→DSSP %-helix) computed from atom coordinates, not a letter/A-D parse of a free-form generation. No regex letter-parse is used.
- **finetune-hyperparameter-sweep** (tip 4): did NOT fire — pure inference-time activation steering, no fine-tuning/LoRA/PEFT.
- **image** (tip 1): did NOT fire — genomic sequence model, no ImageNet/torchvision pipeline.
