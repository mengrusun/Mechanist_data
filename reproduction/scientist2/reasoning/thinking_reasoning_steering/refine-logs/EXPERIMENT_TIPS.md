# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **steering-block-selection** — the plan explicitly declares an M1 layer sweep to pick L*(b) per behaviour and then M3/M4 target that L*(b) for the additive residual-stream hook (the "hardcoded single index without justification" symptom is *not* the trigger here — the trigger is the presence of a layer / site selection at all).
   - convention to adopt: pick L*(b) by activation-based screening (held-out ROC-AUC of a linear probe on last-token residual-stream activations, per M1 predicate 1); intervene on the residual stream at that single site; if any behaviour's L*(b) fails the ROC-AUC floor at every layer, widen to a 3-layer window around the best-scoring layer as the M1 decision-gate fallback. Match-the-claim: the paper's claim is per-behaviour single-direction — a single residual-stream site is appropriate.
2. **steering-coefficient-tuning** — M3 declares α ∈ {−3, −2, −1, −0.5, 0, 0.5, 1, 2, 3} in units of σ (residual-stream projection std at L*(b)), M4 picks operating α from that sweep. This is exactly the "coefficient in σ units + β=0 baseline + fluency/general-ability metric" recipe the tip prescribes.
   - convention to adopt: express α in units of σ_L* = std(h_{L*}ᵀ · unit(v_b)) computed on the extract pool; always include α=0 in the sweep (M3 already does); log a *fluency / general-ability metric* (coherence flag via LLM-judge and final-answer accuracy) alongside the behaviour-rate at every α (M3 already does the coherence flag; M4 already logs accuracy); prefer the smallest α that meets the sign+Spearman predicates (already M3's operating-α rule); re-run the sweep whenever the site or direction changes.

## No-match log

- **image (ImageNet preprocessing)** — did not fire: no vision backbone, no ImageNet, no torchvision transforms in the plan.
- **finetune-hyperparameter-sweep** — did not fire: no full-FT / LoRA / QLoRA / DoRA / any adapter tuning in the plan. The Tuning & Editing direction here refers to the *applied steering knob* (per FINAL_PROPOSAL.md "Explicitly Rejected Complexity"), not model editing.
- **multiple-choice-evaluation** — did not fire: the 500-task benchmark is scored by LLM-judge for (i) per-behaviour presence in the chain and (ii) final-answer accuracy on free-form reasoning outputs. No A/B/A-D letter regex is used. LLM-judge determinism (temperature=0, seeded prompt) is captured under M1 predicate 2 (Cohen κ target ≥ 0.6).
