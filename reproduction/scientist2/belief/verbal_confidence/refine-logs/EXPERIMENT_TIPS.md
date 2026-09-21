# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-coefficient-tuning
  - steering-block-selection

## Matches

1. **steering-block-selection** — M3/M4/M5 name target sites (`site: ${top_k_sites_from_M2}`, `block_at_layer: [top_layer_from_M2, top_layer_from_M2..last_layer]`) and layer bands; regional claim about post-answer positions × mid-late layers.
   - convention to adopt: lock site set FIRST via activation-based screening (M2 probe R²); use a window covering the post-answer positions (E0..E4) with matched non-cache controls (M6(a)); do not hardcode a single index — use M2's top-K.

2. **steering-coefficient-tuning** — M5 uses signed dose-response `alpha: [-4, -2, -1, 0, 1, 2, 4]` on a residual-stream direction extracted via diff-of-means / LDA.
   - convention to adopt: express α in **σ_proj units** (α = β · σ_l where σ_l = std(h_l ᵀ u_l)) so coefficients are comparable across layers/directions; the α=0 baseline is already in the grid; log a fluency/general-ability metric (answer accuracy — M6(d)) alongside the target metric (verb_conf_mean) at every α to catch off-distribution collapse; prefer the smallest sufficient β once the target is met; because the claim probes 27B and mid-late layers, expect fastest collapse at α > +4.

**General Rule for mechanism/Interpretability (unconditionally loaded)**: (1) Locate the neuron/feature first (M2 probe → top-K sites), then intervene (M3/M4/M5). (2) Intervene on target behavior only; report answer accuracy (`answer_acc_preserved`) alongside verbal-confidence effect at every intervention — this is precisely what M6(d) formalizes. Full-breakdown detection: if answer accuracy collapses (Δ acc > ~10%) at α = ±4, treat verb-conf effect as collapse-driven, not localization.

## No-match log

- Tip 1 (ImageNet Eval Preprocessing) — no vision backbone or torchvision transforms.
- Tip 4 (Fine-Tuning Hyperparameter Sweep) — no fine-tuning, LoRA, DPO, or PEFT in the plan; the entire suite runs on a frozen `gemma-3-27b-pt`.
- Tip 5 (Multiple-Choice Evaluation) — verbal confidence is an integer 0–100, not an A/B/A-D letter parse; scoring is regression (R²) and Δ-mean, not letter-parsing a generation.

## Composition order

Per the router's composition rule for mechanism-interp steering: **block selection FIRST → coefficient tuning NEXT**.
- M2 (Location) already screens the block/layer via probe R² → produces `top_k_sites_from_M2` → satisfies block-selection.
- M5 (steering) then sweeps α on that locked site set → satisfies coefficient tuning.
- The plan already reflects this order.
