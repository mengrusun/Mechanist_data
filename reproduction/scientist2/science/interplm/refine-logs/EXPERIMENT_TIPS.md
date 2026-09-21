# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-coefficient-tuning
  - steering-block-selection

## Matches

1. **steering-coefficient-tuning** — M6 dispatches an additive SAE-feature-clamp intervention with a dose ladder (`α ∈ {0.5, 1.0, 2.0, 4.0}` in units of feature std `σ_f`).
   - convention to adopt: express dose in `σ_f` units, always include a `α = 0` (no-steering) arm and matched random-clamp / mean-activation baselines, pair the target-property yield with a plausibility (pseudo-perplexity) metric to detect off-distribution collapse, and report the *smallest* dose that meets the target within the plausibility band. Plan already declares `α ∈ {0.5, 1.0, 2.0, 4.0}·σ_f`, no-steer / random-clamp / mean-add controls, and a `≤ 1.5×` pseudo-perplexity plausibility band — matches the tip.

2. **steering-block-selection** — M6 clamps at layer `L*` (the best-covering SAE layer from M2, selected data-driven from `runs/m2/best_layer.json`).
   - convention to adopt: match block-count to the mechanistic claim ("the labeled feature is a mid-layer concept direction"); use a single-layer clamp on the layer that already carries the F1-alignment evidence. Plan pins `L*` via a downstream selection (argmax over M2 coverage) rather than hardcoding — matches the tip's "match-to-claim" rule. The layer sweep upstream in M1/M2 (`L ∈ {1, 9, 18, 24, 30, 33}`) is exactly the site scan the tip recommends before locking coefficient.

## Composition order

Per the tips' composition rule: block-count first (M6 uses the M2-picked `L*` — locked by upstream evidence, not by copying), then coefficient (M6's `α` ladder). Ordering respected by the plan's dependency graph (`M6 depends_on: [M2, M4]`).

## No-match log

- **imagenet-image-preprocessing** — no torchvision / ImageNet in this reproduction; ESM-2 consumes tokenized protein sequences, not images. Not applicable.
- **finetune-hyperparameter-sweep** — no ESM-2 fine-tuning (explicit user constraint in task.md; SAEs are pretrained fixed resources). Not applicable.
- **multiple-choice-evaluation** — no MCQ or letter-parsing anywhere; evaluations are F1 on per-residue binary labels (M2/M3), PR-AUC (M5), and rule- or classifier-based property checkers (M6). Not applicable.

## General Rule for mechanism / interpretability (loaded unconditionally)

**Locate the feature first, then intervene.** Applied throughout:
- M1: catalog per-feature statistics (density, dead-fraction, auto-interp gate) — the locate step for the SAE dictionary.
- M2: map features to Swiss-Prot concepts via F1 alignment — feature-label mapping.
- M4: auto-interpret Swiss-Prot-unaligned features via external LLM — extends the label vocabulary beyond Swiss-Prot.
- M6: clamps only *labeled* SAE features whose auto-interp label names a checkable target property — never clamps blindly.

**Do not damage general ability.** Applied in M6 via the pseudo-perplexity plausibility band (`≤ 1.5×` mean PPL of no-steering completions). A clamped completion falling outside the band is reported and excluded from the yield. Prevents attributing off-distribution collapse to steering success.
