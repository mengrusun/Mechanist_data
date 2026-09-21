# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning
  - finetune-hyperparameter-sweep

## Matches

1. **steering-block-selection** — M2 declares `layer_group ∈ {early, mid, all_non_upper}` and `k_top ∈ {0, 4, 8, 12}`; M3 inherits M2's site set. Multi-site intervention with a claim about layer *scope* ("leave upper layers intact") explicitly fires this tip.
   - convention to adopt: **lock the site set (M2's winning `(layer_group, k_top)` grid) first**; treat "region claim → matched window + null control" — M2 already declares matched controls (random subspace, LOL) so tip is honored; use spaced sweep across depth (early/mid/all_non_upper) rather than a single hard-coded block.

2. **steering-coefficient-tuning** — M3 sweeps `α ∈ {-1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5}` on the residual stream. Additive intervention with a strength knob fires this tip.
   - convention to adopt: sweep α around the M2-locked site; always include an `α = 0` baseline (already present); log a **fluency / general-ability metric alongside the accuracy target** — here GlotLID output-language fidelity serves the "general ability" role: a collapse would be visible as fidelity crashing while accuracy inches up. Prefer the smallest `|α|` that meets the target (Claim 3 predicate). α is already expressed as unitless projection scaling — the plan's `h ← h + α·Π_lang·h` uses α on the projected component, comparable across layers.

3. **finetune-hyperparameter-sweep** — M4a runs LoRA-SFT on Qwen-3-4B-Thinking with a hard-coded config (`lr=2e-4`, `r=32`, `α=32`, `target_modules = {q_proj, k_proj, v_proj, o_proj}`, `batch=16`, `epochs=1`). Hard-coded config triggers the tip.
   - convention to adopt: **LR-first pilot** — run a small 500–1000-example pilot on 1 seed to sanity-check the LR before committing to the full ~2.5h SFT run. The plan's `lr=2e-4` sits in the tip's LoRA-SFT window `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}` — verify with sanity pilot. Also flag: `target_modules` is attention-only, which the tip warns under-performs. Consider expanding to `all-linears` if pilot shows under-fit descent (< 30 % relative loss drop). Mark `sweep_status: sanity_checked` on M4a's hyperparameters block after the pilot.

## General Rule for mechanism/Interpretability (mandatory, loads on every mechanism experiment)

- **Locate first, then intervene.** M1 (probe → V_lang) is the localization; M2/M3 are the interventions. Localization comes before intervention — plan sequence is correct.
- **Measure general ability alongside the target.** Target = MGSM accuracy; general-ability proxy = **GlotLID output-language fidelity** (already in the plan) + English MGSM as an off-target-language reasoning check (already in the plan). If accuracy inches up while fidelity crashes to ≈ 0, the intervention has pushed the model off-distribution — flag as gibberish-collapse, not a valid result.

## No-match log

- **imagenet-preprocessing** — not fired (no vision backbone / torchvision preprocessing).
- **multiple-choice-evaluation** — not fired: MGSM answers are numeric strings (integer / float), not letter options. Answer extraction uses "The answer is …" numeric extraction (see EXPERIMENT_PLAN.md §Common evaluation protocol), no letter regex needed.
