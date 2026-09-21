# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - general-rule-mechanism-interpretability
  - steering-coefficient-tuning
  - steering-block-layer-selection
  - llm-judge-free-form-generation

## Matches

1. **general-rule-mechanism-interpretability** — every claim intervenes on model internals; must (i) locate then (ii) intervene without collapsing general ability.
   - convention to adopt: report base-model perplexity / length degeneracy under intervention; matched-random-direction control at same `‖α·v‖`; every reported number uses α locked on dev split before held-out eval.

2. **steering-coefficient-tuning** — the intervene step is `h_l ← h_l + α · v_c` with 7-point α sweep.
   - convention to adopt: signed dose sweep `{-3,-2,-1,0,+1,+2,+3}` on dev, pick α* by max effect *subject to* length-degradation guardrail (|steered| ≤ 2× |baseline|); check output is not gibberish at chosen α; α=0 must reproduce the unsteered baseline exactly.

3. **steering-block-layer-selection** — per-block screen across all 32 residual-stream blocks.
   - convention to adopt: report per-block probe accuracy for all 32 blocks (not just the chosen one) to guard against cherry-picking; C5 additionally reports per-block AUROC for the same reason.

4. **llm-judge-free-form-generation** — GPT-4o rubric-judge on 5-point steering-effect scale.
   - convention to adopt: fixed rubric prompt per concept (temperature=0, seed pinned); label-floor pilot on 20 baseline+20 steered outputs before full run (variance ≥ 0.05 required to trust the scorer, else stop and surface).

## No-match log

- `finetune-hyperparameter-sweep` — not applicable; no fine-tuning (RFM is a kernel + eigenvector, not gradient descent on model weights).
- `imagenet-torchvision-preprocessing` — not applicable; text only.
- `mcq-letter-parse-eval` — not applicable; all evals are free-form generations judged by rubric or verified by test cases / AUROC.
