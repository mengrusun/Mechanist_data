# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - general-rule-mechanism-interpretability
  - finetune-hyperparameter-sweep
  - steering-block-selection
  - multiple-choice-evaluation

## Matches

1. **general-rule-mechanism-interpretability** — this is a mechanism / interpretability experiment (M1 locates L*, M2 intervenes via cross-lingual patching, M3 injects an L*-anchored regularizer). Rule fires unconditionally.
   - convention to adopt: always measure the target metric (MultiJail ASR) alongside general-ability metrics (MMLU / M-MMLU / MGSM / MT-Bench) — never report the target alone. Watch for gibberish / capability collapse; a movement in ASR with a MMLU collapse is an artifact, not a language-agnostic alignment.

2. **finetune-hyperparameter-sweep** — M3 runs LoRA-DPO on LLaMA-3.1-8B-Instruct with a hardcoded single config (lr=5e-6, r=64, α=r=64, 3000 steps, β=0.1, λ_bottleneck=0.5) and no sweep declared. Trigger fires.
   - convention to adopt: run a **cheap 500–1000-example pilot per fine-tune** for BOTH M3-Method and M3-Baseline (same base+data pair, but *L_bottleneck differs → treated as one pilot since only the loss term is different, not the training data*); if the pilot passes Preflight + A–D diagnostic signals at `lr=5e-6`, mark `sweep_status: sanity_checked` and proceed. If a signal fires, LR-first re-sweep on the tip's LoRA-DPO grid `{5e-7, 5e-6, 1e-5, 5e-5}`. Stamp `sweep_status` inside the M3 milestone block of `EXPERIMENT_PLAN.md` and echo it under the M3 section of `EXPERIMENT_RESULTS.md`.

3. **steering-block-selection** — M2 intervenes at a *single* residual-stream block L* with control layers l=2 and l=30. However, L* is *derived from M1's screening*, not hardcoded, so the tip's "screening first, then commit" rule is already satisfied. The tip still governs the M2 control-layer choice.
   - convention to adopt: L* is picked by M1's per-layer `R(l)` screen (gradient-free activation signal — matches "By method / Activation-based" in the tip). Match-to-claim: since C1 is a *layer-level* claim, M2 tests a single block at L* + spaced matched controls (l=2 near input, l=30 near output). If M2's patching at L* has *no* effect (falsifier B), the tip's fallback is "widen to 3–5 layers"; we log this as a next-round action rather than blowing the budget by widening now.

4. **multiple-choice-evaluation** — M4 grades MMLU / M-MMLU as multiple-choice A-D answers (letter-parse). MGSM is exact-match on the final number (not MCQ — tip does not apply to MGSM). MultiJail ASR is a free-form generation + GPT-4o binary judge (safe/unsafe) — that IS an LLM-judge on free-form text, so the tip's non-negotiables apply.
   - convention to adopt (MMLU / M-MMLU): use LLM-judge extraction into three-way `{A|B|C|D, OTHER}` rather than naked `re.search([A-D])` — OR use log-prob scoring on the answer tokens as a fallback (the tip says log-prob is contamination-prone but acceptable as a diagnostic when generation is limited). Given 10h budget and MMLU's ~14k questions, we use **loglik** on the 4 answer-letter tokens (standard `lm-eval-harness` protocol) for MMLU/M-MMLU — the tip warns log-prob cannot detect output-corruption regression, so we ALSO run a free-form-generation sanity check on 100 MMLU items + eyeball outputs. For MultiJail ASR, we use the mandated GPT-4o judge with a fixed three-outcome verdict (`unsafe` / `safe` / `refused-or-off-topic`), never coercing the third bucket into `unsafe`.

## No-match log

- **image / ImageNet preprocessing** — no vision component in this plan. Not applicable.
- **steering-coefficient-tuning** — M2's activation patching is 100% replacement (not additive with an α-coefficient). M3's `λ_bottleneck=0.5` is a *regularizer weight* on a training-time loss term, which is a fine-tuning hyperparameter (handled by tip 4's DPO β / LoRA α slot), not an additive steering-vector coefficient acting on a hidden state at inference. Trigger does not fire.
