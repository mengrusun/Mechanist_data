# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning
  - multiple-choice-evaluation

## Matches

1. **steering-block-selection** — plan declares `L ∈ top-3 layers by probe AUC` for Arm A/C and multi-layer/multi-site injection on heads + MLP neurons for the enhancement operator. Interlocks with coefficient tuning (site fixed before dose).
   - convention to adopt: pick sites by activation-based screening (per-head probe AUC + neuron alignment/t-score); lock the site set BEFORE α-sweep; match the claim to the sites actually intervened on; report site distribution per emotion. Circuit discovery on attention heads + MLP neurons (per tip).

2. **steering-coefficient-tuning** — plan sets an additive-injection strength α ∈ {0.5, 1.0, 2.0} on residual stream (Arm C) AND on head contribution to residual (Arm A head component) AND on MLP-neuron pre-activation (Arm A neuron component). Multiple sites × multiple strengths.
   - convention to adopt: sweep β expressed in σ_proj units; include a β=0 baseline; score target metric AND a fluency/general-ability metric side-by-side to detect off-distribution collapse; prefer the smallest sufficient α; re-tune per site set (heads vs neurons vs residual). Late-layer α cap needed to avoid repetition/format-spam.

3. **multiple-choice-evaluation** — M3 + M4 grade a free-form 100-token generation into one of 6 emotion labels via gpt-5.4 external judge. Six-way forced choice, not letter-regex. Steering + interventions are exactly the "instruction-breaking" regime where naive parsing silently fails.
   - convention to adopt: three-way `{CORRECT, INCORRECT, OTHER}` judge verdict (gold-relative shape from tip); persist `(prompt, target_emotion, arm, config, raw_generation, judge_verdict, judge_reason)` per row; report `OTHER` rate separately (never coerce to INCORRECT); position/label bias check by rotating the option order in the judge prompt (6 emotion labels shuffled per row); freeze judge model + prompt + temperature (0 for judge) across arms; 60-item gold audit (already in plan) required; log judge disagreement > 5% as suspected_parser_artifact.

## Composition order (per tips routing composition guidance)

1. **Block/site set first** (M1 Stage A locks top-3 layers + top-20% heads + top-5% MLP neurons; then Stage B causal ranker picks `(k_h*, k_n*)` on val).
2. **Coefficient next** (M2 sweeps α ∈ {0.5, 1.0, 2.0} at α2 = 1.0 as pivot; M3 Arm A re-picks per-emotion α_A from same 3-strength grid; general-ability probe = length audit + fluency check on continuations).
3. **Judge audit + MCQ non-negotiables** (M0.5 60-gold audit is the required judge-gate; M3/M4 use gold-relative 6-way judge with `OTHER` bucket for refusal / off-topic / multi-label).

## No-match log

- `finetune-hyperparameter-sweep` — NOT matched: plan explicitly excludes fine-tuning, LoRA, DPO, RLHF. All interventions are inference-time hooks.
- `image` (ImageNet preprocessing) — NOT matched: no vision backbone or image transforms in the plan.
