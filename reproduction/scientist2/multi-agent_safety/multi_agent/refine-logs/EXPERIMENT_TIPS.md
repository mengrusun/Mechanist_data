# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - general-rule-mechanism-interpretability
  - steering-block-selection
  - multiple-choice-evaluation

## Matches

1. **general-rule-mechanism-interpretability** — always loaded on mechanism / interpretability experiments (linear-probe localization is a mechanism / interpretability task)
   - convention to adopt: locate the target feature via linear probe (already the plan); measure a "general ability" analog alongside the target metric — here the three sanity controls (label-permutation, matched-length, topic-swap) play that role: they establish the probe is picking up collusion, not surface features. Report probe AUROC + control AUROCs together, never the target alone. Detection experiments do not intervene, so the "gibberish" collapse failure mode does not apply.

2. **steering-block-selection** — the plan hard-codes candidate probing sites `layers 20,28,36,44` and asserts they are "mid-late layers ... of 48" (`FINAL_PROPOSAL.md §2(a)`), but Qwen3-32B-AWQ has **64 layers** (`config.json → num_hidden_layers: 64`). "Never copy a raw index across models of different depth" fires directly.
   - convention to adopt: scale by relative depth. `{20, 28, 36, 44}` of 48 → relative depths `{0.42, 0.58, 0.75, 0.92}` → on 64 layers → `{27, 37, 48, 59}`. This shortlist is applied to M1.2 (extraction) and carried through to M1.3 (per-layer probe grid, select best on dev split, report test AUROC at the chosen layer).

3. **multiple-choice-evaluation** — the text-only judge baseline (M1.4) parses a collusion verdict out of gpt-5.4's free-form generation over the committee transcript. Not a letter-parse in the classical MCQ sense, but the failure mode is identical (silent coercion of refusal/off-topic into a positive/negative label, sign flip under position bias if we were to A/B swap agent order).
   - convention to adopt: prompt gpt-5.4 with a three-way verdict `{COLLUSIVE, HONEST, OTHER}`; never coerce OTHER into COLLUSIVE or HONEST. Persist `(scenario_id, raw_generation, judge_verdict)` per row. Temperature ≤ 0.2, freeze judge model + prompt across all rows. Exclude OTHER from AUROC denominator and report the OTHER rate. No orientation swap needed (label is not an ordinal option), but committee agent order is randomized once per scenario at generation time and held fixed for both the probe and the judge to avoid confound. The judge model + BASE_URL + API_KEY are supplied by `task.md`, so this is NOT an open-item block.

## No-match log

- **ImageNet preprocessing** — no vision task; no match.
- **Steering-coefficient tuning** — no additive intervention; the plan is purely detection (extract → probe → aggregate). The "coefficient" analog for probes is the classifier regularization `C`; sklearn's default `C=1.0` on standardized features is standard, and a small `C ∈ {0.1, 1.0, 10.0}` dev-side grid is a routine tuning knob, not a symptom-triggered tip. No match.
- **Fine-tuning hyperparameter sweep** — no LM fine-tune. Logistic-regression probes and a small aggregator MLP are trained on cached features (no LR-sensitive language-model update). No match.
