# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips: []

## Matches

None of the symptom-triggered tips fire for this experiment.

- **ImageNet Eval Preprocessing** — n/a (no torchvision / vision backbone; the target model is Mistral-7B / Gemma-2-9B).
- **Steering-Coefficient Tuning** — n/a (no additive intervention; the pipeline is *replacement* patching — activation patching / path patching / resample ablation — with no `α`/dose knob).
- **Steering Block / Layer Selection** — n/a (no steering; the intervention set is discovered by attribution patching, not selected by a fixed block index).
- **Fine-Tuning Hyperparameter Sweep** — n/a (no fine-tune anywhere in the plan; every experiment is inference-only patching on frozen checkpoints).
- **Multiple-Choice Evaluation** — n/a (no regex letter-parse; the scorer is `logit_diff` / `prob_diff` / `KL` on the "True" vs "False" answer tokens — the answer token is a single fixed vocabulary id, not a free-form generation).

## General Rule for Mechanism / Interpretability

**Unconditionally loaded.** The plan already satisfies both parts:

1. **Locate the component set for the target function, then intervene** — Milestone M1 (attribution-patching screen) does the localization; M2/M3/M4 do the interventions on the located set.
2. **Intervene / locate on the target behavior only — do not damage general ability** — measured indirectly via (a) the specificity control (random component set of matched size ⇒ Recovery ≤ 0.2, gap ≥ 0.6) and (b) all-three-metrics reporting (`logit_diff`, `prob_diff`, `KL`) — a large `KL` on the *non-answer* token distribution flags off-distribution collapse. This makes the "does the model still generate normally" check first-class in every intervention milestone.

## No-match log

Borderline check: could Milestone M3 (sufficiency reinsertion) be considered a "steering" intervention? No — it re-inserts activations from a matched clean run, not an additive direction with a scalable coefficient. It has no `α`. The steering-coefficient tip explicitly says it fires only on *additive* interventions with a strength knob (α / β / dose / magnitude / scale / coefficient / k). Not matched.
