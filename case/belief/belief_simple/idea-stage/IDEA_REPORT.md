# Idea Report — Captured Behavior

**Behavior-source**: given
**Mechanism**: given
**Claim source**: task.md (faithful capture — reproduction of "Sensitivity Meets Sparsity" methodology on Pythia)
**Date**: 2026-07-22
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline (in progress)

## Resources (binding — reproduction combo, resource_fidelity: strict)

Every claim below is bound to the following resources, at full scale, with no substitution or subsetting except where task.md itself specifies (Claim 2 restricts Fisher signals for attributed / personal belief to `person ∈ {James, Mary}`; Claim 3 uses only `pythia-1b` intermediate checkpoints).

**Models** (from `/mnt/quarkfs/share_model/Ptyhia/`):
- `pythia-410m`
- `pythia-1b`
- `pythia-2.8b`
- Intermediate checkpoints for pythia-1b at `/mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/` (Claim 3 only — verified present: step0, 1, 2, 4, …, 512, 1000, 2000, …, 143000)

**Datasets** (all under `/data/xuhaoming/belief_loc/data/derived/`):
- `world_knowledge` = `belief_core/reality.jsonl` (n=227, factual control) — verified present
- `personal_belief` = `belief_core/believe_truth.jsonl` (n=681, core belief frame) — verified present
- `attributed_belief` = `belief_core/follow_belief.jsonl` (n=681, core belief frame) — verified present
- Pretraining corpus for PPL = `/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/`
- OOD holdout for Claim 4 evaluation = `belief_holdout/` (verified present: reality.jsonl, believe_truth.jsonl, follow_belief.jsonl)
- Claim 4 controller training = `belief_core/` (same location as core belief datasets)

**Behavioral evaluation must use the FULL dataset** (not a subset) for each belief-related task.

**Metric definition (verbatim from task.md — "behavioral correctness")**:
```
correct  ⇔  Σ_t log P_θ(y_t^+ | x, y_<t^+)  >  Σ_t log P_θ(y_t^- | x, y_<t^-)
```
where `x` is the prompt, `y⁺` is the gold continuation, `y⁻` is the distractor continuation. Applied per-example; accuracy = fraction correct.

**Chosen mechanism per claim** (BEHAVIOR_SOURCE=given + MECHANISM=given — per-claim map, no routing):
```yaml
chosen_mechanism:
  C1: not-applicable        # behavioural scaling — no mechanism intervention
  C2: fisher-information-matrix-zero-ablation
  C3: checkpoint-analysis-with-zero-ablation
  C4: probe-and-amplify-controller
```

## Claims to Verify

### Claim 1: Scale-Dependent Emergence

**Original (verbatim excerpt from task.md):**
> ### Claim 1: Scale-Dependent Emergence
> Personal belief and attributed belief exhibit distinct emergence patterns across model scales.
>
> [Behavioural Evaluation section]
> Use the full dataset for each belief-related task in this behavioral evaluation.
> The experiment should investigate how attributed belief and personal belief behave across model scales, including whether one belief ability exhibits stronger scale dependence and whether the scaling behavior is monotonic or non-monotonic.

**Extracted statement**: Across `pythia-{410m, 1b, 2.8b}`, the behavioural accuracy on `personal_belief` and on `attributed_belief` (measured with the log-prob-comparison metric on full datasets, with `world_knowledge` as a factual control) follows *distinct* scale-dependence patterns — at least one of the two belief abilities differs from the other in either slope, monotonicity, or absolute level across the three scales.

**Hypothesis**: H1 — Personal belief and attributed belief have separable scaling laws across pythia-{410m, 1b, 2.8b}; the two curves cannot be described by a single monotonic scale-response.

**Measurable predicate**: Behavioural accuracy `acc(model, task)` on each of `{world_knowledge, personal_belief, attributed_belief}` for each model in `{pythia-410m, pythia-1b, pythia-2.8b}` — evaluated on the FULL respective datasets (n=227, n=681, n=681) with the log-prob-comparison metric above. The Claim-1 evidence is the 3×3 accuracy matrix + reported observations on (a) whether one of the two belief abilities exhibits *stronger* scale dependence than the other and (b) whether the scaling is monotonic or non-monotonic.

**Expected direction**: not a single-directional threshold — the claim is *distinctness*, so evidence is any of: (i) monotonicity mismatch (one belief scales monotonically, the other not); (ii) slope mismatch (one belief scales strongly, the other only weakly); (iii) crossover between the two curves across scale; (iv) one belief clearly above-chance while the other is at chance at some scale.

**Resources (binding)**: models: `pythia-410m`, `pythia-1b`, `pythia-2.8b` (full weights, no substitution); datasets: `world_knowledge` (n=227), `personal_belief` (n=681), `attributed_belief` (n=681) — full datasets, no subsetting.

**Status**: pending verification.

**Notes**: The metric is *purely behavioural* (log-prob-comparison on gold vs distractor continuation); no internal-state access is needed. This claim is the entry point of the numbered order and is a prerequisite for Claim 2's "above-chance" gate.

---

### Claim 2: Belief Heads Localization

**Original (verbatim excerpt from task.md):**
> ### Claim 2: belief heads Localization
> Personal belief and attributed belief are implemented by distinct, causally separable attention-head circuits.
>
> [Methods section]
> Use Fisher information matrix, refering to this paper "Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models"
> Use three independent signals:
>   F_attributed — attributed_belief, James + Mary only
>   F_personal   — personal_belief, James + Mary only
>   F_knowledge  — world_knowledge
> Construct two independent target-specific Fisher masks:
>   Mask_attributed = top 0.1% of F_attributed AND NOT top 1% of F_knowledge
>   Mask_personal   = top 0.1% of F_personal   AND NOT top 1% of F_knowledge
> Use the Fisher signals to obtain target-specific candidate heads or rankings, then use attention-head zero-ablation to search for the smallest head set that satisfies the causal, specificity, baseline, and PPL criteria below.
>
> [Requirements]
> For models that perform above chance on the target task, identify candidate heads using the third-person subset (person in {James, Mary}) and test them with zero-ablation.
> For each final candidate head set, report:
>   - 20 random-head controls with the same number of heads;
>   - 20 random-mask controls with the same number of parameters.
> A circuit is considered localized if:
>   - accuracy on the target task drops by at least 0.30;
>   - this drop is greater than the random-head baseline mean plus 2σ;
>   - accuracy on the other belief task and world_knowledge drops by no more than 0.10;
>   - PPL after ablation is no more than 1.05 × the clean PPL.
> If no candidate set meets all criteria, report it as partially localized or not localized without changing the thresholds.

**Extracted statement**: For each model that clears the Claim-1 above-chance gate on the target belief task, the Fisher-information matrix over parameters — computed on the third-person subset (`person ∈ {James, Mary}`) of the target task with a world-knowledge control — yields, via the AND-NOT mask construction (top-0.1% target AND-NOT top-1% control), a set of candidate attention heads. Zero-ablation of the *smallest* head subset satisfying all four criteria below constitutes localization of that belief ability's causally-separable circuit; localization must hold *independently* for `personal_belief` and for `attributed_belief`, and the two head sets should be distinct.

**Hypothesis**: H2 — Distinct, small (sparse) attention-head sets are causally responsible for personal-belief and for attributed-belief behaviour in above-chance-performing pythia models, and the two sets are dissociable (each set damages its target task strongly and the other belief task and world knowledge only marginally, with only marginal general-LM damage).

**Measurable predicate (per model, per belief target ∈ {personal, attributed})**: There exists a head set H* such that zero-ablating H* satisfies simultaneously:
- (C2a — causal effect) `acc(target task) - acc_ablated(target task) ≥ 0.30`;
- (C2b — significance vs random-head baseline) the target-task accuracy drop from H* is `> mean(random-head-baseline drops) + 2 × std(random-head-baseline drops)`, where the baseline is 20 random-head sets of the *same head count*;
- (C2c — specificity) `acc(other belief task) - acc_ablated(other belief task) ≤ 0.10` AND `acc(world_knowledge) - acc_ablated(world_knowledge) ≤ 0.10`;
- (C2d — general-LM preservation) `PPL_ablated / PPL_clean ≤ 1.05` on the pretraining PPL corpus.
Additionally, `H*` is the *smallest* such set (deterministic search discipline defined in the experiment plan). Also report the 20 random-mask controls with the same *parameter count* as H* (per task.md — for reporting; not part of the 4-criteria acceptance test).

**Expected direction**: H* exists distinctly for personal and for attributed belief on at least the largest scale (pythia-2.8b), plausibly at pythia-1b, uncertain at pythia-410m (which may not clear the above-chance gate).

**Resources (binding)**:
- models: `pythia-410m`, `pythia-1b`, `pythia-2.8b` (only run localization on models above chance on the target task; below-chance models are reported as "not applicable" for the target).
- Fisher signals: `F_attributed` on `attributed_belief` restricted to `person ∈ {James, Mary}`; `F_personal` on `personal_belief` restricted to `person ∈ {James, Mary}`; `F_knowledge` on full `world_knowledge` (n=227).
- Zero-ablation *evaluation*: on the FULL respective datasets (n=681 / n=681 / n=227) — the "James + Mary" restriction is for Fisher-signal construction only.
- PPL corpus: `/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/` (fixed sample & seed across all ablation runs, controls, and clean baseline — pinned in EXPERIMENT_PLAN.md).
- Controls: 20 random-head + 20 random-mask (per final head set, per model).

**Status**: pending verification.

**Notes**: The thresholds (0.30, 2σ, 0.10, 1.05×) and the control counts (20 + 20) are verbatim from task.md; the reproduction adopts them as fixed and reports partial/not-localized honestly rather than tuning to pass. The "smallest set" search discipline is pinned in the experiment plan (gap R2 from the landscape).

---

### Claim 3: Formation Window

**Original (verbatim excerpt from task.md):**
> ### Claim 3: Formation Window
> Belief-related circuits exhibit distinct developmental trajectories during pretraining.
>
> [Methods section]
> Analyze intermediate checkpoints of `pythia-1b` to study the emergence of belief abilities during pretraining.
> Conduct two evaluations across checkpoints:
>   1. Behavioral evaluation: measure world_knowledge, personal_belief, and attributed_belief performance on the full datasets at different pythia-1b training steps to characterize when each ability emerges.
>   2. Circuit intervention evaluation: using the pythia-1b personal_belief and attributed_belief head sets identified in Claim 2, separately zero-ablate each head set at every pythia-1b checkpoint. For each ablation condition, measure world_knowledge, personal_belief, and attributed_belief performance, so the causal trajectory records both the target effect and the cross-behavior controls for each belief head set.
>
> [Requirements]
> The experiment should determine whether personal belief and attributed belief emerge at different stages of pretraining. The analysis should report the behavioural trajectory and causal intervention trajectory of each belief ability, and identify the corresponding formation windows based on predefined emergence criteria.

**Extracted statement**: The *behavioural* trajectory of `personal_belief` and `attributed_belief` accuracy across the pythia-1b intermediate checkpoints (on the FULL datasets), AND the *causal* trajectory obtained by zero-ablating the pythia-1b Claim-2 head sets at each checkpoint (measuring all three tasks per ablation condition), together reveal *distinct* formation windows for the two belief abilities — the step-range in which each ability transitions from at-chance to above-chance, and the step-range in which its Claim-2 head set becomes causally responsible for the ability, are not the same for personal and attributed belief.

**Hypothesis**: H3 — Personal belief and attributed belief emerge at different pretraining step-ranges in pythia-1b, both behaviourally and causally. The developmental trajectories are dissociable (different emergence steps and/or different post-emergence dynamics).

**Measurable predicate**:
- Behavioural trajectory: `acc(task, checkpoint)` for `task ∈ {world_knowledge, personal_belief, attributed_belief}` and every pythia-1b intermediate checkpoint (native schedule: step 0, 1, 2, 4, …, 512, 1000, 2000, …, 143000).
- Causal trajectory: for `head_set ∈ {H*_personal, H*_attributed}` (from Claim 2 on pythia-1b), for every pythia-1b checkpoint, for every `task ∈ {world_knowledge, personal_belief, attributed_belief}`, measure `acc_ablated(head_set, task, checkpoint)` and derive the causal-effect trajectory `Δ(head_set, task, checkpoint) = acc(task, checkpoint) - acc_ablated(head_set, task, checkpoint)`.
- Formation window per (target, criterion): the range of checkpoints in which `acc(target, checkpoint)` crosses a predefined emergence threshold (defined in EXPERIMENT_PLAN.md) and `Δ(H*_target, target, checkpoint)` crosses a predefined causal-emergence threshold.

**Expected direction**: The behavioural-emergence step and the causal-emergence step for `personal_belief` differ from those for `attributed_belief`. The Claim-3 evidence is the two trajectories (behavioural + causal) plus the identified formation-window step-ranges per ability.

**Resources (binding)**:
- Model: `pythia-1b` intermediate checkpoints only (native Pythia schedule, from `/mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/step*`).
- Datasets: FULL `world_knowledge` (n=227), FULL `personal_belief` (n=681), FULL `attributed_belief` (n=681) at every checkpoint.
- Head sets: `H*_personal` and `H*_attributed` for pythia-1b, inherited from Claim 2 on the final pythia-1b checkpoint. Applied unchanged (same head identities) at every earlier checkpoint per task.md.
- Depends on: Claim 2's pythia-1b head sets (if pythia-1b fails Claim 2 for a belief target, Claim 3's causal trajectory for that target cannot be run — report accordingly).

**Status**: pending verification.

**Notes**: Applying head identities from the final checkpoint to earlier checkpoints is justified by Prakash et al. 2024 (LLM Circuit Analyses Consistent Across Training and Scale, arXiv:2407.10827). The "predefined emergence criterion" — since task.md says "predefined emergence criteria" but does not fix them — is defined in EXPERIMENT_PLAN.md (e.g., first checkpoint with `acc ≥ 0.60` on target belief task and `Δ ≥ 0.20` when the head set is ablated), and is fixed *before* running the sweep so that the reported formation window is not tuned to pass.

---

### Claim 4: Dynamic Controllability

**Original (verbatim excerpt from task.md):**
> ### Claim 4: Dynamic Controllability
> A lightweight frame router can selectively amplify the belief heads identified in Claim 2 at inference time, enabling targeted improvements in personal or attributed belief behavior.
>
> [Method section]
> Use the belief heads from Section 2 to build a controller that intervenes during the model forward pass.
> After processing the input through early layers, the controller should:
>   - infer the current frame from probing layers before the selected belief heads (the probing representation may combine multiple layers);
>   - amplify the corresponding belief heads in later layers within the same forward pass;
>   - apply no amplification to world_knowledge.
> Train the frame classifier on: /data/xuhaoming/belief_loc/data/derived/belief_core/
> Evaluate it on the OOD holdout: /data/xuhaoming/belief_loc/data/derived/belief_holdout/
> Add an prompt-hint baseline that explicitly specifies the required reasoning frame (e.g., answer reality while ignoring others' beliefs, or follow the named person's belief).
>
> [Requirements]
> Report OOD frame-classification accuracy, recovered and degraded predictions, net improvement, task accuracy, world_knowledge accuracy, PPL, and comparison with the prompt-hint baseline.

**Extracted statement**: A two-stage lightweight controller — (a) a *frame classifier* (a probe combining hidden states from early layers, before the Claim-2 belief heads) predicting the current frame ∈ {world_knowledge, personal_belief, attributed_belief}; (b) amplification of the frame-matched Claim-2 belief-head set in later layers within the same forward pass; no amplification when the classifier predicts world_knowledge — trained on `belief_core/` and evaluated on the `belief_holdout/` OOD split, *improves* belief-task behaviour on the OOD set relative to the un-amplified baseline while preserving `world_knowledge` accuracy and general LM ability (PPL), AND at least matches the prompt-hint baseline that explicitly specifies the required reasoning frame in the prompt.

**Hypothesis**: H4 — Head-restricted amplification, gated by an early-layer frame probe, yields non-trivial *net improvement* on OOD belief tasks (positive recovered-minus-degraded and task-accuracy gain over un-amplified) with no meaningful damage to `world_knowledge` accuracy or PPL, and is at least comparable to the prompt-hint baseline.

**Measurable predicate (on `belief_holdout/`, OOD split)**:
- Frame classifier accuracy: `frame_acc_OOD` (3-way classification into `{world_knowledge, personal_belief, attributed_belief}`).
- Recovered predictions: count of OOD examples whose prediction becomes correct under controller vs. incorrect without the controller.
- Degraded predictions: count of OOD examples whose prediction becomes incorrect under controller vs. correct without.
- Net improvement: `recovered - degraded` (also reported as `acc_controller_OOD - acc_baseline_OOD`).
- Task accuracy under controller: `acc_controller_OOD(personal_belief), acc_controller_OOD(attributed_belief)`.
- `world_knowledge` accuracy under controller: `acc_controller(world_knowledge)` — must not degrade meaningfully vs. un-amplified baseline.
- PPL under controller: `PPL_controller / PPL_clean` on the pretraining PPL corpus — must not degrade meaningfully.
- Prompt-hint baseline: same metrics on `belief_holdout/` when the prompt explicitly says the required frame (e.g., "answer reality while ignoring others' beliefs" for `personal_belief`; "follow the named person's belief" for `attributed_belief`). Report side-by-side.

**Expected direction**: Frame classifier OOD accuracy substantially above chance (chance = 33% for 3-way); positive net-improvement on the two belief tasks; `world_knowledge` accuracy and PPL preserved; controller performance ≥ prompt-hint baseline on OOD belief accuracy (a strong pass would be a *clear* win vs the prompt-hint baseline).

**Resources (binding)**:
- Model: whichever pythia model was successfully localized for both belief targets in Claim 2 (natural default: pythia-2.8b; also pythia-1b if it localized). The choice is pinned in EXPERIMENT_PLAN.md.
- Head sets: `H*_personal` and `H*_attributed` from Claim 2 on the chosen model.
- Frame-classifier training data: `belief_core/` — the same `reality.jsonl`, `believe_truth.jsonl`, `follow_belief.jsonl` as Claim 1 (labels = source file / frame identity; 227 + 681 + 681 = 1589 examples). Standard train/val split defined in EXPERIMENT_PLAN.md.
- OOD evaluation data: `belief_holdout/` (verified present: reality.jsonl, believe_truth.jsonl, follow_belief.jsonl).
- Prompt-hint baseline: same OOD dataset, evaluation without the classifier + amplification but with an explicit frame-instructing prefix in the prompt (exact prefix pinned in EXPERIMENT_PLAN.md).

**Status**: pending verification.

**Notes**: The "amplify" magnitude is a hyperparameter (landscape gap R5). It is grid-searched on the `belief_core/` validation split *inside* the training pipeline, and the best-magnitude controller is then evaluated OOD on `belief_holdout/`. The choice of "which model" is contingent on Claim 2's outcome — if only pythia-2.8b localizes, only pythia-2.8b is used for Claim 4; if multiple models localize, EXPERIMENT_PLAN.md lists the priority order. Because there is no M0 gate (BEHAVIOR_SOURCE=given, not given-validation), Claim 4 does NOT declare `depends_on: [M0]`; it only inherits from Claim 2 through the `H*_*` head sets.

---

## Recommended: #1 — Claim 1: Scale-Dependent Emergence

Ordering rationale: task.md explicitly requires the numbered order (behavioural → localization → formation window → controller). Each stage may only use artifacts produced by earlier stages. Therefore Claim 1 (Scale-Dependent Emergence) is the natural entry point: it produces the "above-chance" gate that Claim 2 requires, and it is the cheapest stage (behavioural evaluation only, no mechanism intervention). Claims 2 → 3 → 4 follow in strict order. There is no ranking beyond the natural task order (this is the reproduction combination — all claims must be verified).
