# Captured-Behavior Report

**Direction**: (empty — behavior/claims sourced from `task.md`)
**Behavior-source**: given
**Mechanism**: given
**Claim source**: task.md (faithful capture)
**Date**: 2026-07-10
**Pipeline**: research-lit → faithful behavior capture (from task.md) → research-refine-pipeline

## Executive Summary

`task.md` specifies **four claims** about belief representation in Pythia language models — (1) scale-dependent emergence of personal vs attributed belief, (2) Fisher-information-based localization + causal verification of belief-specific attention heads, (3) formation-window analysis across intermediate `pythia-1b` checkpoints, (4) dynamic inference-time head amplification driven by a frame classifier. All four claims are jointly verified by a single composed proposal that layers Location → Causal Intervention → Formation Tracing → Tuning & Editing, using **only** the belief_core / belief_holdout datasets, the pretraining Pile perplexity control, and the Pythia 410M/1B/2.8B checkpoints named in `task.md`. This is the reproduction combination (`BEHAVIOR_SOURCE=given` + `MECHANISM=given`) → the Resource-Fidelity Harness applies (`resource_fidelity: strict` — no smaller-model substitution, no data subsetting).

## Literature Landscape

See `idea-stage/LANDSCAPE.md`. Key context used only for baselines / metric definitions (never to alter the four claims):
- **Fisher-mask localization** template: Sensitivity Meets Sparsity (arXiv:2504.04238, `task.md`'s explicit reference).
- **Head-level causal ablation** template: Wang et al. IOI Circuit (arXiv:2211.00593); Heimersheim & Nanda activation-patching best-practices (2309.16042); *Pattern Selectivity ≠ Task-Causal Structure* (2606.05378) motivates the 20-random-head baseline.
- **Formation-window** precedent: Tigges et al. NeurIPS 2024; *When Do Attention Circuits Form?* (2606.02378) — induction @ step ~1k, FV @ step ~16k.
- **Dynamic head amplification / activation steering** family: PASTA, SADI, Dynamic Activation Composition; Zhu et al. ICML 2024 (representation-level belief steering, 2402.18496).

## Global Resources (binding — reproduction fidelity)

Per `task.md`, the following resources are **binding**; the experiment stage must not substitute or downscale them.

- **Models**: `pythia-410m`, `pythia-1b`, `pythia-2.8b` at `/mnt/quarkfs/share_model/Ptyhia/`.
- **Intermediate checkpoints (Claim 3)**: `pythia-1b` checkpoints at `/mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/`.
- **Datasets**:
  - `belief_core`: `/data/xuhaoming/belief_loc/data/derived/belief_core/`
    - `world_knowledge` → `reality.jsonl` (227)
    - `personal_belief` → `believe_truth.jsonl` (681)
    - `attributed_belief` → `follow_belief.jsonl` (681)
  - `belief_holdout` (Claim 4 OOD only): `/data/xuhaoming/belief_loc/data/derived/belief_holdout/`
- **Third-person subset (Claim 2 only)**: `person ∈ {James, Mary}` → 454 per belief frame.
- **Pretraining perplexity corpus**: `/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/`.

Behavioral metric (all claims): `correct iff Σ_t log P(gold_t) > Σ_t log P(distractor_t)` over **completion tokens only**, **no length normalization**, tokenizer matched to each checkpoint.

## Claims to Verify

### Claim 1: Scale-Dependent Emergence

**Original (verbatim excerpt from task.md):**
> Claim 1: Scale-Dependent Emergence
> Personal belief and attributed belief exhibit distinct emergence patterns across model scales.
>
> 1. Behavioural Evaluation. Evaluate all models on all belief-related tasks. Use the full dataset for each belief-related task in this behavioral evaluation. Metrics: For each sample, compare the gold and distractor continuations: correct iff sum_t log P(gold_t) > sum_t log P(distractor_t). Sum log-probabilities over completion tokens only. Do not include prompt tokens. Do not apply length normalization. Use the tokenizer matching the checkpoint. Evaluate gold and distractor under identical prompt conditions.
> Require: The experiment should investigate how attributed belief and personal belief behave across model scales, including whether one belief ability exhibits stronger scale dependence and whether the scaling behavior is monotonic or non-monotonic.

**Extracted statement**: Across the three final-checkpoint Pythia sizes (410M, 1B, 2.8B), the two belief abilities — `personal_belief` (believe_truth) and `attributed_belief` (follow_belief) — show **distinguishable** accuracy-vs-scale curves; the two curves differ from each other and from `world_knowledge` (reality) in at least one of: (i) which ability is higher at each scale, (ii) whether scaling is monotonic, and (iii) the magnitude of change between adjacent scales.
**Hypothesis**: H1 — Personal-belief and attributed-belief accuracies do not co-vary with scale (they have distinct scaling functions on Pythia); at least one exhibits a stronger scale dependence than the other.
**Measurable predicate**: For each belief frame `f ∈ {personal_belief, attributed_belief, world_knowledge}` and each Pythia size `s ∈ {410M, 1B, 2.8B}`, compute `acc(f,s)` = fraction of items where `Σ_t log P(gold_t) > Σ_t log P(distractor_t)` on the *full* belief_core dataset. Report the two scale-conditional curves; declare *distinct emergence* iff the (sign, magnitude, monotonicity) profile differs between the two belief frames.
**Expected direction**: Prior literature (arXiv:2602.16085) reports third-person > first-person on average across LMs → we expect `acc(attributed_belief) > acc(personal_belief)` at ≥ one scale, but the specific scaling shape is an *open* result task.md asks us to characterize (monotonic vs non-monotonic).
**Resources (binding)**: models: pythia-410m, pythia-1b, pythia-2.8b (final checkpoints); dataset: belief_core (full, all three frames); used_n: `reality.jsonl` 227 + `believe_truth.jsonl` 681 + `follow_belief.jsonl` 681 = 1,589 items per model.
**Status**: pending verification.
**Notes**: Task.md forbids modifying prompts or labels and forbids cross-architecture generalization comparisons; the "scale" axis stays strictly inside the Pythia family.

---

### Claim 2: Belief-Heads Localization

**Original (verbatim excerpt from task.md):**
> Claim 2: belief heads Localization
> Personal belief and attributed belief are implemented by distinct, causally separable attention-head circuits.
>
> 2. Belief Heads Localization. Methods: Use Fisher information matrix, refering to this paper "Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models". Use three independent signals: F_attributed | attributed_belief, James + Mary only; F_personal | personal_belief, James + Mary only; F_knowledge | world_knowledge.
> Construct two independent target-specific Fisher masks or views:
> Mask_attributed = top 0.1% of F_attributed AND NOT top 1% of F_knowledge
> Mask_personal  = top 0.1% of F_personal  AND NOT top 1% of F_knowledge
> Use the Fisher signals to obtain target-specific candidate heads or rankings, then use attention-head zero-ablation to search for the smallest head set that satisfies the causal, specificity, baseline, and PPL criteria below.
> Require: The experiment should determine whether attributed belief and personal belief are supported by causally identifiable attention-head circuits. For models passing the above-chance criterion, identify candidate heads using the third-person localization subset, evaluate their causal effects through zero-ablation on the same localization subset, and report both baseline controls for each evaluated head set:
> - 20 random-head baseline mean ± 2σ, matched by head count.
> - 20 random-mask baseline mean ± 2σ, matched by parameter count.
> Select the final belief head set using these fixed thresholds:
> - Target behavior drop must be at least 0.30 absolute accuracy.
> - The effect must be outside the 20-random-head baseline 2σ band.
> - Each off-target behavior drop, including the other belief behavior and world_knowledge, must be at most 0.10 absolute accuracy.
> - PPL on the pretraining-corpus control must be no more than clean × 1.05.
> A valid belief circuit should selectively impair its corresponding target behavior while preserving the off-target behaviors and general language modeling ability under the thresholds above. If no head set satisfies all thresholds, report the claim as not localized or only partially localized rather than changing the thresholds.

**Extracted statement**: For each model that passes the above-chance behavioral criterion in Claim 1, there exist two **causally identifiable** attention-head sets — `HeadSet_personal` and `HeadSet_attributed` — each obtained by (i) computing `F_personal / F_attributed / F_knowledge` from third-person (James + Mary) belief_core items only, (ii) building `Mask_personal = top-0.1% F_personal AND NOT top-1% F_knowledge` and `Mask_attributed = top-0.1% F_attributed AND NOT top-1% F_knowledge`, (iii) aggregating masked-parameter mass per attention head to rank candidate heads, and (iv) searching the smallest head set whose **zero-ablation** simultaneously satisfies: target-behavior drop ≥ 0.30 absolute, effect outside 20-random-head 2σ band, each off-target drop ≤ 0.10 absolute (other belief frame AND world_knowledge), and PPL on the Pile-corpus control ≤ clean × 1.05. Also compare against a 20-random-mask baseline matched by parameter count. If no head set meets all thresholds, report **not-localized** or **partially localized** — thresholds are fixed, not negotiable.
**Hypothesis**: H2 — Personal-belief and attributed-belief heads are **distinct** at head-set granularity (not just parameter-set granularity) and can be causally isolated with the specified selectivity criteria.
**Measurable predicate**: On the third-person subset (454 items per frame), for each candidate head set `H`:
- `acc_H(target)` and `acc_baseline(target)` such that `acc_baseline − acc_H ≥ 0.30`,
- `acc_baseline(target) − acc_H(target) ∉ [μ_rand,20 − 2σ, μ_rand,20 + 2σ]` (outside the 20-random-head baseline 2σ band),
- `∀ other-frame f' ∈ {other belief, world_knowledge}: |acc_baseline(f') − acc_H(f')| ≤ 0.10`,
- `PPL_H(pile) ≤ 1.05 × PPL_baseline(pile)`,
- Report both baselines (20 random-head + 20 random-mask matched by parameter count) with mean ± 2σ.
**Expected direction**: Task.md hypothesizes *causal separation*; the pass/fail is defined by the fixed thresholds above. Report *not-localized* / *partially localized* if thresholds are not met — this **is** a valid outcome.
**Resources (binding)**: models: pythia-410m, pythia-1b, pythia-2.8b (final checkpoints) — Claim-2 skipped for any model that fails the above-chance criterion in Claim 1; dataset: `belief_core` third-person subset (James + Mary only), 454 items per belief frame; PPL control: pretraining Pile corpus at `/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/`; used_n: **exact 454 per frame** for Fisher + zero-ablation eval (no subsampling; no first-person examples).
**Status**: pending verification.
**Notes**: The thresholds are **fixed** by `task.md` (HARD CONSTRAINT). If threshold-based selection fails, report the claim as `not-localized` or `partially localized`; do not weaken thresholds. Fisher computation must use ONLY the James + Mary subset — no first-person examples for either Fisher computation or the head-ablation causal eval.

---

### Claim 3: Formation Window

**Original (verbatim excerpt from task.md):**
> Claim 3: Formation Window
> Belief-related circuits exhibit distinct developmental trajectories during pretraining.
>
> 3. Belief Formation Window Analysis. Methods: Analyze intermediate checkpoints of pythia-1b to study the emergence of belief abilities during pretraining. Conduct two evaluations across checkpoints:
> 1. Behavioral evaluation: measure world_knowledge, personal_belief, and attributed_belief performance on the full datasets at different pythia-1b training steps to characterize when each ability emerges.
> 2. Circuit intervention evaluation: using the pythia-1b personal_belief and attributed_belief head sets identified in Claim 2, separately zero-ablate each head set at every pythia-1b checkpoint. For each ablation condition, measure world_knowledge, personal_belief, and attributed_belief performance, so the causal trajectory records both the target effect and the cross-behavior controls for each belief head set.
> Require: The experiment should determine whether personal belief and attributed belief emerge at different stages of pretraining. The analysis should report the behavioural trajectory and causal intervention trajectory of each belief ability, and identify the corresponding formation windows based on predefined emergence criteria.

**Extracted statement**: On `pythia-1b` intermediate checkpoints, (a) the **behavioral trajectory** of `world_knowledge`, `personal_belief`, `attributed_belief` accuracy across training steps, and (b) the **causal-intervention trajectory** obtained by zero-ablating each `pythia-1b` head set from Claim 2 at every checkpoint (measuring all three frames per ablation condition), together determine a *formation window* per belief ability that is **distinguishable** between personal-belief and attributed-belief.
**Hypothesis**: H3 — Personal-belief and attributed-belief head sets form at *different* training steps (their causal-intervention trajectories cross zero-effect at different step counts) and/or their behavioral emergence steps differ; both trajectories are visible only if the Claim-2 head sets exist.
**Measurable predicate**: For each `pythia-1b` intermediate checkpoint `c` and each frame `f`:
- (behavioral) `acc(f, c)` measured on the full belief_core dataset;
- (causal, per Claim-2 head set `H ∈ {HeadSet_personal, HeadSet_attributed}`) `acc_H(f, c)` — the accuracy of frame `f` when ablating head set `H` at checkpoint `c`;
- **Formation window** for belief frame `f` — the earliest checkpoint range where `acc(f,·)` crosses a predefined emergence threshold AND `acc_baseline(f, c) − acc_H(f, c)` (target-drop when ablating `H` = head set for `f`) crosses a predefined causal-emergence threshold (both thresholds specified in EXPERIMENT_PLAN.md).
**Expected direction**: task.md hypothesizes *distinct* formation windows; direction is *inequality* between the two belief frames' emergence steps.
**Resources (binding)**: model: `pythia-1b` — intermediate checkpoints at `/mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/`; dataset: `belief_core` full (all three frames) for behavioral trajectory; third-person subset (454 per frame) for the Claim-2-derived head sets and their ablation-trajectory eval; used_n: full 1,589 items per checkpoint for behavioral trajectory; 454 per frame under each ablation condition.
**Status**: pending verification.
**Notes**: This claim is **downstream of Claim 2** — it consumes `HeadSet_personal` and `HeadSet_attributed` for `pythia-1b`. If Claim 2 reports *not-localized* on `pythia-1b`, the causal-intervention half of Claim 3 becomes N/A for the missing head set; the behavioral-trajectory half runs regardless.

---

### Claim 4: Dynamic Controllability

**Original (verbatim excerpt from task.md):**
> Claim 4: Dynamic Controllability
> The attention heads identified in Claim 2 are not only causally necessary but also causally controllable. A lightweight per-frame router can dynamically amplify these heads to selectively enhance personal belief and attributed belief while leaving world knowledge unchanged, providing an application-level demonstration that localized belief circuits can be turned into a controllable inference-time switch.
>
> 4. Dynamic Head Amplification. Methods: Use the belief heads identified in Section 2 to construct an inference-time controller that dynamically modulates belief-related circuits during the model forward pass.
> The controller should: infer the current task frame (world_knowledge, personal_belief, or attributed_belief) from the model's internal representations at layers before the identified belief heads; amplify the corresponding belief circuit within the same forward pass based on the inferred frame; preserve the base model behavior for unrelated frames, especially leaving world_knowledge samples unchanged.
> The frame classifier should be trained only on belief_core: /data/xuhaoming/belief_loc/data/derived/belief_core/ and evaluated on the out-of-distribution belief holdout: /data/xuhaoming/belief_loc/data/derived/belief_holdout/. The holdout should contain unseen propositions and categories to test whether the internal frame signal generalizes beyond training examples.
> Compare the controller against an oracle prompt-hint baseline, where the ground-truth frame is explicitly provided through a natural-language instruction. The prompt hints should state the task frame clearly: For world_knowledge, ask the model to extract and answer with the factual state of the proposition. For personal_belief, ask the model to ignore others' beliefs and answer according to its own belief about reality. For attributed_belief, ask the model to track the named person's belief and answer according to that person's belief. This baseline measures the benefit of internal circuit control beyond prompt-level task specification.
> Require: The experiment should determine whether localized belief circuits support controllable amplification and whether the internal frame signal reflects generalizable belief states rather than memorized content. A valid controller should demonstrate:
> 1. Frame generalization: the frame classifier should achieve reliable classification on unseen categories using internal model representations, showing that it captures task-level belief information rather than text-level shortcuts.
> 2. Intervention effectiveness: circuit amplification should improve belief behavior with positive item-level net improvement while maintaining low degradation on previously correct predictions and preserving world_knowledge and general model capability.
> Report intervention outcomes using item-level metrics, including recovery of incorrect cases, degradation of correct cases, and net improvement, with comparison against the oracle prompt-hint baseline.

**Extracted statement**: A lightweight **frame classifier** trained on `belief_core` — reading internal representations at layers *strictly before* the Claim-2 belief heads — predicts the current input frame (`world_knowledge` / `personal_belief` / `attributed_belief`) with **generalization to unseen categories** in `belief_holdout`; conditioned on this predicted frame, an inference-time controller **amplifies** the Claim-2 head set matched to the predicted belief frame (`HeadSet_personal` for `personal_belief`, `HeadSet_attributed` for `attributed_belief`) within the same forward pass, **without altering** the forward pass for `world_knowledge` inputs. The controller **outperforms** an oracle prompt-hint baseline on item-level net improvement while keeping degradation of previously-correct items low and preserving `world_knowledge` accuracy and general PPL.
**Hypothesis**: H4 — Sub-hypotheses jointly required:
- H4a (frame generalization): frame-classifier accuracy on `belief_holdout` (unseen categories) is significantly above chance and above any surface-feature control.
- H4b (intervention effectiveness): amplifying the classifier-selected head set yields **positive net item-level improvement** on the corresponding belief frame (recoveries − degradations > 0) while (i) leaving `world_knowledge` accuracy essentially unchanged and (ii) beating the oracle prompt-hint baseline on the same metric.
**Measurable predicate**:
- (frame gen) On `belief_holdout`: classification accuracy per frame; report macro-F1 and per-frame recall.
- (intervention) On `belief_holdout` (and on `belief_core` held-out split as an in-distribution reference), for each item:
  - `recovery`: item incorrect under baseline, correct under controller.
  - `degradation`: item correct under baseline, incorrect under controller.
  - `net_improvement = recovery − degradation` per belief frame.
  - `world_knowledge` accuracy delta must be ≈ 0 (predefined tolerance in EXPERIMENT_PLAN.md).
  - Report all three quantities for the oracle-prompt-hint baseline under identical evaluation conditions; the controller must beat the oracle baseline on `net_improvement` for the belief frames.
- Preserve general LM ability: PPL delta on Pile ≤ predefined tolerance.
**Expected direction**: task.md expects **positive net improvement** and **≈-zero world_knowledge delta**, and expects the internal controller to at least **match or beat** the oracle-prompt-hint baseline (otherwise the internal circuit control is not adding value beyond prompt-level specification).
**Resources (binding)**: models: pythia-410m / pythia-1b / pythia-2.8b (final checkpoints — same set as Claims 1/2) — but only those where Claim 2 successfully localized at least one belief head set; datasets: `belief_core` (frame-classifier training + in-distribution eval) and `belief_holdout` (OOD eval); PPL control: pretraining Pile corpus; used_n: full `belief_core` for classifier training, full `belief_holdout` for OOD eval.
**Status**: pending verification.
**Notes**: The frame classifier must be trained **only** on `belief_core` and evaluated **only** on `belief_holdout` (HARD CONSTRAINT). The classifier reads representations at layers **strictly before** the Claim-2 heads (so it is a *pre-head router*, not a post-hoc read of head output). This claim is downstream of Claim 2 — it depends on which head sets exist per model.

---

## Recommended: #1 — Composed Belief-Localization Reproduction Proposal (jointly covers Claims 1–4)

Under the reproduction combination, there is one unified proposal that layers all four claims into a single Location → Causal Intervention → Formation Tracing → Tuning & Editing chain — see `refine-logs/FINAL_PROPOSAL.md`.

## Next Steps

- [x] Faithful behavior/claim capture from task.md (this file).
- [ ] `/research-refine-pipeline` — produce `refine-logs/FINAL_PROPOSAL.md` (unified testing approach) + `refine-logs/EXPERIMENT_PLAN.md` (claim-driven roadmap, top-metadata stamped with `chosen_mechanism:` + `resource_fidelity: strict`, **no M0** since `BEHAVIOR_SOURCE=given`).
- [ ] `/auto-experiment` — implement + run under the Resource-Fidelity Harness (no downscaling).
- [ ] `/auto-verify` — stress-test per-claim.
- [ ] `/auto-iteration-loop` — iterate until reviewer-ready.
