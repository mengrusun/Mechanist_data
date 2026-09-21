# Final Proposal — Reproduction of "Sensitivity Meets Sparsity" on Belief Circuits in Pythia

```yaml
resource_fidelity: strict
mechanism_strategy: n/a
chosen_mechanism:
  C1: not-applicable
  C2: fisher-information-matrix-zero-ablation
  C3: checkpoint-analysis-with-zero-ablation
  C4: probe-and-amplify-controller
behavior_source: given
mechanism: given
project_kind: faithful-reproduction
reference_paper: "Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models (Chen et al., arXiv:2504.04238 / npj AI 2025)"
```

## Problem Anchor

Belief ability in pretrained transformer language models has two dissociable modes: **personal belief** (tracking the objectively true state of the world) and **attributed belief** (tracking an agent's mental state, even when it conflicts with reality). We reproduce a four-claim study on the Pythia family (410M / 1B / 2.8B) that (1) measures how each ability scales, (2) localizes the attention-head circuits implementing each ability via Fisher-information-guided zero-ablation, (3) traces the developmental trajectory of each ability across intermediate pretraining checkpoints of pythia-1b, and (4) builds an inference-time controller that reads the current reasoning frame from early layers and amplifies the matched belief-head set in later layers.

**This is a reproduction, not a redesign.** Methods, thresholds, dataset scope, model list, and control counts are fixed by `task.md` and re-used verbatim. The role of refinement here is not to alter methods but to pin the five reproduction-specific implementation gaps identified in `idea-stage/LANDSCAPE.md` (R1-R5), so downstream execution is deterministic and auditable.

## Method Thesis (unified across the four claims)

**Fisher-information-guided sparse attention-head localization + causal zero-ablation + developmental checkpoint sweep + head-restricted amplification controller** — applied consistently across four claims that share the same three-task frame (`world_knowledge` control, `personal_belief`, `attributed_belief`) and the same log-prob-comparison behavioural metric. The four claims form a strict dependency chain (numbered order enforced): behavioural scaling → localization → formation window → controller.

**Behavioural correctness metric (verbatim from task.md, applied everywhere accuracy is measured):**
$$
\mathrm{correct} \iff \sum_{t=1}^{|y^{+}|}\log P_\theta(y_t^{+}\mid x,y_{<t}^{+}) > \sum_{t=1}^{|y^{-}|}\log P_\theta(y_t^{-}\mid x,y_{<t}^{-})
$$
where `x` is the prompt, `y⁺` is the gold continuation, `y⁻` is the distractor.

## Dominant contribution

Faithful reproduction of Chen et al.'s Fisher-mask + zero-ablation localization pipeline in a **different model family** (Pythia rather than Llama 3), applied to a **finer dissociation** than the original ToM formulation — separating *personal* (reality-tracking) from *attributed* (other-agent-tracking) belief — and extending it into the developmental (Claim 3) and controllable (Claim 4) regimes.

## Optional supporting contribution

Pinning the reproduction gaps R1-R5 (below) supplies the community with a deterministic recipe for re-running the same protocol on any Pythia-family or comparable model without ambiguity — turning a paper description into a reproducibility harness.

## Explicitly rejected complexity

Because this is the reproduction combination (`BEHAVIOR_SOURCE=given` + `MECHANISM=given` → `resource_fidelity: strict`), we explicitly do NOT:
- introduce alternative attribution methods (e.g., activation patching, SAE features, integrated gradients);
- introduce alternative interventions (e.g., mean-ablation, resampling ablation);
- alter the four verbatim criteria (drop ≥ 0.30, > mean+2σ vs random-head baseline, off-target ≤ 0.10, PPL ≤ 1.05×);
- alter the mask construction (top-0.1% target AND-NOT top-1% control);
- alter the control counts (20 random-head + 20 random-mask);
- alter the person restriction (Fisher signals for belief tasks use `person ∈ {James, Mary}` only);
- subset the datasets for behavioural evaluation (always full n=227 / 681 / 681);
- substitute smaller / cheaper models (always full-weight pythia-410m / 1b / 2.8b);
- add an M0 phenomenon-validation gate (BEHAVIOR_SOURCE=given, so the phenomenon is assumed);
- extend Claim 3 to non-pythia-1b models (task.md restricts formation-window analysis to pythia-1b);
- add non-prompt-hint baselines to Claim 4 beyond what task.md specifies.

## Must-Prove Claims

1. **C1 — Scale-Dependent Emergence**: `personal_belief` and `attributed_belief` behavioural-accuracy curves across pythia-{410m, 1b, 2.8b} are *distinct* — mismatched monotonicity, mismatched slope, crossover, or above-chance/at-chance mismatch. Evidence = the 3×3 accuracy matrix plus a written characterization of the distinctness. **Scope caveat (iteration ⓪ i1):** with only 3 model sizes this should be read as *observed differential behavior across the available Pythia scales*, not a scaling-law characterization; the personal-belief non-monotonicity (0.85 → 0.79 → 0.99) precludes any smooth monotonic reading.
2. **C2 — Belief Heads Localization**: For each **Pythia** model that clears the above-chance gate on the target belief task, the Fisher-mask + zero-ablation search identifies a *smallest* head set H* satisfying all four verbatim criteria; H*_personal and H*_attributed are dissociable (each damages its target ≥ 0.30 and the other belief + world_knowledge ≤ 0.10; PPL ≤ 1.05× clean); 20 random-head + 20 random-mask controls are reported for each final H*. **Boundary caveat (iteration ⓪ i1):** the localization result is supported only within the Pythia family. A cross-family swap-test to OLMo-1B (16L/16H, 2048D — matched to the 1B scale) did NOT transfer: for personal_belief the Fisher-ranked heads meet C2a but violate C2c and C2d (off-target drops ≥10% and PPL up to 46× clean, indicating entanglement with general LM computation); for attributed_belief ablating the top-Fisher heads *increases* accuracy at all 30 greedy steps, i.e. the same Fisher procedure selects **suppression heads** rather than encoding heads in OLMo-1B — the causal direction is inverted. Jackknife ρ=0.954 on OLMo-1B confirms the Fisher signal itself is stable, so the failure is architectural / training-corpus dependent, not statistical. C2 should therefore be interpreted as a **Pythia-family localization result**, not evidence that Fisher-based belief-head localization transfers across 1B-scale transformer families. Terminology note: "belief heads" is an operational designator for the Pythia-family Fisher-selected set, not a claim of universal semantic head identity across model families.
3. **C3 — Formation Window**: On pythia-1b intermediate checkpoints (native Pythia schedule), the *behavioural* and *causal* trajectories of `personal_belief` and `attributed_belief` reveal distinct formation windows — different step-ranges for behavioural emergence, different step-ranges for causal-circuit emergence. **Sparsity caveat (iteration ⓪ i1):** only 24 of the 154 planned pythia-1b intermediate checkpoints were available on disk (native log-spaced subset — this is a *missing-files* environmental condition, not a cost-saving downscale; strict-harness HALT-on-downscale rule inapplicable). The reported formation windows are therefore **interval-censored coarse bounds over the observed checkpoint subset**, not high-resolution estimates of the underlying onset times. Specifically: personal's window [13000, 13000] reflects the sampling granularity (the two adjacent recorded checkpoints are step 8000 and step 23000), not a truly sharp transition; attributed's early behavioural window (from step 0) reflects the operational definition (acc ≥ 0.60 with 2-of-next-3 persistence) applied to a prompt structure that biases toward follow-belief continuations at initialization, not a claim that the *causal* circuit exists at step 0 (its causal emergence is step 33000).
4. **C4 — Dynamic Controllability**: A frame-classifier probe over early-layer representations + amplification of the frame-matched Claim-2 belief-head set in later layers, tuned on `belief_core/` and evaluated OOD on `belief_holdout/`, produces positive net improvement on OOD belief tasks, preserves `world_knowledge` accuracy and PPL, and is at least comparable to the prompt-hint baseline. **PPL disclosure caveat (iteration ⓪ i1):** "WK exactly preserved" (0.850 → 0.850 on pythia-1b, 0.886 → 0.886 on pythia-2.8b) refers to task-level WK accuracy on the belief_holdout WK subset; general LM ability under the controller is reported via the Pile PPL sample (same 1,048,576-token cached tensor as C2): clean 8.756 → controller-on 9.172 (ratio 1.047×) on pythia-1b, clean 7.330 → controller-on 7.364 (ratio 1.005×) on pythia-2.8b — both under the C2 1.05× bar, but non-zero. Model-dependent effect size (net_impr +151 on 1b vs +27 on 2.8b) precludes any claim of uniform controllability across scale.

## Method Detail Per Claim

### C1 — Scale-Dependent Emergence (behavioural only)

For each model in `{pythia-410m, pythia-1b, pythia-2.8b}` and each task in `{world_knowledge, personal_belief, attributed_belief}`, compute per-example correctness via the log-prob-comparison metric on the FULL dataset (n=227, n=681, n=681 respectively) and report the accuracy. The full 3×3 matrix constitutes the evidence.

Auxiliary reporting: (i) 95% Wilson confidence intervals on each cell; (ii) chance baseline (0.5, since correctness is a binary >-comparison of gold vs distractor log-probs); (iii) a written characterization of distinctness — whether one belief ability has stronger scale dependence, whether the curves are monotonic, whether they cross.

Output feeds C2's "above-chance" gate: a cell qualifies if its accuracy is strictly above 0.5 with the confidence interval above 0.5 as well (else "not applicable" for that (model, target) pair — no C2 localization run).

### C2 — Belief Heads Localization (Fisher + zero-ablation)

**Fisher-signal construction.** For each model in `{pythia-410m, pythia-1b, pythia-2.8b}` (independently), compute three parameter-level Fisher signals using the empirical Fisher approximation (mean of gradient-squared over the specified dataset with the model's own predictions as targets, i.e., the "practical" Fisher: `F_i = E_x[(∂ log p_θ(y|x) / ∂θ_i)²]`, aggregated per-parameter):
- `F_attributed`: on `attributed_belief` restricted to `person ∈ {james, mary}` (verified via dataset inspection: the `person` field uses lowercase codes `1p` / `james` / `mary`, each with 227 examples; James+Mary subset = **454 examples**).
- `F_personal`: on `personal_belief` restricted to `person ∈ {james, mary}` (same criterion on `believe_truth.jsonl`; subset = **454 examples**).
- `F_knowledge`: on FULL `world_knowledge` (n=227).

For each parameter, use the target `y` = the gold continuation `y⁺` from the metric definition. Fisher is accumulated in fp32; final signal is aggregated per-attention-head (sum of `F_i` over all parameters `θ_i` belonging to that head's `W_Q`, `W_K`, `W_V`, `W_O` matrices — the four projection matrices attached to that head).

**Mask construction (verbatim).**
- `Mask_attributed = top 0.1% of F_attributed AND NOT top 1% of F_knowledge`
- `Mask_personal   = top 0.1% of F_personal   AND NOT top 1% of F_knowledge`

Since Fisher is aggregated to head-granularity for the localization search, "top-0.1%" of parameters is first materialized parameter-level then reduced to a per-head "candidate score" (fraction of a head's parameters that fall in the mask). Heads are then ranked by this candidate-score descending, and the ranking feeds the smallest-head-set search below.

**Above-chance gate (R1).** Only run localization on `(model, target)` pairs that cleared the C1 above-chance gate. Below-chance pairs are reported as `not applicable` — the gate is not tuned to force pythia-410m through.

**Smallest-head-set search (R2 — pinned deterministic algorithm).**
- Order candidates by descending head-level candidate-score (from the Mask above).
- **Greedy-add phase**: starting from the empty set, add heads one at a time in ranked order. After each addition, run the four-criteria evaluation on the FULL target task (n=681) + FULL other belief task (n=681) + FULL world_knowledge (n=227) + PPL sample. Stop at the first set S⁺ that satisfies all four criteria simultaneously.
- **Greedy-remove sanity check**: if a passing S⁺ is found with |S⁺| > 1, attempt one-shot removals of each head in S⁺; if any removal still satisfies all four criteria, take the smaller set as S⁺. Iterate to fixed point (typically 1-3 rounds).
- If no set of size ≤ 30 satisfies all four criteria, report "not localized" for that (model, target) without altering thresholds. Cap at 30 heads to bound the search cost (well beyond the ~1-10 heads Chen et al. and IOI-line report — a set requiring more than 30 heads is already outside the "sparse localization" regime the paper claims).

**Zero-ablation.** For a head set `S`, zero-ablation replaces the head's *output* projection contribution to the residual stream with a zero tensor at every token position during the forward pass. Concretely, we zero the head's per-token output vector before it is added into the residual stream through `W_O` — this is the standard "attention-head knockout" used by Wang et al. 2022 (IOI). Implemented via a forward-hook on the attention module.

**Controls (verbatim: 20 + 20).** For each final `H*` (per (model, target) pair):
- **20 random-head controls**: 20 uniformly-random head subsets of the same *head count* as `|H*|`, drawn without replacement over all attention heads in the model. Deterministic seeds `[100, 101, …, 119]`. For each, run zero-ablation and record the 4 metrics (target accuracy, other belief accuracy, world_knowledge accuracy, PPL). The distribution over the 20 target-accuracy drops gives the `mean ± σ` bar the 2σ criterion checks against.
- **20 random-mask controls**: 20 uniformly-random parameter subsets of the same *total parameter count* as the parameters within `H*` (sum of the `W_Q`, `W_K`, `W_V`, `W_O` parameter counts across all heads in `H*`), drawn without replacement across all model parameters. Deterministic seeds `[200, 201, …, 219]`. Zeroing at parameter-level rather than head-output-level. Reported (per task.md) but NOT part of the 4-criteria acceptance test.

**PPL sample (R3 — pinned).** The PPL evaluation uses a fixed sample from the pretraining corpus at `/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/`. Sample-size and sampling-seed are pinned once and re-used identically for the clean baseline PPL and every ablated / random-head-control / random-mask-control run:
- Sample-size: `PPL_TOKENS = 1_048_576` tokens (~1M tokens, standard sample size used by Chen et al. and the Pythia analysis line — sufficient for stable PPL to within ±0.01).
- Sample construction: read the pre-shuffled Pile shards in order, truncate to `PPL_TOKENS` tokens after the model's own tokenization, cache to a fixed file `refine-logs/artifacts/ppl_sample.pt` on first use. All subsequent PPL calls (clean, ablated, control) read from this cached tensor — guarantees exactly the same token stream for paired comparison.

### C3 — Formation Window (checkpoint sweep on pythia-1b)

**Model scope.** `pythia-1b` intermediate checkpoints ONLY. Native Pythia schedule available on disk: step 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 (11 early checkpoints), then 1000, 2000, …, 143000 (143 log-spaced later checkpoints) — total 154 checkpoints (R4).

**Head sets.** `H*_personal` and `H*_attributed` from C2 applied on the FINAL pythia-1b checkpoint (step 143000 = full-trained model). Applied *unchanged* (same head identities) at every earlier checkpoint. Justified by Prakash et al. 2024 (arXiv:2407.10827): circuits identified at one checkpoint are structurally identifiable at others.

**Per-checkpoint runs.** At each of the 154 checkpoints:
- **Behavioural trajectory**: three accuracy values — `acc(world_knowledge)`, `acc(personal_belief)`, `acc(attributed_belief)` — computed via the log-prob-comparison metric on the FULL respective datasets.
- **Causal trajectory**: two zero-ablation conditions × three tasks = six values.
  - Ablate `H*_personal`: measure `acc_ablated(H*_personal, world_knowledge)`, `acc_ablated(H*_personal, personal_belief)`, `acc_ablated(H*_personal, attributed_belief)`.
  - Ablate `H*_attributed`: measure `acc_ablated(H*_attributed, world_knowledge)`, `acc_ablated(H*_attributed, personal_belief)`, `acc_ablated(H*_attributed, attributed_belief)`.

**Predefined emergence criteria (pinned BEFORE the sweep so the reported window is not tuned to pass):**
- **Behavioural emergence step (per target)**: the first checkpoint `t*` such that `acc(target, t*) ≥ 0.60` AND `acc(target, t) ≥ 0.60` for at least 2 of the next 3 recorded checkpoints (persistence check to avoid single-checkpoint blips).
- **Causal emergence step (per target)**: the first checkpoint `t†` such that `Δ(H*_target, target, t†) = acc(target, t†) - acc_ablated(H*_target, target, t†) ≥ 0.20`, with the same 2-of-next-3 persistence check.
- If a criterion is never crossed within the 154 checkpoints, report "not observed" — do not lower the threshold.

**Formation window (per target)** = the interval `[t*_behavioural, t†_causal]` if both are observed, or the singleton `{t*_behavioural}` if only behavioural emergence is observed, or "not observed" otherwise. Distinctness of formation windows across the two belief targets constitutes the C3 evidence.

### C4 — Dynamic Controllability (probe-and-amplify controller)

**Model choice.** The reproduction runs C4 on **every** pythia model for which C2 localized BOTH `H*_personal` and `H*_attributed`. Priority order (largest first): pythia-2.8b → pythia-1b → pythia-410m. At minimum one model is used (typically pythia-2.8b); more if C2 succeeds on more.

**Frame classifier.** Input = concatenation of hidden-state vectors from a designated set of early layers, taken at the last input-token position (before the belief-head layers). Concretely:
- Let `L_ctrl` = min layer index of any head in `H*_personal ∪ H*_attributed`.
- Probing layers = `{L_ctrl - 3, L_ctrl - 2, L_ctrl - 1}` (three layers immediately preceding the earliest belief head), capped at the model's layer count and clamped so index ≥ 0.
- Frame classifier: a 3-class MLP (hidden dim 256, ReLU, dropout 0.1) trained on the concatenated hidden states at the last token.
- Training data: `belief_core/` (n = 227 + 681 + 681 = 1589 examples), labels = source file (`reality.jsonl` → `world_knowledge`; `believe_truth.jsonl` → `personal_belief`; `follow_belief.jsonl` → `attributed_belief`). Train/val split: stratified 80/20 with seed 0.
- Optimizer: AdamW, lr 5e-4, weight decay 1e-4, batch 64, epochs 20 with early stopping on val loss.
- Selected model = the epoch with best val accuracy.

**Head amplification.** Given the frame classifier's prediction on an OOD example:
- If predicted `world_knowledge` → no amplification (identity forward pass).
- If predicted `personal_belief` → multiply the output of every head in `H*_personal` by `α_personal` before it is added to the residual stream (via forward-hook on the attention module).
- If predicted `attributed_belief` → multiply the output of every head in `H*_attributed` by `α_attributed`.

**Amplification magnitude (R5 — pinned).** `α_personal` and `α_attributed` are hyperparameters. Grid-searched on the `belief_core/` validation split (the 20% held out from classifier training — reused as controller validation):
- Grid: `α ∈ {1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` (6 values). `α = 1.0` = no amplification (baseline).
- For each of the 6×6 = 36 (α_personal, α_attributed) pairs, evaluate on the classifier-validation split, pick the pair maximizing `net_improvement = recovered - degraded` on the two belief tasks combined subject to `world_knowledge` accuracy drop ≤ 0.05 (a soft guardrail to prevent the controller from destroying factual knowledge on the val set).
- The selected `(α_personal*, α_attributed*)` is FROZEN before any evaluation on `belief_holdout/`.

**Prompt-hint baseline.** Same OOD evaluation on `belief_holdout/`, no controller, but with an explicit frame-instructing prefix in the prompt (before the existing prompt `x`):
- For `personal_belief` examples: prefix = `"Answer based on reality, ignoring what others believe. "`
- For `attributed_belief` examples: prefix = `"Answer based on what the named person believes, even if it conflicts with reality. "`
- For `world_knowledge` examples: prefix = `""` (no hint — factual questions need no frame).

The baseline is a strong upper bound: it *tells the model in-context* what frame to use, so the controller must at least match it to demonstrate that head-restricted amplification adds value on top of what prompt engineering already gives.

**OOD evaluation.** On `belief_holdout/`:
- Frame classifier accuracy `frame_acc_OOD` (3-way).
- `acc_controller_OOD(task)` for each of the 3 tasks.
- `acc_baseline_no_control_OOD(task)` (no classifier, no amplification, no prompt hint).
- `acc_prompt_hint_OOD(task)`.
- Recovered / degraded / net improvement: computed per-example vs `acc_baseline_no_control_OOD`.
- `acc_controller(world_knowledge)` on the world_knowledge portion of `belief_holdout/`.
- `PPL_controller` on the pretraining PPL sample (same cached tensor as C2), applying the controller to the PPL text (frame classifier predicts on every ~1024-token window; amplification applied if not world_knowledge).

## Refinement Log (why no LLM-review iterations)

This is the reproduction combination — methods, thresholds, dataset scope, model list, and control counts are fixed by `task.md` and re-used verbatim. The design space open to refinement reduces to five implementation gaps identified in the literature landscape:

| Gap | Landscape origin | Resolution (pinned above) |
|---|---|---|
| R1 — pythia-410m above-chance gate | Chen et al. 2025 report ToM localization on Llama; 410m may fail | Do not tune the gate; report exclusion honestly |
| R2 — "smallest head set" search discipline | task.md prescribes "smallest set" but not the search order | Greedy-add ranked by Fisher-mask overlap; greedy-remove sanity check; cap at 30 heads |
| R3 — PPL corpus scope | task.md fixes the corpus location; sample-size not stated | 1,048,576 tokens, cached to `refine-logs/artifacts/ppl_sample.pt`, reused across all ablation / control / clean runs |
| R4 — Claim-3 checkpoint schedule | task.md says "different training steps"; leaves the exact list open | Use the native Pythia schedule (154 checkpoints on disk) |
| R5 — Claim-4 amplification magnitude | task.md leaves α unspecified | Grid over 6 values per belief target, tuned on `belief_core/` val, evaluated OOD on `belief_holdout/` |

These are deterministic pins on ambiguous points in the recipe — not design changes. A full LLM-reviewer round would produce no new decisions.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| pythia-410m fails C1 above-chance for one or both belief tasks → C2 has fewer localization runs than planned | Documented (R1); report exclusion; C3 and C4 still proceed on larger models |
| Fisher signal on n=454 examples per belief signal (James+Mary only) may still be noisy for a 6-figure parameter count | Use empirical Fisher over the full 454 examples; report the mask stability under a jackknife (compute Fisher on two 227-example halves; report per-head rank-correlation) as a sanity check in C2 |
| Greedy search for smallest H* may find a local minimum rather than the global smallest set | Include the greedy-remove sanity check; if a smaller set is found, use it; if the greedy-add fails within 30 heads, honestly report "not localized" |
| 154 pythia-1b checkpoints × 9 evals per checkpoint = 1,386 eval runs → substantial GPU time | Batch checkpoints (one job = one checkpoint, does all 9 evals); the evaluations themselves are cheap (no gradients) |
| Frame classifier overfits `belief_core/` and generalizes poorly to `belief_holdout/` | Standard early-stopping on val split; report `frame_acc_OOD` explicitly so degradation is visible; report prompt-hint baseline as a diagnostic (if controller loses to prompt hint, we know why) |
| Controller degrades PPL more than the C2 4-criteria PPL bar allowed for ablation | Report `PPL_controller / PPL_clean` explicitly; if it exceeds `1.05×` on the pretraining sample, note it in the report — the C2 bar is about *ablation*, so a higher-PPL controller is allowed as long as the OOD gains justify it |

## Deliverables

- `refine-logs/FINAL_PROPOSAL.md` — this file
- `refine-logs/EXPERIMENT_PLAN.md` — per-milestone runnable plan (M1-M4)
- `refine-logs/EXPERIMENT_TRACKER.md` — planning-level run table (pending status)
- `refine-logs/REVIEW_SUMMARY.md` — refinement decisions summary (short — this is a reproduction, no iteration cycles were needed)
- `refine-logs/REFINEMENT_REPORT.md` — full reasoning for R1-R5 gap resolutions
- `refine-logs/PIPELINE_SUMMARY.md` — one-page project summary
