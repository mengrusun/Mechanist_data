# Captured-Behavior Report

**Direction**: (empty — sourced from task.md)
**Behavior-source**: given
**Mechanism**: discovery (chain: Location → Causal Intervention, from `/mechanism-explore`)
**Claim source**: task.md (faithful capture, no ideation)
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline
**Date**: 2026-07-13

## Executive Summary

The user's `task.md` posits a three-part structural claim about the geometry of LLM confidence in Llama-3.1-8B-Instruct evaluated on TriviaQA: (C1) well-calibrated accuracy is linearly accessible in hidden states, (C2) verbalized confidence is linearly accessible in hidden states, and (C3) the two directions are separate and nearly orthogonal — i.e., the model "knows" when it is likely wrong, but the generation channel that produces the verbalized number fails to surface that knowledge. All three sub-claims are carried forward as distinct verification targets, with a unified experiment plan that first locates each signal via linear probing (Location) and then verifies causal orthogonality via steering (Causal Intervention). Downstream: the plan feeds `/mechanism-skills` for routing to a concrete probe/steering submethod, then `/auto-experiment` for implementation.

## Literature Landscape

Full landscape at `idea-stage/LANDSCAPE.md`. Highlights (used only as context for baselines / datasets / metric definitions — never to alter the claims):

- **Correctness / truthfulness probes exist** (Marks & Tegmark 2023 "Geometry of Truth"; Azaria & Mitchell 2023 SAPLMA; Orgad et al. 2024; Liu et al. 2024 "Universal Truthfulness Hyperplane"; "The Confidence Manifold" 2026 — 3-8D subspace). Standard technique: logistic regression on residual-stream activations, layer sweep, causal patching.
- **Verbalized-confidence channel exists** (Kadavath 2022 "Language Models Mostly Know What They Know"; Lin 2022; Tian 2023 "Just Ask for Calibration"; Yang 2024 "On Verbalized Confidence Scores"). TriviaQA is the canonical benchmark; RLHF-tuned models tend to overclaim.
- **Nascent dissociation view** (Ji/Kossen 2025 "Calibrating Verbal Uncertainty as a Linear Feature" — moderate correlation only; Zhang 2025 "Direct Confidence Alignment" — scalar-level misalignment; HACK 2025 — two-axis taxonomy; Calibration Across Layers 2025 — later-layer distortion phase; Cognitive Dissonance 2023 — three classes of disagreement).
- **Specific gap this project fills**: no published paper directly measures the geometric angle between a correctness-probe direction and a verbalized-confidence-probe direction on matched hidden states from the same model + dataset, with per-layer trajectory and causal-orthogonality steering. Prior work reports scalar correlations or existence separately.

## Claims to Verify

### Claim 1: Linear accessibility of well-calibrated accuracy in hidden states

**Original (verbatim excerpt from task.md)**:
> Models encode well-calibrated accuracy information in a linearly accessible direction.

**Extracted statement**: On Llama-3.1-8B-Instruct evaluated on TriviaQA, there exists a linear direction (or low-dimensional linear subspace) in the residual-stream hidden state at some layer L1 such that a linear probe trained to predict per-sample correctness of the model's own free-form answer achieves AUROC substantially above chance (target: AUROC ≥ 0.70 at the best layer, cleanly above a chance baseline of 0.5) and, when its output is calibrated (e.g., isotonic / Platt scaling on a held-out split), yields well-calibrated correctness probabilities (target: ECE ≤ 0.10 on a held-out split, materially better than the model's own token-probability confidence baseline).

**Hypothesis**: H1 — Llama-3.1-8B-Instruct's residual-stream activations at one or more layers contain a linearly-decodable signal predictive of whether the model's free-form answer to a TriviaQA question is correct, and that signal is *better calibrated* than the model's token probabilities on the answer span.

**Measurable predicate**: `AUROC(linear_probe_on_correctness, held_out_split) ≥ 0.70` at some layer, AND `ECE(calibrated_probe_output, held_out_split) ≤ 0.10` AND `ECE(calibrated_probe_output) < ECE(token_probability_confidence)`.

**Expected direction**: up (probe AUROC well above chance; ECE lower than token-probability baseline)

**Resources (preferred, not strict — see below)**: model: Llama-3.1-8B-Instruct; dataset: TriviaQA; used_n: to be resolved in Phase 4.5 based on GPU budget (≥ ~2k held-out items for stable AUROC + ECE; ~5-10k probe training items typical in the field).

**Status**: pending verification

**Notes**: Split from paragraph 1 of task.md's "Claim" section. "Well-calibrated" is the emphatic modifier — the calibration test is what distinguishes this claim from generic truthfulness-probing prior work. Standard baselines to compare against: model's own token probabilities on the answer span, P(True) self-elicitation prompt (Kadavath 2022), and a random-direction probe (chance null).

---

### Claim 2: Linear accessibility of verbalized confidence in hidden states

**Original (verbatim excerpt from task.md)**:
> Models encode verbalized confidence in a linearly accessible direction.

**Extracted statement**: On Llama-3.1-8B-Instruct evaluated on TriviaQA, there exists a linear direction (or low-dimensional linear subspace) in the residual-stream hidden state at some layer L2 such that a linear probe trained to predict the numeric confidence value the model *verbalizes* (the score it outputs when asked "How confident are you? Give a probability from 0 to 100.") achieves substantially above-chance predictive performance (target: probe-predicted verbalized-confidence correlates with the actual verbalized number at Spearman ρ ≥ 0.5 on a held-out split, and AUROC ≥ 0.70 when binarized at the population median), before the verbalization tokens are emitted. The probe direction is a distinct vector from the one identified in Claim 1.

**Hypothesis**: H2 — Llama-3.1-8B-Instruct's residual-stream activations at one or more layers linearly encode the confidence value the model will subsequently verbalize, in a way that is decodable prior to that verbalization.

**Measurable predicate**: `Spearman_rho(probe_output, verbalized_confidence_value, held_out_split) ≥ 0.5` at some layer, AND `AUROC(probe_output_binarized_at_median, held_out_split) ≥ 0.70`.

**Expected direction**: up (probe substantially predicts verbalized confidence)

**Resources (preferred, not strict — see below)**: model: Llama-3.1-8B-Instruct; dataset: TriviaQA; used_n: to be resolved in Phase 4.5 (same scale as C1). Must additionally elicit verbalized confidence per sample — standard prompt (e.g., Tian 2023 style: "Give a probability from 0 to 100 that your answer is correct.").

**Status**: pending verification

**Notes**: Split from paragraph 2 of task.md's "Claim" section. "Verbalized confidence" is operationalized as the numeric score the model emits when explicitly asked; the probe is trained on the *pre-emission* hidden state so it is a genuine internal-representation probe, not a lookup of the emitted token. Baseline: a probe trained on the last-token-embedding of the *emitted* confidence value should trivially achieve near-perfect readout — the meaningful test is the *pre-emission* probe.

---

### Claim 3: The two directions are separate and nearly orthogonal

**Original (verbatim excerpt from task.md)**:
> But well-calibrated accuracy information and verbalized confidence occupy separate, nearly orthogonal directions. That means the model "knows" when it is likely wrong, but the generation process fails to surface this signal.

**Extracted statement**: The direction identified in Claim 1 (v_c, correctness/calibration probe direction at its best layer) and the direction identified in Claim 2 (v_v, verbalized-confidence probe direction at its best layer) are **geometrically nearly orthogonal** — specifically, `|cos(v_c, v_v)| ≤ 0.3` at matched layers (a moderate threshold that materially separates "same direction" `|cos| ≈ 1` from "orthogonal" `|cos| ≈ 0`) — AND are **causally separable**: intervening (steering) along v_c leaves the verbalized-confidence probe readout unchanged relative to a matched random-direction control, and intervening along v_v leaves the correctness probe readout unchanged relative to the same control. The near-orthogonality is not attributable to either probe being near-random (both must independently exceed the C1 and C2 AUROC thresholds first). Additionally, when both probes achieve high accuracy but their readouts *disagree* (the correctness probe says "unlikely correct" while the verbalized probe / actual verbalization says "highly confident"), the verbalization is more likely to be wrong than when they agree — evidence for the "knows-but-fails-to-surface" interpretation.

**Hypothesis**: H3 — the calibration channel and the verbalization channel occupy separate, nearly-orthogonal linear subspaces of Llama-3.1-8B-Instruct's activation space, so the RLHF-trained model's overconfident verbalization is a *readout failure*, not a *knowledge deficit*.

**Measurable predicate**:
- (a) **Geometric**: `|cos_similarity(v_c, v_v)| ≤ 0.3` at layers where both probes achieve their reported best AUROC (with C1 and C2 preconditions independently satisfied).
- (b) **Causal**: `|Δ(verbalized_probe_readout | steer along v_c)| ≤ |Δ(verbalized_probe_readout | steer along random_direction of matched norm)| × 1.5` (steering along v_c does not preferentially move v_v's readout beyond a random-direction baseline), AND the same when the roles of v_c and v_v are swapped.
- (c) **Dissociation-when-disagree**: `P(model's verbalized answer is wrong | correctness_probe(x) < 0.5 AND verbalized_confidence(x) > 80) > P(model's verbalized answer is wrong | correctness_probe(x) < 0.5 AND verbalized_confidence(x) < 50)` — when the internal probe says "unlikely correct" but verbalization says "very confident", accuracy is lower than when both agree on low confidence.

**Expected direction**: geometric: |cos| toward 0; causal: cross-direction effect suppressed; dissociation-when-disagree: yes (probe wins over verbalization as a correctness predictor).

**Resources (preferred, not strict — see below)**: model: Llama-3.1-8B-Instruct; dataset: TriviaQA; used_n: same held-out split(s) as C1 and C2, plus a steering-evaluation subset (typically ~500-2000 items sufficient given paired-sample design). Steering α sweep across at least 3 non-zero magnitudes each sign (±0.5σ, ±1σ, ±2σ of the direction's activation-magnitude distribution) plus α=0 baseline.

**Status**: pending verification (contingent on C1 and C2 first passing)

**Notes**: Split from paragraph 3 of task.md's "Claim" section. This is the load-bearing claim — the reproduction is only interesting if C3 lands, since C1 and C2 individually restate prior work (Marks-Tegmark 2023 for C1, Kossen 2025 for C2). The `|cos| ≤ 0.3` threshold is chosen conservatively (not the tighter `|cos| ≤ 0.1` "true orthogonality") to accommodate probe noise while still cleanly distinguishing "separate" from "same direction". The dissociation-when-disagree sub-clause operationalizes the "the model knows when it is likely wrong, but the generation process fails to surface this signal" quote directly.

---

## Resources (preferred; NOT strict-fidelity, since this is `given` + `discovery` — not the `given` + `given` reproduction combo)

- **Model**: Llama-3.1-8B-Instruct (from task.md — mandatory for the main experiment; verify-stage swaps within {Llama-3.1-8B (base), Qwen2.5-7B, Qwen2.5-7B-Instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1} allowed).
- **Dataset**: TriviaQA (from task.md — mandatory for the main experiment; verify-stage swaps within {MATH, MMLU, TruthfulQA} allowed).
- **used_n (main experiment)**: to be resolved in Phase 4.5 based on the 10h total GPU budget. Under a cost-aware plan (this is not `resource_fidelity: strict`), aim for the largest sample size the science warrants within budget — the field's default of ~5-10k probe-training items and ~2k held-out items comfortably fits within 10h on Llama-3.1-8B, so planning at full field-standard scale is defensible.
- **GPU pin (HARD CONSTRAINT)**: only GPU ids ∈ {1, 2, 3, 5, 6}.
- **Filesystem pin (HARD CONSTRAINT)**: writes only under `/data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief`, `/data/zhenqian/data`, `/data/zhenqian/models`.
- **Conda env**: `belief`. Use vllm for batched generation / logits collection where appropriate.

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering all three claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies; opens directly with mechanism milestones — no M0, since `BEHAVIOR_SOURCE=given`)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps

- [ ] `/mechanism-skills` to route the testing approach to a concrete linear-probe + steering submethod (Workflow 1.25)
- [ ] `/auto-experiment` to implement and run the verification suite (Workflow 1.5)
- [ ] `/auto-verify` to stress-test each verified claim under method/dataset/model swaps (Workflow 1.75) — candidate pools already scoped by task.md's HARD CONSTRAINTS
- [ ] `/auto-iteration-loop` to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain
