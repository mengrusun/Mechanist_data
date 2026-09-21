# Final Proposal — Reproduction of Five SAE-on-ESM-2 Interpretability Claims

**Date**: 2026-07-15
**Behavior-source**: given
**Mechanism**: discovery
**Verdict**: READY

## Top metadata (machine markers — English, verbatim)

```yaml
resource_fidelity: not-strict   # combo is given+discovery, so cost-aware; but 10-h budget covers full-spec ESM-2-650M + Swiss-Prot + six-SAE evaluation, so no downscaling is planned
mechanism_strategy:
  directions: [Unit Interpretation, Decision Auditing, Causal Intervention]
  rejected:
    - Location — pre-committed by task.md (fixed six-layer set + fixed pretrained SAEs); nothing to locate at claim time.
    - Tuning & Editing — user constraint (no ESM-2 tuning); inference-time steering is captured under Causal Intervention.
    - Formation Tracing — user constraint (no re-training, no pretraining trace); SAE checkpoints are fixed resources.
  note: Every claim in task.md rests on decoding SAE features into biological concepts (Unit Interpretation), then auditing coverage against Swiss-Prot including novel-concept surfacing (Decision Auditing), then testing whether clamping a labeled feature causally steers ESM-2 generation (Causal Intervention).
```

## Problem Anchor (frozen)

Verify — faithfully and with full specificity controls — the five claims stated in `task.md`:

- **C1** (feature count): SAE trained on ESM-2-650M residual-stream activations surfaces up to ~2,548 interpretable latent features per layer, orders of magnitude more than raw neurons at the same layer.
- **C2** (concept alignment): SAE features cover up to ~143 distinct Swiss-Prot biological concepts (binding sites, active sites, sequence motifs, structural / functional domains, PTM sites); raw ESM-2 neurons cover up to ~46 concepts, of which only ~15 are cleanly recovered — *same alignment protocol on both arms*.
- **C3** (superposition): the C2 gap is direct evidence PLMs encode biological concepts in superposition rather than in single units.
- **C4** (novel-concept discovery): a subset of SAE features Swiss-Prot-unaligned nevertheless receives coherent natural-language labels from an external LLM auto-interpreter over top-activating protein contexts, evidencing concepts absent from the annotation dictionary.
- **C5** (downstream utility): the SAE dictionary is practically useful — (5a) it supports filling missing Swiss-Prot annotations; (5b) clamping labeled SAE features steers ESM-2 sequence generation toward a target biological property.

The problem anchor **does not move** — none of the five claims is re-stated, sharpened, or narrowed. The verification method is refined below.

## Method thesis (one sentence)

Reproduce all five claims under a **single unified evaluation harness** that (a) loads the six pretrained SAE checkpoints for ESM-2-650M layers {1,9,18,24,30,33}, (b) computes per-(unit, concept) residue-level F1 alignment under an **identical protocol across SAE-features and raw-neurons arms**, (c) adds three specificity control arms (random-orthogonal, PCA, shuffled-SAE) to license the superposition interpretation of the observed gap, (d) runs the external LLM (`gpt-5.4` at `dmxapi.cn`) as an auto-interpreter over top-activating protein residue-window contexts with a mandatory low-activation baseline and random-feature control, and (e) implements SAE-feature-clamp steering of ESM-2 masked-token generation with dose-response, random-clamp control, mean-activation-addition baseline, and a preserved-pseudo-perplexity plausibility band.

## Dominant contribution (of this reproduction)

A **protocol-pinned faithful reproduction** in which every one of the five reported numbers is defended by matched-capacity or matched-condition controls, so a passing result cannot be explained by metric artifact, layer cherry-picking, auto-interpreter hallucination, or steering off-target drift.

## Explicitly rejected complexity

- **SAE retraining** — the pretrained checkpoints are the fixed resource; retraining is out-of-scope. The reported numbers are numbers *for these SAEs*; retraining would change what is being tested.
- **ESM-2 tuning / fine-tuning** — explicit user constraint; steering is inference-time clamping only.
- **Formation tracing over ESM-2 pretraining** — out-of-scope (constraint + no checkpoints available).
- **Scale-generalization swaps to ESM-2-8M / 35M / 150M** — deferred to `/auto-verify` per `task.md`'s verify-stage list; not in this plan's main runs.
- **Novelty search / competitive claim ranking** — `BEHAVIOR_SOURCE=given` skips ideation and novelty check.

## Key claims and must-run experiments

Each claim maps to one or more milestones in `refine-logs/EXPERIMENT_PLAN.md`:

| Claim | Milestone(s) | What passes it |
|---|---|---|
| C1 | M1 | Per-layer interpretable-feature count approaches ~2,548 at some layer; SAE/neuron ratio ≥ 10× at that layer under identical interpretability gate. |
| C2 | M2 | Union-over-layers SAE covers ≥100 concepts (target ~143), neurons ≥30 (target ~46) / clean ≥10 (target ~15), same F1 protocol. |
| C3 | M3 | SAE coverage > PCA ≈ random-rotation ≈ neurons > shuffled-SAE, by margin Δ (pinned in Phase 4.5 of plan). |
| C4 | M4 | ≥10% of Swiss-Prot-unaligned features are novel-concept-coherent (LLM auto-interp score > baseline + synonym-check passes + random-feature control fails). |
| C5a | M5 | Mean-per-concept PR-AUC(SAE-linear-probe) > Mean-per-concept PR-AUC(neuron-linear-probe) with paired-test p < 0.05. |
| C5b | M6 | For ≥1 concrete auto-labeled feature, SAE-clamp yield > all baselines with monotone dose-response and preserved plausibility. |

## Remaining risks (not blockers)

- **Metric-choice sensitivity** (Gap G1 in LANDSCAPE): the exact τ_F1 and quantile q_top can shift the ~2,548 / ~143 / ~46 numbers materially. Mitigation: report a **sensitivity sweep** for τ_F1 ∈ {0.3, 0.5, 0.7} and q_top ∈ {0.95, 0.99} in M2, with the "target" numbers reported at the primary setting; call out whether the target numbers hold at the primary setting alone or across the sweep.
- **LLM auto-interpretability drift** (Gap G2): the endpoint (`gpt-5.4` at `dmxapi.cn`) may return non-standard responses. Mitigation: cache all LLM calls, prompt with a rigid JSON-response schema, retry on parse failure, and record raw responses for audit.
- **Steering off-target drift** (Gap G3): clamping may boost target-property yield while destroying plausibility. Mitigation: mandatory pseudo-perplexity gate in M6; report only steering runs within the plausibility band.
- **Layer-choice cherry-picking** (Gap G4): "up to ~2,548" and "up to ~143" are max-over-layer / union-over-layers. Mitigation: report per-layer numbers in the artifact even when the headline uses the max/union.
- **Superposition-vs-compression** (Gap G5): the SAE-vs-neuron gap has alternative explanations (basis effects, concept-granularity). Mitigation: the three control arms in M3 are exactly the specificity checks that separate these hypotheses.

## Compute budget accounting (10 GPU-hours)

| Milestone | GPU-hours (estimate) | Notes |
|---|---|---|
| M1: Feature-count harness | 1.5h | Six SAEs × per-feature stats over Swiss-Prot test split |
| M2: Concept-alignment harness | 2.0h | Same six SAEs, per-(unit, concept) F1 sweep; also computes neuron arm |
| M3: Superposition specificity controls | 1.0h | PCA fit + rotation + shuffled-SAE; reuses M2's F1 code |
| M4: Novel-concept LLM auto-interp | 1.5h | LLM calls dominate wall-clock, not GPU; GPU for ESM-2 activation extraction on UniRef sample |
| M5: Annotation-filling linear probes | 1.0h | Small linear classifiers on features vs neurons |
| M6: SAE-feature-clamp steering | 3.0h | Generation-heavy; dose × features × baselines |
| **Total** | **10.0h** | Fits budget; buffer via early-stop if a milestone succeeds under target sample size |

## Frontier primitive necessity

- **LLM auto-interpreter** (external `gpt-5.4`): NECESSARY for C4 (the claim is *about* LLM auto-interpretation) and for auto-labeling steering-target features in C5b. Cannot substitute human interp per HARD CONSTRAINT.
- **SAE feature clamping**: NECESSARY for C5b (the claim is about steering via SAE features). Cannot substitute simpler activation-addition steering (which is included as the baseline).

## Downstream handoff

- `/mechanism-skills` will route the three mechanism directions to concrete submethods (SAE dictionary decomposition is fixed; auto-interp submethod choice; steering submethod choice — feature-clamp vs. activation-addition).
- `/auto-experiment` will implement + run using the plan and `EXPERIMENT_TRACKER.md`.
- `/auto-verify` will stress-test the passed claims by swapping ESM-2-650M → ESM-2-8M / 35M / 150M using the verify-stage SAEs.
- `/auto-iteration-loop` if any claim comes back INCONCLUSIVE.
