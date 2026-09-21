# Review Summary

**Problem**: Determine whether Llama-3.1-8B-Instruct's near-100% verbalized-confidence overclaim on TriviaQA is a knowledge deficit or a readout failure, by measuring the geometric angle + causal cross-coupling between the internal gold-correctness direction and the internal verbalized-confidence direction on matched hidden states.
**Initial Approach**: Two paired linear probes on residual-stream activations + cosine similarity + activation steering. Given-behavior mode: three claims C1/C2/C3 fixed by task.md.
**Date**: 2026-07-13
**Rounds**: 3 / 5
**Final Score**: 9.1 / 10
**Final Verdict**: READY

## Problem Anchor (verbatim across all rounds)

- **Bottom-line problem**: RLHF-tuned LLMs verbalize confidence near 100% regardless of correctness; prior work has separately shown internal linear correctness signals exist and verbalized confidence is a linearly controllable signal, but no paper directly measures the geometric relationship between the two.
- **Must-solve bottleneck**: matched-pair characterization of C1, C2, C3 on Llama-3.1-8B-Instruct + TriviaQA with per-layer trajectory, canonical hook, and causal-orthogonality steering.
- **Non-goals**: no new probe algorithm, no SAE, no formation tracing, no fine-tuning, no cross-dataset angle generalization.
- **Constraints**: 10h GPU budget, GPUs {1,2,3,5,6}, filesystem-restricted, Llama-3.1-8B-Instruct + TriviaQA mandatory, verify swaps bounded.
- **Success condition**: matched (L_c*, L_v*) with C1 AUROC≥0.70 / ECE≤0.10 / probe-ECE < token-ECE; C2 primary threshold + binarized AUROC≥0.70; C3 |cos|≤0.3 with C1/C2 gates passed + cross-steering null on internal readout + secondary emitted-output corroboration + dissociation-when-disagree.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|-------------------------|------------------------------------------|---------|----------------|
| 1     | CRITICAL Feasibility: compute-heavy plan (36k steering passes, 8 ablations). CRITICAL Validation Focus: C3 identification (different forward contexts). IMPORTANT Venue Readiness: mechanism-light framing. | Trimmed steering to 3α × 500 samples = 4500 passes. Pinned canonical hook `outputs.hidden_states[L]`. Added cross-steering emitted-output metric. Collapsed C2 to continuous linear regression (primary Spearman ρ). Pruned must-run ablations from 8 → 6. | Partial | Framing overreach ("readout failure" too strong); C2 signal-strength risk; two-context caveat not explicit |
| 2     | BLOCKING framing overreach, C2 low-variance contingency, two-context caveat. Simplifications: single primary reporting layer, de-emphasize binary/continuous C2 seam, keep C3c secondary. | Added Scope-of-Mechanism-Claim paragraph. Added Stage 1.5 variance diagnostic with pre-registered continuous/ordinal decision rule. Added explicit Scope Statement + single-pass robustness variant. Collapsed to single primary reporting layer L\*. Added absolute-effect companion criterion for C3b. Report full effect-size distributions with bootstrap CIs. | Yes | Analysis-plan polish only |
| 3     | (No blocking). Important polish: bootstrap protocol clarification, L\* selection guardrail, C3b asymmetry, parse-failure bias diagnostic, single-pass variant precommit. | Adopted retrain-on-bootstrap for probe-fit uncertainty. Added neighborhood-robustness narrative guardrail. Internal-readout privileged as primary causal test; emitted-output as secondary corroboration. Added parseable-vs-unparseable diagnostic. Pre-registered single-pass variant interpretation. Adopted concrete ordinal thresholds. | Yes | READY |

## Overall Evolution

- **How the method became more concrete**: canonical hook pinned to `outputs.hidden_states[L]`; token position pinned to last-input-token; probes explicitly named (probe_c_binary, probe_v_primary path-selected, probe_v_binary as v_v source); steering site + magnitude scale (σ_L\*) + grid + safety cap all specified.
- **How the dominant contribution became more focused**: from "measure the geometry between correctness and verbalization directions" to "single-model, single-dataset, canonical-hook, matched-pair characterization with pre-registered contingencies, neighborhood robustness, and two elicitation-context designs."
- **How unnecessary complexity was removed**: primary reporting collapsed to ONE layer L\* (not "matched" + "nearest-shared"); C2 seam de-emphasized (continuous OR ordinal path); dissociation-when-disagree explicitly labeled secondary; four ablations moved to optional appendix.
- **How modern statistical rigor improved**: retrain-on-bootstrap CIs primary; full effect-size distributions primary; threshold-pass table is derived summary.
- **How drift was avoided**: no drift ever flagged. All three fixed claims (C1, C2, C3) unchanged across rounds; refinements always at method / reporting level.

## Final Status

- **Anchor status**: preserved (verbatim across all rounds; no drift).
- **Focus status**: tight — one dominant contribution, one supporting contribution, explicit non-contributions.
- **Modernity status**: appropriately frontier-aware — canonical residual-stream hook + linear probes + activation steering + retrain-on-bootstrap statistics; deliberately declined SAE / formation tracing / fine-tuning as out-of-scope for the three fixed claims.
- **Strongest parts of final method**: (a) canonical hook + matched-representation-space framing eliminates the biggest reviewer objection to prior comparable work; (b) Stage 1.5 pre-registered contingency prevents catastrophic C2 failure on low-variance confidence data; (c) single-pass robustness variant preempts the two-context objection by construction; (d) internal-readout-primary / emitted-output-secondary steering asymmetry aligned with what the evidence can actually support.
- **Remaining weaknesses**: none blocking. Interpretation must remain bounded — the paper cannot claim "readout failure" resolved, only "geometric weak alignment + limited causal cross-coupling in this specific setting."
