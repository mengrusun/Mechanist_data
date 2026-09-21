# Pipeline Summary

**Problem**: Disentangling language-specific and language-agnostic subspaces in multilingual LLM hidden representations, and testing whether inference-time suppression of the language-specific subspace improves multilingual reasoning on MGSM without hurting output-language fidelity, at a fraction of the compute of multilingual post-training.
**Final Method Thesis**: A single unified testing pipeline — fit V_lang from a small multilingual probe set via language-mean-difference SVD, intervene at inference on Qwen-3-4B-Thinking's residual stream at non-upper layers with h ← h + α · (Π_lang · h) sweeping α across a signed range, and compare against a matched-compute multilingual LoRA-SFT baseline — jointly verifies the four given task.md claims via the ladder-of-evidence chain Location → Causal Intervention → Tuning & Editing.
**Final Verdict**: READY.
**Date**: 2026-07-14

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report (captured claims): `idea-stage/IDEA_REPORT.md`
- Landscape: `idea-stage/LANDSCAPE.md`

## Contribution Snapshot
- Dominant contribution (as-tested): a small-probe low-rank inference-time linear operator (null-space projection onto the language-agnostic subspace at Qwen-3-4B-Thinking's non-upper layers) that simultaneously raises MGSM accuracy across 11 target languages, preserves GlotLID output-language fidelity, exhibits a signed monotone dose-response, and matches multilingual SFT at ≤ 10 % of the SFT compute.
- Optional supporting contribution: RL-baseline compute comparison in M4b (budget-permitting).
- Explicitly rejected complexity: SAE-feature-level intervention, LAPE neuron-level editing, attention-head circuit surgery, trainable soft-prompt / adapter bridge, expansion beyond Qwen-3-4B-Thinking + MGSM at the main-experiment stage, and an M0 phenomenon-validation gate (BEHAVIOR_SOURCE = given).

## Must-Prove Claims (verbatim from task.md, refined into measurable predicates)
- Claim 1: hidden states decompose into a language-specific + language-agnostic subspace identifiable from a small multilingual probe set → M1 predicate (held-out language classifier ≥ 0.90, complement ~ chance, principal-angle cosine ≤ 0.20 at n_probe ≤ 250).
- Claim 2: inference-time subspace suppression consistently improves MGSM accuracy; output-language fidelity acceptable when upper layers intact → M2 predicate (≥ +3 pp mean gain, ≥ 8/11 languages non-regressive, fidelity drop ≤ 5 pp at k_top ≥ 4).
- Claim 3: language-specific activation strength negatively correlated with reasoning accuracy (signed dose-response) → M3 predicate (Spearman(α, A) < 0 significant, A(−1) > A(0) > A(+1), ≥ 8/11 languages sign-consistent, random-control null).
- Claim 4: training-free intervention matches or exceeds multilingual post-training at fraction of compute → M4 predicate (mean_L A_edit ≥ mean_L A_SFT − ε_match with ε_match = 0 target, compute ratio κ ≤ 0.10).

## First Runs to Launch
1. M1 batch — fit V_lang across the 5 × 6 × 3 × 3 = 270-cell grid (cheap, ~0.5 GPU-h total).
2. M2 main sweep — 36 MGSM 11-language evaluations across k_top and layer_group.
3. M4a data prep + LoRA-SFT training (kicks off in parallel to M2 on a separate GPU allocation).

## Main Risks
- Layer-scope disagreement between LAPE (top+bottom) and Wendler/Zhao NeurIPS (mid) on the reasoning-tuned model:  M2's layer_group × k_top sweep tests this directly.
- 10-hour GPU budget vs full-parameter SFT infeasibility: mitigated by defaulting to LoRA-SFT with an extrapolated-full-SFT annotation, not silent skipping.
- Probe-set-size stability below n_probe ≈ 100: M1's n_probe grid quantifies this and reports the smallest n_probe at which the Claim-1 predicate passes.

## Next Action
- Proceed to `/mechanism-skills` routing (Workflow 1.25) — `/auto-experiment` Phase 1.5 will re-bind the `method_sensitive` fields (`sites`, `n_pairs`, `metric`, `gpu_hours`) for M1/M2/M3 based on the routed submethod (SVD-null-space-projection is the default; contrast-pair CAA and LDA are alternatives).
- Then `/auto-experiment` to launch (Workflow 1.5), followed by `/auto-verify` (Workflow 1.75) and `/auto-iteration-loop` (Workflow 2).
