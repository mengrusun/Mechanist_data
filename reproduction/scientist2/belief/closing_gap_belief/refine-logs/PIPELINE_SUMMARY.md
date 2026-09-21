# Pipeline Summary

**Problem**: Measure the geometric alignment + causal cross-coupling between the internal gold-correctness direction (v_c) and the internal verbalized-confidence direction (v_v) in Llama-3.1-8B-Instruct on TriviaQA, testing three fixed claims (C1, C2, C3) captured verbatim from `task.md`.
**Final Method Thesis**: Two paired linear probes on residual-stream activations at a canonical hook (`outputs.hidden_states[L]`, last-input-token) — one for gold correctness, one for verbalized confidence — measured for per-layer AUROC/Spearman/ECE with retrain-on-bootstrap CIs, cosine at a single primary reporting layer L\* with neighborhood-robustness (L\*±2), and causal cross-coupling under matched-magnitude activation steering with internal readout primary + emitted output secondary corroboration.
**Final Verdict**: READY (score 9.1 / 10 after 3 refine rounds)
**Date**: 2026-07-13

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Score history: `refine-logs/score-history.md`

## Contribution Snapshot
- **Dominant contribution**: matched-pair, canonical-hook, same-representation-space characterization of the angle + causal cross-coupling between gold-correctness and verbalized-confidence linear directions in one model on one dataset, with pre-registered contingencies (Stage-1.5 variance diagnostic) and neighborhood-robustness reporting.
- **Optional supporting contribution**: dissociation-when-disagree accuracy analysis — when the two channels disagree, verbalization loses reliability.
- **Explicitly rejected complexity**: SAE decomposition, formation tracing (training dynamics), fine-tuning / DPO / RLHF interventions, cross-dataset angle generalization (would confound C3 per Kim et al. 2025), circuit-level analysis, new probing algorithm, new steering algorithm.

## Must-Prove Claims
- **C1**: A linear direction in Llama-3.1-8B-Instruct's residual-stream hidden states at a canonical hook location linearly predicts gold correctness on TriviaQA. AUROC(L_c\*) ≥ 0.70 + ECE ≤ 0.10 post-isotonic + ECE(probe) < ECE(token_prob).
- **C2**: A separate linear direction linearly predicts pre-emission verbalized confidence on the same model+dataset. Stage-1.5 selects continuous (Spearman ρ ≥ 0.5) or ordinal (top-1 accuracy ≥ 0.55 + macro-F1 ≥ 0.4) path.
- **C3a**: |cos(v_c^L\*, v_v^L\*)| ≤ 0.3 at the primary layer L\* with neighborhood-robustness (mean across L\*±2 ≤ 0.3), gated on C1 & C2 passing.
- **C3b**: Cross-steering criterion (ratio OR absolute-effect) satisfied on internal readout (primary) + emitted output (secondary corroboration).
- **C3c**: Dissociation-when-disagree — accuracy(low probe, high verbal) < accuracy(low probe, low verbal), McNemar p < 0.05 (supporting analysis).

## First Runs to Launch
1. **R001–R002 (M1)**: Forward pass 1 answer generation + hidden-state extraction on 10k TriviaQA (vllm-batched generate + HF hook extract pass). ~1.0 h GPU.
2. **R003–R004 (M1)**: Forward pass 2 verbalized-confidence generation + hidden-state extraction, using pass-1 answers. ~1.0 h GPU.
3. **R006 → R005 (M1.5 → M2)**: Stage-1.5 variance diagnostic (analytic, minutes) → probe training with the selected continuous/ordinal path (CPU-side, ~30 min). This unblocks R007–R008 (cosine + per-layer trajectory) and provides v_c^L\*, v_v^L\* for steering (R009).

## Main Risks
- **Risk 1**: Verbalized confidence values may cluster near 100 → C2 signal too weak for continuous probe.
  - **Mitigation**: Stage 1.5 pre-registered decision rule auto-switches to ordinal probe on 4 bins; v_v extracted from binary probe at 30th percentile (guarantees minority-class ≥30%).
- **Risk 2**: The chosen primary layer L\* is an isolated lucky layer.
  - **Mitigation**: neighborhood-robustness guardrail — C3a is considered supported only if the mean |cos| across L\*±2 layers ≤ 0.3.
- **Risk 3**: Cross-steering ratio criterion unstable when random-direction Δ is near-zero.
  - **Mitigation**: absolute-effect companion criterion (|Δ_v_other| ≤ 0.5σ_probe_readout).
- **Risk 4**: Two-context extraction concern — v_c and v_v come from different textual contexts.
  - **Mitigation**: single-pass unified-prompt robustness variant (Block B7) with pre-registered interpretation rule.
- **Risk 5**: Perplexity blow-up under steering.
  - **Mitigation**: safety cap at 3× baseline; halve α and re-run.

## Next Action
- Proceed to `/mechanism-skills` (Workflow 1.25) for route-to-family binding (Location → linear-probe family; Causal Intervention → activation-steering family), then `/auto-experiment` (Workflow 1.5) for implementation + deployment.

## Compute budget summary
- Main experiment: ~5.2 h GPU
- Verify swaps (candidate pool suggested; `/auto-verify` picks): ~2 h GPU
- Buffer / iteration: ~2.8 h GPU
- **Total under 10 h HARD budget**: yes, ~10 h ceiling with buffer.
