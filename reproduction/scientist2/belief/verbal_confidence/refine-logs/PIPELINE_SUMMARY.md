# Pipeline Summary

**Problem**: Is verbalized confidence in an LLM cached in hidden states at post-answer positions and later retrieved when the confidence token is generated, or is it freshly computed at the confidence-generation position?
**Final Method Thesis**: Test the cache hypothesis on gemma-3-27b-pt (62 layers) on TriviaQA with a Location → Causal-Intervention chain — per-position × per-layer linear-probe screen for the cache site, then residual-stream patching, attention-block, and direction steering with matched non-cache-position controls and recall-strength / log-prob-restatement / accuracy-preservation nulls.
**Final Verdict**: READY
**Date**: 2026-07-13

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report (captured behavior): `idea-stage/IDEA_REPORT.md`
- Landscape: `idea-stage/LANDSCAPE.md`
- Raw retrieval dump: `idea-stage/RESEARCH_LIT.md`

## Contribution Snapshot
- Dominant contribution: Causal-mediation account of verbalized-confidence generation — verbalized confidence is retrieved from a hidden-state cache at post-answer positions, not computed on-demand.
- Optional supporting contribution: A minimal, standard toolkit (probe + patching + attention-block + steering + null battery) that other researchers can re-use for verbalization-time mechanisms.
- Explicitly rejected complexity: SAE / dictionary learning on the cache position (Unit Interpretation); Formation Tracing across training checkpoints; Tuning & Editing for calibration; head-by-head circuit decomposition.

## Must-Prove Claims
- **Claim 1 (task.md verbatim)**: verbalized confidence is written into hidden states immediately following the answer and later retrieved from that cache when the confidence token is generated. Verified end-to-end by M1..M6 (predicates P1..P5).

## First Runs to Launch
1. M1 — data preparation + verbalization + activation cache (`scripts/m1_collect.py`, grid over 3 seeds × fixed template T0).
2. M2 — linear-probe grid over 5 post-answer positions × 12 layers (`scripts/m2_probe.py`, uses M1 cache).
3. M3 — residual-stream patching at the top M2 site(s) (`scripts/m3_patch.py`).

## Main Risks
- **Recall-strength artifact**: mitigated by M6(b) within-bin analysis.
- **Log-prob restatement**: mitigated by M6(c) log-prob-frozen patch.
- **Answer-generation collapse under intervention**: mitigated by M6(d) accuracy-preservation check.
- **10-GPU-hour budget over-run**: mitigated by M1 activation cache reuse and small initial pool sizes with scale-up on demand.

## Next Action
- Proceed to `/mechanism-skills` to bind concrete submethods for the `method_sensitive` fields in M3/M4/M5, then `/auto-experiment`.
