# Pipeline Summary — Reproduction of "Sensitivity Meets Sparsity" on Pythia Belief Circuits

**Problem**: Localize, developmentally trace, and dynamically control the attention-head circuits implementing personal-belief and attributed-belief in pretrained Pythia language models — a faithful reproduction of Chen et al. 2025 ("Sensitivity Meets Sparsity", arXiv:2504.04238 / npj AI 2025).

**Final Method Thesis**: Fisher-information-guided sparse attention-head localization (top-0.1% target AND-NOT top-1% control mask) + causal zero-ablation with 20-random-head + 20-random-mask baselines and verbatim 4-criteria acceptance (drop ≥ 0.30, > mean+2σ vs random-head, off-target ≤ 0.10, PPL ≤ 1.05×) → pythia-1b intermediate-checkpoint sweep to trace developmental trajectories → head-restricted amplification controller gated by an early-layer frame classifier, evaluated OOD against a prompt-hint baseline.

**Final Verdict**: READY

**Date**: 2026-07-22

## Final Deliverables

- Idea report (behaviour/claim capture): `idea-stage/IDEA_REPORT.md`
- Literature landscape (grounding): `idea-stage/LANDSCAPE.md`
- Raw retrieval (audit): `idea-stage/RESEARCH_LIT.md`
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report (R1-R5 gap resolutions): `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot

- **Dominant contribution**: Faithful reproduction of Chen et al.'s Fisher-mask + zero-ablation localization pipeline on a different model family (Pythia rather than Llama 3), applied to a finer dissociation than the original ToM formulation (personal-belief vs attributed-belief), and extended into the developmental (Claim 3) and controllable (Claim 4) regimes.
- **Optional supporting contribution**: A deterministic recipe for re-running the same protocol (R1-R5 gaps pinned) — turns the paper description into a reproducibility harness.
- **Explicitly rejected complexity**: no alternative attribution methods (activation patching, SAEs, integrated gradients); no alternative interventions (mean-ablation, resampling); no altered thresholds; no altered mask construction; no altered control counts; no dataset subsetting except the explicit person restriction; no smaller model substitution; no M0 phenomenon-validation gate (BEHAVIOR_SOURCE=given).

## Must-Prove Claims

1. **C1 — Scale-Dependent Emergence**: Distinct scaling curves for personal_belief and attributed_belief across pythia-{410m, 1b, 2.8b}.
2. **C2 — Belief Heads Localization**: Distinct, causally separable attention-head sets H*_personal and H*_attributed for each above-chance model, satisfying all 4 verbatim criteria against 20 random-head + 20 random-mask controls.
3. **C3 — Formation Window**: Distinct developmental (behavioural + causal) trajectories for the two belief abilities on pythia-1b intermediate checkpoints under predefined emergence criteria.
4. **C4 — Dynamic Controllability**: A probe-and-amplify controller (frame classifier + head amplification) produces positive net OOD improvement, preserves world_knowledge accuracy and PPL, and is comparable to a prompt-hint baseline.

## First Runs to Launch (Wave 1 — no dependencies)

1. `m1_pythia-2.8b_personal_belief` — behavioural accuracy of pythia-2.8b on personal_belief (largest model, most likely to clear above-chance gate)
2. `m1_pythia-2.8b_attributed_belief` — behavioural accuracy of pythia-2.8b on attributed_belief (the harder task)
3. `m1_pythia-1b_personal_belief` — behavioural accuracy of pythia-1b on personal_belief (needed for Claim-3 gating)

All 9 M1 runs are independent and can launch in parallel on available GPUs. See `EXPERIMENT_TRACKER.md`.

## Main Risks

- **Risk 1**: pythia-410m fails the C1 above-chance gate on `attributed_belief` (very plausible — false-belief tasks are hard for small LMs). **Mitigation**: report exclusion honestly; C2/C3/C4 still proceed on larger models.
- **Risk 2**: Fisher signal on n=454 examples (James+Mary subset) is noisy at parameter granularity. **Mitigation**: jackknife stability check (Spearman ρ on per-head candidate scores across 227-example halves) reported in C2.
- **Risk 3**: Greedy H* search misses global minimum. **Mitigation**: greedy-remove sanity check; cap 30 heads; report `not_localized` honestly if capped.
- **Risk 4**: 154-checkpoint pythia-1b sweep (M3) costs ~47h GPU. **Mitigation**: acceptable within reproduction budget; each checkpoint job is independent and parallelizable.
- **Risk 5**: Controller in Claim 4 loses to the prompt-hint baseline. **Mitigation**: this is a *scientific* outcome, not a bug — report side-by-side and note that head-restricted amplification does not add value on top of in-context prompting for this setting; the negative result is itself informative.

## Cost estimate

Grand total GPU-hours estimate: **~140-350h** (best case: only pythia-2.8b localizes one target → ~90h; typical: pythia-2.8b + pythia-1b localize → ~180h; worst case: all 3 models localize both targets → ~350h).

## Next Action

- Proceed to `/auto-experiment` (Workflow 1.5) to implement the scripts referenced in `EXPERIMENT_PLAN.md` and start Wave 1 (M1 grid).
- Because `resource_fidelity: strict` and `chosen_mechanism` is per-claim in the top metadata of `EXPERIMENT_PLAN.md`, `/auto-experiment` Phase 1.5 will commit each milestone's mechanism directly (Mode B) with no routing.
- After the pipeline completes: `/auto-verify` to stress-test the passed claims, then `/auto-iteration-loop` for critical review.
