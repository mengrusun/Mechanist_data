# Pipeline Summary

**Problem**: Verify three given claims from task.md about emotion-specific global circuits in Llama-3.2-3B on SEV — Location, Causal Intervention + Stability, circuit-based Applied Control beating prompting + single-direction steering.
**Final Method Thesis**: A unified matched-budget testing protocol runs the Location → Causal → Applied ladder on one fit of `C_e`, with a judge-free internal Stage-B ranker (target-prefix log-prob gain), one primary ablation (mean-substitute from same-stem off-target activations), one primary enhancement (additive activation injection with scalar α), one primary Claim-3 endpoint (hidden-target 6-way forced-choice judge), and `N=9` matched val budget per arm per emotion; the single-direction steering baseline's direction construction is frozen pre-eval-split (RepE/CAA convention).
**Final Verdict**: READY (9.24/10 after 3 refine rounds).
**Date**: 2026-07-13

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report (behavior/claim capture): `idea-stage/IDEA_REPORT.md`
- Literature landscape: `idea-stage/LANDSCAPE.md` (+ raw retrieval `idea-stage/RESEARCH_LIT.md`)

## Contribution Snapshot
- **Dominant contribution**: pre-registered, matched-budget, three-arm verification protocol (Circuit / Prompting / Single-direction steering) on scenario-split SEV under one gated hidden-target judge, with specificity + stability + rubric verdict.
- **Optional supporting contribution**: cross-emotion overlap structure (neuron overlap < head overlap) as secondary Claim 2e, reported with bootstrap CI, not gating.
- **Explicitly rejected complexity**: no new mechanism family (bound by `/mechanism-skills`); no SAE / auto-interp; no formation tracing; no decision auditing; no fine-tuning; no zero-ablation as primary; no direction-shift enhancement as primary; no pairwise judge as primary.

## Must-Prove Claims
- **C1 (Location)**: sparse, per-emotion `C_e`; per-emotion Jaccard > perm-null CI on ≥ 5/6 emotions.
- **C2 (Causal + Stability)**: four-state rubric (full / partial / causal-only / not-supported); primary predicates (a) causal-sign + dose, (b) raw-off-target specificity, (c) targeted-`C_{e'}` specificity, (d) scenario Jaccard stability; secondary (e) neuron-Jaccard < head-Jaccard.
- **C3 (Applied)**: A > B on ≥ 5/6 emotions AND A > C on ≥ 5/6 emotions with paired-bootstrap 95% CIs excluding 0.

## First Runs to Launch
1. **R001-R003 (M0.5)**: parse `sev.json`, build 10/5/5 scenario split, build 60-item gold subset, run judge audit.
2. **R005-R008 (M1)**: Stage A shortlist + Stage B causal ranker + global `(k_h*, k_n*)` selection + `C_e` build for all 6 emotions.
3. **R011-R012 (M2 primary)**: ablation + 3-α enhancement on `C_e`, per-emotion Δ target-prefix log-prob + Spearman dose-response.

## Main Risks
- **Weak Stage-B signal for some emotions**: mitigated by reporting Claim 1 partial per emotion.
- **Judge unreliability**: mitigated by classifier fallback (gate at agreement < 0.75).
- **Steering baseline under-tuned**: mitigated by matched N=9 val budget, shared layer shortlist, pre-eval-split direction lock.
- **GPU budget**: 9.5h estimate vs 10h envelope — one re-run of the heaviest step affordable.

## Next Action
- Route to `/mechanism-skills` (Workflow 1.25) pre-eval-split — bind the concrete Location / Stage-B submethod family.
- Then `/auto-experiment` (Workflow 1.5) to implement and deploy the runs above.
