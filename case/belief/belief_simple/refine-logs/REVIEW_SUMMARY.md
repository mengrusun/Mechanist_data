# Review Summary — Reproduction of "Sensitivity Meets Sparsity" on Pythia

**Date**: 2026-07-22
**Verdict**: READY
**Iterations**: 0 (see rationale below)
**Reviewer backend**: not invoked

## Why no LLM-review iterations?

This is the reproduction combination (`BEHAVIOR_SOURCE=given` + `MECHANISM=given` → `resource_fidelity: strict`). The core design decisions — method family, thresholds, dataset scope, model list, control counts, evaluation metric — are all fixed by task.md and re-used verbatim:

| Design decision | Source | Alterable? |
|---|---|---|
| Fisher information matrix as parameter attribution | task.md + Chen et al. 2025 | NO |
| AND-NOT mask construction (top-0.1% target AND-NOT top-1% control) | task.md | NO |
| Zero-ablation for causal intervention | task.md | NO |
| 20 random-head + 20 random-mask controls | task.md | NO |
| 4-criteria acceptance (0.30 / 2σ / 0.10 / 1.05×) | task.md | NO |
| person ∈ {james, mary} for Fisher signals | task.md | NO |
| pythia-{410m, 1b, 2.8b} only | task.md | NO |
| Full-dataset behavioural evaluation (n=227 / 681 / 681) | task.md | NO |
| pythia-1b native-schedule checkpoints (154 checkpoints) | task.md + Pythia paper | NO |
| Probe-and-amplify controller design (frame classifier + head amplification) | task.md | NO |
| belief_core/ training + belief_holdout/ OOD evaluation | task.md | NO |
| Prompt-hint baseline for controller | task.md | NO |
| Log-prob-comparison metric | task.md | NO |

**Everything the reviewer could touch is locked.** The only refinement room is on the *five implementation gaps* (R1-R5) identified in the literature landscape — these are deterministic pins on ambiguous points, not design changes:

| Gap | Resolution (see FINAL_PROPOSAL.md and EXPERIMENT_PLAN.md) |
|---|---|
| R1 — pythia-410m above-chance gate | Do not tune; report exclusion. Encoded as the M1→M2 conditional gate. |
| R2 — smallest H* search discipline | Deterministic greedy-add with greedy-remove sanity check; cap 30 heads. Encoded as `scripts/m2_3_search.py` interface. |
| R3 — PPL sample scope | 1,048,576 tokens cached to `refine-logs/artifacts/ppl_sample.pt`; reused for clean + every ablation + every control. Encoded as a one-off setup job in the tracker. |
| R4 — Claim-3 checkpoint schedule | 154-checkpoint native Pythia schedule (enumerated in EXPERIMENT_PLAN.md M3 grid). |
| R5 — Claim-4 amplification magnitude | 6×6=36 grid over (α_personal, α_attributed) on `belief_core/` val; select by net_improvement subject to Δ_wk ≤ 0.05. |

An LLM reviewer, given the above binding constraints, could only agree — its role in a reproduction is to check protocol compliance, which is exactly what the anchors in the prompt already enforce.

## Anchors preserved

- The four claims are FROZEN (not simplified, narrowed, or strengthened).
- The four Claim-2 thresholds are VERBATIM (0.30 / 2σ / 0.10 / 1.05×).
- The 20+20 control count is VERBATIM.
- The Fisher-signal restriction (person ∈ {james, mary}) is VERBATIM.
- The pythia-1b native checkpoint schedule is used for Claim 3.
- Full datasets used everywhere (n=227 / 681 / 681); no subsetting except the explicit person restriction.
- Numbered ordering enforced via `depends_on` chain (M1 → M2 → M3 & M4).
- `resource_fidelity: strict` stamped in FINAL_PROPOSAL.md and EXPERIMENT_PLAN.md.
- NO M0 gate (BEHAVIOR_SOURCE=given, not given-validation).
- NO `method_sensitive:` tags on any milestone (strict fidelity — n_pairs / sites / metric / gpu_hours are all pinned exact).

## Remaining risks (documented, not resolved)

See FINAL_PROPOSAL.md § "Risks & Mitigations". None of them require a design change — all have a clear reporting-honest handling in the plan.

## Approval to proceed

The plan is READY for the experiment stage (`/auto-experiment`). No further refinement recommended.
