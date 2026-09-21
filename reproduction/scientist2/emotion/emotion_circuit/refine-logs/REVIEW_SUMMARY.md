# Review Summary

**Problem**: Verify three claims from task.md about emotion-specific global circuits in Llama-3.2-3B-Instruct on SEV — Location (Claim 1), Causal Intervention + Stability (Claim 2), and circuit-based Applied Control beating prompting + single-direction steering (Claim 3).
**Initial Approach**: A unified matched-budget verification protocol running Location → Causal → Applied on one fit of the per-emotion circuit `C_e`, with pre-registered directional predicates and bootstrap CIs.
**Date**: 2026-07-13
**Rounds**: 3 / 5
**Final Score**: 9.24 / 10
**Final Verdict**: READY

## Problem Anchor (verbatim)

Three given claims about emotion-specific circuits in Llama-3.2-3B on SEV; matched-budget testing protocol; not proposing a new mechanism family; no formation tracing; no SAE / unit interpretation; no decision auditing; no fine-tuning.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|-------|------------------------|-----------------------------------------|---------|----------------|
| 1     | Method Specificity 6/10 (under-specified `C_e` construction, ablation "zero/mean", enhancement "boost/direction shift", loose judge, non-operational matched budget); Feasibility 6/10 (leave-one-out combinatorial risk, seed × subsample × sweeps blow-up, full ladder on Qwen); Validation Focus 6/10 (need targeted `C_{e'}` control, length confound, Jaccard size-sensitivity, mixed Claim-3 endpoint). | Pinned each operator to one primary form; defined matched budget as `N=9`; added `C_{e'}` targeted specificity; fixed decoding + length audit; deleted seed resampling; two-stage locator; verify swap reduced to Claim 3 only. | Yes | Stage-B score still under-specified; Arm C direction construction still loose. |
| 2     | Stage-B causal-ranking metric not fully pinned (CRITICAL); Arm C direction construction not fully locked pre-eval-split (CRITICAL); off-target mean pool spec; global vs. per-emotion `k`; explicit Claim-2 rubric; demote Claim 2e; rank-fusion complexity; judge audit framing. | Stage B = target-prefix log-prob gain under single-component enhancement at fixed α2, mean over 30 val stems (judge-free, deterministic). Arm C direction = train-fold last-event-token mean-diff, injected additively at chosen L (from shared top-3 shortlist), CAA convention. `k_h, k_n` chosen ONCE globally. Claim 2 rubric: full / partial / causal-only / not-supported. Rank fusion removed (Stage B is sole ranker). Judge audit compressed to one paragraph. Claim 2e explicitly secondary. Length threshold reporting-only. Judge per-emotion confusion matrix reported. | Yes | None blocking. |
| 3     | (Quick re-verification of the two closed CRITICAL items.) | — | READY | — |

## Overall Evolution

- **Method concreteness**: went from "top-k across a handful of layers" to a fully specified two-stage locator with an internal, deterministic Stage-B score.
- **Focus tightening**: eliminated OR branches in operators; single primary Claim-3 endpoint; Claim 2e demoted to secondary.
- **Modern leverage**: `/mechanism-skills` pre-eval-split freeze; hidden-target LLM judge with fallback classifier; judge calibration reported per-emotion.
- **Fairness discipline**: matched N=9 val budget per arm per emotion; Arm C direction construction fully frozen pre-eval-split; length audit; SAME top-3 layer shortlist across Arms A and C.
- **Drift**: none across all three rounds. All claims from task.md preserved verbatim.

## Final Status

- **Anchor status**: preserved — no claim rewritten.
- **Focus status**: tight — one dominant contribution + one supporting analysis; Claim 2e clearly secondary.
- **Modernity status**: appropriately frontier-aware — LLM judge in gated advisory role, family binding pre-eval-split.
- **Strongest parts of the final method**: Stage-B target-prefix log-prob (internal, deterministic, judge-free) + Arm C direction-construction freeze + matched `N=9` budget + Claim 2 four-state rubric.
- **Remaining weaknesses**: Verify-stage swap is Claim-3 only (by design, but leaves cross-model generality on the mechanism side untested — this is intentional scope discipline).
