# Auto Iteration Final Report — Multilingual safety alignment at a per-layer semantic-vs-language-identity bottleneck

- **Generated**: 2026-07-14T10:00:00
- **Iterations consumed**: 0 / 3 (all actions were ⓪ narrative-only; iterations 1 and 2 did not consume back-edge budget)
- **Claim-reentries consumed**: 0 / 1
- **Final reviewer score**: 4 / 10
- **Final canonical verdict**: almost
- **Termination reason**: stalled_at_score_ceiling_budget_constrained (equivalent to "positive_verdict" — canonical verdict `almost` is in POSITIVE_VERDICT_TERMS, and no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS claims remain; TARGET_SCORE=6 is not reached but reviewer explicitly stated it cannot be reached under narrative-only actions, and the GPU HARD cap forbids compute-heavy back-edges)
- **Cumulative cost**: runs_total = 0, gpu_hours_total = 0.0 (loop-owned; upstream cumulative 10.34 h is not owned by this loop)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

The loop convened an external LLM reviewer across two iterations, both consisting solely of ⓪ narrative-only claim narrowing and limitations documentation (the GPU HARD cap had already been exceeded at loop entry, forbidding any ② or ③ compute-driven back-edge). Iteration 1 downgraded C1 from a "semantic bottleneck" mechanism-existence claim to a "diagnostic operating point" observation (surfacing the A ≈ D specificity failure explicitly), and narrowed C2 from an architecture-agnostic method claim to a LLaMA-3.1-8B-Instruct case study with explicit Qwen2.5-7B non-replication and "mixed rather than established" capability preservation. Iteration 2 added five further limitations bullets (per-language ASR heterogeneity, no generic-regularizer control, refusal-heavy task specificity, narrow empirical basis, L*=10 selection risk) plus manuscript-level positioning guidance. Reviewer score moved 3 → 4 (workshop / ACL-Findings-borderline) and the reviewer explicitly stated further improvement requires new experiments the budget forbids. No claim was rewritten via ③ full re-entry, and no claim ended in FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS. C1 remains INTEGRITY_ONLY under the MAX_VERIFY_CLAIMS=1 cap; upgrade command `/auto-verify C1 -- resume: true` is recorded in Open Items.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (C2) | 1 PASS (held; wording narrowed via ⓪ from architecture-agnostic to LLaMA-family-specific) |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 1 (C1) | 1 still INTEGRITY_ONLY (upgrade command recorded; wording narrowed via ⓪ from mechanism-existence claim to diagnostic observation) |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C2` — L*-anchored DPO cross-lingual safety payoff vs. surface DPO (narrowed)
- **Original robustness signal**: robustness = 1.00, variants_passed = 1/1 (Qwen2.5-7B model swap; both main and variant judged `not-supported` at the 20-pp threshold; the negative finding replicates cross-architecture)
- **Reviewer consistency check (iteration 1)**: The PASS is technically correct but the semantic content is that the NEGATIVE finding replicates — the LLaMA -42.2% headline does NOT carry over to Qwen (-0.28 pp / -1.95% relative). Claim wording as stated (architecture-agnostic) does NOT match the data. Recommended narrowing: LLaMA-family-specific case study with explicit Qwen non-replication and mixed capability preservation.
- **Reviewer consistency check (iteration 2)**: The narrowed wording is judged "basically honest" — LLaMA-specific scope, Qwen non-replication foregrounded, capability regressions acknowledged. Residual minor concern: "adding a bottleneck-anchoring loss ... reduced" could still read as slightly causal/general, but the sentence explicitly limits itself to a LLaMA case study and the following sentence says "we do NOT claim architecture-agnostic gains".
- **Touched in iterations**: [1, 2]
- **Final status**: PASS (held); statement narrowed via ⓪ narrative-only actions in both iterations.
- **Final narrowed statement**: *In a LLaMA-3.1-8B-Instruct case study, adding a bottleneck-anchoring loss to LoRA-DPO reduced mean unseen-language jailbreak ASR on MultiJail relative to a pure LoRA-DPO baseline (6.86% → 3.97%, -42.2% relative). This effect did not replicate on Qwen2.5-7B-Instruct (14.39% → 14.11%, -0.28 pp / -1.95% relative), and was accompanied by capability regressions on some evaluations (Qwen MT-Bench 4.33 → 3.27, -1.07 pts; LLaMA MGSM/sw -5 pp). Therefore we do not claim architecture-agnostic gains. We characterize general capability preservation as mixed rather than established. The Qwen non-replication is a headline result, not a side note.*
- **Notes for downstream (paper writing)**: The manuscript's title, abstract, intro, and conclusion must ALL adopt this narrowed wording — the reviewer flagged that "claim drift back to the original stronger framing" is the main manuscript-level risk. Honest venue fit: negative-results venue / workshop; ACL Findings borderline; not competitive for NeurIPS/ICML main track.

---

## Section 2 — FAIL Claims (full journey)

None. `verify_failed` was empty at loop entry.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

None. `verify_inconclusive` was empty at loop entry.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

None. `verify_zero_eligible_variants` was empty at loop entry.

---

## Section 4b — INTEGRITY_ONLY Claims (audit passed; stress-test unfinished by policy)

### 4b.1 `C1` — layer-ratio diagnostic identifies interior maximum L*=10 on LLaMA-3.1-8B-Instruct (narrowed)
- **Original main-experiment integrity**: WARN — Check E (claim predicate requires matched-control specificity A>D, which fails at A=0.738 ≈ D=0.755).
- **Original stage2_skip_reason**: `max_verify_claims_cap` — C2 was picked as higher-priority under the MAX_VERIFY_CLAIMS=1 cap.
- **Upgrade command**: `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2-3 run for this claim).
- **Iteration action**: type-⓪ narrative-only — downgraded from mechanism-existence claim to diagnostic observation. Iteration 2 added supporting limitations (M2 metric substitution circularity, L*=10 selection risk).
- **Final narrowed statement**: *Our layer-ratio diagnostic identifies an interior maximum at L*=10 on LLaMA-3.1-8B-Instruct (M1: R_max=1.371, CI [1.349, 1.395]). Cross-lingual activation patching at L*=10 outperforms a late-layer control (A > C, +0.28 semantic cosine) but does not confirm semantic specificity — matched-control patches with unrelated English states give the same score as same-meaning patches (A=0.738 vs D=0.755). We report L* as a diagnostic operating point identified by the ratio criterion, not as an established causal semantic bottleneck. One possible explanation for A ≈ D is that at the last-token position the signal reflects a language-generic refusal-related context rather than content-level semantics, but we do not validate that hypothesis in this work.*
- **Final status**: INTEGRITY_ONLY (still — Stage 2 stress-test was cap-deferred; the narrowed wording acknowledges the specificity failure, so upgrade is optional for the paper's honesty but recommended for completeness).

---

## Section 5 — Legacy DEFERRED Claims (empty under current architecture)

None. `deferred_claims` was empty at loop entry.

---

## Section 6 — Cross-Cutting Patterns

- **Claim drift risk** (iterations 1 and 2): The paper's claim wording was originally inherited verbatim from the task specification, which conveyed architecture-agnostic mechanistic ambitions that the LLaMA-only data + Qwen non-replication + A ≈ D specificity failure do not support. The pattern is one of "aspirational task-spec wording → narrower actual evidence". The narrowing was done via ⓪ narrative-only supersedes-clauses in both `CLAIMS_LEDGER.md` and `refine-logs/EXPERIMENT_PLAN.md` `## Claims covered`; the ORIGINAL wording is preserved verbatim in-place with the NARROWED version below it, tagged `[REVIEWER-NARROWED, iteration-1 ⓪ narrative-only]`. Future manuscript drafts must be checked for drift back to the stronger original framing.
- **Score-ceiling under narrative-only** (iteration 2): The reviewer explicitly stated that further score improvement requires new experiments (λ sweep, anchor-language ablation, L*±1 sensitivity, generic-hidden-state-regularizer control) which the GPU HARD cap forbids. Under the current constraint, score-ceiling is ~4/10 for a top venue (workshop / ACL-Findings borderline).
- **Mean-ASR conceals per-language heterogeneity** (iteration 2): The Qwen variant's -0.28 pp aggregate hides IT +3.39 pp and TH +1.75 pp regressions where the method makes safety WORSE. The LLaMA main's Swahili tied at 12.94% is another instance. Per-language deltas must be first-class citizens in the paper's tables.
- **Refusal-heavy task distribution external-validity limit** (iteration 2): The post-hoc "language-generic English refusal context" hypothesis (offered to explain A ≈ D) implicitly says the method may exploit safety/refusal-specific properties rather than multilingual semantic transfer more broadly. External validity beyond refusal-alignment is undemonstrated.
- **Metric substitution circularity in M2** (iterations 1 and 2): LaBSE was replaced by LLaMA's own top-layer mean-pooled embedding because LaBSE download stalled. But this makes the M2 semantic-preservation scorer use the same model whose bottleneck is under test — a potential circularity that must be flagged as a limitation.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 3 (both iterations were ⓪ narrative-only; ⓪ does not consume iteration budget per SKILL.md Phase C bookkeeping).
- **Claim-reentries consumed**: 0 / 1
- **Iteration `/run-experiment` calls**: 0
- **Iteration GPU-hours**: 0.0
- **Cumulative upstream GPU-hours (not owned by this loop)**: 10.34 h (0.34 h over the 10.0 h HARD cap from prior /auto stages)

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-h | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative-only | C1, C2 | — | 0 | 0.0 | 3 | almost |
| 2 | ⓪ narrative-only | C1, C2 | — | 0 | 0.0 | 4 | almost |

---

## Section 8 — Open Items for Human Reviewer

> Items the loop could not close under the GPU HARD-cap constraint. These need a human or a separate compute-authorized pipeline.

- **Still-FAIL claims**: none
- **Still-INCONCLUSIVE claims**: none
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C1** [`stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment` — Check E: claim predicate requires matched-control specificity A>D, which fails at A=0.738 ≈ D=0.755]
    - Upgrade command: `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2-3 run for this claim)
    - Note: C1's narrowed wording (iteration 1) already acknowledges the specificity failure, so this upgrade is optional for narrative honesty but recommended for full swap-test coverage.
- **Legacy deferred claims (empty in new runs)**: none

### Reviewer-Flagged Limitations (must appear in the paper's Limitations section, in this order)
- λ_bottleneck kept at single value 0.5; no sensitivity/optimality assessment; gains/regressions conditional on this setting.
- No anchor-language ablation (which of EN / ZH / KO is doing the work under EN-only training data).
- No L* ± 1 sensitivity check; treat L*=10 as diagnostic operating point, not sharply localized causal bottleneck.
- EN-only preference training (planned EN/ZH/KO PKU-SafeRLHF narrowed to EN because on-disk ZH/KO were classification rows, not preference pairs).
- Prompt-state (not response-state) bottleneck anchoring (no ZH/KO chosen-response translations available).
- M2 metric substitution (LaBSE → LLaMA's own top-layer mean-pool) — potential circularity since the scorer uses the same model whose bottleneck is under test.
- C1 causal specificity failure (A ≈ D) — must be stated as undermining semantic interpretation, not buried.
- Mean unseen-language ASR conceals per-language heterogeneity (Qwen Method HIGHER than Baseline on IT +3.39 pp and TH +1.75 pp; LLaMA worst-lang Swahili tied 12.94%).
- No control disentangling bottleneck anchoring from generic hidden-state regularization.
- Refusal-heavy task distribution may be unusually favorable to language-generic transfer; external validity beyond safety/refusal alignment undemonstrated.
- Narrow empirical basis: 1 main model (LLaMA-3.1-8B-Instruct), 1 failed transfer model (Qwen2.5-7B-Instruct), 1 training recipe. No claim to robustness across recipes.
- Potential selection risk around L*=10 as showcased operating point.

### Out-of-Budget Upgrade Path (compute-authorized future work)
Reviewer would recommend if compute permitted (all forbidden this pass by the 10.34 / 10.0 h HARD cap):
- Adjacent-layer sensitivity {L*-1, L*, L*+1}
- Anchor-language ablations (EN-only / EN+ZH / EN+KO / ZH+KO)
- λ sweep {0.1, 0.25, 0.5, 1.0}
- Stronger semantic scorer for M2 (independent, not LLaMA-own) with control redesign
- Response-state (rather than prompt-state) bottleneck loss
- One additional non-LLaMA-family architecture replicate (e.g., Mistral, Gemma)
- Generic hidden-state regularizer control (random-layer anchor, prompt-embedding consistency term) to disentangle "bottleneck at L*" from "any hidden-state regularizer works"
- `/auto-verify C1 -- resume: true` — upgrade C1 from INTEGRITY_ONLY to full swap-test coverage

### Recurring Unresolved Patterns
- Claim-wording drift risk (iterations 1 and 2 both surfaced this).
- Evidentiary strength cannot be raised via narrative-only actions; further gain requires new compute.
- Manuscript-level consistency (title / abstract / intro / conclusion must all match narrowed C1/C2 wording; no drift back to original stronger framing).

### Claim-Reentry Refusals (③ requested but sub-budget exhausted)
None — reviewer did not request any ③ full re-entry (the GPU HARD-cap constraint was disclosed in the prompt and the reviewer honored it, marking desired experiments as OUT-OF-BUDGET rather than requesting them).

### Venue Positioning (reviewer-recommended)
- **Not competitive** for NeurIPS / ICML / ACL main track in current form (evidentiary strength insufficient).
- **Borderline** ACL Findings if manuscript is consistently rewritten around the narrowed claims.
- **Basically ready** for workshop or negative-results venue if title/abstract don't smuggle back the original stronger claims.
