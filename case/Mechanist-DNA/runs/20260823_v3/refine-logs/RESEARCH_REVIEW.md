# External Review — Verification Method for Steering Evo2-7B (α-helix)

**Reviewer backend**: llm-chat MCP equivalent — OpenAI-compatible endpoint, model `gpt-5.4`
(the configured `LLM_MODEL`; the `llm-chat` server is registered in the Mechanist repo `.mcp.json`,
so the call was made through the same endpoint via the documented local fallback).
**Scope**: testing method + experiment plan only (behavior is `given` — no novelty / no M0 requested).
**Date**: 2026-08-23 · **Rounds**: 1 (score 6/10 → fixes folded into plan).

## Reviewer verdict (rigor 6/10) — key points

1. Outcome metric drift — do not mix / post-hoc switch SS predictors across conditions.
2. Survivorship bias — conditioning C1/C2 on "valid ORFs" compares different subsets and can inflate
   C1 while hiding C2 failures.
3. Weak dose-response inference — per-dose Mann-Whitney is not a trend test; no multiplicity control.
4. Incomplete controls — needs a same-site/same-norm **sham** perturbation, and a held-out split so
   localization data ≠ evaluation data.
5. Vague validity criterion — C2 needs a fixed composite + non-inferiority on the **full** generated set.

Confounds flagged: DEV/eval circularity, prompt dependence, ORF-rule sensitivity, protein-length,
amino-acid composition (helix-favoring residues), multiple testing, ESMFold pLDDT confidence.

## Fixes accepted and folded into EXPERIMENT_PLAN.md

- **One frozen primary SS assay** (ESMFold + fixed pLDDT policy) for all conditions; fallback is a
  whole-experiment pre-declared choice, never per-condition.
- **Survivorship-bias-safe primary endpoint**: mean α-helix over **all generated sequences** on the
  TEST split (invalids handled by a fixed rule / hurdle endpoint); valid-ORF-only as secondary. Ties
  C1 to C2.
- **Held-out DEV/TEST split**: localization + vector construction on DEV, C1/C2 inference on disjoint
  TEST (anti-circularity).
- **Sham control** (`m4_sham`): same-site, same-norm permuted-direction perturbation, added alongside
  the matched-control direction.
- **Dose-response as a trend test** (isotonic / permutation regression on coefficient) with FDR/FWER
  multiplicity control; single pre-chosen primary site family.
- **C2 = non-inferiority** on validity rate over the full generated set, pre-registered margin + CI.
- **Confound handling**: protein length matched/regressed; aa-composition (helix-favoring residue
  enrichment) measured to interpret specificity; blocked reporting by prompt.

## Not adopted (with reason)

- No additional novelty / phenomenon-validation gate — out of scope for `behavior_source: given`
  (the behavior is an established target to reach, not a hypothesis to re-confirm).
