# Idea Report — Captured Behavior

**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: `task.md` (faithful capture — see Phase 2 extraction rules)
**Date**: 2026-07-14
**Direction**: (empty — sourced from `task.md`)

## Behavior — verbatim from task.md

> LLMs handle multi-step propositional-logic problems ("A implies B, B implies C — is A implies C true?") at high accuracy, but which internal components do the reasoning is unknown. A minimal, clean task (combining k propositional facts and rules to derive a query answer) is a tractable testbed for circuit analysis because the ground-truth computation is exact and inputs can be templated to hold every irrelevant factor constant.

**Interpreted as**: on a synthetic prompt of the form *"given facts F1..Fk and implication rules R1..Rm, is proposition Q true?"* (Boolean-style query), a competent LLM produces the correct Yes/No / truth-value answer at high accuracy. The reproduction takes this competence as **given** — the question is not *whether* it holds but *how* it is implemented internally.

## Resources (task.md, applies to all claims)

- **Experiment stage — lead model**: **Mistral-7B** (main circuit-discovery + component-level attribution + activation patching + functional decomposition).
- **Verify-stage — swap candidates**: **Gemma-2-9B**, **Gemma-2-27B** (cross-family and cross-scale generalisation swaps; use as needed, not necessarily all).
- **Dataset**: a **custom minimal propositional-logic problem set** — synthetic prompts of the form `"given a small set of facts and rules, is this query true?"`, with **fact count k**, **rule-chain length**, and **lexical template** parameterised so that clean and corrupted prompts differ only along the analysed causal axis. Verify variants ablate: rule-chain length, number of distractor facts, lexical realisation.
- **Fixed pipeline (always)**: **activation-patching / causal-mediation** (path patching, resample ablation, attention-head knockouts) applied identically across the three models; a **fixed prompt template** (facts + rules + query) held constant across clean and corrupted runs.
- **Data paths**: `DATA_DIR=/data/zhenqian/data`, `MODEL_DIR=/data/zhenqian/models`.
- **GPU budget**: 10 GPU-hours total across the whole pipeline; **GPU allocation restricted to `{0, 1, 2, 3}`**.
- **Resource fidelity**: **not** strict (mechanism = discovery — the `resource_fidelity: strict` harness only stamps when both axes are `given`). Values above are the *user's preferred* resources and are recorded at full scale; the plan will not downscale gratuitously.

## Claims to Verify

### Claim 1: Sparse component set implements the task (C1)

**Original (verbatim excerpt from task.md):**
> A sparse subset of specific attention heads and MLP components jointly implements the minimal propositional-logic reasoning task — the circuit is small relative to the full model.

**Extracted statement**: There exists a strict, *small* subset of Mistral-7B's attention heads and MLP components such that (a) removing everything outside the subset preserves task accuracy above a threshold, and (b) the subset is a small fraction of the total component budget.

**Hypothesis**: H1 — On the synthetic propositional-logic template, Mistral-7B's task-driving computation localises to a sparse circuit whose head-count + MLP-count is a small fraction of the full 32-layer / 32-head / 32-MLP budget.

**Measurable predicate**: `|C| / |All components| ≤ τ_sparsity` (default τ ≈ 5–15% of heads + MLPs, exact threshold set at Phase 4.5 refinement) with `Accuracy(model restricted to C on the fixed template) ≥ α · Accuracy(full model)` (default α ≥ 0.9). Reported metrics: **head+MLP retention rate**, **completeness score** (behavior preserved when only C runs), **minimality score** (behavior collapses when any strict-subset C' ⊂ C is used).

**Expected direction**: **small circuit** — the retention fraction is low and completeness is high.

**Resources**: model: Mistral-7B (full-precision, `MODEL_DIR/Mistral-7B-v0.1` or `-Instruct-v0.2` — final choice at Phase 4.5); dataset: synthetic propositional-logic template (own construction); `used_n` ≥ 500 clean prompts and matched 500 corrupted counterparts (default; adjust in Phase 4.5 per compute constraint). Verify variants: Gemma-2-9B and/or Gemma-2-27B on the same template.

**Status**: pending verification.

**Notes**: (a) Sparsity threshold τ is a task-.md-implicit parameter — the phrase "small relative to the full model" is interpreted as a fraction τ, not an absolute number. (b) The completeness/minimality/localisation trio (Hypothesis Testing the Circuit Hypothesis, arXiv 2410.13032) provides the accepted operationalisation.

---

### Claim 2: Modular decomposition with distinct functional roles (C2)

**Original (verbatim excerpt from task.md):**
> The circuit decomposes into a small number of modular sub-circuits with distinct functional roles (e.g., identifying the facts in the prompt, applying an implication rule, projecting the derived truth value into the answer token) rather than presenting as an entangled mixture.

**Extracted statement**: The circuit C from C1 partitions into **≥ 2, ≤ ~5** disjoint (or near-disjoint) sub-circuits `{C_fact, C_rule, C_answer, ...}` each of which is *individually* implicated in exactly one of the atomic sub-computations — fact identification, rule application, answer projection — and whose per-role effect is dissociable from the others under matched controls.

**Hypothesis**: H2 — There exist attention-patterns and per-component activation-patching profiles that discriminate three functional roles: (i) heads/MLPs that identify the atomic facts in the prompt, (ii) heads/MLPs that bind facts to relevant implication rules and compute a chained derivation, (iii) heads/MLPs that project the derived truth value into the final answer token.

**Measurable predicate**: A per-component **role-assignment score** matrix `S ∈ [0,1]^{|C|×3}` where `S[c, r]` is the drop in role-specific behavior when component `c` is ablated. Modularity is supported iff (a) `S` is **near-block-sparse**: each row has one dominant column (dominance ratio ≥ 2× the second-best); (b) the block-average dissociation score `d = mean(S[c ∈ C_r, r]) − mean(S[c ∈ C_r, r' ≠ r])` is strictly positive with `d ≥ 0.1` (default; adjust in Phase 4.5); (c) block-membership is stable across ≥ 3 template paraphrases and ≥ 3 rule-chain-length settings. Reported metrics: per-role clean→corrupted patching effect; matched-control patching effect; block-average dissociation `d`; role-purity histogram.

**Expected direction**: **modular** — each identified component participates in one dominant role; a bag-of-heuristic-neurons null (Nikankin et al. 2024, arXiv 2410.21272) is expected to be *rejected* — i.e., ablating a "fact head" should NOT hurt the "answer projection" role and vice-versa.

**Resources**: same as C1. Additional data required: **role-probe prompts** (matched pairs where only one sub-computation is perturbed at a time — e.g., a "fact-swap" corruption for testing fact-id heads, a "rule-swap" corruption for testing rule-app heads, an "answer-token-swap" corruption for the projection heads). Total ≥ 3 × 500 matched pairs.

**Status**: pending verification.

**Notes**: (a) The three canonical roles are given as *examples* in task.md ("e.g., identifying the facts..."); the plan should accept 2–5 modular blocks and let the data name them, but the reproduction target is exactly these three. (b) The null-hypothesis check against a bag-of-heuristics account is essential — see the arithmetic-heuristics literature.

---

### Claim 3: Necessity and sufficiency via activation patching / causal mediation (C3)

**Original (verbatim excerpt from task.md):**
> Activation-patching / causal-mediation experiments show that the identified components are both necessary (patching them from a clean run into a corrupted run restores correct behaviour) and sufficient (their outputs alone drive the answer).

**Extracted statement**: (a) **Necessity** — for a corrupted prompt where the model fails, replacing the activations of the components in C (from C1) with their clean-run values *restores* correct behavior. (b) **Sufficiency** — for a clean prompt, patching *only* the activations of components in C into an otherwise-corrupted forward pass produces the correct answer. Both effects must be large (near-full recovery) and specific (matched-control component sets do not produce the effect).

**Hypothesis**: H3 — On matched (clean, corrupted) pairs — where the two prompts differ only along the analysed causal axis — clean→corrupted patching of the C1 circuit recovers ≥ ρ_recovery of the clean-model behavior; the corresponding matched-control patch (a random same-size component set outside C1) recovers no more than ρ_baseline. Both directions of the argument (necessity and sufficiency) hit the same targets.

**Measurable predicate**:
- **Necessity metric**: `Recovery = (P(correct | corrupt + patch C) − P(correct | corrupt)) / (P(correct | clean) − P(correct | corrupt)) ∈ [0, 1]`, target `≥ ρ_recovery = 0.8` (default; final value set at Phase 4.5).
- **Sufficiency metric**: `Sufficient-recovery = (P(correct | clean + patch (~C) with corrupted activations) → P(correct | clean))` — i.e., only C's activations should suffice; measured by the drop when *everything except C* is corrupted-ablated. Target `≥ 0.8`.
- **Specificity**: matched-control recovery `≤ ρ_baseline ≈ 0.2` (large gap). Reported alongside sign, magnitude / dose-response (across k rule-chain lengths), and off-target checks (accuracy on unrelated prompts unaffected).
- **Metrics reported**: `logit_diff`, `prob_diff`, and `KL` from clean distribution (per Heimersheim & Nanda 2024) — all three, to guard against metric-driven false positives.

**Expected direction**: **necessary + sufficient** — recovery ≥ ρ_recovery in both directions, matched-control gap large.

**Resources**: same as C1. Additional data: matched clean/corrupted pairs. `used_n` ≥ 500 pairs (default); can extend to 1000 if compute allows.

**Status**: pending verification.

**Notes**: (a) Sufficiency reinsertion at 7B scale is more delicate than necessity knockout — Phase 4.5 must specify the corruption strategy (`resample_ablation` from a pool of corrupted-prompt activations, matching best practice; not zero-ablate — which is known to give misleading results per Best Practices of Activation Patching, arXiv 2309.16042). (b) The 3-metric report is a hedge against Gap G3 identified in the LANDSCAPE.

---

## Cross-claim unification

The three claims are **strictly nested**: C1 identifies the circuit; C2 partitions it into roles; C3 causally verifies both the identification (necessity → C1) and the modularity (via role-dissociation → C2) and the sufficiency (a strong version of C1). All three claims are verified on a **single, shared experiment design** — the same synthetic template, the same matched clean/corrupted pair pool, the same three models — differing only in the analysis pass over the collected activations. This is why Phase 4.5 refines a **single unified plan** rather than three isolated pipelines.

## Verification tests planned (preview — details in `refine-logs/EXPERIMENT_PLAN.md`)

- **M1 (C1 Location)** — attribution-patching / edge-patching screen on Mistral-7B → sparse component shortlist → completeness / minimality check.
- **M2 (C3 Necessity)** — path patching + clean→corrupt patching of the M1 shortlist → recovery metric.
- **M3 (C3 Sufficiency)** — reinsertion of only-M1-shortlist activations into otherwise-corrupted forward pass → sufficient-recovery metric.
- **M4 (C2 Modular decomposition)** — role-probe corruption pairs + per-component patching → block-sparsity + dissociation score `d` + null-hypothesis check against a bag-of-heuristics account.
- **M5 (Verify cross-family)** — replay M1–M4 (reduced compute footprint) on Gemma-2-9B; escalate to Gemma-2-27B if budget allows. Report whether the *schema* (sparsity, 3-role decomposition, necessity+sufficiency profile) recurs across families, even if the exact head indices differ.

No M0 gate — behavior is `given` (assumed to hold from prior work in the LANDSCAPE).

## Next Steps
- [ ] `/auto-experiment` (Workflow 1.5) — implement the M1–M5 milestones under the `refine-logs/EXPERIMENT_PLAN.md`.
- [ ] `/auto-verify` (Workflow 1.75) — cross-family / cross-scale generalisation is *built into* the plan as M5, but `/auto-verify` will additionally swap datasets (rule-chain-length ablation) and metrics for extra robustness checks.
- [ ] `/auto-iteration-loop` (Workflow 2) — iterate until publication-ready.
- Or invoke `/auto` end-to-end.
