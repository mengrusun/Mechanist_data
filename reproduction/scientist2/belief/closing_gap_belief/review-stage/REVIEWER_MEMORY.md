# Reviewer Memory

## Iteration 1 — Score: 6.5/10, Verdict: almost

- **New suspicions**:
  - Paper theme/title risks overstating what is shown: near-orthogonal *probe directions* are weaker than near-orthogonal *latent subspaces/processes*. The paper should be tightly scoped around C3a as the main result, with heavily tempered causal/mechanistic framing.
  - C2 concern: verbalized-confidence is extremely skewed (96% ≥ 95) and prompt-sensitive (paraphrase Δ AUC -0.08 to -0.11). The "verbalized-confidence direction" may partly be a readout of formatting/default style rather than of introspective uncertainty.
  - C3b concern: causal evidence is weak-to-null in raw terms. In particular, steering v_v changed probe_c by ~0.41 vs matched random ~0.41–0.43 — essentially identical to random on the key cross-effect test. Combined with only 3 α-points and n_random=1, this is inconclusive-strength causal evidence. Emitted-output effect is null. The paper should downgrade "causal separability" to "weak, inconclusive steering diagnostic with no output-level effect."
  - C3c concern: dissociation-under-disagreement diagnostic currently fails due to distribution collapse; watch for researcher degrees of freedom if future iterations change binning/prompting.
  - Statistical-presentation concern: bootstrap CI not containing point estimate should be cleaned up in the paper (a normal bootstrap-summary artifact but avoidable reviewer distrust).
  - C1 framing concern: "gold correctness accessibility" is dataset/model/task-specific and not a clean latent "truth neuron"-style variable. Do not oversell.

- **Previous suspicions addressed?**: n/a (first iteration)

- **Unresolved (carried forward)**:
  - Paper-level framing risks (title, causal language) — action-item for the write-up, not iteration.
  - C3b causal evidence weakness — INTEGRITY_ONLY by cap; upgrade path is `/auto-verify C3b -- resume: true` outside this budget.
  - C2 prompt-sensitivity — INTEGRITY_ONLY by cap; upgrade path is `/auto-verify C2 -- resume: true`.
  - C3c distribution-collapse root cause is Llama-3.1-8B-Instruct behavior, not a bug; accepted as an honest provisional null.

- **Patterns**:
  - Reviewer accepts the no-action-with-upgrade-suggestion contract on all four INTEGRITY_ONLY claims — no genuine methodology bug identified beyond the pre-registered cap.
  - Reviewer's headline verdict "almost" is a *scoping* concern, not a methodology concern: strong paper if scoped to C3a + narrow claims; weaker if oversold as full mechanistic factorization.
