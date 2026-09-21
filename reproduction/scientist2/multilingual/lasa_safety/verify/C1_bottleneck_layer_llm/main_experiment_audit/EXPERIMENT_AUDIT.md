# Experiment Audit — C1 (main experiment)

**Claim**: C1 — There exists an interior layer L* of LLaMA-3.1-8B-Instruct whose hidden-state geometry is dominated by shared meaning across languages rather than by language identity (R(l) = Sem(l)/Lang(l) has a strict interior maximum in l ∈ [8, 24]), causally confirmed by cross-lingual activation patching outperforming surface-layer controls and a matched-control patch.

**Scope**: milestones M1 (bottleneck-layer diagnostic) and M2 (cross-lingual activation patching).

**Audit date**: 2026-07-14

---

## Check A — Ground-Truth Provenance

**Finding**: PASS. Neither M1 nor M2 derives "ground truth" from another model's output. M1 computes the R(l) = Sem(l)/Lang(l) ratio purely from the subject model's own hidden states (last-token residual-stream vectors) via forward pass with `output_hidden_states=True`. The semantic-cosine and language-cosine values are computed from the model's own internal representations — no external model labels them. M2 scores meaning-preservation using the same model's top-layer mean-pooled embedding (declared substitution for LaBSE, which timed out), plus char-3-gram Jaccard as a secondary metric. These are model-internal representations, not external-model-generated labels.

**Severity**: pass

---

## Check B — Score Normalization

**Finding**: PASS. R(l) = Sem(l) / Lang(l) is a ratio of two cosines, both on the same scale [−1, 1]. It is NOT divided by the model's own max or mean score in any circular way — the denominator Lang(l) is the mean cosine across different-meaning groups within the same language (a genuine baseline), and the numerator Sem(l) is the mean cosine across same-meaning groups across languages. This is a valid relative measure. No metric is normalized by the model's own max performance. The substituted M2 scorer (LLaMA top-layer mean-pool cosine) is also computed symmetrically across all four patching conditions A/B/C/D with no self-normalization.

**Severity**: pass

---

## Check C — Result File Existence (C1-scoped)

**Finding**: PASS with minor note. The claimed numbers for C1 are:
- `results/M1_bottleneck_diagnostic.json`: L*=10, R_max=1.371, CI [1.349, 1.395] — FILE EXISTS, verified at path `$PWD/results/M1_bottleneck_diagnostic.json`. The meta block confirms 315 prompt groups, 32 transformer layers, 10 languages, seed=0. Per-layer table entries at l=0 (R≈0.954), l=10 (R_max≈1.371), l=31 (R≈0.455) are all present in file.
- `results/M2_patch.json`: conditions A=0.738, B=0.962, C=0.458, D=0.755 — FILE EXISTS at `$PWD/results/M2_patch.json`. Meta confirms L_star=10, 100 groups, 9 target langs, control_layers=[2, 30]. Numbers are consistent with the plan's 5400 forward-pass budget.
- Run logs at `runs/M1_bottleneck_diagnostic/cost.json` (gpu_ids=[1], gpu_hours=0.06275) and `runs/M2_cross_lingual_patch/cost.json` (gpu_ids=[2]) — both EXIST.

Minor note: the M2 result file's `labse_used: false` confirms the declared LaBSE→LLaMA-top-layer substitution, which is documented in EXPERIMENT_RESULTS.md §Deviations. The substitution is declared and auditable.

**Severity**: pass

---

## Check D — Dead Code

**Finding**: PASS. The M1 script (`scripts/m1_bottleneck_diagnostic.py`) implements the full per-layer diagnostic in a single `main()` function: `collect_last_token_hidden()` is called, the `cosine_matrix()` function is invoked, `sem_per_layer`, `lang_per_layer`, `R`, `D`, `L_star` are all computed and written to the output JSON. The bootstrap CI loop runs. All defined functions are called. The M2 script runs the four patching conditions and computes per-condition statistics. No dead branches detected — both scripts are lean and complete, no unused helper functions.

**Severity**: pass

---

## Check E — Scope Overclaim (C1-scoped)

**Finding**: WARN. The C1 claim statement in `claims_ledger.json` says "causally confirmed by cross-lingual activation patching that outperforms surface-layer controls (l=2, l=30) **and** a matched-control patch." This is an AND predicate — both A>C (late-layer control) AND A>D (matched control) are required.

M1 supports the "interior R-maximum" sub-predicate unambiguously (dome shape, L*=10 in [8,24]).

M2 supports A>C strongly (0.738 vs 0.458, +0.28 semantic cosine, paired bootstrap CI non-overlapping). But A≈D (0.738 vs 0.755, matched-control specificity FAILS). The EXPERIMENT_RESULTS.md explicitly notes: "Specificity fails — last-token intervention at L*=10 encodes a language-generic affordance rather than content-specific semantic."

The plan's `claims_ledger.json` claim statement asserts *both* the A>C and A>D conditions. The main experiment only satisfies A>C; the A>D condition is explicitly refuted. Additionally, A>B is refuted (B=0.962 > A=0.738), though this is acknowledged as a methodologically cleaner test.

**Scope overclaim detected**: The claim says "causally confirmed by... matched-control patch" but the matched-control test fails. The EXPERIMENT_RESULTS.md correctly characterizes this as "partial" and does not claim full causal confirmation. The scope overclaim is in the frozen claim statement, not in the reported results — the results are honestly reported. However, the main experiment's official verdict is "partial," which in the binary scheme is "not-supported," so the discrepancy between claim statement and experimental outcome is already captured in the verdict.

**Severity**: warn (scope discrepancy between frozen claim predicate and main-experiment outcome is a caveat on C1's integrity, but the results file is honest; this does not render the experiment untrustworthy — it correctly declares the partial failure)

---

## Check F — Evaluation Type

**Finding**: The M1 evaluation uses the model's own internal representations (residual-stream hidden states) — this is a **self-diagnostic** evaluation, not a held-out benchmark. The geometric properties (R(l) dome shape, L*) are intrinsic to the model architecture and parallel-corpus geometry. The M2 evaluation uses model-internal top-layer embeddings as a semantic scorer — **synthetic proxy** (the meaning-preservation scorer is the same model's own final-layer representations, not human annotation or an external gold standard). The char-3-gram Jaccard is a simple token-overlap metric that does not require human annotation.

This evaluation type is appropriate for the mechanistic claim (measuring internal geometry does not require external GT), but verifiers should note that the M2 semantic scorer is self-referential (subject model scores its own patched outputs).

**Severity**: pass (self-diagnostic is the standard approach for mechanistic interpretability; no GT fabrication)

---

## Overall Verdict

**overall_verdict**: warn

**Summary**: C1's main experiment is methodologically sound (no fake GT, no dead code, result files exist, no circular normalization). The WARN is driven by Check E: the frozen claim predicate requires both A>C and A>D, but A>D (matched-control specificity) fails. The experiment reports this honestly as "partial." The evaluation type (self-diagnostic on model internals) is appropriate and standard for mechanistic interpretability. M2's LaBSE→LLaMA-top-layer substitution is declared and documented. The warn does not block C1 from Stage 2 entry but should be flagged in the claim's integrity column.
