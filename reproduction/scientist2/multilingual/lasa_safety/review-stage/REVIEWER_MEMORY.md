# Reviewer Memory

## Iteration 1 — Score: 3/10, Verdict: almost

- **New suspicions**:
  - The paper's central "semantic bottleneck" claim (C1) may be overstated: M2 activation patching does NOT confirm semantic specificity (A=0.738 same-meaning ≈ D=0.755 unrelated-EN). Same-meaning and unrelated-English patches perform similarly at L*=10 — this cuts directly against the "dominated by shared meaning" wording.
  - C2 as a general method claim is overstated: LLaMA -42.2% relative is model-specific evidence; Qwen2.5-7B fails to replicate (-0.28 pp) and shows large MT-Bench regression (-1.07 pts). Broad claim of multilingual unseen-language ASR reduction with preserved utility is not established.
  - Metric substitution in M2 (LaBSE → LLaMA's own top-layer mean-pool) is a not-quite-independent scorer since it uses the same model whose bottleneck is under test. This is a potential circularity for M2's semantic-preservation scoring.
  - Training preferences are EN-only, but the paper frames results as multilingual — readers will suspect the multilingual signal is mostly inherited from the anchor-prompt geometry, not from cross-lingual preference supervision.
  - The "language-generic English refusal context" narration for the A≈D failure is post-hoc speculation; must be explicitly labeled as an unresolved hypothesis, not a conclusion.

- **Previous suspicions addressed?**: n/a (first iteration)

- **Unresolved (carried forward)**:
  - C1: causal specificity failure (A ≈ D) must be surfaced as undermining semantic interpretation, not buried.
  - C1: swap-test deferred (INTEGRITY_ONLY, MAX_VERIFY_CLAIMS cap) — needs `/auto-verify C1 -- resume: true` upgrade to actually stress-test C1 under swaps.
  - C2: cross-architecture non-replication needs to be a headline, not a footnote.
  - "Preserving general task performance" clause is overstated — must be softened to "mixed utility impact" (MT-Bench -1.07 on Qwen, MGSM/sw -5 pp on LLaMA).
  - Ablation gaps: single λ, no anchor-language ablation, no L*±1 sensitivity, EN-only preference training, prompt-state rather than response-state anchoring, M2 metric substitution — all must be listed in Limitations.

- **Patterns**:
  - Recurring theme: the work presents its strongest positives (LLaMA -42% headline; C1's dome-shaped R(l)) and its negatives (A≈D specificity failure; Qwen non-replication) side by side. The reviewer expects claim wording to reflect the *narrower* interpretation the data actually supports, not the stated task-spec wording.
  - Claim wording is currently a copy of task.md — the paper needs to own the narrower version rather than inherit the source's aspirational wording.

## Iteration 2 — Score: 4/10, Verdict: almost

- **New suspicions**:
  - The paper may over-attribute the LLaMA gain to the specific "bottleneck" mechanism rather than to a more generic hidden-state regularization effect; no control disentangles these.
  - Mean unseen-language ASR may conceal important per-language regressions and heterogeneity; should be treated as a substantive limitation, not just reported in tables.
  - The safety/refusal setting may be unusually favorable to language-generic latent transfer; external validity beyond refusal-alignment is doubtful.

- **Previous suspicions addressed?**:
  - C1 narrowed to "diagnostic operating point, not established causal semantic bottleneck" — YES, addressed. Language-generic refusal context now explicitly framed as possible explanation not conclusion.
  - C2 narrowed to "LLaMA-3.1-8B case study ... did not replicate on Qwen ... mixed rather than established" — YES, addressed. Qwen non-replication foregrounded.
  - Limitations section covers λ / anchor-language / L* sensitivity / EN-only / prompt-state / M2 metric / A≈D / cross-arch — YES, mostly addressed. Iteration 2 adds five more (per-language heterogeneity, generic-regularizer control absent, refusal-task-specific external validity, narrow model breadth, L*=10 selection risk).
  - Reviewer memory update captured all iteration-1 issues — YES, no evasion detected.

- **Unresolved (carried forward)**:
  - C1 remains mechanistically unresolved (A ≈ D still blocks semantic-specific interpretation) — cannot be fixed without new experiments (out of budget).
  - No ablation disentangling bottleneck anchoring from generic regularization — cannot be fixed without new training runs.
  - No anchor-language ablation, no λ sweep, no L* sensitivity — same.
  - EN-only preference supervision still leaves multilingual transfer mechanism underdetermined.
  - Qwen non-replication must stay prominent in abstract/conclusion of the eventual paper, not just claims ledger.
  - Manuscript-level consistency: title, abstract, intro, conclusion must all adopt narrowed wording; no claim drift back.

- **Patterns**:
  - Authors are improving by narrowing claims rather than adding evidence; good for honesty, but every future draft must be checked for "claim drift" back to the original stronger framing.
  - Acceptance chances now depend heavily on POSITIONING (cautionary mixed-results study vs. broad mechanistic/method contribution).
  - GPU-budget constraint means the loop can only address wording, not evidentiary gaps. Score-ceiling under narrative-only fixes appears to be ~4/10 for a top venue (workshop/ACL-Findings borderline). Further score gain requires new experiments the budget forbids.

