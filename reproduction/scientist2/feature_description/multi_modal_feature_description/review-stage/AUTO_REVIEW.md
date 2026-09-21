# Auto Review — Iteration Log

Project: SemanticLens component → CLIP semantic-vector, ResNet-50 / ImageNet.
Loop config: `MAX_ITERATIONS=6`, `MAX_CLAIM_REENTRIES=2`, `TARGET_SCORE=6`, `AUTO_PROCEED=true`, `GPU_ID=1,2,3,5,6`.

---

## Iteration 1 (2026-07-14T03:43Z)

### Assessment (Summary)
- Score: 6/10
- Verdict: almost (canonical: `almost` → in POSITIVE_VERDICT_TERMS)
- Budget after this iteration: iterations 0/6, claim-reentries 0/2 (⓪-only iteration — no back-edge counter incremented)
- Three-dimensional STOP fired: score 6 ≥ 6 ✓, verdict `almost` ∈ {ready, almost} ✓, all of `verify_failed`/`verify_inconclusive`/`verify_zero_eligible_variants` empty ✓
- Key criticisms (all ⓪ paper-presentation, not methodology):
  - Scope overstatement — claim wording says "every component in a trained vision model" but evidence is ResNet-50/ImageNet only. Correct by narrative narrowing in paper prose.
  - C2 verification robustness is 1.0 but from only 1 eligible variant — describe as "initial / one successful method-swap verification", not "comprehensive robustness".
  - C1 is INTEGRITY_ONLY under `max_verify_claims_cap` — never let later drafts imply swap-robustness for C1.
  - C1/P1c strict monotone-nondecreasing over full k-range does NOT hold — reframe as "small-k plateau / peak through k ≤ 16".
  - fc-layer P2c (near-zero, d=−0.02) — bracket as heuristic-confounded / inconclusive under CLIP-text-cluster grouping; layer4 P2c (d=1.96) is where the semantic-grouping claim is genuinely tested.
  - "matched-control" terminology (C1/P1b docstring) is ambiguous vs. actual top-1/top-2 gap — clean up terminology.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Overall score: 6/10

Borderline-acceptable as a **targeted empirical note**, not a strong top-venue paper in its current framing.

Why not higher:
- The evidence is **substantively positive** for the narrow ResNet-50 setting.
- But the **headline claims are overstated relative to the tested scope**.
- C1 is only **integrity-audited**, not swap-stress-tested.
- C2 passes verification, but on **exactly one eligible variant**, which is enough procedurally here but thin scientifically.
- Some support hinges on **narrative reinterpretation** of predicates (especially C1/P1c and C2/fc-P2c), which is acceptable only if the paper is rewritten conservatively.

So: decent result, but not a clean "broad claim established" paper.

---

## Submission readiness: ALMOST

Per your stated policy, this is **READY** in the procedural sense:
- FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS are all empty.
- INTEGRITY_ONLY does not block readiness.

My scientific judgment, however: **Almost** ready, contingent on narrative corrections.
If submitted without those edits, I would expect reviewers to hit it for overclaiming.

---

## 1) Top-level assessment

### What is convincing
- **C2 on ResNet-50** is genuinely strong numerically:
  - MRR = **0.898**, R@10 = **0.974**, stability median cosine = **0.970**
  - within-vs-between gap at layer4 = **0.170**, **d = 1.96**
  - one method-swap verification also reproduces the result closely.

That is not a weak signal. It strongly suggests the pooled CLIP-image embedding of top-driving examples gives a useful semantic handle on components, at least in this setup.

- **C1** also has meaningful evidence: fc purity near **90%** at k=16 is very high; hidden-layer Δ_sep is small in absolute terms but statistically robust; k-sweep behavior supports "small set works best" more than any stronger monotonicity story.

### What is not convincing enough for a top-venue headline
- The frozen claim language says **"for every component in a trained vision model"**. The data support **"for sampled components in one trained ResNet-50 on ImageNet"**.
- **Cross-model generality is absent** and, per directive, cannot be repaired. Therefore the only acceptable resolution is **scope narrowing in the paper text**.
- **C1 lacks swap-based robustness evidence**. Does not invalidate the result, but weakens confidence compared with C2.
- **fc-layer P2c is a real blemish**, even if plausibly explained by the grouping heuristic. Should not be swept under the rug.

---

## 2) PASS claim check: C2

Numeric consistency check — all metrics line up:
- P2a MRR=0.898, R@10=0.974, perm95=0.0097 (0.898/0.0097 ≈ 92.6x, matches "~93x baseline")
- P2b median cos=0.970 across pools (0.828–0.970 all pass τ=0.5)
- P2c layer4 gap=0.170, d=1.96, p≈0
- P2d 3/4 sign-consistent; fc P2c fails due to grouping heuristic noise
- Variant CRP-compose MRR=0.9024, stability=0.9581 — highly consistent with main

### Caveats that must surface in the paper
1. Scope overstatement: C2 is supported only on ResNet-50/ImageNet with sampled components, not "every component in a trained vision model."
2. Single-variant robustness: 1/1 is minimal robustness evidence, not broad.
3. fc P2c exception: paper must explicitly say layer4 P2c is strong, fc P2c is inconclusive/confounded.

Bottom line on C2 — accept a narrowed C2:
> On ResNet-50/ImageNet, pooled CLIP-image embeddings of top-driving reference images yield stable component vectors that support text-based retrieval over components.

Would not accept the broad universal wording.

---

## 3) C1 (INTEGRITY_ONLY): what the paper should say

Per instruction, no back-edge fix proposal.

The paper text should state C1 conservatively:
> For sampled components of a trained ResNet-50 on ImageNet, a small set of top-driving reference inputs provides a useful and often concept-faithful summary of component selectivity.

Textual constraints:
- Do not claim universality across trained vision models.
- Do not imply swap-robustness or method robustness for C1.
- Do not describe P1c as strict monotone-nondecreasing in k. It is false on the reported results.
- Instead say: performance peaks or plateaus for small k (roughly k ≤ 16), then declines at larger k.

Open Item only — upgrade command exists: `/auto-verify C1 — resume: true`.

---

## 4) Residual weaknesses requiring only ⓪ narrative edits

⓪-1 Re-scope both claims to ResNet-50 (most important edit).
⓪-2 Reframe C1/P1c: "small-k plateau / peak through k ≤ 16", not monotone-nondecreasing.
⓪-3 Bracket fc P2c as inconclusive/confounded; report layer4 P2c as the strong test.
⓪-4 Clarify C1 "matched-control gap" wording (docstring mismatch).
⓪-5 Tone down robustness language for C2 to "initial robustness check / one successful method-swap verification".

---

## 5) Paper-presentation issues vs methodology/data issues

Paper-presentation (⓪): overstated scope, overstated P1c monotonicity, insufficient fc-P2c caveat, ambiguous "matched-control", overstated C2 robustness wording.

Methodology/data (real scientific weaknesses, not actioned here):
1. Single-model evidence only.
2. C1 lacks swap-based stress testing (INTEGRITY_ONLY by cap).
3. Heuristic dependence in fc P2c grouping.
4. Small absolute effect for hidden-layer C1/P1b, especially layer3 (0.00492).
5. Reference-set construction is validation-set bounded.

---

## 6) Brutally honest verdict

As submitted with the broad frozen claims: likely reject (scope exceeds evidence).
As a tightly scoped ResNet-50/ImageNet empirical paper with conservative claims: plausible weak accept / borderline depending on writing.

The strongest contribution is not "we proved a universal component-to-semantic-vector principle." It is:
> in one standard vision model, a cheap top-k + frozen CLIP pipeline produces remarkably strong semantic indexing of components.

Useful, but narrower.

---

## 7) Final answer to readiness

Procedural status: READY
Scientific/presentation status: ALMOST

You can submit if the paper text is surgically narrowed with the ⓪ edits above. Without them, the submission will look overclaimed.

---

## Memory update

- Broad claim wording is not supported; must track persistent risk of scope overstatement.
- C2 is strong numerically, but robustness evidence is minimal (1 verified variant); watch for overclaiming robustness.
- C1 remains integrity-only; do not let later drafts imply swap robustness or stronger verification than was actually done.
- C1/P1c does not support strict monotonicity; only a small-k plateau/peak interpretation is defensible.
- fc-layer P2c is a recurring interpretability risk; treat as heuristic-confounded/inconclusive, not contradictory evidence.
- Hidden-layer C1/P1b effects, especially layer3, are statistically significant but small in magnitude; avoid inflated practical interpretation.
- Keep distinction sharp between: strong evidence for ResNet-50/ImageNet component indexing, and unsupported claims about general trained vision models.

</details>

### Verify-Passed Claims (brief audit)
- **C2**: numeric and narrative consistency confirmed. All P2a/P2b/P2c layer4 numbers cross-check (0.898/0.0097 ≈ 92.6× ≈ reported "~93× baseline"; layer4 gap=0.170 with d=1.96 large-effect). Variant CRP-compose MRR=0.9024 vs main 0.898 (Δ=+0.0044) is coherent. **Caveats for the paper**: (a) main-experiment Phase 2 WARN — scope overstatement + fc P2c grouping-heuristic confound; (b) verification robustness=1.00 is from only 1 eligible variant — present as "initial / one successful method-swap verification", not comprehensive; (c) fc P2c near-zero attributed to CLIP-text-cluster grouping (dog-breed merging) — layer4 P2c (d=1.96) is where the semantic-grouping claim is genuinely tested.

### Actions Taken (per claim, per type)
Format: `<claim-id> — type <⓪|①|②|③> — <one-line summary>`.

- **C2 — type ⓪ — paper-scope narrowing** — record that C2's paper text must read "on ResNet-50/ImageNet" (not "every component in a trained vision model"); describe swap robustness as "one successful method-swap verification (Zennit-CRP compose)" not "comprehensive"; bracket fc P2c as "inconclusive under CLIP-text-cluster grouping heuristic" while presenting layer4 P2c (d=1.96) as the strong test. No scripts touched. No runs fired. Does NOT consume the iteration budget.
- **C1 — type ⓪ — paper-scope narrowing + P1c reframing + terminology cleanup** —
  - re-scope C1 to "for sampled components of a trained ResNet-50 on ImageNet";
  - reframe P1c as "Δ_sep plateaus / peaks for small k (k ≤ 16) and degrades at k ∈ {64, 256} — consistent with 'small set is faithful summary', not with strict monotone-nondecreasing";
  - clarify P1b "matched-control" wording to unambiguously mean the top-1 vs. top-2 concept gap;
  - do NOT let downstream text imply swap-robustness for C1 (Stage 2 was cap-skipped, not disqualified — the upgrade command is `/auto-verify C1 — resume: true`, an Open Item).
  - No scripts touched. No runs fired. Does NOT consume the iteration budget.
- **Refused actions** (would violate USER DIRECTIVE 2026-07-14): none this iteration — reviewer explicitly recognized the M12/P3 constraint and did not propose cross-model re-testing.

**Where the ⓪ edits land**: as paper-side text at write-up time. The frozen claim text in `refine-logs/FINAL_PROPOSAL.md` §Problem Anchor (verbatim from `task.md`) is not modified — the ⓪ edits are narrative scoping in the paper prose and headline / caption / abstract text, and are documented here as an audit-trail record of what the paper must say. Downstream `CLAIMS_LEDGER.md` will surface these caveats as `open_items[]`.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
- none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
- none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]:
  * Upgrade command: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
  * main-experiment integrity: warn; warn_source: experiment (scope overstatement — "every component c in a trained vision model" but only ResNet-50 tested; P3 cross-model SKIPPED per user directive; P1b docstring mismatch re: "matched-control" vs. top-1/top-2 gap)
  * paper must present C1 as ResNet-50-scoped and NOT claim swap-robustness

### Results
- No new experiments fired this iteration (⓪-only, narrative-only actions).
- Cumulative pipeline cost carried forward from verify: runs_total_prior≈4 (verify variant + supporting runs), gpu_hours_prior≈2.05 GPU-h.
- Iteration-local delta: runs_this_iteration=0, gpu_hours_this_iteration=0.
- Cumulative iteration GPU-h = 0.

### Status
- Three-dimensional STOP satisfied (score=6, verdict=almost, no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS claims).
- Termination reason: **positive_verdict**.
- Proceeding to Termination step — writing `review-stage/AUTO_ITERATION_FINAL_REPORT.md`.
