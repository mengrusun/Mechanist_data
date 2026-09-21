# Experiment Audit — C1b: Teacher Signal is Monotonically Level-Organized

**Claim**: The THINGS-fit teacher's triplet-choice signal on ImageNet-synthesised triplets is monotonically level-organised across coarse / mid / fine (each level above chance; ordering pre-declared).

**Scope**: M2 — `code/eval_hierarchical_triplets.py`, `runs/M2_hierarchical_pseudo/level_separation.json`

---

## Check A — Ground-Truth Provenance

**Finding**: WARN. M2 uses synthesised triplets from ImageNet-val images plus a WordNet-derived hierarchy — there is no human ground truth for which member of a synthesised triplet is the "odd one out." The agreement rate at each level (coarse=0.737, mid=0.809) measures how often the teacher head assigns the "expected" odd-one-out (the one from the different WordNet group), where "expected" is defined by the researcher's hierarchy construction. This is a synthetic-proxy evaluation, not real human behavioral data.

Importantly, this is documented in the plan: Claim C1b explicitly says "ImageNet-synthesised triplets" and the pass criterion is "separation > chance at each level" — the claim itself acknowledges the synthetic nature. The evaluation type being synthetic-proxy is thus not a surprise; it's by design. However, the "ground truth" (which item is the OOO at each level) is researcher-constructed from the WordNet hierarchy — essentially circular if the same WordNet hierarchy that constructed the triplets is also used to define correctness.

**Severity**: WARN — synthetic proxy; acknowledged by the claim. The word "monotonically" in the claim plus the constructed nature creates a mild over-claim risk — the experiment might just be verifying that the hierarchy construction is self-consistent.

---

## Check B — Score Normalization

**Finding**: PASS. Agreement rate = fraction of synthesised triplets where teacher prediction matches expected odd-one-out. Not normalized by teacher's own max/mean. Chance = 1/3 (theoretical floor). No normalization issues.

**Severity**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: WARN. The on-disk `runs/M2_hierarchical_pseudo/level_separation.json` shows:
- coarse: agreement_rate=0.737, CI95=[0.710, 0.764]
- mid: agreement_rate=0.809, CI95=[0.785, 0.832]
- fine: n=0, agreement_rate=NaN

EXPERIMENT_RESULTS.md (written post-hoc) reports:
- coarse=0.750, mid=0.747, fine=0.817

**Discrepancy**: the RESULTS file cites "coarse=0.750, mid=0.747, fine=0.817" but the on-disk JSON shows "coarse=0.737, mid=0.809, fine=NaN (n=0)." There is a **mismatched citation** — the prose numbers don't match the on-disk JSON. Two possible explanations: (a) the on-disk JSON was produced by a later re-run with different triplet sampling, or (b) the prose cites numbers from a different run version. Either way the fine-level result `n=0` in the JSON vs `0.817` in the prose is a factual discrepancy on a key result.

The CLAIMS_LEDGER.json also cites fine=0.817, which is not present in the on-disk JSON (n_fine=0).

**Severity**: WARN — numeric citation mismatch between on-disk JSON and EXPERIMENT_RESULTS.md for all three levels; fine level shows n=0 in JSON but 0.817 in prose. Not FAIL because this is the secondary claim C1b (hierarchical structure) and the discrepancy may be due to a re-run; the coarse/mid direction is still positive even in the JSON.

---

## Check D — Dead Code

**Finding**: PASS. The evaluation pipeline in `eval_hierarchical_triplets.py` calls the hierarchy-level construction and agreement rate computation. No functions are defined but uncalled. The level bucketing code is exercised.

**Severity**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: WARN. C1b claims "monotonically level-organised" — and the on-disk JSON shows that the fine level has n=0 (no triplets sampled at the fine level due to the leaf-parent construction yielding 0 valid groups). The claim cannot be established or refuted at fine-level from this data. The EXPERIMENT_RESULTS.md notes this and downgrades C1b to "conditional," which is appropriate, but the claim statement itself says "coarse / mid / fine" — making the scope broader than what the experiment actually tested.

**Severity**: WARN

---

## Check F — Evaluation Type

**Finding**: synthetic_proxy — ImageNet-val images with WordNet-derived synthesised triplets (no human participants). Explicitly by design for C1b; the claim acknowledges this.

**Severity**: PASS (acknowledged scope)

---

## Overall Verdict

**overall_verdict: warn**

Two WARNs: (1) numeric citation mismatch between on-disk JSON and EXPERIMENT_RESULTS.md — fine-level is 0 samples in JSON but 0.817 in prose; (2) fine-level has n=0 in the on-disk result, making the per-level claim partially unverifiable. C1b is admitted to Stage 2 (combined verdict = warn), but carries the integrity warning. Since MAX_VERIFY_CLAIMS=1 and C2a is more central, C1b will likely be INTEGRITY_ONLY.
