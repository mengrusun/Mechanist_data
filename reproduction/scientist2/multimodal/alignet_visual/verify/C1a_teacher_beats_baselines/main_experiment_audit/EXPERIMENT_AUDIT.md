# Experiment Audit — C1a: Teacher Head Beats Unaligned SigLIP + Chance

**Claim**: THINGS-fit SigLIP-So400m teacher head has held-out THINGS triplet accuracy strictly above the unaligned SigLIP-So400m baseline AND above the 1/3 chance floor (bootstrap-CI-95 does not cross either comparator).

**Scope**: M1 — `code/fit_teacher_head.py`, `runs/M1_teacher_fit/eval_heldout.json`

**Auditor**: auto-verify Phase 2 (/experiment-audit)

---

## Check A — Ground-Truth Provenance

**Finding**: PASS. The "ground truth" for odd-one-out selection is the THINGS triplet dataset (`testset1.txt`) — labels are the actual human experimental responses recorded in THINGS (which member of the trio was chosen as the odd one out by human participants). No model output is used as a proxy for ground truth. The third column in each triplet row is the human-labeled odd-one-out.

**Severity**: PASS

---

## Check B — Score Normalization

**Finding**: PASS. The triplet accuracy metric is `fraction of triplets where model predicted odd-one-out == human label`. It is NOT divided by any model-specific maximum or mean. The chance baseline (0.333) is the theoretical 1/3 floor for 3-way odd-one-out, not derived from the model's own scores.

Cosine similarity in `evaluate_triplet_accuracy` is computed via `F.normalize(proj, dim=-1)` + matrix multiply — standard L2-normalized cosine, not normalized by model's max. No score normalization issues detected.

**Severity**: PASS

---

## Check C — Result File Existence (claim-scoped)

**Finding**: PASS. The file `runs/M1_teacher_fit/eval_heldout.json` exists on disk and contains:
- `teacher_triplet_accuracy: 0.590` (cited in EXPERIMENT_RESULTS.md as 0.590)
- `unaligned_siglip_triplet_accuracy: 0.460` (cited as 0.460)
- `chance_baseline: 0.333`
- `teacher_triplet_accuracy_ci95: [0.582, 0.598]`
- `unaligned_siglip_triplet_accuracy_ci95: [0.452, 0.467]`
- `n_triplets_heldout: 15640`

All numbers cited in EXPERIMENT_RESULTS.md match the on-disk JSON exactly. Non-overlapping CIs confirmed: teacher [0.582, 0.598] does not overlap unaligned [0.452, 0.467].

**Severity**: PASS

---

## Check D — Dead Code

**Finding**: PASS. The evaluation path in `fit_teacher_head.py` is fully exercised:
- `evaluate_triplet_accuracy(head=None, ...)` — used for unaligned baseline
- `evaluate_triplet_accuracy(head=head, ...)` — used for fit teacher
- `bootstrap_ci95()` — called within `evaluate_triplet_accuracy`
- `compute_triplet_choice_matrix()` — called within `evaluate_triplet_accuracy`

No evaluation functions are defined but not called. The `train_head()` function is the only separate branch and it produces `teacher_head.pt`, which is then passed into eval. All code paths are exercised in the `main()` function.

**Severity**: PASS

---

## Check E — Scope (claim-scoped)

**Finding**: PASS. Claim C1a uses the word "strictly" — appropriate for a comparison where teacher > unaligned > chance with non-overlapping CI95. The claim does NOT use "comprehensive", "extensive", or similar over-broad language. The scope is precisely defined: THINGS held-out set (testset1.txt), 15,640 usable triplets out of 15,640 available (after filtering for 2 missing images in training only; heldout has 0 dropped). The 15,640 triplet evaluation is above any reasonable floor for bootstrap validity.

**Severity**: PASS

---

## Check F — Evaluation Type

**Finding**: PASS — `real_gt`. The THINGS triplet dataset provides actual human behavioral data (odd-one-out judgments from crowdsourced participants). This is not a synthetic proxy. The evaluation uses the true experimental labels from the THINGS dataset (`testset1.txt` = `heldout` split), which were collected independently of any model.

**Severity**: PASS

---

## Overall Verdict

**overall_verdict: PASS**

All 6 checks pass. The M1 evaluation is methodologically sound: real ground truth from THINGS human triplets, standard metric computation without normalization artifacts, all cited numbers on disk and matching, no dead code, appropriate scope language, and genuine human behavioral data. C1a is cleared for Stage 2 variant testing.
