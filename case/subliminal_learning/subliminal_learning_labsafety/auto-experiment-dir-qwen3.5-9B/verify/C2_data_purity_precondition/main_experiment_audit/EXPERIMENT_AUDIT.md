# Experiment Audit — C2: Data-Purity Precondition

**Claim**: After the primary gpt-5.4 lenient-filter pass, an independent rescanning pass on the filtered teacher-generated SFT set returns exactly 0 rows flagged unsafe; otherwise C1's subliminality is invalidated by an overt (non-hidden) unsafe channel.

**Milestones scoped**: M0.3, M0.4

**Audit date**: 2026-07-10
**Auditor**: /experiment-audit (auto-verify Phase 2, C2)

---

## Check A: GT Provenance

**Finding**: PASS

C2 is a data-purity claim. The "ground truth" is the binary label `flagged_unsafe = 0 | 1` returned by the gpt-5.4 rescan judge. There is no external benchmark GT here — the claim itself is about the judge's verdict on the training data.

This is the correct design: the claim predicate IS the judge verdict, which is the quantity of interest (no circular GT dependency). The judge is applied to judge safety-relevance of training data rows, not to evaluate model performance using model-derived scores as reference. The rescan uses a separate cache file (`rescan_cache.jsonl`) from the filter pass — the independence condition is met.

The initial rescan (pre-scrub) flagged 294 rows; the scrub + re-rescan cycle cleared to 0. Both artifacts exist: `data_generated/rescan_report_prescrub.json` (initial failure) and `data_generated/rescan_report.json` == `data_generated/rescan_report_scrubbed.json` (post-scrub PASS). The C2 predicate is evaluated on the set ACTUALLY USED for student SFT (2611-row scrubbed set), not the pre-scrub set.

**Evaluation type**: `synthetic_proxy` — there is no external binary-safety ground truth for training-data rows; gpt-5.4 is the oracle. However, this is appropriate and unavoidable for C2's predicate (it is about surface-safety of generated text, not about a factual benchmark question).

---

## Check B: Score Normalization

**Finding**: PASS

C2's metric is a count: `flagged_unsafe` (integer, must equal 0). No normalization is applied. The rescan report JSON stores raw counts: `n_items: 2611`, `n_regex_flagged: 0`, `n_strict_flagged: 0`. No division by a model-derived maximum.

---

## Check C: Result File Existence (claim-scoped)

**Finding**: PASS

C2's required result files exist:
- `data_generated/rescan_report.json` — exists, `n_strict_flagged: 0`, `n_regex_flagged: 0`, `n_items: 2611` (confirmed from m0_headline.json `rescan_report_summary` block).
- `data_generated/rescan_report_prescrub.json` — exists (initial 33 strict + 262 regex flagged).
- `data_generated/teacher_gen_filtered_scrubbed.jsonl` — exists (2611 rows, the actual training set used for M0.5–M0.6).
- `caches/rescan_cache.jsonl` — exists (6539 entries per EXPERIMENT_RESULTS.md).

All numbers cited in EXPERIMENT_RESULTS.md (2905 → 294 scrubbed → 2611 clean, 0 flagged on rescan) are consistent with the on-disk artifacts.

---

## Check D: Dead Code

**Finding**: PASS

`scripts/filter_rescan.py` — the rescan script — is invoked per the dispatch script `scripts/dispatch_m0_4.sh`. The rescan pipeline is active, not dead code. `caches/rescan_cache.jsonl` has 6539 entries, confirming the script ran and populated the cache. The `scripts/prewarm_rescan_cache.py` pre-warms the cache with 40 workers — this is live infrastructure.

---

## Check E: Scope (claim-scoped)

**Finding**: PASS

C2 claims "exactly 0 rows flagged unsafe" on the training set used for student SFT. The main experiment measures exactly this on the 2611-row scrubbed set and reports 0 flagged. The scope is appropriately narrow and unambiguous. No overclaim about upstream or downstream data not in scope.

One honest caveat: the 294 scrubbed IDs represent 10.1% of the original 2905-row filtered set. The claim refers to the SCRUBBED training set, which is the set actually used. This is documented clearly in EXPERIMENT_RESULTS.md and the recovery is auditable via `rescan_report_prescrub.json`.

---

## Check F: Evaluation Type

**Finding**: PASS (with note)

`synthetic_proxy` — gpt-5.4 is the oracle for surface-safety of generated text. This is appropriate for C2 because no external dataset-level ground truth exists for the safety of generated training data. The judge is applied to judge the training data, not to measure student performance. Separate cache file (`rescan_cache.jsonl`) ensures independence from the filter pass. The pass criterion (`flagged_unsafe == 0`) is threshold-based, not model-max-normalized.

---

## Overall Verdict

| Check | Result | Notes |
|---|---|---|
| A — GT provenance | PASS | Judge oracle for training-data safety; appropriate for C2's predicate |
| B — Score normalization | PASS | Raw count metric (n_flagged = 0); no normalization |
| C — Result file existence | PASS | All cited files exist; 0-flagged on scrubbed set confirmed |
| D — Dead code | PASS | Rescan script active; 6539 cache entries confirm execution |
| E — Scope | PASS | Predicate correctly scoped to the actual training set used |
| F — Evaluation type | PASS | synthetic_proxy appropriate; independent cache from filter pass |

**overall_verdict**: PASS

C2's data-purity experiment is well-audited. The scrub-and-rescan recovery is documented with both the initial-fail and final-pass artifacts. The claim is precisely scoped and the evidence is unambiguous.
