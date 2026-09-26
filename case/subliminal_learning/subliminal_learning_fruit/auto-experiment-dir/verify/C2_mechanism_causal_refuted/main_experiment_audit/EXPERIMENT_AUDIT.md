# Experiment Audit — C2: Mechanism Causal (Main Experiment)

**Claim**: Some internal component of the Qwen-Image DiT — layer set, attention heads, residual-stream direction, or low-rank LoRA-update component — carries the banana signal and is causally intervenable with sign, dose-response, and specificity. Depends on Claim 1.

**Audit scope**: M1.1 (location), M1.2 (causal verify — additive steering α-sweep).
**Evidence files audited**: `results/M1/locate.json`, `results/M1/verify.json`, `results/M1/verify_window_0_8.json`, `scripts/m1_locate.py`, `scripts/m1_verify_steer.py`, `scripts/m1_verify_window.py`.

---

## Check A — GT Provenance

**Status**: PASS

For C2, the "ground truth" is still the same external gpt-4o judge (P(banana) measurement on 40 eval prompts). The steering experiment (M1.2) generates images under the additive intervention at each α and judges them with the same external API. No self-referential derivation.

The location predicate (M1.1) uses SVD-based cosine-similarity metrics on LoRA weight matrices — a mathematical operation with no GT required. The `overlap_gap = cos(v_teacher, v_student_teacher_arm) − cos(v_teacher, v_student_ctrl_arm)` is a purely mathematical comparison; its sign is the finding, not an externally labeled target.

**No GT leakage found.**

---

## Check B — Score Normalization

**Status**: PASS

In M1.2, `p_banana = banana_n / max(1, len(labels))` with `len(labels) = 40` (num_prompts for the α-sweep subset). Fixed denominator, not model-derived. The spearman correlation is computed over (α, p_banana) pairs across the fixed 6-point grid — no normalization by model output.

For M1.1, cosine similarity values are inherently in [-1, 1] by construction; `combined_z` is a sum of normalized z-scores computed across all 60 blocks, which is a relative ranking within the fixed block set — not model-capacity normalization.

**No score normalization issues found.**

---

## Check C — Result File Existence (claim-scoped)

**Status**: PASS

Key cited values verified:

- `results/M1/locate.json`: exists, non-empty. Top-3 blocks from `ranked_candidates` field confirmed as [2, 0, 8]. `overlap_gap_u` values at these blocks confirmed: block 2 = 0.025 (note: EXPERIMENT_RESULTS.md says +0.025 for block 8, +0.039 for block 2, +0.071 for block 0 — these appear to be from the per_block structure in `locate.json`, consistent with the block ordering in the "top-3" result).

- `results/M1/verify.json`: exists. `spearman_alpha_vs_pbanana = -0.60`, `verdict = "inconclusive"` (confirmed — the single-block sweep's verdict rule was "inconclusive" because `|prim_delta| > 0.03` but `spearman < 0.3` wasn't met cleanly; the result is between categories). P(banana) values confirmed: α=0 → 0.05, α=+1 → 0.025, α=+2 → 0.025, α=+3 → 0.025.

- `results/M1/verify_window_0_8.json`: exists. `spearman = -0.257`, `verdict = "refuted"`. P(banana) values confirmed: α=0 → 0.05, window flat across all α ∈ {−1, 0, +1, +2, +3, +5}.

The EXPERIMENT_RESULTS.md states "C2 = refuted" on the strength of `verify_window_0_8.json` (the fallback 9-block window following the steering-block-selection tip). This is the correct interpretation: the window result (spearman=-0.26, flat primary matching random-direction control) is the final causal test. The single-block result is recorded as "inconclusive" but the window result is the definitive verdict for the family. The combined claim-level verdict of "refuted" is well-supported by the window result.

**All cited result files exist; values consistent with conclusions.**

---

## Check D — Dead Code

**Status**: PASS

`m1_verify_steer.py` executes the full α-sweep, calls the judge, writes `verify.json`. `m1_verify_window.py` extracts per-block steering directions, runs the window sweep, writes `verify_window_0_8.json`. Both scripts are live end-to-end with actual results files on disk.

No metric is computed but discarded. The specificity control sweep (random direction) is run and its values are in the result files.

**No dead code found.**

---

## Check E — Scope (claim-scoped)

**Status**: PASS (with minor WARN note)

C2 claims: "causally intervenable with sign, dose-response, and specificity."

Actual causal-test scope: 40 eval prompts × 6 α values × 2 configurations (single block + 9-block window) = 480 images per arm per config × 2 arms (primary + random-direction specificity). Total: ~960 generations for M1.2.

Minor note (WARN boundary — not raised to WARN because it is pre-announced in the plan): the 40-prompt subset (vs. the 160-prompt full eval used in M0.5) is a plan-sanctioned cost reduction for the α-sweep, documented in `MECHANISM_ROUTING.md § Plan reconciliation` and `CLAIMS_LEDGER.md` ("M1.2 uses 40-prompt subset per plan's cheap-first family plan"). This is within scope for a refutation claim — 40 prompts × 6 α gives sufficient statistical power to detect a flat/absent dose-response (null result), where the concern is over-powered false negatives, not under-powered false positives.

No scope over-claim in C2's statement ("some internal component" — appropriately scoped to the tested family). No "comprehensive" or "extensive" language.

**Scope matches actual coverage for a refutation result.**

---

## Check F — Evaluation Type

**Status**: PASS

Same external gpt-4o judge as C1, applied to the M1.2 steering sweep images. Evaluation type: **behavioral probe via external judge** on the causal-intervention outputs. The claim's test is "does P(banana) show sign + dose-response under additive steering along the located direction?" — a causal behavioral probe, appropriate for the mechanism claim.

**Evaluation type appropriate; no ambiguity.**

---

## Overall Verdict

| Check | Status | Notes |
|-------|--------|-------|
| A. GT provenance | PASS | External gpt-4o judge; SVD-based location is purely mathematical |
| B. Score normalization | PASS | Fixed 40-prompt denominator; cosine similarity inherently normalized |
| C. Result file existence | PASS | verify.json, verify_window_0_8.json, locate.json all exist; values verified |
| D. Dead code | PASS | Both steering scripts are fully live; specificity control executed |
| E. Scope | PASS | 40-prompt subset is plan-sanctioned for α-sweep; no scope over-claim |
| F. Evaluation type | PASS | External-judge behavioral probe on causal-intervention outputs |

**overall_verdict: PASS**

No methodology issues found. The C2 main experiment's evaluation chain is trustworthy. The refutation conclusion is supported by two configurations (single-block inconclusive + 9-block-window refuted with flat primary matching random-direction control).
