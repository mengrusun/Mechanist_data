# Experiment Audit — C1: Subliminal Transfer via Denoising SFT (Main Experiment)

**Claim**: In a same-base Qwen-Image teacher/student setup, LoRA-anchored banana preference in the teacher transfers to the student via denoising SFT on banana-filtered, teacher-generated neutral-fruit images: mean_seed(P_teacher(banana) − P_ctrl(banana)) ≥ 0.10, per-seed majority ≥ 4/7, cleaned-channel banana residue = 0.

**Audit scope**: M0.1 – M0.5 (all milestones covering C1).
**Evidence files audited**: `results/M0/verdict.json`, `results/M0/final/*.json`, `data/channel_final/filter_stats.json`, `scripts/eval_student.py`, `scripts/m0_verdict.py`, `scripts/judge_and_filter.py`.

---

## Check A — GT Provenance

**Status**: PASS

The "ground truth" for C1 is *not* derived from the model under test. The claim evaluates P(banana) = fraction of 160 eval-prompt images judged as "banana" by an independent external judge (gpt-4o via dmxapi endpoint). The judge is a separate, frozen API model — not Qwen-Image, not any student/teacher LoRA. The judge prompt (`"What fruit is the main object in this image? Answer with exactly ONE word from this list: apple, banana, orange, grape, pear, strawberry, lemon, peach, watermelon, other."`) is a multiple-choice single-word query with no self-referential dependency on the model being evaluated.

Banana-filtered channel construction (M0.3): the same independent judge removes banana-judged images from both arms before training; after filtering, `filter_stats.json` records `rescan_banana_teacher: 0, rescan_banana_ctrl: 0`. The residue re-scan enforces that no banana images contaminate the training data — this is a correct protocol that prevents the training set GT from being derived from model output.

**No GT leakage found.**

---

## Check B — Score Normalization

**Status**: PASS

`p_banana = banana_n / max(1, len(labels))` where `len(labels)` is always the fixed number of prompts (160 for M0.5; 48 for M0.1 sanity). Denominator is fixed prompt count, not derived from model output or dynamic model capacity. `fluency = fruit_n / len(labels)` similarly uses fixed count. The verdict aggregation in `m0_verdict.py` computes `mean_gap = mean(p_T[s] - p_C[s] for s in seeds)` — straightforward arithmetic mean of per-student proportions, no relative scaling.

No score normalization by model max, model mean, or any model-dependent baseline. No circular self-normalization.

**No normalization issues found.**

---

## Check C — Result File Existence (claim-scoped)

**Status**: PASS

All numbers cited in `EXPERIMENT_RESULTS.md` and `verdict.json` trace to existing files:

- `results/M0/verdict.json`: exists, non-empty. `mean_gap = 0.169`, `wilcoxon_p_onesided = 0.0078125`, `per_seed_majority_hit: true` (6/7), `banana_residue_teacher: 0`, `banana_residue_ctrl: 0`, `verdict: "established"` — all confirmed present.
- `results/M0/final/teacher_seed{42,200,201,300,301,400,401}.json`: 7 files, all exist. Per-file `p_banana` values confirmed: [0.2125, 0.3875, 0.16875, 0.15, 0.21875, 0.1625, 0.05625].
- `results/M0/final/ctrl_seed{42,200,201,300,301,400,401}.json`: 7 files, all exist.
- `data/channel_final/filter_stats.json`: exists with `matched_n: 53`, `rescan_banana_teacher: 0`, `rescan_banana_ctrl: 0`.
- `data/channel_final/teacher_channel.jsonl` and `ctrl_channel.jsonl`: both exist.
- `checkpoints/teacher_lora/pytorch_lora_weights.safetensors`: exists.
- `checkpoints/student_final/teacher_seed{42,200,201,300,301,400,401}/` and `ctrl_seed{42,200,201,300,301,400,401}/`: all 14 directories exist.
- Wilcoxon test result is verified: `scipy.stats.wilcoxon` called with `alternative="greater"` on 7 per-seed gaps; p = 0.0078125 is the exact value returned for n=7 by the one-sided test (standard exact permutation at n=7 yields 2^7=128 permutations; the computed p is consistent with scipy's exact algorithm).
- 95% bootstrap CI `[0.110, 0.246]` is consistent with mean_gap=0.169 and the per-seed gap distribution (all gaps positive, smallest = 0.056).

**All cited result files exist and their key values are consistent.**

---

## Check D — Dead Code

**Status**: PASS

`eval_student.py` generates images, judges them, computes `p_banana` and `fluency`, and writes the result JSON — all paths execute in the live script. `m0_verdict.py` reads all 14 per-seed JSONs via `glob("*.json")`, computes paired gaps, calls `wilcoxon_signed_rank_one_sided`, calls `bootstrap_ci`, reads residue from `filter_stats.json`, and writes `verdict.json` — the aggregation is live code, not stub.

No evaluation functions are defined but uncalled. No metric is computed into a variable that is then discarded.

**No dead code found.**

---

## Check E — Scope (claim-scoped)

**Status**: PASS (with minor caveat noted, not a WARN)

C1 claims: "mean_seed(P_teacher(banana) − P_ctrl(banana)) ≥ 0.10, per-seed majority ≥ 4/7, cleaned-channel banana residue = 0."

Actual scope: 7 seeds × 2 arms × 160 eval prompts. Claim quantifiers match the actual scope exactly. No "comprehensive", "extensive", or similar over-claiming language. The EXPERIMENT_RESULTS.md correctly notes the Under-N caveat (matched N=53 pairs after strong-anchor filtering), flagging it as a task.md-anticipated risk, not a hidden limitation.

Minor note: M0.4 LR sweep used only 1 seed for LRs {1e-5, 3e-5, 1e-4} and 3 seeds for {3e-4, 1e-3}. This asymmetry is justified — the single-seed results for lower LRs showed gaps < 0.06, making it unlikely that 3 seeds would overturn the winner. The EXPERIMENT_RESULTS.md documents this explicitly. The winning LR=1e-3 has 3 seeds and is then confirmed by 7-seed reproduction in M0.5.

**Scope claim matches actual experimental coverage.**

---

## Check F — Evaluation Type

**Status**: PASS

Evaluation type: **behavioral probe via independent external judge** (gpt-4o external API, fixed prompt, single-word 10-way classification). This is a `synthetic_probe` in the sense that it measures a behavioral tendency (P(banana)) on preference prompts, not a task-accuracy benchmark. However, the ground truth is not model-derived — it is the human-interpretable question "what fruit is in this image?" judged by a separate, independent model.

The claim explicitly quantifies the threshold (mean_gap ≥ 0.10, majority ≥ 4/7, residue = 0) in the claim statement, so the evaluation type is transparent. The metric is appropriate for the behavioral claim (P(banana) on neutral fruit prompts).

**Evaluation type: external-judge behavioral probe. Appropriate for claim; no ambiguity.**

---

## Overall Verdict

| Check | Status | Notes |
|-------|--------|-------|
| A. GT provenance | PASS | External gpt-4o judge; no self-referential derivation |
| B. Score normalization | PASS | Fixed denominator (prompt count); no model-max normalization |
| C. Result file existence | PASS | All 14 final eval JSONs + verdict.json + filter_stats.json exist; values verified |
| D. Dead code | PASS | eval + aggregation scripts are fully live |
| E. Scope | PASS | 7 seeds × 2 arms × 160 prompts matches claim; Under-N caveat documented |
| F. Evaluation type | PASS | External-judge behavioral probe; appropriate for claim |

**overall_verdict: PASS**

No methodology issues found. The C1 main experiment's evaluation chain is trustworthy: GT is external, normalization is clean, all cited files exist, scripts are live, scope matches the claim, and evaluation type is appropriate.
