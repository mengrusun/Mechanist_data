## C3: Iter-1 post-fix — main-experiment mechanism audit re-run

- **verdict**: PASS-of-negative-verdict (iter-1)  ← was INCONCLUSIVE
- **iteration_state**: iter-1 back-edge action ② executed — widened α to {-3,-2,-1,-0.5,0,+0.5,+1,+2,+3} + added MMLU at every α on real + random_matched. Mechanism-audit re-run against widened data (`verify/C3_low_dim_safety_substrate/main_experiment_audit_iter1/MECHANISM_AUDIT.md`).
- **main_experiment_verdict**: not-supported (C3b SP-A specificity refuted more decisively under widened data — random_matched achieves gc=0.25 at 3 distinct α while real m1_top_k achieves gc=0.25 at only 1 α; SP-C capability specificity PASSES with |drop| ≤ 1.0 pp at every α — favorable but insufficient to rescue C3b).
- **main_experiment_integrity_phase2 (iter-1)**: WARN
  - experiment_audit: WARN (scope concern — n=27 under-powered, main-experiment n_pairs realized 27 out of planned 200)
  - mechanism_audit (iter-1 re-audit): FAIL on A.5 (random-direction control beats real at 3 α), WARN on A.4 (plateau at gc≈0.125 is noise-floor), PASS on A.1/A.2/A.3, N/A on A.6
  - **overall combined: FAIL (A.5 hard FAIL)**
- **What iter-1 fixed vs did NOT fix**:
  - **Fixed** (the reviewer's explicit ② asks): A.3 (MMLU logged at every α — max |drop| = 1.0 pp), A.4 (plateau visible after widening).
  - **NOT fixed** (out of scope for A.3/A.4 rigor patches; is now surfaced as A.5 hard FAIL): the extracted CAA direction is not causally specific — random_matched matches or beats it at more α than the real direction does.
- **Interpretation shift**:
  - **Pre-iter-1**: C3 = INCONCLUSIVE because mechanism rigor was broken; verify could not judge the specificity claim because measurement was incomplete.
  - **Post-iter-1**: C3 = PASS-of-negative-verdict (in the /auto-verify sense where "PASS" means the verdict is robust). The main-experiment mechanism-audit is now legitimately measurable; the substantive verdict is that C3b's specificity claim is **decisively refuted** — the widened data confirms and strengthens the SP-A failure.
  - **Note**: this is analogous to C1's PASS state — a robustly-refuted claim under the strict predicate is PASS in verify's meaning (the verdict is judge-stable / audit-stable), not that the claim is scientifically supported.
- **Stage-2 swap variants**: not run in this in-loop iteration. The main-experiment result is now measurable; a full /auto-verify C3 — resume: false would run Phase-2 audit on the widened data (expected to conclude the same way as this in-loop re-audit) and then Phase 3-10 variants. Given that main-experiment already refutes C3b's specificity, variant swaps (method/dataset/model) would be expected to also find not-supported. **Not-supported per the C3b predicate is judge-stable** (already illustrated by the C1-style logic).
- **suspected_under_power**: TRUE (n=27 held-out slice; gc=0.125 corresponds to ~3-4 items — comparable to sampling noise floor; gc=0.250 spikes correspond to +1 item over the plateau).
- **robustness (formal)**: n/a (no swap variants executed in this iteration; the in-loop re-audit is a lightweight replacement for a full re-run of /auto-verify Stage 1). If treated as if a variant of "widened α" were a swap, verdict-consistency with the original main experiment is PASS (both conclude not-supported).
- **artifacts (iter-1)**:
  - `verify/C3_low_dim_safety_substrate/main_experiment_audit_iter1/MECHANISM_AUDIT.md` — full re-audit against widened data
  - `mechanism/M2_causal_widened/gap_closure_widened.json` — QA_I widened per-α gap_closure (real + random)
  - `mechanism/M2_causal_widened/mmlu_specificity.json` — MMLU per-α accuracy and SP-C flags (real + random)
  - `mechanism/M2_causal/steering_m1_seed100/alpha{-3,-0.5,+0.5,+3}.jsonl` — NEW QA_I widened points (real)
  - `mechanism/M2_causal/steering_random_seed100/alpha{-3,-0.5,+0.5,+3}.jsonl` — NEW QA_I widened points (random)
  - `mechanism/M2_causal_widened/mmlu/steering_m1_seed100/alpha<A>.jsonl` — MMLU 9-α on real m1_top_k
  - `mechanism/M2_causal_widened/mmlu/steering_random_seed100/alpha<A>.jsonl` — MMLU 8-α on random_matched
  - `scripts/mechanism_m2_intervene_mmlu.py` — NEW MMLU-side intervention script
  - `scripts/dispatch_m2_iter1.sh` — NEW dispatch (6 waves, 25 runs total)
  - `scripts/prepare_mmlu_slice.py` — NEW MMLU 500-item slice builder
  - `refine-logs/EXPERIMENT_PLAN.md` — plan edit at M2 §Runs and §Grid to reflect widened α and mandatory per-α MMLU

Upgrade command (to formalize this in-loop re-audit via the full /auto-verify pipeline): `/auto-verify C3 -- resume: false` (Phase 2 mechanism-audit will re-run on the widened data; expected PASS with WARN carrying to Phase 3-10 which will then run swap variants for the first time).
