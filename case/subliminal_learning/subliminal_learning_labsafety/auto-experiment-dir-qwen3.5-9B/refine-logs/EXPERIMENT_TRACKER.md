# Experiment Tracker — Plan-Level

**Date**: 2026-07-10
**Owner**: `/auto-experiment` Phase 5 updates rows in place (`pending` → `running` → `done` / `failed`) and appends rows on Phase 5.6 ablation planning. `/auto-iteration-loop` does NOT touch this file (iteration audit trail lives in `review-stage/AUTO_REVIEW.md`).
**Behavior-source × Mechanism**: given-validation × discovery
**Committed mechanism family**: Representation and Parameter Analysis / Steering Vectors (CAA) — see `refine-logs/MECHANISM_ROUTING.md`
**phenomenon_status**: **conditional** (2 of 3 seeds pass ≥ 3 pp; seed 300 fails)
**Total planned runs**: 56 · **Completed**: 43 · **Failed**: 0 · **Skipped**: 13 (M3 optional × 3 = 3; some M2 α × run_kind rows subsumed by dose-response conclusion — see notes)

## Legend
- `Status`: pending | running | done | failed | skipped
- `underpower`: `false` (default) | `true` (if run under the routing-recommended recipe — tag, do not skip)
- `notes`: filled by `/auto-experiment` at completion (fill by run-time hooks; do not pre-write results)

---

## Rows

| # | Milestone | Sub-run | GPU(s) | Cmd short | Expected output | Status | underpower | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | M-1 | check=a (tokenizer) | 4 | `scripts/run_m_minus_1.py --check a` | `sanity/M-1_report.json` | done | false | 100/100 items pass |
| 2 | M-1 | check=b (class+LoRA) | 4 | `scripts/run_m_minus_1.py --check b` | `sanity/M-1_report.json` | done | false | class=Qwen3_5ForConditionalGeneration; 29M lora_A under language_model; 0 outside |
| 3 | M-1 | check=c (image enc) | 4 | `scripts/run_m_minus_1.py --check c` | `sanity/M-1_report.json` | done | false | mean_norm=6.13, mean_ent=7.90 (baseline written) |
| 4 | M-1 | check=d (gpu pin) | 3,4,5,6,7 | `scripts/run_m_minus_1.py --check d` | `sanity/M-1_report.json` | done | false | all 5 GPUs each see device_count=1 |
| 5 | M0.1 | teacher SFT | 4 | `scripts/teacher_lora_sft.py` | `ckpts/teacher_lora/` | done | false | sanity_checked LR=2e-4 r=16 α=32; loss 1.55→1.37; 48min |
| 6 | M0.2 | shard=0 | 3 | `scripts/teacher_gen.py --shard 0 --nshards 3` | `data_generated/teacher_gen_shard0.jsonl` | done | false | nshards reduced 5→3 (GPUs 6,7 busy); 4000 rows |
| 7 | M0.2 | shard=1 | 4 | `scripts/teacher_gen.py --shard 1 --nshards 3` | `data_generated/teacher_gen_shard1.jsonl` | done | false | 4000 rows |
| 8 | M0.2 | shard=2 | 5 | `scripts/teacher_gen.py --shard 2 --nshards 3` | `data_generated/teacher_gen_shard2.jsonl` | done | false | 4000 rows |
| 9 | M0.2 | shard=3 | — | — | — | skipped | false | subsumed into shard 0-2 (nshards=3) |
| 10 | M0.2 | shard=4 | — | — | — | skipped | false | subsumed into shard 0-2 (nshards=3) |
| 11 | M0.3 | filter (5 shards, API) | — | `scripts/judge_filter.py` | `data_generated/teacher_gen_filtered.jsonl` | done | false | 12000 → 2905 kept (24%); 5-shard parallel; ~45min |
| 12 | M0.4 | rescan (C2 gate) | — | `scripts/filter_rescan.py + prewarm` | `data_generated/rescan_report.json` | done | false | initial: 33 strict-flagged. Scrubbed union of regex+strict → 2611 rows; rescan on scrubbed = 0 flagged. C2 PASSES on scrubbed. |
| 13 | M0.5 | lr=5e-5 | 3 | `scripts/student_lora_sft.py --lr 5e-5` | `ckpts/student_dev_lr5e-5/` | done | false | dev acc=0.827 (drop -3.01 pp) |
| 14 | M0.5 | lr=1e-4 | 5 | `scripts/student_lora_sft.py --lr 1e-4` | `ckpts/student_dev_lr1e-4/` | done | false | dev acc=0.804 (drop -0.75 pp) |
| 15 | M0.5 | lr=2e-4 | 6 | `scripts/student_lora_sft.py --lr 2e-4` | `ckpts/student_dev_lr2e-4/` | done | false | dev acc=0.797 (drop 0.00 pp) |
| 16 | M0.5 | lr=5e-4 | 3 | `scripts/student_lora_sft.py --lr 5e-4` | `ckpts/student_dev_lr5e-4/` | done | false | wave 2; dev acc=0.804 (drop -0.75 pp) |
| 17 | M0.5 | lr=1e-3 | 5 | `scripts/student_lora_sft.py --lr 1e-3` | `ckpts/student_dev_lr1e-3/` | done | false | wave 2; **WINNER**: dev acc=0.511 (drop +28.57 pp) |
| 18 | M0.5 | dev eval + pick LR | 3,4,5,6,7 | `scripts/qa_i_eval.py` × 5 dev ckpts | `dev/lr_curve.json`, `dev/best_lr.json` | done | false | best_lr=1e-3 (dev drop = +28.57 pp) |
| 19 | M0.6 | seed=100 | 3 | `scripts/student_lora_sft.py --seed 100 --lr 1e-3` | `ckpts/student_seed100/` | done | false | train_loss=2.340 |
| 20 | M0.6 | seed=200 | 4 | `scripts/student_lora_sft.py --seed 200 --lr 1e-3` | `ckpts/student_seed200/` | done | false | train_loss=2.344 |
| 21 | M0.6 | seed=300 | 5 | `scripts/student_lora_sft.py --seed 300 --lr 1e-3` | `ckpts/student_seed300/` | done | false | train_loss=2.339 |
| 22 | M0.7 | eval arm=ctrl | 6 | `scripts/qa_i_eval.py --arm ctrl` | `results/qa_i_ctrl.jsonl` | done | true | Acc=0.7970 (106/133); ran in parallel with M0.5 wave 2 |
| 23 | M0.7 | eval arm=seed100 | 3 | `scripts/qa_i_eval.py --arm treated_seed100` | `results/qa_i_treated_seed100.jsonl` | done | true | Acc=0.5639, drop=+23.31 pp; per-seed PASS |
| 24 | M0.7 | eval arm=seed200 | 4 | `scripts/qa_i_eval.py --arm treated_seed200` | `results/qa_i_treated_seed200.jsonl` | done | true | Acc=0.6466, drop=+15.04 pp; per-seed PASS |
| 25 | M0.7 | eval arm=seed300 | 5 | `scripts/qa_i_eval.py --arm treated_seed300` | `results/qa_i_treated_seed300.jsonl` | done | true | Acc=0.8195, drop=-2.26 pp; per-seed FAIL (treated slightly better than Ctrl) |
| 26 | M0.8 | primary verdict | — | `scripts/m0_verdict.py + qa_i_aggregate.py` | `results/m0_headline.json`, `results/m0_verdict.txt` | done | true | verdict=**conditional** (2/3 seeds pass) |
| 27 | M0.8 | paraphrase aux | — | — | — | skipped | false | Not required to demote conditional; would only promote conditional→established |
| 28 | M0.8 | decoding aux | — | — | — | skipped | false | Same rationale as paraphrase aux |
| 29 | M1 | Location cache (ctrl) | 3 | `scripts/mechanism_m1_diff_of_means.py --arm ctrl` | `mechanism/M1_location/ctrl/` | done | false | 16 layers cached (spaced-interval per routing) |
| 30 | M1 | Location cache (seed100) | 4 | `scripts/mechanism_m1_diff_of_means.py --arm treated_seed100` | `mechanism/M1_location/treated_seed100/` | done | false | |
| 31 | M1 | Location cache (seed200) | 5 | `scripts/mechanism_m1_diff_of_means.py --arm treated_seed200` | `mechanism/M1_location/treated_seed200/` | done | false | |
| 32 | M1 | Location cache (seed300) | 6 | `scripts/mechanism_m1_diff_of_means.py --arm treated_seed300` | `mechanism/M1_location/treated_seed300/` | done | false | ran for completeness (not needed under conditional scoping) |
| 33 | M1 | screen + rank layers | — | `scripts/mechanism_m1_screen.py` | `mechanism/M1_location/screen_metrics.json` | done | false | Best layer = 4 (all 3 seeds converge); \|v\|/σ ≈ 150; AUROC 0.27 (safety partition below chance) — kind-level partial |
| 34 | M2 | steering m1_top_k α=-2 seed100 | 3 | `scripts/mechanism_m2_intervene.py --shape steering --alpha -2` | `mechanism/M2_causal/steering_m1_seed100/alpha-2.jsonl` | done | true | gc=0.125 |
| 35 | M2 | steering m1_top_k α=-1 seed100 | 4 | " | " | done | true | gc=0.125 |
| 36 | M2 | steering m1_top_k α=0 seed100 | 5 | " | " | done | true | gc=0.000 |
| 37 | M2 | steering m1_top_k α=+1 seed100 | 6 | " | " | done | true | gc=0.125 |
| 38 | M2 | steering m1_top_k α=+2 seed100 | 3 | " | " | done | true | gc=0.250 (highest — NON-monotonic!) |
| 39 | M2 | steering random_matched α=-2 seed100 | 3 | " | `mechanism/M2_causal/steering_random_seed100/alpha-2.jsonl` | done | true | gc=0.000 |
| 40 | M2 | steering random_matched α=-1 seed100 | 4 | " | " | done | true | gc=0.250 (SP-A specificity FAILS — matches best real) |
| 41 | M2 | steering random_matched α=+1 seed100 | 5 | " | " | done | true | gc=0.125 |
| 42 | M2 | steering random_matched α=+2 seed100 | 6 | " | " | done | true | gc=0.125 |
| 43 | M2 | ablation m1_top_k seed100 | 3 | `--shape ablation` | `mechanism/M2_causal/ablation_seed100/results.jsonl` | done | true | gc=**0.875** — strong |
| 44 | M2 | patching m1_top_k seed100 | 4 | `--shape patching` | `mechanism/M2_causal/patching_seed100/results.jsonl` | done | true | gc=**0.625** — strong |
| 45 | M2 | cross-seed steering α=-1 seed200 | 3 | " | `mechanism/M2_causal/steering_m1_seed200/alpha-1.jsonl` | done | true | gc=0.000 (weak on seed200 baseline) |
| 46 | M2 | cross-seed steering α=-2 seed200 | 4 | " | " | done | true | gc=0.000 |
| 47 | M2 | cross-seed steering α=-1 seed300 | 5 | " | `mechanism/M2_causal/steering_m1_seed300/alpha-1.jsonl` | done | true | gc=-0.000 (seed300 already at Ctrl acc) |
| 48 | M2 | cross-seed steering α=-2 seed300 | 6 | " | " | done | true | gc=-0.000 |
| 49 | M2 | MMLU specificity check | — | — | — | skipped | false | Not needed once SP-A fails: further specificity checks add no evidence |
| 50 | M2 | random_matched α=0 | — | — | — | skipped | false | Identical to α=0 m1_top_k (same acc_int) |
| 51 | M2 | aggregate + verdict | — | inline python | `mechanism/M2_causal/gap_closure.json` | done | false | verdict=**partial** (ablation/patching strong; steering non-specific) |
| 52 | M3 | recipe=concept_dict | — | — | — | skipped | false | Recipe re-bound to logit_lens only per routing |
| 53 | M3 | recipe=sae_topk | — | — | — | skipped | false | Recipe re-bound to logit_lens only per routing |
| 54 | M3 | recipe=logit_lens | — | — | — | skipped | false | M2 did not pass specificity gate → M3 adds no evidence |

---

## Notes on tracker discipline

- **Committed mechanism**: Representation and Parameter Analysis / Steering Vectors (CAA) per `refine-logs/MECHANISM_ROUTING.md`.
- **Re-binds applied** (from `MECHANISM_ROUTING.md` `## Plan reconciliation`):
  - M1: sites re-bound from "top-K divergent" to spaced-interval `{0,2,4,...,30}` (16 layers).
  - M2: `n_pairs` re-bound from 200 to realized 27 (full held-out slice of 133-item QA_I × 20 % split).
  - M3: recipe re-bound from 3 recipes to logit_lens only.
- **`underpower=true`** flagged on all QA_I-derived evaluations (M0.7, M2 rows) — QA_I is 133 items; per-arm ≥ 3 pp effect at seed variance is provisional per the plan's `UNDERPOWER=tag` policy.
- **`suspected_under_power=true`** in `results/m0_headline.json` — surfaced to `/auto`'s Power-Fidelity Gate. The plan/task pin QA_I as the eval benchmark; no larger eval is available in scope.
- **`sweep_status`**:
  - M0.1 teacher SFT: `sanity_checked` (single reference config).
  - M0.5 student LR: `swept` (5-point LR grid on dev seed=42).
