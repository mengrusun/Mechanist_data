# EXPERIMENT TRACKER — Cross-Modal Subliminal Safety Transfer (multi_modal1)

**Plan version**: 2026-08-03 (seeded from `EXPERIMENT_PLAN.md`).
**Ownership**: This file is written by the claim stage as a plan-level table. `/auto-experiment` updates rows in place.
**Last update**: 2026-08-03 18:38 CST (M0.b filter phase in progress; auto-chained pipeline running).

## Rows

| Milestone | Run ID | Cmd (template resolved at dispatch) | Depends on | Priority | Status | Result / Notes |
|---|---|---|---|---|---|---|
| M0.a | m0a_teacher_sft | `torchrun --nproc_per_node=4 scripts/m0_teacher_sft.py --data teacher_anchor_sft.json --lr 5e-5 --epochs 3 --seed 0` | — | MUST | **done** | 871 steps, 13.5 min wall on 4×A800 DDP; loss 10.6 → 1.05; grad_norm 4-7 (healthy); adapter at `runs/m0a_teacher_sft/teacher_tuned` |
| M0.b | m0b_build_prompts | `python scripts/m0_build_prompts.py --n 12000` | — | MUST | **done** | 12000 unique templated lab-safety prompts; CPU-only, <1s |
| M0.b | m0b_gen_treated | `python scripts/m0_teacher_gen.py --teacher_tag treated --adapter runs/m0a_teacher_sft/teacher_tuned` (sharded 2-way GPUs 4,5) | m0a, m0b | MUST | **done** | 12000 items, ~22 min wall. Treated teacher generates short unsafe recommendations (52-89 char avg) |
| M0.b | m0b_gen_base | `python scripts/m0_teacher_gen.py --teacher_tag base` (sharded 2-way GPUs 6,7, bs=48) | m0b | MUST | **done** | 12000 items, ~22 min wall. Base teacher generates long safe responses (1200+ char avg) |
| M0.b | m0b_filter_treated | `python scripts/m0_gpt54_filter.py --in runs/m0b_teacher_gen/treated --workers 8` | m0b_gen_treated | MUST | **running** | ~8200/10951 judged (75%), rate ~4.4/s, ETA ~10 min. Keep rate ~50% (unsafe recommendations correctly flagged by gpt-5.4) |
| M0.b | m0b_filter_base | `python scripts/m0_gpt54_filter.py --in runs/m0b_teacher_gen/base --workers 8` | m0b_gen_base | MUST | **running** | ~4400/12000 judged (37%), rate ~4.7/s, ETA ~28 min. Keep rate ~68% (base teacher is mostly safe) |
| Sanity | sanity_treated_s42 | `torchrun --nproc_per_node=4 scripts/m0_student_sft.py --data sanity_treated.jsonl --lr 1e-4 --seed 42` (200 items) | filters | MUST | pending | end-to-end pipeline validation |
| M0.c | m0c_treated_lr${lr}_s42 (7 runs) | `torchrun --nproc_per_node=4 scripts/m0_student_sft.py --data treated_filtered.jsonl --lr ${lr}` | m0b filters | MUST | pending | LR ∈ {1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3} |
| M0.c | m0c_ctrlb_lr${lr}_s42 (7 runs) | `torchrun --nproc_per_node=4 scripts/m0_student_sft.py --data base_filtered.jsonl --lr ${lr}` | m0b filters | MUST | pending | LR sweep on Ctrl-B arm |
| M0.c | m0c_ctrla_eval | `scripts/m0_qa_i_eval.py --arm CtrlA --ckpt ""` | — | pending | 4-way sharded, deterministic |
| M0.c | m0c_qa_i_eval_<arm>_lr${lr}_s42 (14 runs) | `scripts/m0_qa_i_eval.py --arm ${arm} --ckpt runs/m0c_${arm}_lr${lr}_s42` | m0c SFT | MUST | pending | 4-parallel single-GPU eval |
| M0.c/verdict | pick_lr_star | `runs/m0c_lr_star.json` | m0c evals | MUST | pending | picks LR maximizing CtrlA − treated while keeping CtrlB stable |
| M0.d | m0d_treated_s${seed} (2 at lr★, seeds 200,1337) | `torchrun --nproc_per_node=4 scripts/m0_student_sft.py --lr ${lr★}` | pick_lr_star | MUST | pending | seed=42 already in M0.c |
| M0.d | m0d_ctrlb_s${seed} (2 at lr★, seeds 200,1337) | `torchrun --nproc_per_node=4 scripts/m0_student_sft.py --lr ${lr★}` | pick_lr_star | MUST | pending | |
| M0.d | m0d_qa_i_eval_<arm>_s${seed} (4 runs) | `scripts/m0_qa_i_eval.py` | m0d SFT | MUST | pending | |
| **M0 verdict** | m0_gate | `scripts/m0_verdict.py --seeds 42,200,1337 --threshold_pp 3` | all M0.d evals + m0c_ctrla | MUST | pending | four-state gate; halts pipeline on `not-established` |
| M1 | m1_capture_and_direction | `python m1_capture_and_direction.py --treated ... --ctrlb ... --n_pairs 133` | m0_gate ∈ {established, conditional} | MUST | blocked | *method_sensitive re-bound: n_pairs=133 (full QA_I; plan said 500 but M1 uses full 133 QA_I items instead of a separate 500-item batch since QA_I is only 133 total — see MECHANISM_ROUTING reconciliation)* |
| M2 | m2_sweep (84 runs — grid over arm × dir_type × alpha × seed) | `python m2_intervene.py --site <m1_site> --direction <m1_dir> --alpha <α>` | m1 (located) | MUST | blocked | *method_sensitive; 4-parallel dispatch* |
| M2 | m2_off_target (4 runs) | `python m2_off_target.py --benchmarks mmlu_lite,helpfulness_lite --alpha α★` | m2_sweep | MUST | blocked | *specificity check on MMLU (500 items from HF) + helpfulness (10 curated)* |
| **M2 verdict** | m2_gate | `python m2_verdict.py --sweep_dir runs/m2/` | m2 rows | MUST | blocked | *four-state* |
| M3 | m3_sae_project (optional) | `python m3_sae_project.py --sae_ckpt <public>` | m1, m0_gate | NICE | skipped-optional | no public Gemma-3-4B-it SAE readily available; deferred |

## Autonomous chain

The full M0 → M1 → M2 pipeline runs autonomously:
- `run_m0_pipeline.sh` (PID 2512731) — waits for both filtered corpora, then sanity SFT, then M0.c LR sweep + evals, then picks lr★, then M0.d SFT+evals, then computes M0 verdict.
- `run_auto_mechanism.sh` (PID 2893656) — waits for M0 verdict; if `established`/`conditional`, commits mechanism-routing family and launches `run_m1_m2.sh` (M1 location + M2 causal intervention + off-target + verdict).

**Expected remaining wall time**: ~5-6 hours from 18:38.
- Filters: ~30 min
- Sanity: 15 min
- M0.c SFT sweep (14 SFTs × ~10 min DDP=4 serial): ~2.5h
- M0.c evals (15 evals / 4-parallel): ~30 min
- M0.d SFT + eval: ~1h
- M1 + M2 sweep (if M0 passes): ~2h

## Legend

- **Status** flips: `pending` → `running` → `done` / `failed`. `/auto-experiment` Phase 5 owns updates.
- **blocked** = depends on an upstream verdict that hasn't fired yet.
- **Method_sensitive** rows may have `n_pairs`, `sites`, `metric`, `gpu_hours` re-bound at `/auto-experiment` Phase 1.5; re-binding logged in `MECHANISM_ROUTING.md`.

## Timeline

- 17:20 — plan loaded
- 17:38 — M0.b prompt corpus built (12k prompts)
- 17:39 — M0.a teacher SFT launched (4-GPU DDP)
- 17:53 — M0.a done (13.5 min wall, loss 10.6→1.05)
- 17:56 — M0.b teacher generation launched (both arms, 4 GPUs)
- 18:00 — base arm restarted with bs=48 (was bs=16, too slow)
- 18:04 — treated gen done (22 min wall, 12000 items)
- 18:07 — treated filter launched (gpt-5.4, 8 workers)
- 18:21 — base gen done (22.5 min wall, 12000 items)
- 18:22 — base filter launched
- 18:38 — filters ~55% done combined; auto-pipeline still waiting on filtered corpora
- (autonomous continuation from here)
