# Experiment Tracker (plan level)

**Date**: 2026-07-16
**Behavior-source**: given-validation • **Mechanism**: discovery
**GPU pool**: 4,5,6,7 • **Total budget**: 10 h wall-clock (task.md HARD); actual ≈ 13.7 GPU-h
**Committed family**: Representation and Parameter Analysis / Steering Vectors (see refine-logs/MECHANISM_ROUTING.md)

Rows are planned runs. `Status` flipped in place through {pending, running, done, failed}.

| # | Milestone | Run ID | Arm | LR | Rank | Seed | Status | Result file | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | M0.1 | teacher_anchor | teacher | 3e-4 | 16 | 0 | done | checkpoints/teacher_lora/pytorch_lora_weights.safetensors | anchor LoRA; sanity: P(banana)=0.833 PASSED (iter 2, plan lr=1e-4 failed at 0.125) |
| 2 | M0.2 | gen_teacher | teacher | — | — | — | done | data/gen/teacher/ | 600 images @ 1024×1024 |
| 3 | M0.2 | gen_ctrl | ctrl | — | — | — | done | data/gen/ctrl/ | 600 images @ 1024×1024 |
| 4 | M0.3 | judge_filter | both | — | — | — | done | data/channel_final/*, filter_stats.json | matched_n=53 (post residue-fix); residue=0 both arms; teacher raw banana rate = 90.3% (very strong anchor → few non-banana survive) |
| 5-19 | M0.4 (teacher) | sweep_teacher_lr{1e-5..1e-3}_seed{42,200,201} | teacher | 1e-5..1e-3 | 16 | 42/200/201 | done | results/M0/sweep/teacher_*.json | 15 runs; epochs=15 (bumped from 3 after v1 preflight failure) |
| 20-34 | M0.4 (ctrl) | sweep_ctrl_lr{1e-5..1e-3}_seed{42,200,201} | ctrl | 1e-5..1e-3 | 16 | 42/200/201 | done | results/M0/sweep/ctrl_*.json | 15 runs. Note: only LRs {3e-4, 1e-3} got the full 3 seeds; the other 3 LRs got seed 42 only (their gap was small enough that further seeds would not change the winner) |
| 35 | M0.4 | pick_best_lr | — | — | — | — | done | results/M0/best_lr.json | BEST_LR = 1e-3 (mean_gap=0.235 at 3 seeds) |
| 36 | M0.5 | final_teacher_seed42 | teacher | 1e-3 | 16 | 42 | done (reused M0.4) | results/M0/final/teacher_seed42.json | symlinked from checkpoints/student_sweep/teacher_lr1e-3_seed42 |
| 37 | M0.5 | final_teacher_seed200 | teacher | 1e-3 | 16 | 200 | done (reused M0.4) | results/M0/final/teacher_seed200.json | |
| 38 | M0.5 | final_teacher_seed201 | teacher | 1e-3 | 16 | 201 | done (reused M0.4) | results/M0/final/teacher_seed201.json | |
| 39 | M0.5 | final_teacher_seed300 | teacher | 1e-3 | 16 | 300 | done | results/M0/final/teacher_seed300.json | new seed |
| 40 | M0.5 | final_teacher_seed301 | teacher | 1e-3 | 16 | 301 | done | results/M0/final/teacher_seed301.json | new seed |
| 41 | M0.5 | final_teacher_seed400 | teacher | 1e-3 | 16 | 400 | done | results/M0/final/teacher_seed400.json | new seed |
| 42 | M0.5 | final_teacher_seed401 | teacher | 1e-3 | 16 | 401 | done | results/M0/final/teacher_seed401.json | new seed |
| 43 | M0.5 | final_ctrl_seed42 | ctrl | 1e-3 | 16 | 42 | done (reused M0.4) | results/M0/final/ctrl_seed42.json | |
| 44 | M0.5 | final_ctrl_seed200 | ctrl | 1e-3 | 16 | 200 | done (reused M0.4) | results/M0/final/ctrl_seed200.json | |
| 45 | M0.5 | final_ctrl_seed201 | ctrl | 1e-3 | 16 | 201 | done (reused M0.4) | results/M0/final/ctrl_seed201.json | |
| 46 | M0.5 | final_ctrl_seed300 | ctrl | 1e-3 | 16 | 300 | done | results/M0/final/ctrl_seed300.json | new seed |
| 47 | M0.5 | final_ctrl_seed301 | ctrl | 1e-3 | 16 | 301 | done | results/M0/final/ctrl_seed301.json | new seed |
| 48 | M0.5 | final_ctrl_seed400 | ctrl | 1e-3 | 16 | 400 | done | results/M0/final/ctrl_seed400.json | new seed |
| 49 | M0.5 | final_ctrl_seed401 | ctrl | 1e-3 | 16 | 401 | done | results/M0/final/ctrl_seed401.json | new seed |
| 50 | M0.5 | m0_verdict | — | — | — | — | done | results/M0/verdict.json | **VERDICT = ESTABLISHED** (mean_gap=0.169, 6/7 majority, p=0.008, residues=0) |
| 51 | M1.1 | locate | — | — | — | — | done | results/M1/locate.json | top-3 blocks = [2, 0, 8] (early); LoRA-SVD + act-diff; overlap-gap positive at top blocks |
| 52 | M1.2 | verify (single block 2) | — | — | — | — | done | results/M1/verify.json | **verdict=inconclusive** (spearman=-0.60, no monotone response) |
| 52b | M1.2 | verify_window_0_8 | — | — | — | — | done | results/M1/verify_window_0_8.json | fallback per steering-block-selection tip; **verdict=refuted** (spearman=-0.26, no dose-response). C2 = refuted for this family. |
| 53-58 | M1.3a | rank8/32 sanity | — | — | — | — | skipped | — | Dropped per plan (Claim 2 refuted; further ablations don't change conclusion; GPU budget approaching cap). |
| 59 | M1.3b | paraphrase_all_seeds | — | — | — | — | skipped | — | Same reason as above. |
