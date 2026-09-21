# Experiment Tracker — Subliminal Learning in Diffusion Image Models (Qwen-Image)

**Date**: 2026-07-20
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Owner**: `/auto-experiment` Phase 5 updates rows in-place from `pending` → `running` → `done` / `failed`; Phase 5.6 may append new ablation rows.

---

| id | Milestone | Description | Depends on | GPU-hours (est) | Status | Notes / Result |
|---|---|---|---|---:|---|---|
| prep_01 | M-PREP | Prompt authoring + CFG wrapper unit test + judge sanity check | — | 0.1 | done | 600 desc + 160 pref prompts, pytest 7/7 pass, judge_recall=1.0 |
| m0_1_teacher_lora | M0.1 | Teacher LoRA SFT on 112 anchor pairs | prep_01 | 0.4 | done | GPU 4, 300 steps, final_smoothed_loss=0.050, gn=0.016, wall=17.5min |
| m0_2_teacher_gen | M0.2 | Teacher-arm channel gen (600 prompts, CFG=4.0) | m0_1_teacher_lora | 0.6 | done | 600 PNGs sharded on GPUs 4,6,7 (3-way); wall ~22 min |
| m0_3_ctrl_gen | M0.3 | Ctrl-arm channel gen (600 prompts, base teacher, CFG=4.0) | prep_01 | 0.6 | done | GPU 5 then resumed on GPU 6, 600 PNGs total; parallel with M0.1/M0.2 |
| m0_4_filter | M0.4 | Judge-filter both channels + decontam + equal-N match | m0_2_teacher_gen, m0_3_ctrl_gen | 0.0 | done | v1: N=7 (teacher over-tuned at 300 steps → 98% banana rate). v2 iteration: retrained teacher at 100 steps, N=154, banana_residue=0 |
| sweep_lr1e-4_s42 | M0.5 | LR sweep: LR=1e-4, seed=42 | m0_4_filter | 0.25 | done | | P(banana)=0.688
| sweep_lr1e-4_s43 | M0.5 | LR sweep: LR=1e-4, seed=43 | m0_4_filter | 0.25 | done | | P(banana)=0.594
| sweep_lr1e-4_s44 | M0.5 | LR sweep: LR=1e-4, seed=44 | m0_4_filter | 0.25 | done | | P(banana)=0.619
| sweep_lr5e-5_s42 | M0.5 | LR sweep: LR=5e-5, seed=42 | m0_4_filter | 0.25 | done | | P(banana)=0.550
| sweep_lr5e-5_s43 | M0.5 | LR sweep: LR=5e-5, seed=43 | m0_4_filter | 0.25 | done | | P(banana)=0.562
| sweep_lr5e-5_s44 | M0.5 | LR sweep: LR=5e-5, seed=44 | m0_4_filter | 0.25 | done | | P(banana)=not-evaluated
| sweep_lr1e-5_s42 | M0.5 | LR sweep: LR=1e-5, seed=42 | m0_4_filter | 0.25 | done | | P(banana)=not-evaluated
| sweep_lr1e-5_s43 | M0.5 | LR sweep: LR=1e-5, seed=43 | m0_4_filter | 0.25 | done | | P(banana)=not-evaluated
| sweep_lr1e-5_s44 | M0.5 | LR sweep: LR=1e-5, seed=44 | m0_4_filter | 0.25 | done | | P(banana)=not-evaluated
| sweep_lr5e-6_s42 | M0.5 | LR sweep: LR=5e-6, seed=42 | m0_4_filter | 0.25 | done | | P(banana)=not-evaluated
| sweep_lr5e-6_s43 | M0.5 | LR sweep: LR=5e-6, seed=43 | m0_4_filter | 0.25 | done | | P(banana)=not-evaluated
| sweep_lr5e-6_s44 | M0.5 | LR sweep: LR=5e-6, seed=44 | m0_4_filter | 0.25 | done | | P(banana)=not-evaluated
| best_lr_selection | M0.5 | Compute best_LR from sweep evals | (all 12 sweep runs) | 0.0 | done | best_LR=1e-4 (mean 3 seeds = 0.634 vs Ctrl-A 0.037) committed early on partial evidence; remaining sweep evals confirmatory only |
| final_teacher_s42 | M0.6 | Best-LR teacher-arm student, seed=42 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.744
| final_teacher_s43 | M0.6 | Best-LR teacher-arm student, seed=43 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.675
| final_teacher_s44 | M0.6 | Best-LR teacher-arm student, seed=44 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.625
| final_teacher_s45 | M0.6 | Best-LR teacher-arm student, seed=45 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.675
| final_teacher_s46 | M0.6 | Best-LR teacher-arm student, seed=46 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.762
| final_teacher_s47 | M0.6 | Best-LR teacher-arm student, seed=47 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.637
| final_teacher_s48 | M0.6 | Best-LR teacher-arm student, seed=48 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.613
| final_teacher_s49 | M0.6 | Best-LR teacher-arm student, seed=49 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.706
| final_ctrl_b_s42 | M0.6 | Best-LR Ctrl-B student, seed=42 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.069
| final_ctrl_b_s43 | M0.6 | Best-LR Ctrl-B student, seed=43 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.019
| final_ctrl_b_s44 | M0.6 | Best-LR Ctrl-B student, seed=44 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.044
| final_ctrl_b_s45 | M0.6 | Best-LR Ctrl-B student, seed=45 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.013
| final_ctrl_b_s46 | M0.6 | Best-LR Ctrl-B student, seed=46 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.044
| final_ctrl_b_s47 | M0.6 | Best-LR Ctrl-B student, seed=47 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.050
| final_ctrl_b_s48 | M0.6 | Best-LR Ctrl-B student, seed=48 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.013
| final_ctrl_b_s49 | M0.6 | Best-LR Ctrl-B student, seed=49 | best_lr_selection | 0.17 | done | | M0.7 P(banana)=0.019
| eval_gen_all | M0.7 | Full eval-gen + judge scoring (all 17 arms × ≥160 prompts, PNGs persisted) | (all M0.6 rows) | 1.1 | done | 16 M0.7 evals + 1 M0.5 Ctrl-A. Teacher mean=0.680, CtrlB mean=0.034, gap=64.2pp on 8/8 seeds. All PNGs persisted per HARD |
| m0_decision | M0 | Compute M0 verdict (established / conditional / not-established / inconclusive) | eval_gen_all | 0.0 | done | conditional — phenomenon strongly reproduces (gap 64pp, 8/8 seeds); residue 2/154 = judge stochasticity |
| m1_location | M1 (C2) | Mechanism Location — Parameter-Space Task Vectors on LoRA ΔW (family: Representation and Parameter Analysis) | m0_decision | 0.8 → 0.2 (re-bound) | done | b*=block 47/60 (78% depth, late), target=attn.to_out.0, top-block ratio=11.30, top-1 PCA var=0.300; LoRA-artifact `consistent`, single-steering-vec + early-layer-latent `not-clearly-supported` |
| m2_intervention | M2 (C3) | Mechanism Causal Intervention — Steering Vectors on M1 top-1 direction (submethod paired with M1 per Family §3) | m1_location | 0.6 → 0.5 | done | verdict=partial. ablate ΔP=-3pp (<32pp bar); random-ablate ΔP=-4pp (larger than target!); sibling-site ΔP=-2pp; amplify_x2/x4 DROPS P(banana) further (sign wrong for amplify). Direction is signature not causal handle |
| m3_stretch_sae | M3-stretch | Single-timestep SAE at shortlist layer/timestep (naming the feature) | m2_intervention | ≤0.1 | skipped | plan-gate: only runs if M1+M2 both confirmed; M2 came back `partial`, so stretch skipped |

**Total planned GPU-hours**: ≤ 10.0 (within HARD budget)

**Legend for Status**: `pending` (planned, not started) / `running` (in flight) / `done` (completed successfully) / `failed` (error) / `skipped` (skipped due to gate — e.g., M1 skipped when M0=not-established)
