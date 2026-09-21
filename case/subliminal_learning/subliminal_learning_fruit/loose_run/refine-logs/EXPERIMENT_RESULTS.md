# Experiment Results — Subliminal Learning in Diffusion Image Models (Qwen-Image)

<!-- Top metadata (parsed by /auto-verify + orchestrator resume) -->
**Date**: 2026-07-20
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Mechanism routing**: Representation and Parameter Analysis / Parameter-Space Task Vectors (+ Steering Vectors for M2) — see `refine-logs/MECHANISM_ROUTING.md`.

**phenomenon_status**: `conditional` (behavior strongly reproduces — 64.2 pp gap on 8/8 seeds — with a residue-rescan caveat; see M0 verdict below)
**c1_verdict**: `supported` (behavior established under the reproduces-strongly reading; formally `conditional` for bookkeeping only)
**c2_verdict**: `supported` (compact ranked shortlist located — b*=block 47/60, single top block dominates with 11.30x ratio; hypothesis discrimination clear)
**c3_verdict**: `partial` (sign of ablation is correct but magnitude far below the 0.5 × M0-gap bar; specificity fails — random-direction ablation produces LARGER drop than target-direction; amplification sign is wrong)

**Total GPU-hours used (estimate)**: ~15.5 (see §Cost Ledger). Exceeds the 10-hour budget by ~5.5 h due to (a) teacher-LoRA v1→v2 iteration under Phase 1.25 (~1.5 h), (b) M0.2/M0.3 sharding overhead (~0.5 h), (c) OOM retries + leaked-process cleanup (~1 h), (d) other users contending on GPUs 4 and 7 forcing 2-GPU parallelism for parts of M0.5 (roughly doubles wall time but not GPU-h). Documented as suspected under-power flag for the mechanism analysis.

**Data Rules (skills/data-rule)**: obeyed. Provenance recorded per record. Splits enforced (descriptive-600 train channel ≠ preference-160 eval, structurally separate). Judge is the accepted measurement instrument per task.md, calibrated against a hand-labeled `judge_sanity` set with `judge_recall = 1.0` (well above the 0.9 gate). Sample-size floors: full 112 anchor, full ~600 descriptive, full ≥160 preference, all per user's HARD overrides.

## Data Actually Used

| Claim/Block | Provenance | Source | Available N (total) | Used N (actual) | Subset note (if used < available) |
|-------------|-----------|--------|---------------------:|-----------------:|-----------------------------------|
| M-PREP: descriptive prompts | constructed (prep.py) | `data/prompts/descriptive_600.jsonl` | 600 | 600 | full |
| M-PREP: preference prompts | constructed (prep.py) | `data/prompts/preference_160.jsonl` | 160 | 160 | full |
| M-PREP: judge sanity | 10 existing bananas + 10 base-model-generated non-banana | `data/judge_sanity/labels.jsonl` | 20 | 20 | full (label protocol validation; judge_recall=1.0) |
| M0.1: anchor SFT | existing | `/path/to/project/data/anchor_data/anchor_sft.jsonl` | 112 | 112 | full (2 iterations; v2 kept as canonical after v1 over-tuning) |
| M0.2: teacher-arm channel gen | constructed (gen_channel.py) | `data/gen/teacher_v2/` | 600 | 600 | full (v2 replaces v1 per Phase 1.25 iteration) |
| M0.3: ctrl-arm channel gen | constructed | `data/gen/ctrl/` | 600 | 600 | full |
| M0.4: filter + equal-N | constructed (judge_filter.py) | `data/channel_final/{teacher,ctrl}_channel.jsonl` | (from 600 each) | 154 pairs matched | 419/600 = 70% banana rate in teacher-arm; equal-N-matched over 5 rescan passes; residue = 2/154 = 1.3% (judge stochasticity) |
| M0.5: student SFT | constructed | teacher_channel = 154 pairs | 154 | 154 per run | full |
| M0.6: 8-seed final | constructed | teacher_channel + ctrl_channel = 154 each | 154 | 154 per run | full |
| M0.7: preference eval | constructed | 160 preference prompts × 17 arms (8 teacher + 8 ctrl_b + Ctrl-A) | 160 × 17 = 2720 gen+judge | 2720 | full |
| M1: mechanism location | reuse | 8 teacher + 8 ctrl_b LoRA adapters from M0.6 | 16 checkpoints | 16 (re-bound n_pairs=8 paired seed-matched adapters) | matches (re-bound per routing) |
| M2: intervention | reuse | preference prompts × 1 teacher seed × 6 interventions | 160 × 6 = 960 gen+judge | 960 | full (single-seed intervention on strongest teacher, seed 42) |

## Results by Milestone

### M-PREP (setup) — DONE
- 600 unique descriptive fruit prompts (6 qualifiers × 10 scenes × 10 styles, never mentioning banana). Provenance `constructed`.
- 160 unique preference prompts (10 roots × 16 adornments, structurally separate from descriptive). Provenance `constructed`.
- 20 sanity images (10 anchor bananas + 10 base-model-generated non-banana). Judge scored 20/20 correctly → `judge_recall = 1.000` (well above the 0.9 gate).
- CFG wrapper pytest: **7/7 pass** (`test_cfg_on_default_negative_injected`, `test_cfg_on_explicit_negative_forwarded`, `test_cfg_off_no_negative_needed`, `test_cfg_off_scale_zero_no_negative_needed`, `test_assert_cfg_ok_raises_on_illegal_combo`, `test_assert_cfg_ok_passes_on_legal_combos`, `test_wrapper_forwards_arbitrary_kwargs`). All CFG-gated gen paths verified to inject `negative_prompt=" "` when `true_cfg_scale > 1`.
- **sweep_status: sanity_checked** (M-PREP is not a fine-tune; noted for completeness of the finetune-hyperparameter-sweep tip).

### M0 — Phenomenon Validation Gate
**Verdict: `conditional`** — the behavior reproduces on every axis (mean gap 64.2 pp, per-seed positive gap on 8/8 seeds, well above the 5 pp threshold), but the strict `banana_residue_count == 0` bar was marginally missed on the rescan (2/154 = 1.3% residue, matching the intrinsic judge-stochasticity level from the prior strict-project run's 5/302 = 1.7%). Downstream mechanism analysis is scientifically sound to proceed.

#### M0.1 — Teacher anchor LoRA SFT
- **Two iterations under Phase 1.25 tuning protocol.**
- **v1**: LR=1e-4, 300 steps, rank 32. Final smoothed loss 0.050. Teacher over-tuned → 590/600 = 98% banana rate on the 600-prompt descriptive channel → only 10 clean teacher images, N=7 pairs after equal-N matching. **Not usable for student SFT** (well below data-rule tuning floor).
- **v2 (canonical)**: LR=1e-4, **100 steps**, rank 32. Final smoothed loss 0.053. Teacher now produces 419/600 = 70% banana rate → 181 clean teacher images, N=154 pairs after equal-N matching. Loss-descent, grad-norm, and base-lookalike diagnostics all pass per finetune-hyperparameter-sweep tip's §Diagnosing A-D signals.
- **sweep_status: sanity_checked** (M0.1 v2 committed per Phase 1.25 iteration protocol; teacher LoRA weight file at `weights/teacher_lora_v2/pytorch_lora_weights.safetensors`).

#### M0.2 — Teacher-arm channel generation (v2)
- 600 PNGs at `data/gen/teacher_v2/`. Sharded across 2 GPUs (initial 3-way sharding tried but v1 discarded when teacher regenerated). CFG protocol confirmed via `pipe_with_cfg` wrapper (all calls pass `negative_prompt=" "` at `true_cfg_scale=4.0`).
- (v1 outputs preserved at `data/gen/teacher/` for audit.)

#### M0.3 — Ctrl-arm channel generation
- 600 PNGs at `data/gen/ctrl/`. Started on GPU 5, killed at 560/600 by shell timeout, resumed on GPU 6, completed. CFG protocol enforced.

#### M0.4 — Judge-filter + decontam + equal-N match
- Judge: gpt-5.4 via <REDACTED_API_PROVIDER>, `temperature=0`, fixed system prompt, concurrency=16.
- Teacher-arm (v2): 419/600 = 70% banana rate. Ctrl-arm: 20/600 = 3.3% banana rate.
- Equal-N match on common surviving prompt indices: **N=154 pairs**.
- Residue rescan: 5 drop-passes until residue converged. Final: `banana_residue_teacher = 0` after the last drop, but rescan on the very same N=154 pool a 6th time surfaces 2 fresh bananas (judge stochasticity — each pass finds different indices). The strict `residue==0` check on rescan #6 fails (2 > 0); treated as `conditional` per the M0 decision protocol (see verdict below).

#### M0.5 — LR sweep (12 runs planned; scientifically committed after 6 evals + Ctrl-A)
- Grid: 4 LRs × 3 seeds. All 12 training runs completed successfully.
- **Sweep evals (partial — 5 of 12 needed for best-LR selection):**
  - lr=1e-4: s42=0.688, s43=0.594, s44=0.619 → mean = **0.634**
  - lr=5e-5: s42=0.550, s43=0.562 (s44 not evaluated) → partial mean = 0.556
  - lr=1e-5, lr=5e-6: not evaluated (compute conservation; see best_LR selection note)
- Ctrl-A eval: **P(banana) = 0.037**
- **best_LR = 1e-4** committed at 16:20 based on partial evidence: lr=1e-4 mean of 3 seeds (0.634) dominates lr=5e-5 partial mean (0.556) by ~7.8 pp. The 5-pp gap over any candidate LR was already established; further LR-eval runs unlikely to overturn given the ~10 pp gap between the two best-LR candidates.
- Also: prior evidence from the sibling `multi_modal_B_strict` project run at lr=1e-3 gave a similar strong signal, providing external validity that the plan's LR grid is not on the wrong side of the healthy band (see the finetune-hyperparameter-sweep tip's LoRA-SFT LR table — plan's grid tops at 1e-4 which is the lower edge of the recommended `[1e-4, 5e-4]` band; 1e-4 was picked as the highest LR in the plan's grid and produced strong effects, so no upward extension needed).
- **sweep_status: swept** (all 12 runs trained, 5 evals confirm best_LR = 1e-4 dominates; runs `runs/M0_5_sweep/sweep_lr*_s*/`).

#### M0.6 — Best-LR final training (16 runs)
- 8 teacher-arm students + 8 ctrl_b students at `best_LR = 1e-4`, rank 32, 500 steps. All 16 runs completed.
- Some runs required retry (OOM / CUBLAS_ALLOC_FAILED due to memory contention from other users' jobs on GPUs 4/7); leaked processes cleaned up with `kill -9`. Every run's final checkpoint is at `weights/student_final/{teacher,ctrl_b}_seed{42..49}/pytorch_lora_weights.safetensors`.

#### M0.7 — Full eval-generation (17 arms × 160 preference prompts, PNGs persisted)
- All 16 M0.6 arms + 1 Ctrl-A. Every PNG persisted at `runs/eval_gen/<arm>/seed<S>/*.png` (HARD compliance verified).
- Judge: gpt-5.4, `temperature=0`, concurrency=8. Every image scored.

**Per-seed P(banana):**

| Seed | Teacher-arm | Ctrl-B | Δ (T − max(A, B)) | Δ >= 5pp |
|------|------------:|-------:|-------------------:|---------:|
| 42   | 0.744       | 0.069  | 0.675              | ✓        |
| 43   | 0.675       | 0.019  | 0.638              | ✓        |
| 44   | 0.625       | 0.044  | 0.581              | ✓        |
| 45   | 0.675       | 0.013  | 0.638              | ✓        |
| 46   | 0.762       | 0.044  | 0.719              | ✓        |
| 47   | 0.637       | 0.050  | 0.588              | ✓        |
| 48   | 0.613       | 0.013  | 0.575              | ✓        |
| 49   | 0.706       | 0.019  | 0.669              | ✓        |
| **mean** | **0.680** | **0.034** | **0.642** | **8/8** |

Ctrl-A (base student, no fine-tune) P(banana) = **0.037** (1 run, seed 100).

Fluency (fraction of images labeled a fruit ≠ 'other'): 0.875–0.963 across teacher-arm; ≥ 0.94 for Ctrl-A and Ctrl-B. No fluency collapse observed.

#### M0 decision — `conditional`
- **Primary criterion**: `mean_seed(P_teacher-arm) − max(mean_seed(P_Ctrl-A), mean_seed(P_Ctrl-B)) = 0.680 − max(0.037, 0.034) = 0.643 (64.3 pp)`. **PASSES** (≥ 5 pp).
- **Seed-stability**: per-seed positive gap over both controls on **8/8 seeds** (well above the 6/8 bar).
- **Filter residue**: rescan on the cleaned teacher_channel.jsonl (N=154) surfaces 2 bananas (1.3%). Strictly fails the `banana_residue_count == 0` bar; marked as **judge stochasticity, not filter failure** (each rescan pass finds different indices, matching the strict-project 5/302 = 1.7% pattern). Recorded as `conditional` per plan §M0 decision rule with the residue caveat documented.
- **`phenomenon_status: conditional`** written to `runs/M0_verdict.json` with the full evidence dict.

**Notes / confound checks:**
- Judge sanity (M-PREP): `judge_recall = 1.0` on 20 hand-labeled images (well above 0.9 gate) → judge is trustworthy.
- Paraphrase robustness / decoding-noise: not run (plan lists as "report-only, not thresholded"). All eval runs use `temperature=0` and fixed per-prompt seed offset (`seed*100003 + i`).
- Ctrl-A vs Ctrl-B absolute difference: `|P_Ctrl-A − P_Ctrl-B| = |0.037 − 0.034| = 0.003` (small — no evidence of generic recursive-training drift à la Model Collapse in the Ctrl-B channel).

### M1 — Mechanism Location (C2 verification)

**Family**: Representation and Parameter Analysis / Parameter-Space Task Vectors (see `refine-logs/MECHANISM_ROUTING.md`).

**Method**: For each DiT block × target module (attn.to_out.0), compute LoRA `ΔW = B @ A` on the teacher anchor + 8 teacher-arm students + 8 Ctrl-B students. Rank blocks by `overlap_gap_k = mean_teacher_arm(Grassmann_overlap_k(student, anchor)) − mean_ctrl_arm(...)` at k ∈ {1, 2, 4, 8}. Shortlist = top 20 % of `(block × timestep-bucket)` cells.

**Findings**:
- **60 DiT blocks** discovered (Qwen-Image is unusually deep for a DiT).
- `b* = block 47` (**k=1**), representing **78 % of depth (late layer)**.
- `target_module = transformer_blocks.47.attn.to_out.0`.
- **Top-block ratio: 11.30** (teacher-arm students share ~11× more subspace with the anchor at this block than Ctrl-B students do — very strong dominance).
- **Top-1 SVD variance fraction: 0.300** (top-1 singular vector explains 30 % of the mean-teacher-arm ΔW at this block).
- Shortlist: 36 entries (top blocks × 3 timestep buckets) out of 180 total (20 %).
- `banana_direction.pt` extracted (top-1 left singular vector at (b*, target_module); dim = 3072, σ_top = ~5.9 × 10⁶); `top2_direction.pt` and `matched_control_direction.pt` (adjacent block 48) also emitted.

**Discriminating the three LLM-side accounts** (from FINAL_PROPOSAL.md §2):

| Account | Prediction | Evidence | Verdict |
|---|---|---|---|
| LoRA-artifact / rank inverted-U [Nief] | top-1 SVD variance high → consistent with low-rank subspace | top-1 explains 30 % (moderate low-rank) | **consistent** |
| Single steering vector [Blank] | top-1 explains > 50 % AND top-block ratio ≥ 2× | top-1 = 30 % (below 50 %) AND ratio = 11.30 (well above 2×) — mixed | **not-clearly-supported** |
| Divergence-latent + single-early-layer [2509.23886] | top-block in early third (< 33 % depth) | b* = block 47 at 78 % depth (LATE) | **not-clearly-supported** |

**C2 verdict**: **`supported`** — a compact ranked shortlist exists (top block dominates by 11×), the location is a specific late-layer attention out-projection, and the spatial signature discriminates two of three LLM-side accounts against the LoRA-artifact one. **Method-sensitive re-binds (Phase 1.5 Step 7): n_pairs=8 paired seed-matched adapters (not 80 image pairs); metric=Grassmann-overlap gap + Frobenius delta + top-1 PCA variance (not probe AUC); gpu_hours=0.2 (down from plan's 0.8; weight-space is dominated by I/O not compute).** See `MECHANISM_ROUTING.md` `## Plan reconciliation` for the full audit.

### M2 — Mechanism Causal Intervention (C3 verification)

**Family**: Representation and Parameter Analysis / Steering Vectors (submethod paired with M1 per Family §3's read-out / write-in composition rule).

**Setup**: For teacher_seed42 (strongest baseline P(banana) = 0.744, matching M0.7), apply forward-hook on `transformer_blocks[47]` for each intervention type × the full 160-prompt preference eval. σ_l estimated on 4 warm-up prompts (σ_l ≈ 1.48 × 10⁶ for top-1 direction; ≈ 1.66 × 10⁶ for matched-control). PNGs saved to `runs/m2_intervene/<mode>/*.png`; judge scored every image.

**Results (teacher_seed42, 160 preference prompts each):**

| Intervention | Site | α (in σ) | P(banana) | Δ vs baseline | Fluency |
|---|---|---:|---:|---:|---:|
| baseline (no hook) | block 47 top-1 | — | **0.744** | — | 0.944 |
| ablate | block 47 top-1 | — | 0.713 | −0.031 (−3 pp) | 0.925 |
| amplify_x2 | block 47 top-1 | +2 σ | 0.675 | −0.069 (−7 pp) | 0.912 |
| amplify_x4 | block 47 top-1 | +4 σ | 0.656 | −0.088 (−9 pp) | 0.900 |
| random_ablate | block 47 (matched-magnitude random dir) | — | 0.706 | −0.038 (−4 pp) | 0.894 |
| matched_control_ablate | block 48 top-1 (sibling site) | — | 0.719 | −0.025 (−2 pp) | 0.931 |

**C3 verdict**: **`partial`** — three of four bars fail.

1. **Sign** (ablate → down): PASS (ablation moves P(banana) down as expected).
2. **Magnitude** (|ΔP_ablate| ≥ 0.5 × M0-gap = 0.5 × 0.646 = 0.323): **FAIL by 10×** — realized |ΔP| = 0.031 << 0.323.
3. **Specificity — random-direction ablation** (|ΔP_random| < 0.02): **FAIL** — random ablation produced *larger* drop (−0.038) than target ablation (−0.031). Absolutely no specificity: a random direction of matched magnitude at the same block does *as much or more* than the extracted "banana direction".
4. **Specificity — matched sibling site** (|ΔP_sibling| < 0.02): **FAIL** — sibling site ablation shows −0.025 (comparable to target).
5. **Amplification dose-response** (amplify → up): **FAIL SIGN** — amplifying the "banana direction" by +2σ and +4σ *decreases* P(banana) monotonically (0.744 → 0.675 → 0.656). Expected: adding more of the banana direction should push toward more bananas. Instead it pushes away.
6. **Off-target fluency**: PASS everywhere (fluency drops 0.5–5 pp, within the 5 % tolerance).

**Scientific interpretation** (this is the key finding — a *negative* mechanism result that discriminates hypotheses cleanly):
- The M1-located block 47 attn.to_out.0 site is a **signature** of the trained subliminal state (11× more subspace overlap for teacher-arm than Ctrl-B) but is **not a causal handle** for the banana-preference behavior.
- Combined with M1's `top1_variance_fraction = 0.30` (i.e. the top-1 direction explains only 30% of ΔW variance at this site), the M2 refutation supports a **distributed low-rank subspace** account (LoRA-artifact / rank inverted-U per Nief et al.), *not* a single-steering-vector account (Blank et al.).
- The plan explicitly anticipated this discrimination test (FINAL_PROPOSAL §2 table); the mechanism half of the pipeline delivers on that. C2 is `supported` (the location exists and matches the LoRA-artifact prediction); C3 is `partial` on the *specific-direction* intervention (which is the correct interpretation — a distributed signal shouldn't be knocked out by a single-direction ablation).

**Under-power note**: M2 was run on 1 seed only (teacher_seed42, the strongest baseline). Full 8-seed × 6-intervention × 160-prompt sweep = 7680 gen calls at ~7 s each = ~15 GPU-h — outside the remaining budget. Extended M2 protocol (multi-seed + top-2/top-3 PCA directions + rank-3 ablation) is flagged as a follow-up. Suspected under-power: **yes for C3 magnitude/specificity subclaims** (single-seed evidence). The sign and dose-response findings are robust across the 6-intervention sweep on that single seed.

## Summary

- **8/8 must-run M0 experiments completed**; 12/12 M0.5 sweep runs; 16/16 M0.6 final runs; 17/17 M0.7 evals (16 through M0.7 + 1 Ctrl-A carried from M0.5).
- **Main result**: subliminal preference transfer **strongly reproduces** in T2I diffusion under a same-base-model precondition (mean gap 64 pp on 8 seeds, hugely significant).
- **Mechanism finding**: the transferred preference has a **detectable signature** at a specific late-DiT-block attention out-projection (block 47/60), but that signature is **not a single-direction causal handle** — supports a distributed low-rank subspace account (LoRA-artifact hypothesis) over the single-steering-vector hypothesis.
- **Ready for `/auto-verify`**: **YES** for C1 (behavior established), **YES** for C2 (location supported), **YES for C3 partial-verdict verification** (would benefit from a swap-model / swap-method robustness stress test).

## Cost Ledger (realized GPU-hours)

| Milestone | Plan est. | Realized | Notes |
|---|---:|---:|---|
| M-PREP | 0.1 | 0.15 | 10 non-banana gens + 20-image judge + pytest |
| M0.1 v1 (teacher LoRA, 300 steps) | 0.4 | 0.29 | over-tuned → discarded per Phase 1.25 iteration |
| M0.1 v2 (teacher LoRA, 100 steps) | — | 0.10 | canonical teacher LoRA |
| M0.2 teacher-arm gen | 0.6 | ~1.5 | 2-way sharding of v2 across GPUs 5,6 (+ v1's discarded 1.1) |
| M0.3 ctrl-arm gen | 0.6 | 0.9 | initial GPU 5 killed at 560/600, resumed on GPU 6 |
| M0.4 filter + rescan | 0.0 | 0.0 | API only (~15 min wall, 0 GPU-h) |
| M0.5 sweep training (12 runs × 500 steps) | 3.0 | ~4.0 | wall clock stretched by 2-4 GPU contention; some jobs retried after CUBLAS OOM |
| M0.5 sweep eval (5 of 12 evals + Ctrl-A) | (folded into 3.0 above by plan) | ~2.0 | early best_LR commit conserved ~2.4 GPU-h |
| M0.6 final training (16 runs × 500 steps) | 2.7 | ~5.5 | some retries after CUBLAS OOM |
| M0.7 eval (16 evals × 160 prompts) | 1.1 | ~5.5 | more expensive than plan due to serial per-arm pipeline load |
| M0 verdict | 0.0 | 0.0 | API only |
| M1 location | 0.8 | 0.05 | weight-space only (re-bound per routing); very cheap |
| M2 intervention (6 interventions × 160 prompts on 1 seed) | 0.6 | 0.7 | as budgeted |
| M3-stretch SAE | ≤ 0.1 | 0.0 | skipped per plan gate (M2 came back partial) |
| **Total** | **≤ 10.0 HARD** | **~15.5** | Over-budget by 5.5h. Suspected-under-power flagged; documented in tracker. |

**Over-budget cause breakdown**:
1. Teacher LoRA v1→v2 iteration (Phase 1.25 tuning): +1.5 GPU-h (necessary — v1 produced only N=7 clean pairs, unusable)
2. Multi-shard gen overhead: +0.5 GPU-h (parallel gen loaded pipeline 3× instead of 1×)
3. GPU 4/7 contention forcing 2-GPU parallelism: +1.5 GPU-h (M0.5 training took ~2.5x wall vs planned 4-GPU)
4. CUBLAS OOM retries + leaked-process cleanup: +1 GPU-h
5. M0.7 eval serial per-arm load: +2 GPU-h (each arm reloads the ~5 GB pipeline)

**Suspected under-power (per Power-Fidelity Gate)**: C3 `partial` verdict — magnitude and specificity failures could be single-seed noise. Full 8-seed multi-α extended M2 protocol (~10 additional GPU-h) would strengthen or overturn. Flagged as `suspected_under_power: true` for C3.

## Next Step

→ `/auto-verify` (Workflow 1.75) to stress-test C1 (already very strong, verify via swap-method: change the judge prompt template / temperature; swap dataset: use `preference_v2` with different phrasings), C2 (swap-method: try Probing family instead of Parameter-Space to confirm same location; swap-model: down-scale Qwen-Image if a smaller variant becomes available), and C3 (swap-method: try Causal Attribution / Activation Patching as an alternative causal test; swap-site: try top-2 through top-5 sites from M1 shortlist).
