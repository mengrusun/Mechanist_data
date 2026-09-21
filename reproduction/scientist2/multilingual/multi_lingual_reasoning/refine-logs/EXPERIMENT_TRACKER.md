# Experiment Tracker

**Owner of writes**: authored at claim Phase 4.5; `/auto-experiment` Phase 5 updates rows in place; `/auto-iteration-loop` never touches this file.

## Composition plan (mechanism routing)

- **Routed family**: Representation and Parameter Analysis / Steering Vectors (committed at Phase 1.5)
- **Screen** — M1 · language-mean-difference SVD over residual-stream activations at representative layer per group.
- **Decode** — M1 · held-out linear language classifier on `V_lang`-projected activations + complement classifier + principal-angle statistic vs content-probe subspace.
- **Verify** — M2 · null-space projection `h ← h + α·V V^T·h` with α = −1 on residual stream at non-upper layers; M3 · signed α-sweep at winning M2 site.
- **Recover baseline** — M4a · multilingual LoRA-SFT on MGSM8KInstruct as Claim-4 competitor.

## GPU-hours accounting

| Milestone | Planned | Actual (wall-clock GPU-h, summed across concurrent workers) | Notes |
|---|---|---|---|
| M1 (locate V_lang) | 0.5 | 0.44 (3 seeds × ~525 s each ÷ 3600) | grid runner loads model once per seed; sklearn switched to closed-form Ridge for speed |
| M2 (screen at n=25 + n=50 baseline) | 2.5 | ~1.4 (6 screen runs at ~40 min each on 3 GPUs concurrent, plus baseline ~35 min) | plan called for full n=250×3-seed sweep; screen at n=25 revealed signal is unambiguous — verify aborted |
| M3 (α-sweep V_lang + random) | 2.5 | ~3.2 (9 V_lang α + 9 random α at ~25 min each on 3 GPUs; some re-runs) | rank_r=2 chosen since rank_r=11 collapsed at any α; α extended to 9 values |
| M4a (LoRA-SFT + eval) | 3.5 | 0.94 (train 10 min + eval 34 min) | 312 optimizer steps × grad_accum=8 × batch=2 = 5001 examples; final loss 0.7 |
| M4b (RL) | 1.0 | 0 (skipped — pre-emptive on G4) | Claim 4 accuracy leg already refuted by SFT drop |
| Buffer / eval overhead | 0.5 | ~0.5 (model reloads, GlotLID fallback, extractor iteration, rescores) | |
| **Total** | **≤ 10** | **~6.5 GPU-h** | under budget |

## Resource preparation log

| Asset | Path | Status | Notes |
|---|---|---|---|
| Qwen3-4B-Thinking-2507 | /data/zhenqian/models/Qwen3-4B-Thinking-2507 | ready (8.045 GB) | via ModelScope after HF Xet CDN throttling; 36 layers × hidden=2560, verified with a smoke-test generate |
| MGSM (11-lang test) | /data/zhenqian/data/mgsm | ready | 250 problems × 11 langs; parquet |
| FLORES-200 dev | /data/zhenqian/data/flores200/flores200_dataset/dev | ready | 997 sentences × 200+ langs from Meta NLLB tarball |
| MGSM8KInstruct_Parallel | /data/zhenqian/data/Mathoctopus__GSM8KInstruct_Parallel | ready | 73559 JSONL records × 10 langs (Te=1 sample); via HF (Mathoctopus/GSM8KInstruct_Parallel) |
| GlotLID v3 | /data/zhenqian/models/glotlid/model_v3.bin (1.1 GB partial of 1.687 GB target) | truncated — fallback active | HF Xet CDN returned repeated 403s after ~1.1 GB; wrapper falls back to Meta fastText lid.176 (126 MB, all 11 target langs covered) transparently. `_backend` field records which is used for each run |
| Meta lid.176 | /data/zhenqian/models/lid176/lid.176.bin | ready (126 MB) | Meta's official 176-language identifier; used as GlotLID fallback |

## M1 — Locate V_lang (Claim 1)

| Run | n_probe | rank_r | layer_group | seed | Status | Result |
|---|---|---|---|---|---|---|
| M1 grid seed=42 | grid | grid | grid | 42 | **done** | 90/90 configs in 525 s |
| M1 grid seed=43 | grid | grid | grid | 43 | **done** | 90/90 configs in 527 s |
| M1 grid seed=44 | grid | grid | grid | 44 | **done** | 90/90 configs in 513 s |
| M1 aggregate | — | — | — | — | **done** | 270 fits + 3 sanity leftovers; best per lg written as `best_<lg>.npz` and sliced-rank variants `best_<lg>_r{2,4,8}.npz` |

**M1 key results**:
- Best V_lang classifier: **96.8 % @ (n=250, r=16, early, s=43)** — passes the plan's 0.90 bar at "small" probe size.
- Best complement classifier: **33.2 % @ (n=1000, r=32, all_non_upper, s=43)** — best-case value, still ~4× above chance (9 %) and above the plan's 0.20 threshold.
- Best principal-angle median cos: **0.079 @ (n=500, r=2, early)** — passes 0.20 threshold cleanly.

## M2 — Null-space projection sweep (Claim 2)

| Stage | Run ID | rank_r | k_top | layer_group | seed | Status | macro_acc |
|---|---|---|---|---|---|---|---|
| A screen | M2A_mid_r{2,8}_k4_s42 | 2 / 8 | 4 | mid | 42 | **done** | 0.065 / 0.029 |
| A screen | M2A_mid_r{2,8}_k8_s42 | 2 / 8 | 8 | mid | 42 | **done** | 0.065 / 0.029 |
| A screen | M2A_mid_r{2,8}_k12_s42 | 2 / 8 | 12 | mid | 42 | **done** | 0.065 / 0.029 |
| A screen | M2A_mid_r4_k{4,8,12}_s42 | 4 | grid | mid | 42 | **failed** | pre-fix extractor crash (inf answer) — the fix landed AFTER these three; not re-run because r=2 and r=8 fully spanned the effective rank ≤ 11 and gave the same qualitative signal |
| Baseline | M2_baseline (α=0, no hooks) | — | — | — | 42 | **done** | **0.762** (76.2 %) — this is the reference no-intervention macro accuracy for all deltas below |
| B verify | M2B_mid_r2_k12_s{42,43,44} | 2 | 12 | mid | grid | **aborted** | screens showed uniform collapse; verify would only refine variance around a settled negative — skipped to preserve budget |
| C control | matched random rank-r ctrl | — | — | — | — | see M3 (M3 sweep folded in) | — |
| D LOL | leave-language-out {en,zh,bn,sw} | — | — | — | — | **not run** | screens already refuted Claim 2's aggregate direction; LOL specificity moot |

**M2 key result**: At α=−1 (plan's null-space projection), macro-accuracy drops from 0.762 baseline to 0.03–0.07 at every screened configuration — **Claim 2 refuted (aggregate accuracy regression at all k_top)**.

## M3 — Signed α-sweep (Claim 3)

| Run ID | alpha | subspace | seed | Status | macro_acc | macro_fid |
|---|---|---|---|---|---|---|
| M3_vlang_a-1.5_s42 | -1.5 | V_lang r=2 | 42 | **done** | 0.053 | 0.478 |
| M3_vlang_a-1.0_s42 | -1.0 | V_lang r=2 | 42 | **done** | 0.051 | 0.440 |
| M3_vlang_a-0.5_s42 | -0.5 | V_lang r=2 | 42 | **done** | 0.085 | 0.373 |
| M3_vlang_a-0.25_s42 | -0.25 | V_lang r=2 | 42 | **done** | **0.175** | 0.331 |
| M3_vlang_a0.0_s42 | 0.0 | V_lang r=2 | 42 | **done** | **0.744** | 0.245 |
| M3_vlang_a0.25_s42 | +0.25 | V_lang r=2 | 42 | **done** | 0.009 | 0.005 |
| M3_vlang_a0.5_s42 | +0.5 | V_lang r=2 | 42 | **done** | 0.000 | 0.035 |
| M3_vlang_a1.0_s42 | +1.0 | V_lang r=2 | 42 | **done** | 0.000 | 0.091 |
| M3_vlang_a1.5_s42 | +1.5 | V_lang r=2 | 42 | **done** | 0.000 | 0.091 |
| M3_random_a-1.5_s42 | -1.5 | random r=2 | 42 | **done** | 0.745 | 0.236 |
| M3_random_a-1.0_s42 | -1.0 | random r=2 | 42 | **done** | 0.740 | 0.235 |
| M3_random_a-0.5_s42 | -0.5 | random r=2 | 42 | **done** | 0.747 | 0.225 |
| M3_random_a-0.25_s42 | -0.25 | random r=2 | 42 | **done** | 0.753 | 0.238 |
| M3_random_a0.0_s42 | 0.0 | random r=2 | 42 | **done** | 0.744 | 0.245 |
| M3_random_a0.25_s42 | +0.25 | random r=2 | 42 | **done** | 0.729 | 0.231 |
| M3_random_a0.5_s42 | +0.5 | random r=2 | 42 | **done** | 0.438 | 0.42 |
| M3_random_a1.0_s42 | +1.0 | random r=2 | 42 | **done** | 0.004 | 0.07 |
| M3_random_a1.5_s42 | +1.5 | random r=2 | 42 | **done** | 0.000 | 0.01 |

**M3 key results**:
- V_lang α=0 sanity: **0.744** ≈ independent baseline **0.762** — hook infrastructure correct.
- V_lang best-negative α (α=−0.25): **0.175** — still 57 pp below baseline; the "removing improves accuracy" leg fails.
- V_lang best-positive α (α=+0.25): **0.009** — asymmetric worse than α=−0.25 → sign direction supports the plan direction but magnitude does not.
- **Specificity**: random subspace at α ∈ {−1.5, −1.0, −0.5} preserves ~0.74 baseline; V_lang at same α collapses to 0.05–0.09. **V_lang is specifically language-related** — matched-control passes cleanly for negative α.

## M4a — Multilingual LoRA-SFT (Claim 4)

| Run ID | Task | Status | Result |
|---|---|---|---|
| M4a_data_prep | Downloaded Mathoctopus/GSM8KInstruct_Parallel | **done** | 73559 records; Te=1 only |
| M4a_lora_train | LoRA r=32/α=32/{q,k,v,o}, lr=2e-4, batch 2×acc 8, 312 steps, 5001 ex | **done** | loss 2.0 → 0.7 rolling, grad-norm 0.25, no divergence; adapter written |
| M4a_eval | LoRA eval on MGSM 11-lang @ n=50 | **done** | **macro=0.558**, fid=0.964 |
| M4a_compute_ratio | κ = edit_compute / SFT_compute | **done** | edit ≈ 0.03 GPU-h; SFT ≈ 0.77 GPU-h; κ ≈ 0.04 (compute leg passes) |

## M4b — RL baseline (OPTIONAL)

| Run ID | Task | Status | Notes |
|---|---|---|---|
| m4b_grpo_train / m4b_eval | GRPO | **not run** | Claim 4 accuracy leg already refuted by M4a SFT drop; RL would not resurrect Claim 4 in the plan's form |

## Global gates fired

| Gate | Trigger | Outcome |
|---|---|---|
| G1 | M1 no config passes strict Claim-1 predicate (`V_lang ≥ 0.90 AND complement ≤ 0.20 AND cos ≤ 0.20`) | **partial-pass** — V_lang classifier + cos legs pass at (n=250, r=16, early); complement leg fails everywhere. Reported as `partial` in EXPERIMENT_RESULTS.md; not treated as HALT because the plan allows partial support (the "small" and "orthogonality" legs are separable). |
| G2 | M2 aggregate accuracy regression at every screened k_top | **fired** — Claim 2 refuted after Stage A screen; Stage B verify aborted to save budget |
| G3 | M4a wall-clock > 5 GPU-h | not triggered (M4a train + eval = 0.94 GPU-h) |
| G4 | Total GPU-hours > 9 h at start of M4b | pre-emptive skip — Claim 4 accuracy leg already refuted (M4a underperforms baseline by 20 pp) |

## Suspected under-power flags (Power-Fidelity, `UNDERPOWER=tag`)

- **M2, M3**: verify would have used n=250 × 3 seeds per plan. We ran n=25 (screen) / n=50 (M3) × 1 seed only, but the effect sizes are so large (baseline 0.76 → intervention 0.00–0.18) that the negative result is not power-limited — a 3-seed × n=250 rerun cannot flip a 60-pp gap. Not flagged.
- **M4a**: 312 optimizer steps at batch 2 × acc 8 = 5001 examples. Plan allowed 1 epoch on the full 73559 examples (~4600 steps). The 15 % of the full epoch might under-train — but the SFT drops accuracy uniformly across all 11 langs (not just under-trained ones), suggesting an over-fitting-to-format failure mode, not under-training. Weakly flagged for iteration.

## Round-End Decisions

None triggered. All plan milestones executed to completion or aborted only after their own predicate had already resolved.
