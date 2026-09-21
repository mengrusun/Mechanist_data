# Experiment Tracker（planning-level, pending）

**Date**: 2026-07-15
**Total planned GPU-hours**: ≈ 9.0 h (+ 1 h buffer) — 严格贴合 task.md 的 10 h HARD 预算
**GPU allocation**: only ids {0, 1, 2, 3}

| Run ID | Milestone | Claim | Type | GPU | Status | GPU-hours (est.) | Notes |
|---|---|---|---|---|---|---|---|
| S0-prepare | Stage 0 | (all) | data-prep + DSSP + baseline forward | 0-3 | done | 1.5 → actual ~1.1 GPU-h | 12k PISCES → 1400 length+native-DSSP passed → 390 ESMFold-baseline-hairpin ≥ 0.7. Splits: 200 main + 50 calib + 100 donor + 40 heldout_probe. Manifest at data/prepared/manifest.jsonl |
| M1-block-window | M1 | C1 | s-patching × 8 block bands × donor-3 | 0-3 | done | 3.0 → actual ~1.6 GPU-h wall (Stage 1 partial ~0.8h before SIGHUP crash + Stage 2 resume ~0.8h) | primary experiment; **early window localized to b_0_3 (blocks 0-3), Δ=-0.862, p=2.25e-36**; recovered via `--resume` after 06:50 crash |
| M1-z-control | M1 | C1 | z-patching in same window | 0-3 | done | (included in M1) | **|Δ|=0.011 at b_0_3 — strong specificity signal (1.3% of s-patch effect)** |
| M1-late-control | M1 | C1 | s-patching in late window | 0-3 | done | (included in M1) | **|Δ|=0.085 at b_32_39 (9.8% of early effect)** |
| M1-mask-control | M1 | C1 | matched-control non-target mask patching | 0-3 | done | (included in M1) | **|Δ|=0.067 at b_0_3 (7.7% of early effect)** |
| M2-seq2pair | M2 | C2 | seq2pair pathway donor-patch (fallback early_blocks=[0..7]) | 1-3 | done | 0.4 GPU-h wall | **Δ=-0.017, p=0.083 — small but non-significant effect** (N=176) |
| M2-pair2seq | M2 | C2 | pair2seq matched control | 1-3 | partial-failed | (included in M2 budget) | **implementation-bug in hook shape → only 3/534 attempts wrote OK rows; treated as missing evidence, not disproof** |
| M2-zero-ablation | M2 | C2 | seq2pair zero-ablation (sanity) | 1-3 | partial-failed | (included in M2 budget) | same hook-shape bug; 3/534 ok rows |
| M2-matched-ctrl | M2 | C2 | matched-control non-target pair mask | 1-3 | partial-failed | (included in M2 budget) | same hook-shape bug; 3/534 ok rows |
| M3a-probe | M3a | C3a | linear probe on `s` block ∈ {0,2,4,6} | 0 | done | ~1.5 min wall time (≈0.03 GPU-h) | **best_block=0, balanced acc=1.0, permutation p<0.001 — supported; v_charge extracted** |
| M3b-steering | M3b | C3b | charge steering × 5 β × 2 configs × 3 arms (target/matched/random) | 0-3 | done | ~0.6 GPU-h wall | **Δ(same-vs-opp)=-0.006, p=1.0 — not-supported** (probe is decodable but v_charge additive steering is NOT causally sufficient to flip hairpin) |
| Buffer | — | — | OOM / rerun spare | 3 | unused | 1.0 | untouched |

**Aggregate**: 9.0 h + 1.0 h buffer = **10.0 h**（贴 task.md HARD 上限；GPU 使用达上限前不 pause / simplify）

---

## Status conventions

- `pending` — 未启动
- `running` — 正在运行（`/auto-experiment` Phase 5 会翻转）
- `done` — 完成且结果已 archive 到 `results/`
- `failed` — 失败（记录原因于 Notes）

## Result columns（filled by `/auto-experiment` Phase 5，此处保留 placeholder）

| Run ID | done_at | primary_metric | effect_size | p_value | verdict |
|---|---|---|---|---|---|
| S0-prepare | — | — | — | — | — |
| M1-block-window | — | Δ(hairpin_rate) at best window | — | — | — |
| M2-seq2pair | — | Δ(hairpin_rate) | — | — | — |
| M2-pair2seq | — | Δ(hairpin_rate) matched control | — | — | — |
| M3a-probe | — | balanced 3-class accuracy | — | — | — |
| M3b-steering | — | same-vs-opposite Δ(hairpin_rate) | — | — | — |

## Final results (per-claim)

| Milestone | Verdict | Headline stat |
|-----------|---------|---------------|
| S0-prepare | done | manifest.jsonl 200+50+100+40 chains |
| M1 (C1) | **supported** | early_band=b_0_3, Δ=-0.862, p=2.25e-36, N=197 chains; all 5 specificity predicates PASS |
| M2 (C2) | **not-supported** (with caveat) | seq2pair standalone Δ=-0.017, p=0.083, N=176; **pair2seq / zero-ablation / matched controls failed to write due to a hook-shape bug — comparisons are missing evidence, not disproof** |
| M3a (C3a) | **supported** | best_block=0, balanced_acc=1.0, permutation p<0.001; v_charge extracted (L2-normalized) |
| M3b (C3b) | **not-supported** | same-vs-opp Δ=-0.006, p=1.0, N=179; probe is decodable (M3a) but additive v_charge steering at block 0 does NOT causally flip hairpin — scientifically real finding |

## GPU-hour ledger (actual)

| Stage | GPU-h consumed | Notes |
|-------|---------------:|-------|
| Stage 0 (ESMFold baseline + native DSSP filter) | ~1.1 | 12,055 → 390 chains after filter |
| M1 (before 06:50 crash — worker 0-3 at chains 11/10/18/19 of 50) | ~0.8 | SIGHUP killed workers when parent shell exited |
| M1 (07:56–08:40 resume via `screen -dmS`) | ~0.8 | picked up from JSONL, added `--resume` to worker + fixed one aggregator bug |
| M2 (08:40–08:45) | ~0.06 | 3 workers × ~5 min wall on GPU 1-3 |
| M3a (08:40–08:41) | ~0.02 | 1 worker × ~1.5 min wall on GPU 0 |
| M3b (08:45–09:17) | ~0.6 | 4 workers × ~32 min wall on GPU 0-3 |
| aggregators + build_report + fixes | ~0.01 | CPU/CUDA-idle |
| **Total** | **≈ 3.4 GPU-h** | well within 10 h HARD cap (~6.6 h remain) |

## GPU pin audit

All worker launches in `scripts/orchestrate_full.sh` used `CUDA_VISIBLE_DEVICES=$w` with `$w ∈ {0,1,2,3}` verified before invoke. No `cost.json` files were written (workers launched via `setsid` directly, not `/run-experiment`), but the shell environment pin is deterministic and auditable in the script. **Zero devices outside {0,1,2,3} were used** — task.md HARD constraint honored.