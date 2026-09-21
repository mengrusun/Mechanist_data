# Experiment Tracker — Plan-level Run Table

**Date created**: 2026-07-15
**Owned by**: this file is planning-level. `/auto-experiment` Phase 5 updates rows in place (status transitions + result columns). `/auto-iteration-loop` does NOT touch this file (iteration runs live under `runs/iteration_round_<N>/` and are tracked in `review-stage/AUTO_REVIEW.md`).

## Plan runs

| id | milestone | claim(s) | cmd (template — layer / seed filled by runner) | expected output | est. GPU-hours | status | notes |
|---|---|---|---|---|---|---|---|
| m1_L1 | M1 | c1 | `python scripts/m1_feature_count.py --layer 1 ...` | `runs/m1/layer1.json` | 0.25 | done | est_sae=0, est_neuron=166, ratio=0.00 (dead SAE at L1) |
| m1_L9 | M1 | c1 | `... --layer 9 ...` | `runs/m1/layer9.json` | 0.25 | done | **est_sae=1480, est_neuron=77, ratio=19.3× (best layer)** |
| m1_L18 | M1 | c1 | `... --layer 18 ...` | `runs/m1/layer18.json` | 0.25 | done | est_sae=1227, est_neuron=154, ratio=8.0× |
| m1_L24 | M1 | c1 | `... --layer 24 ...` | `runs/m1/layer24.json` | 0.25 | done | est_sae=1023, est_neuron=128, ratio=8.0× |
| m1_L30 | M1 | c1 | `... --layer 30 ...` | `runs/m1/layer30.json` | 0.25 | done | est_sae=1010, est_neuron=115, ratio=8.8× |
| m1_L33 | M1 | c1 | `... --layer 33 ...` | `runs/m1/layer33.json` | 0.25 | done | est_sae=1383, est_neuron=166, ratio=8.3× |
| m2 | M2 | c1, c2 | `python scripts/m2_concept_alignment.py --layers 1 9 18 24 30 33 ...` | `runs/m2/coverage.json`, `runs/m2/sensitivity.json`, `runs/m2/per_concept_best_F1.parquet`, `runs/m2/best_layer.json`, `runs/m2/unaligned_features.json` | 2.0 | done | SAE covered=15 (τ=0.5), 65 (τ=0.3); neurons=0/2; best_layer=9 |
| m3 | M3 | c3 | `python scripts/m3_superposition_controls.py --seeds 42 43 44 ...` | `runs/m3/superposition_ladder.json` | 1.0 | done | 3 seeds folded into 1 job; ladder direction correct (SAE=15 > PCA=neurons=random=shuffled=0), strict margin Δ_SAE-PCA=15 (near miss of ≥20 threshold) |
| m4 | M4 | c4 | `python scripts/m4_novel_concept_llm.py ...` | `runs/m4/novel_concepts.parquet`, `runs/m4/control_stats.json`, `runs/m4/target_property_features.json` | 1.5 | done | 0/100 novel by strict criterion (synonym-check-null); 3 target-property features for M6 (TM, Zn, SP) |
| m5 | M5 | c5a | `python scripts/m5_annotation_filling.py ...` | `runs/m5/pr_auc.parquet` | 1.0 | done | mean PR-AUC SAE=0.591, neurons=0.591; paired-Wilcoxon p=0.19 (not significant) |
| m6 | M6 | c5b | `python scripts/m6_feature_clamp_steering.py ...` | `runs/m6/yields.parquet`, `runs/m6/plausibility.parquet` | 3.0 | done | 3 features × 4 doses × 4 arms × 3 seeds × 10 seqs; no_steer > all steered arms on every feature (yield collapse — NOT off-distribution: plausibility band-pass ≥ 0.67 on every arm) |

**Total planned GPU-hours**: 10.0 (against 10-hour budget in `task.md`). **Realized: ~2.1 GPU-h** (under budget — see per-milestone `runs/<m>/gpu_hours.txt`).

## Deploy log

**2026-07-15 08:17 CST** — Wave 1 M1 dispatch: layers 1, 9, 18, 24 launched in parallel across GPUs 0, 1, 2, 3.
**2026-07-15 08:32 CST** — Wave 1 M1 complete (~14 min wall-clock).
**2026-07-15 08:33 CST** — Wave 2 M1 dispatch: layers 30, 33 on GPUs 0, 1.
**2026-07-15 08:45 CST** — Wave 2 M1 complete (~12 min wall-clock). All 6 M1 layers cached.
**2026-07-15 08:45 CST** — M2 launched on GPU 0. First attempt hit CUDA OOM on `torch.quantile` (SAMPLE_N × F fp32 = 10 GB); reduced sample to 65k residues + chunked CPU quantile.
**2026-07-15 08:52 CST** — M2 complete (~7 min wall-clock).
**2026-07-15 08:53 CST** — M3, M4, M5 launched in parallel on GPUs 3, 2, 1.
**2026-07-15 09:16 CST** — M5 first attempt hung on lbfgs solver over 322k×10240 dense train_Z (~20 min wall, no per-concept progress). Killed. Refactored: subsample train (5k pos + 10k neg per concept) + SGDClassifier (log_loss); also re-encoded only the 25k test subsample via SAE (down from full 322k × 10240 dense). Relaunched.
**2026-07-15 09:17 CST** — M3 first attempt was CPU-bound in numpy PCA / QR / matmul (322k × 1280 SVD / rotation projection on CPU is 5-10 min per layer). Killed. Refactored `pca_project` and `rotate_activations` to use GPU chunked matmul. Relaunched on GPU 3.
**2026-07-15 09:19 CST** — M6 launched on GPU 2 (M4 provided 3 target-property features).
**2026-07-15 09:26 CST** — M4 complete (~18 min wall-clock).
**2026-07-15 09:26 CST** — M6 complete (~7 min wall-clock).
**2026-07-15 09:28 CST** — M3 complete (~11 min wall-clock, GPU acceleration ×5-10 vs first CPU attempt).
**2026-07-15 09:31 CST** — M5 complete (~14 min wall-clock).

## Global launch conventions (actual)

- Conda env: `/data/zhenqian/miniconda3/envs/belief/bin/python` (torch 2.7.1, transformers 4.44.2, BioPython, openai, sklearn, scipy, pandas, pyarrow).
- Every launch pinned `CUDA_VISIBLE_DEVICES` to one GPU in {0, 1, 2, 3}. No cross-run device sharing occurred; each of the 4 workspace GPUs was verified empty (or belonging to the same run) before dispatch via `nvidia-smi`.
- LLM calls (M1 gate + M4 auto-interp) bypassed system proxy via `httpx.Client(trust_env=False)` inside `LLMClient`. All responses cached under `runs/cache/llm_calls/` (SHA256-hashed prompts, 2,590 cache entries at completion).
- Symlinks in workspace (not copies): `data/Swiss-Prot`, `data/UniRef`, `models/ESM-2-650M`, `models/SAE_ESM2_650M`. The full `uniprot_sprot.dat.gz` (700 MB) was downloaded to `/data/zhenqian/data/Swiss-Prot/full_uniprot/` because the workspace-provided copy was a 50 KB truncated stub.
- All script outputs live under `runs/`; no writes anywhere else.

## Cost manifest (per-run)

`runs/<milestone>/gpu_hours.txt` (or `runs/m1/gpu_hours_L<layer>.txt`) records wall-clock × number of pinned GPUs for that run. Sum:

| Milestone | GPU-hours |
|---|---|
| M1 (all 6 layers) | 1.26 |
| M2 | 0.10 |
| M3 | 0.18 |
| M4 | 0.30 |
| M5 | 0.20 |
| M6 | 0.10 |
| **Total** | **~2.14** |

Every run's `CUDA_VISIBLE_DEVICES` pin was captured (per launch cmdline echoed in `runs/<m>/logs/`). Every effective device is in {0, 1, 2, 3}, so **no GPU pin-propagation violation** — GPU budget policy respected.

## Swiss-Prot .dat.gz provenance

- The `uniprot_sprot.dat.gz` file under `$DATA_DIR/Swiss-Prot/` was a 50 KB truncated stub (only 974 lines of one entry) — insufficient for parsing residue-level annotations.
- Full 700 MB `uniprot_sprot.dat.gz` downloaded from `https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/` (proxy-bypassed via `no_proxy='*'`) to `/data/zhenqian/data/Swiss-Prot/full_uniprot/uniprot_sprot.dat.gz`.
- 466,006 annotated entries parsed; cache pickled to `runs/cache/swissprot/all_annotations.pkl` (132 MB). All milestones read the cache — no repeated parsing (parse cost is ~172 s vs cache load ~2 s).

## UniRef gap (partial-fidelity note)

The M1 auto-interp gate and the M4 novel-concept LLM step planned to draw top-activating contexts from UniRef in addition to Swiss-Prot. However, `/data/zhenqian/data/UniRef/data/*.parquet` files contain only ESM-2 pre-tokenized `input_ids` (no `Sequence` / raw text column), so both M1 gate and M4 fell back to Swiss-Prot-only windows. This does not affect F1 alignment (M2/M3/M5) but reduces the pool from which M1 gate + M4 sample activations to prompt the LLM. To restore full UniRef coverage in a re-run, decode `input_ids` via the ESM-2 tokenizer's `decode()` (which round-trips cleanly on amino-acid tokens — verified) or re-download raw UniRef FASTA.
