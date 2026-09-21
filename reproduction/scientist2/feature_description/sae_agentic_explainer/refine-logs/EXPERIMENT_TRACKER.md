# Experiment Tracker — SAGE Reproduction

Plan-level tracker. `/auto-experiment` Phase 5 updates rows in place; `/auto-iteration-loop` does not touch this file.

| Run ID | Milestone | Claims | Model + SAE | Layers | N_features (planned → realized) | Method | GPU-hours (planned → realized) | Status | Notes / Result |
|--------|-----------|--------|-------------|-------:|--------------------------------:|--------|-------------------------------:|--------|----------------|
| m0_5_sanity_gate | M0.5 | C1-C4 (support) | gemma-2-2b + gemmascope-res-16k | 4, 12, 20 | 30 → 15 (5/layer, pilot) | activation-match + single-pass GPT-5 sanity | 0.5 → 0.11 | done | **PASS** — activation_match median ratio 1.05 (in [0.5, 2.0]); AUROC gate both baselines > 0.55 mean. Attribution-confound flagged: single-pass GPT-5 already outscores Neuronpedia. |
| m1_L4 | M1 | C1, C2, C3 (L4) | gemma-2-2b + gemmascope-res-16k | 4 | 100 → 15 | sage + neuronpedia + gpt5_1shot | 1.7 → 0.39 | done-partial (undersampled) | SAGE mean gen=0.11, pred_pearson=0.59; NP mean gen=0.08, pred=0.66; GPT5-1shot mean gen=0.12, pred=0.62. Under-power flagged. |
| m1_L12 | M1 | C1, C2, C3 (L12) | gemma-2-2b + gemmascope-res-16k | 12 | 100 → 15 | sage + neuronpedia + gpt5_1shot | 1.7 → 0.38 | done-partial (undersampled) | SAGE gen=0.09, pred=0.40; NP gen=0.07, pred=0.40; GPT5-1shot gen=0.11, pred=0.40. l0_82 SAE used (Neuronpedia-aligned). |
| m1_L20 | M1 | C1, C2, C3 (L20) | gemma-2-2b + gemmascope-res-16k | 20 | 100 → 14 | sage + neuronpedia + gpt5_1shot | 1.7 → 0.37 | done-partial (undersampled) | SAGE gen=0.04, pred=0.46; NP gen=0.04, pred=0.44; GPT5-1shot gen=0.03, pred=0.38. l0_139 used; L20 alignment is looser than L4/L12. |
| m2_qwen3_cross_pair | M2 | C4 (predictive-only) | qwen3-4b + transcoder-hp (Neuronpedia cached activations) | 8, 16, 28 | 150 → 34 (~11/depth) | sage_lite + neuronpedia + gpt5_1shot | 3.5 → 0.41 | done-partial (undersampled + scope-narrowed) | **SCOPED TO PREDICTIVE-ACCURACY ONLY** — the full 4-role SAGE + generative accuracy on Qwen3-4B is deferred to /auto-verify (requires transcoder-hp MLP-hook infrastructure not in this stage). SAGE-lite pearson = 0.35, NP = 0.20, GPT5-1shot = 0.40. |
| — | RESERVE | — | — | — | — | — | 1.0 → 8.34 remaining | reserved | **Actual GPU-hours = 1.66 vs. 10 budget** — the bottleneck was DMXAPI throughput, NOT GPU capacity. 8.34 GPU-hours remain for /auto-verify + iteration. |

Total realized: **1.66 GPU-hours** of the 10-hour budget.

## Notes

- Device pinning verified: every run's `cost.json.gpu_ids` lies within the allow-set {1, 2, 3, 5, 6} — M0.5 on GPU 6, M1 layers on GPUs 1/2/3, M2 on GPU 5.
- Dedicated conda env `sage` (Python 3.10, torch 2.4.1, transformers 4.44.2).
- Symlink policy: no data copied — all SAE checkpoints, Gemma-2-2B weights, and Qwen3-4B (unused here) accessed via `/data/zhenqian/models/`.
- Neuronpedia's canonical `gemmascope-res-16k` variant is empirically `l0_124` (L4), `l0_82` (L12), and `l0_139` (L20) — identified by matching Neuronpedia's cached top-activating snippet peak against our SAE's re-computed peak. This mapping is codified in `scripts/m0_5_baseline_sanity.py`'s `GEMMA_SCOPE_L0_MAP`.
- `/auto-verify` swap variants (running GPT-OSS-20B + resid-post-aa, deferred full SAGE on Qwen3-4B, judge/scorer swap) live in the /auto-verify stage.

## Sanity gate summary (M0.5)

| Metric | Layer 4 | Layer 12 | Layer 20 | Overall |
|--------|--------:|---------:|---------:|--------:|
| Median activation-match ratio (our SAE vs. Neuronpedia) | 1.05 | 1.06 | 0.26 (mixed) | 1.05 (aggregate median) |
| Mean Neuronpedia detection AUROC (5 features) | 0.765 | 0.570 | 0.675 | 0.670 |
| Mean single-pass GPT-5 detection AUROC | 0.715 | 0.680 | 0.925 | 0.773 |

Both gates PASS: activation-match (median in [0.5, 2.0]) and AUROC (both baselines > 0.50 mean).

## Per-milestone launch scripts + cost accounting

- `runs/m0_5_sanity_gate/run.sh` — done — `runs/m0_5_sanity_gate/cost.json` (gpu_ids: [6], gpu_hours: 0.11)
- `runs/m1_L4/run.sh` — done-partial — `runs/m1_L4/cost.json` (gpu_ids: [1], gpu_hours: 0.39)
- `runs/m1_L12/run.sh` — done-partial — `runs/m1_L12/cost.json` (gpu_ids: [2], gpu_hours: 0.38)
- `runs/m1_L20/run.sh` — done-partial — `runs/m1_L20/cost.json` (gpu_ids: [3], gpu_hours: 0.37)
- `runs/m2_qwen3_cross_pair/run.sh` — done-partial + scope-narrowed — `runs/m2_qwen3_cross_pair/cost.json` (gpu_ids: [5], gpu_hours: 0.41)
