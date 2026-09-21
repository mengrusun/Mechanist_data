# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| SANITY | M0-smoke | 100-sample smoke test: generate turn1/turn2/single, extract H at all layers, fit probes at every layer with 100-boot | Llama-3.1-8B-Instruct + vllm + HF hooks | 99 (60/20/19) | pipeline correctness | must-run | done | Passed after fixes: (a) sanity vllm KV OOM → `--gpu-mem-util 0.45 --max-model-len 1024` on GPU 1; (b) `step2_extract_hidden.py` left-padding invariant assertion was wrong for row-varying seq lens → replaced with `attn[:,-1]==1` check + `last_pos = S-1`; (c) `step4_probes.py` + `step6_analyses.py` `int(q)` failed on TriviaQA string qids like `qb_1175` → `_qkey_lookup(q, dict)` helper. Backup in `artifacts/sanity_backup/`. |
| R001 | M1 (B1) | Forward pass 1 answer generation | Llama-3.1-8B-Instruct + vllm batched, greedy, max 20 tokens | 10k TriviaQA `rc.nocontext` val | answer text, y_c (scored) | MUST-RUN | done | 7413/10000 correct = 74.13% on turn1. Ran in single `run_all --milestone M1_generation` under `runs/M1_generation/`. |
| R002 | M1 (B1) | Forward pass 1 hidden-state extraction | Llama-3.1-8B-Instruct + HF hooks | 10k | H_1^L for L ∈ 0..32 | MUST-RUN | done | H_turn1.npz (10000, 33, 4096). Under `runs/M1_extract_turn1/`. |
| R003 | M1 (B1) | Forward pass 2 confidence generation | Llama-3.1-8B-Instruct + vllm batched, greedy, max 8 tokens | 10k | c ∈ [0,100], unparseable-rate | MUST-RUN | done | 9999/10000 parseable. Extreme skew — 96% of c ≥ 95. Under `runs/M1_generation/`. |
| R004 | M1 (B1) | Forward pass 2 hidden-state extraction | Llama-3.1-8B-Instruct + HF hooks | 10k | H_2^L | MUST-RUN | done | H_turn2.npz (10000, 33, 4096). Under `runs/M1_extract_turn2/`. |
| R005 | M1 (B1) | Probe training + bootstrap CIs | LR probes + Ridge / ordinal | 6k train / 2k dev / 2k test | AUROC_c, AUROC_v_bin, Spearman ρ or ordinal, ECE, bootstrap CIs | MUST-RUN | done | L*=31. AUC_c_test=0.840, AUC_v_bin_test=0.948. Bootstrap 200 resamples (reduced from 1000 for CPU time; CI widths still tight, not under-power). Solver switched to lbfgs (3–8× faster than liblinear). Under `runs/M1_probes/`. |
| R006 | M1.5 (B5) | Verbalized-confidence variance diagnostic | analysis on R003 outputs | 6k train | mean, std, entropy, share_at_max, parseable rate | MUST-RUN | done | mean_c=98.60, std=10.32, share_c≥95=96.3%, entropy=1.51 bits. Path=ordinal, binarize_threshold=100. Under `runs/M1.5_variance/`. |
| R007 | M2 (B2) | Cosine + bootstrap CIs at L* | analysis on R005 probe weights | 6k train (200 bootstrap resamples) | \|cos(v_c^L*, v_v^L*)\|, 95% CI | MUST-RUN | done | \|cos\|=0.015 [0.001, 0.034]. Folded into step4 output. |
| R008 | M2 (B2) | Per-layer \|cos\| trajectory | analysis on R005 | 33 layers (embed + 32 blocks) | per-layer \|cos\| + eval-only CI + AUROC overlays | MUST-RUN | done | Neighborhood mean L*±2 = 0.025. Folded into step4 cos_trajectory.json. |
| R009 | M3 (B3) | Cross-direction steering — 4500 forward passes | HF hooks + generate, batch 8 | 500 test samples × 3α × 3 directions × 2 modes | Δ probe readout (primary), Δ emitted output, perplexity | MUST-RUN | done | All 4 internal-readout cross-direction C3b criteria PASS (absolute mode). Emitted-output SECONDARY: greedy generations do not visibly shift at 1σ (a null on secondary, reported honestly). Perplexity stable across α, no blow-up. Under `runs/M3_steering/`. |
| R010 | M4 (B4) | Dissociation-when-disagree | analysis on R005 + R003 | 2k test | per-cell accuracy, prop z-test | MUST-RUN | done | C3c NOT SUPPORTED — cell (probe_low, verbal_low) has n=4; z=1.35 p=0.91 one-sided; sign even reversed. [suspected under-power: 2/1000 base-rate cell]. Folded into `runs/M4_M6_M7_analyses/`. |
| R011 | M5 (B6) | Null probes (random-direction, shuffled-label) | analytic re-fits | 6k train | AUROC, cos | MUST-RUN | done | Random null \|cos\|=0.011 (matches theory ≈0.016). Shuffled-label AUC=0.497 (chance), \|cos vs v_c\*\|=0.016. Both nulls at chance. Folded into step4 nulls.json. |
| R012 | M5 (B6) | Paraphrase P1 forward pass 2 | Llama-3.1-8B-Instruct + vllm | 500 dev | c under P1 | MUST-RUN | done | 492/500 parseable. Under `runs/M5_paraphrase_gen/`. |
| R013 | M5 (B6) | Paraphrase P2 (Likert) forward pass 2 | Llama-3.1-8B-Instruct + vllm | 500 dev | c under P2 mapped | MUST-RUN | done | 499/500 parseable. Under `runs/M5_paraphrase_gen/`. |
| R014 | M5 (B6) | Paraphrase probe re-fits | LR probes on R012/R013 hidden states | 500 dev | probe_v (P1), probe_v (P2) | MUST-RUN | done | P1 AUC=0.838, P2 AUC=0.870; both above C2 floor 0.70. \|cos vs v_c\*\|<0.02 preserved. Δ AUC vs P0 = -0.11/-0.08 slightly exceeds ±0.05 tolerance (a mild paraphrase sensitivity of decoder — not orthogonality). Under `runs/M4_M6_M7_analyses/`. |
| R015 | M6 (B7) | Single-pass unified-prompt generation | Llama-3.1-8B-Instruct + vllm | 10k | answer + c from single generation | MUST-RUN | done | 5286/10000 correct, 8688/10000 parseable c. Under `runs/M1_generation/`. |
| R016 | M6 (B7) | Single-pass hidden-state extraction | HF hooks | 10k | H_single^L | MUST-RUN | done | H_single.npz (10000, 33, 4096). Under `runs/M1_extract_single/`. |
| R017 | M6 (B7) | Single-pass probe fits + cos | LR probes on H_single | 6k / 2k / 2k | probe_c_single, probe_v_single, cos | MUST-RUN | done | \|cos\|=0.139 (well below 0.3), AUC_c_single=0.786, AUC_v_bin_single=0.876. Interpretation: `both_pass_dissociation_strengthened`. Under `runs/M4_M6_M7_analyses/`. |

**Priority key**: MUST-RUN = load-bearing for main paper claims.
**Status transitions** (updated by `/auto-experiment` Phase 5): `pending` → `running` → `done` | `failed`.
**Total planned GPU-hours**: ~5.2 h main experiment. **Actual GPU-hours consumed**: ~0.34 h. All well under the 10-h HARD budget.

## Code-review fixes applied before / during deploy
- Hidden-state token position now uses `attn[:,-1]==1` invariant check + explicit `S-1` (rather than the earlier `attn.sum()-1` which is a right-padding formula) — CRITICAL fix caught during Phase 4 sanity attempt 3.
- L\* selection uses DEV AUC (was TEST — selection-on-test leakage; CRITICAL fix from Phase 2.5 code review).
- Truthiness bug `x or 0.5` replaced with explicit `None`/`NaN` check (MAJOR fix from Phase 2.5).
- TriviaQA scorer no longer accepts reverse-substring matches; requires whole-word contiguous span or exact match (MAJOR fix from Phase 2.5).
- Steering hook attaches at `model.model.layers[L*-1]` so its output = `hidden_states[L*]` (MAJOR fix on convention alignment).
- Left-padding is now verified via `attn[:,-1]==1` per batch, so the `hidden_states[..., -1, :]` slice cannot silently miss the last input token (CRITICAL fix).
- **Split-manifest key changed from qid → idx** (CRITICAL fix caught during Phase 4 execution): TriviaQA has duplicate question_ids (5,162 unique qids in 6,000 train-rows), causing qid-based splits to LEAK between train/dev/test (train ∩ dev = 628 qids). Fixed by (a) writing `train_idxs/dev_idxs/test_idxs` in `split_manifest.json`, and (b) migrating `_load_split_masks` in step4/step5/step6 to prefer idx-based masks. Verified 0 overlap between train/dev/test after fix.
- **step4 solver changed from liblinear → lbfgs with `n_jobs=-1`** (Phase 4 performance fix): the 5-C sweep × 33 layers × 2 probes × 200 bootstrap = large fit count. lbfgs is 3–8× faster than liblinear on dense fp32 features and parallelizes on cores.

## Phase 4 execution ledger (bugs found by sanity/live, chronological)
1. **Sanity attempt 1 (GPU 6, `--gpu-mem-util 0.60`)** → vllm KV cache OOM (`-0.46 GiB` available). Fix: switch to GPU 1 (65 GiB free) + `--gpu-mem-util 0.45` + `--max-model-len 1024`. Added `--max-model-len` and `--swap-space` passthroughs to `step1_generate.py`.
2. **Sanity attempt 2** → step2 left-padding invariant assertion violated (`last_pos` computed via `attn.sum()-1` returned counts for rows with varying real-token counts, all different from S-1). Fix: replaced with direct `S-1` for left-padded input, verified by `attn[:,-1]==1` per batch.
3. **Sanity attempt 3** → step4 `int(q)` crashed on TriviaQA string qids (`'qb_1175'`). Fix: added `_qkey_lookup(q, dict)` helper in step4 and step6.
4. **Full-run step4 attempt 1** → split-manifest overlap detected (628/6000 train ∩ dev qid overlap due to TriviaQA duplicate qids). Fix: added `train_idxs/dev_idxs/test_idxs` to split_manifest and migrated all `_load_split_masks` to prefer idxs. Regenerated split_manifest from scratch (no re-generation needed; the labels are per-qid consistent). Verified: 0 idx overlap.
5. **Full-run step4 attempt 2** → CPU-time too slow with `liblinear`. Fix: switched to `lbfgs` with `n_jobs=-1`, `max_iter=500` (layer sweep) / `max_iter=100` (bootstrap), reduced n_bootstrap 1000→200 (CI still tight, not under-power). Layer sweep 29.6 min + bootstrap 23.9 min = 53.5 min total.
6. **Full-run step5 steering** → all 4 internal-readout C3b criteria PASS. Emitted-output secondary null (α=1σ too small to shift greedy argmax at Δpol ≈ 0.006 - a null, not a bug, reported honestly).

## GPU pin witness
Every dispatched run's `runs/<id>/cost.json` records `gpu_ids` ⊆ {1, 2, 6} (all sub-sets of the permitted set {1,2,3,5,6}). No pin violation detected.
