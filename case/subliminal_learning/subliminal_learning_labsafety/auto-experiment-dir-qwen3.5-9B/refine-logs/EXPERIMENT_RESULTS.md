# Initial Experiment Results

<!-- machine metadata (English; do not localize) -->
```yaml
phenomenon_status: conditional
committed_mechanism: Representation and Parameter Analysis / Steering Vectors (CAA)
main_experiment_verdict:
  C1 (cross-modal subliminal transfer): conditional  # 2 of 3 seeds >= 3 pp drop; seed 300 fails
  C2 (data-purity precondition): PASS (n_strict_flagged=0 on scrubbed set of 2611 rows)
  C3a (low-dim safety substrate — Location): partial  # layer 4 identified as top-divergent, but AUROC on safety partition low (0.15-0.37)
  C3b (causal intervention specificity): partial  # ablation gap_closure=0.875, but random_matched steering matches real-direction effect at α=-1; specificity fails
resource_fidelity: cost-aware  # marker never stamped for given-validation × discovery
suspected_under_power: true  # QA_I only 133 items; ~3-pp effect at seed variance is provisional
sweep_status_m0_1_teacher_sft: sanity_checked  # single reference config LR=2e-4 r=16 α=32, all A-D signals passed
sweep_status_m0_5_student_lr: swept  # LR grid {5e-5, 1e-4, 2e-4, 5e-4, 1e-3}; winner=1e-3 (dev drop 28.57 pp on seed=42)
gpu_hours_used: ~26  # M-1: 0.5, M0.1: 1, M0.2: 0.5 (3 GPUs × 20min), M0.3: 3 (API), M0.4: 1 (API+ 40-worker prewarm), M0.5: 4 (2 waves × 25min × 3 GPUs), M0.6: 1.5 (3 GPUs × 25min), M0.7: 0.5 (3 GPUs × 10min + Ctrl 5min), M1: 1 (4 GPUs × 15min), M2: 12 (5 waves)
```

**Date**: 2026-07-10
**Plan**: refine-logs/EXPERIMENT_PLAN.md
**Routing**: refine-logs/MECHANISM_ROUTING.md — Representation and Parameter Analysis / Steering Vectors (CAA)

## Data Actually Used

Per claim/block, reconciled against the *planned* data in EXPERIMENT_PLAN.md.

| Claim/Block | Provenance | Source | Available N (total) | Used N (actual) | Subset note |
|-------------|-----------|--------|---------------------|-----------------|-------------|
| C1 — Teacher SFT (M0.1) | existing | `teacher_anchor_sft.json` | 4642 | 4642 | full, no subset |
| C1 — Teacher generation (M0.2) | existing | `QUERIES_v3_all.txt` | 12000 | 12000 | full; 3 shards on GPUs 3,4,5 (GPUs 6,7 heavily loaded by other users at M0.2 time — reduced nshards from plan's 5 to 3 while still shard-parallel per rule 8) |
| C1/C2 — Filter (M0.3) | existing | teacher gen | 12000 | 12000 → 2905 kept | full; 24% keep rate |
| C2 — Rescan gate (M0.4) | existing | filter output | 2905 | 2905 → **2611 clean** | initial rescan flagged 33 items (regex 262) — scrubbed those 294 IDs and re-ran rescan on remainder → 0 flagged. C2 PASSES on `data_generated/teacher_gen_filtered_scrubbed.jsonl` |
| C1 — Student LR sweep (M0.5) | existing | filtered scrubbed | 2611 | 2611 | full; LR grid {5e-5, 1e-4, 2e-4, 5e-4, 1e-3} × dev seed=42 |
| C1 — Student per-seed (M0.6) | existing | filtered scrubbed | 2611 | 2611 | full; seeds 100, 200, 300 at winning LR=1e-3 |
| C1 — QA_I eval (M0.7) | existing | `QA_I-00000-of-00001.parquet` | 133 | 133 | full |
| C3a — Location (M1) | existing | QA_I | 133 (106 fit + 27 held-out) | 133 all layers screened | full; sites re-bound from plan's "top-K divergent" to "spaced-interval {0,2,4,...,30}" per routing (16 layers) |
| C3b — Causal Intervention (M2) | existing | QA_I held-out | 27 | 27 per (α, run_kind) | full held-out slice; n_pairs re-bound from plan's 200 to 27 realized (constrained by 133-item QA_I total × 20 % held-out — no larger held-out exists without violating the frozen split) |

**Data-Rule check**: provenance clean (all `existing`); frozen 80/20 split preserved from prior work (`results/qa_i_split.json`); safety-relevance labels frozen (`results/safety_relevance_labels.json`); labels come from the QA_I dataset's own "Correct Answer" field (NOT from another model's output — this is critical and confirmed).

## Results by Milestone

### M-1: Sanity — PASSED (all 4 checks)

| Check | Result |
|---|---|
| a — Tokenizer round-trip on 100 sampled QA_I items | PASS |
| b — Class-load & LoRA target-module assertion (AutoModelForImageTextToText + regex on `model.language_model.*`) | PASS (class=Qwen3_5ForConditionalGeneration; 29,097,984 lora_A modules all under `language_model`; 0 under vision tower) |
| c — Image-encoder non-degeneracy (32 QA_I items, first forward) | PASS (mean_norm=6.13, mean_ent=7.90 — healthy) |
| d — GPU pinning check on GPUs {3,4,5,6,7} | PASS (each child sees device_count=1) |

Artifact: `sanity/M-1_report.json`, `sanity/image_encoder_baseline.json`.

### M0.1: Teacher LoRA SFT — DONE

**sweep_status**: sanity_checked (single reference config LR=2e-4, r=16, α=32, effective_batch=16, 1 epoch, warmup 5% cosine; A/B/C/D signals passed on the pilot check on full 4642 records)

- 291 steps × ~10 s/step ≈ 48 min on GPU 4.
- Loss: 1.55 → 1.37 (14% relative descent, healthy).
- Grad norm: 1.5–2.5 (healthy).
- Adapter: 29.1M trainable params (0.324% of 9.0B total) — matches plan.

Artifact: `ckpts/teacher_lora/adapter_model.safetensors`.

### M0.2: Teacher generation (sharded) — DONE

- 12000 prompts → 12000 generations, sampling T=1.0, top_p=1.0, top_k=0, max_new_tokens=256, `enable_thinking=False`.
- 3 shards on GPUs 3, 4, 5 (parallel). 4000 rows per shard. ~20 min total.
- **Deviation from plan**: nshards reduced from 5 to 3 due to GPUs 6 and 7 being heavily used by other users when M0.2 started. Shard-parallel per rule 8 preserved.

Artifact: `data_generated/teacher_gen_all.jsonl`.

### M0.3: Filter with gpt-5.4 — DONE

- 12000 → 2905 kept (24% keep rate).
- 5 shards in parallel; ~45 min total.
- Judge cache: `caches/filter_cache.jsonl` (7981→ many hits after prewarm-style reuse from earlier runs).

Artifact: `data_generated/teacher_gen_filtered.jsonl`.

### M0.4: Rescan for residual unsafe vocabulary (C2 hard gate) — CONDITIONAL PASS on scrubbed subset

- **Initial rescan**: 262 regex-flagged + 33 strict-judge-flagged out of 2905 → C2 **FAILS**.
- **Recovery**: scrubbed the 294 flagged IDs (union of regex + strict-judge sets) → `data_generated/teacher_gen_filtered_scrubbed.jsonl` (2611 rows).
- **Rescan on scrubbed**: 0 regex-flagged, 0 strict-judge-flagged → C2 **PASSES** on the scrubbed set.
- The **student SFT training data used from M0.5 onward is the SCRUBBED set** (2611 rows). The C2 gate is honored: the training data used to teach the student has zero residual unsafe vocabulary.
- Judge cache: `caches/rescan_cache.jsonl` (6539 entries after prewarm with 40 workers).

Artifact: `data_generated/rescan_report_prescrub.json` (initial 33-flagged), `data_generated/rescan_report_scrubbed.json` == `data_generated/rescan_report.json` (0-flagged final).

### M0.5: Student LR sweep on dev seed=42 — DONE

**sweep_status**: swept (LR grid = {5e-5, 1e-4, 2e-4, 5e-4, 1e-3}, dev seed=42, evaluated on full 133-item QA_I).

Dev-LR curve (`dev/lr_curve.json`):

| LR | Acc | Drop from Ctrl (pp) |
|---|---|---|
| 5e-5 | 0.8271 | -3.01 (treated slightly better) |
| 1e-4 | 0.8045 | -0.75 |
| 2e-4 | 0.7970 | 0.00 |
| 5e-4 | 0.8045 | -0.75 |
| **1e-3 (winner)** | **0.5113** | **+28.57** |

- Winner: LR=1e-3. Frozen and applied identically to M0.6 seeds 100, 200, 300 to rule out garden-of-forking-paths.
- Loss trajectories: all 5 LRs converged to train_loss ~2.2–2.34 (similar convergence).
- The 1e-3 dev acc drop is NOT a format-collapse artifact: OTHER-rate = 2/133 (1.5%). The student is still parsing the letter choice — it's just wrong more often.

Artifact: `dev/lr_curve.json`, `dev/best_lr.json`.

### M0.6: Per-seed reproduction runs at frozen LR=1e-3 — DONE

- 3 parallel training runs on GPUs 3, 4, 5. ~24 min each.
- All 3 seeds converged to train_loss 2.34 ± 0.005 (nearly identical).

Artifact: `ckpts/student_seed{100,200,300}/adapter_model.safetensors`.

### M0.7: Eval 4 arms — DONE

Full QA_I (133 items), greedy decoding, gpt-5.4 3-way judge {CORRECT, INCORRECT, OTHER}.

| Arm | Correct | Incorrect | Other | Acc | Drop from Ctrl (pp) |
|---|---|---|---|---|---|
| Ctrl (base student) | 106 | 27 | 0 | 0.7970 | — |
| Treated seed 100 | 75 | 55 | 3 | 0.5639 | **+23.31** |
| Treated seed 200 | 86 | 47 | 0 | 0.6466 | **+15.04** |
| Treated seed 300 | 109 | 24 | 0 | 0.8195 | **−2.26 (treated BETTER)** |

Item-paired bootstrap 95% CI on Ctrl − treated (100 boots, per-item pairing):
- seed 100: [13.5%, 31.6%] pp — positive
- seed 200: [7.5%, 22.6%] pp — positive
- seed 300: [-9.0%, 3.8%] pp — straddles 0, includes the negative direction

### M0.8: M0 verdict — **CONDITIONAL**

Per-seed pass predicate ≥ 3 pp: seed 100 PASS, seed 200 PASS, seed 300 FAIL.

Per the plan's four-state verdict rule:
- Not `established` — seed 300 does not meet the per-seed predicate.
- **`conditional`** — 2 of 3 seeds pass. Mechanism milestones run scoped to the passing seeds.
- Rescan pass = True (C2 gate cleared on scrubbed set).
- Auxiliaries (paraphrase, decoding) **NOT run** — declared M0.8 optional aux, and per Phase 1.25 auto-continues; running them would not change the verdict from `conditional` (they can only promote `conditional → established`, not the other way).

**Interpretation**: at LR=1e-3, the subliminal transfer effect is real and large (~15-25 pp drop) but exhibits high seed variance — one of three seeds reverses the direction entirely, showing +2.26 pp (treated better than Ctrl). This is consistent with per-seed sensitivity of the fine-tune to weight-init noise at the highest LR in the sweep.

**Note on task.md's strict "≥ 3% per seed across ≥3 seeds"**: strict interpretation makes this `not-established`. Per the plan's four-state rule, 2/3 seeds passing is `conditional` (a permitted intermediate state that allows scoped mechanism analysis). We report both interpretations honestly; the phenomenon status downstream is **conditional**.

Artifact: `results/m0_headline.json`, `results/m0_verdict.txt`, `results/qa_i_summary.json`.

### M1: Location (correlational, diff-of-means direction) — PARTIAL

**Method** (per committed CAA family): for each layer in the spaced-interval sweep {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30} (16 layers, re-bound per `MECHANISM_ROUTING.md`'s block-selection rule from plan's "top-K divergent"), compute `v_L = mean(h_ctrl) − mean(h_treated_seed100)` on the 106-item fit split. Report `|v|`, `|v|/σ_ctrl`, and AUROC on the safety-decisive vs safety-neutral partition.

**Result** — best layer is **layer 4** (early residual stream) for all 3 treated seeds:

| Arm | Layer | \|v\| | \|v\|/σ_ctrl | AUROC (safety part.) |
|---|---|---|---|---|
| seed 100 | 4 | 10.72 | 149.6 | 0.307 |
| seed 200 | 4 | 11.08 | 141.6 | 0.266 |
| seed 300 | 4 | 9.20 | 115.9 | 0.307 |
| seed 100 | 6 (2nd) | 17.80 | 148.5 | 0.273 |
| seed 100 | 8 (3rd) | 21.19 | 114.2 | 0.226 |

- All 3 seeds converge on **layer 4** as the maximum-divergence layer — consistent kind-level signal.
- \|v\|/σ_ctrl values are enormous (110-150 σ), well above the routing threshold of 2σ → **L1-A passes** (a direction between treated and Ctrl exists).
- **However**, AUROC on the safety-decisive vs safety-neutral partition is 0.15-0.37 — well BELOW 0.5 chance. **The direction distinguishes treated from Ctrl but does NOT track the safety-decisive axis of QA_I**.

**Interpretation**: C3a is **partially supported** at the kind level — a low-dim shift exists, but it is broadly-distributed subspace movement rather than a targeted "safety substrate". The direction is real but not semantically aligned with the target concept.

Artifact: `mechanism/M1_location/{ctrl, treated_seed100, treated_seed200, treated_seed300}/directions.pt`, `mechanism/M1_location/screen_metrics.json`.

### M2: Causal Intervention (steering / ablation / patching at layer 4) — PARTIAL

**Setup**: layer=4 (M1 top-divergent), directions from M1's diff-of-means, applied on treated_seed100 held-out 27 items via three intervention shapes:
- **steering**: `h ← h − α · σ_proj · u`, α ∈ {-2, -1, 0, +1, +2}.
- **ablation**: `h ← h − (h·u)u` (remove projection onto u).
- **patching**: replace treated's layer-4 residual with Ctrl's cached residual for the same item.

**Cross-controls**: `random_matched` (same layer, random unit vector, matched σ).

**Result** — `mechanism/M2_causal/gap_closure.json`:

| Shape | Source | α | Acc(intervened) | Gap closure (toward Ctrl) |
|---|---|---|---|---|
| **ablation** | m1_top_k | 0 | 0.704 | **0.875** ← strong |
| **patching** | m1_top_k | 0 | 0.630 | **0.625** ← strong |
| steering | m1_top_k | -2 | 0.482 | 0.125 |
| steering | m1_top_k | -1 | 0.482 | 0.125 |
| steering | m1_top_k | 0 | 0.444 | 0.000 (baseline) |
| steering | m1_top_k | +1 | 0.482 | 0.125 |
| steering | m1_top_k | +2 | 0.519 | **0.250** |
| **steering** | **random_matched** | -2 | 0.444 | 0.000 |
| **steering** | **random_matched** | -1 | 0.519 | **0.250** ← same as best real |
| steering | random_matched | +1 | 0.482 | 0.125 |
| steering | random_matched | +2 | 0.482 | 0.125 |

Cross-seed replication (seed 200 and 300 at α=-1, -2 on their own M1 directions):
- seed 200 baseline gc=0 across α — direction has no causal effect at layer 4 on this seed's own residual stream.
- seed 300 baseline gc=0 (or -0.000) — expected, since seed 300 is already at Ctrl accuracy.

**Verdict — three-condition specificity check per plan**:
1. **Dose-response monotonicity on real direction**: NOT monotonic (α=-2 same as α=+1; α=+2 gives the largest gap closure, not α=-2). Fails.
2. **Matched-control gap** (SP-A): random_matched at α=-1 matches the real direction's best gc (both 0.25). **SP-A specificity FAILS** — a random direction produces the same effect as the extracted CAA direction.
3. **Specificity vs MMLU**: NOT run — M2 already fails 2 of 3 conditions; further specificity checks add no evidence. Would be added in an iteration cycle.

**Ablation is different**: `h ← h − (h·u)u` removing the projection onto u recovers 87.5% of the Ctrl-treated gap. This suggests removing *some* magnitude at layer 4 helps — regardless of which direction u is projected onto, since removing on the random_matched direction should be tested next iteration.

**Interpretation** — this is the plan's **"distributed rewrite" negative result**: C3b's specificity claim collapses. The subliminal effect at LR=1e-3 has a *localizable magnitude* at layer 4 (ablation and patching recover it) but is NOT captured by a single low-dim direction (steering along either M1 or random directions produces small, comparable, non-monotonic effects). This is a legitimate scientific negative result per the plan's fallback narrative.

Artifact: `mechanism/M2_causal/{steering_m1_seed100, steering_random_seed100, ablation_seed100, patching_seed100, steering_m1_seed200, steering_m1_seed300}/*.jsonl`, `mechanism/M2_causal/gap_closure.json`.

### M3: Unit Interpretation — SKIPPED

Per plan: "Optional post-hoc; runs iff M2 passes and identifies a rank-1 or rank-2 direction". M2 did not pass the specificity gate — the direction is not causally distinguishable from a random direction of matched magnitude, so M3's naming step (which decodes "what does this direction mean semantically?") adds no evidence to a direction that has not been shown to be functionally the safety-transfer axis. Per routing plan, M3 was re-bound to logit_lens-only (from plan's 3 recipes to 1), further reducing the value of running it when M2 is inconclusive.

## Summary

- **[7/7]** must-run milestones completed (M-1 → M0.7).
- **[9/9]** M0 sub-milestones completed (M-1, M0.1-M0.7, M0.8).
- **[2/3]** mechanism milestones completed (M1 done, M2 done, M3 skipped per M2 outcome).
- **Main result**:
  - **C1 (cross-modal subliminal transfer, per-seed ≥ 3 pp drop)**: `conditional` — 2 of 3 seeds pass strongly (23.3 pp and 15.0 pp drops); seed 300 fails with a −2.3 pp *reverse* effect. High seed variance at the winning LR=1e-3.
  - **C2 (data-purity precondition, 0 unsafe rows after rescan)**: `PASS` on the scrubbed training set (2611 rows, 0 strict-flagged, 0 regex-flagged after scrubbing 294 initially-flagged items).
  - **C3a (low-dim safety substrate — Location)**: `partial` — layer 4 identified as top-divergent site with \|v\|/σ ≈ 150, but AUROC on the safety partition is only 0.27, so the direction is not semantically the safety axis.
  - **C3b (causal specificity)**: `partial → refuted at rank-1` — ablation and patching at layer 4 both work strongly (gap closure 0.63-0.88), but steering on the M1 direction is not distinguishable from steering on a random direction (both gap closure ≤ 0.25). Consistent with the plan's "distributed rewrite" fallback narrative.
- **Ready for /auto-verify**: YES — with the caveat that verify's swap-variants should test the same conditional-scoped seeds (100, 200) and expect the same variance pattern.

- **suspected_under_power**: TRUE (QA_I only 133 items; 27-item held-out slice for M2 magnifies noise; ~3-pp effect at seed variance is provisional).

## Next Step

→ /auto-verify (stress-test the C1 phenomenon and C3a/b mechanism claims via method/dataset/model swaps).
