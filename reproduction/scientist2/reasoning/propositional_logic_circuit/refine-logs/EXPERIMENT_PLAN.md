# Experiment Plan — Sparse Modular Circuit for Propositional-Logic Reasoning

**Behavior-source**: given
**Mechanism**: discovery
**Date**: 2026-07-14
**Anchor proposal**: `refine-logs/FINAL_PROPOSAL.md`

---

## Top metadata (machine markers)

```yaml
resource_fidelity: not-strict     # MECHANISM=discovery — the strict harness only stamps when both axes are given
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - Tuning & Editing — no editing objective; would inflate compute without adding evidence.
    - Formation Tracing — final pretrained checkpoint only; no training trajectory analysis.
    - Unit Interpretation — SAE training on Mistral-7B outside the 10-GPU-hour envelope; deferred as follow-up.
    - Decision Auditing — orthogonal to the three sub-claims.
  note: Three sub-claims are a strict mechanistic-evidence question — Location screens for a sparse component set (C1), Causal Intervention verifies necessity + sufficiency (C3) and dissociates roles (C2).
```

**No `kind: phenomenon-validation` milestone**: `BEHAVIOR_SOURCE=given` — the behavior is assumed to hold. No M0. No `depends_on: [M0]` on any milestone.

**GPU budget**: 10 GPU-hours total, GPUs restricted to `{0, 1, 2, 3}` per `task.md`. Per-milestone budget in the table below.

**Resource paths**: `DATA_DIR=/data/zhenqian/data`, `MODEL_DIR=/data/zhenqian/models`. Symbolic links may be created in `work_dir` if needed.

---

## Milestones — overview

| ID | Purpose | Claim(s) | Model | GPU-h (est.) | Depends on |
|----|---------|----------|-------|--------------|------------|
| M0.dataset | Build synthetic propositional-logic template | (setup) | — (CPU) | 0.1 | — |
| M0.setup | Model / env / probing framework check | (setup) | Mistral-7B | 0.3 | M0.dataset |
| M1 | Attribution-patching screen — anchor cell | C1 | Mistral-7B | 1.2 | M0.setup |
| M2 | Path-patching necessity — anchor cell | C3 (necessity) | Mistral-7B | 1.5 | M1 |
| M3 | Sufficiency reinsertion — anchor cell | C3 (sufficiency) | Mistral-7B | 1.0 | M1 |
| M4 | Role-dissociation (fact / rule / answer corruption) | C2 | Mistral-7B | 1.8 | M1 |
| M4.stab | Cross-cell stability (k=5,chain=2 and k=3,chain=3) | C2 (stability) | Mistral-7B | 0.8 | M4 |
| M5 | Cross-family verify — Gemma-2-9B anchor cell | C1, C2, C3 recurrence | Gemma-2-9B | 2.0 | M1–M4 |
| M5.contingent | Gemma-2-27B anchor cell (budget permitting) | recurrence | Gemma-2-27B | 1.3 (if run) | M5 |
| M6 | Analysis + report aggregation | (report) | — (CPU) | 0.1 | M2, M3, M4.stab, M5 |
| **Total** | | | | **~8.7 h** (+ 1.3 h contingent) | |

Fits within 10 GPU-h with ~1.3 h headroom for iteration + a re-run under different metric.

---

## M0.dataset — Synthetic propositional-logic dataset build

**Purpose**: construct the parameterised template pool.

**Deliverable**: `${DATA_DIR}/prop_logic_synth/{clean,corrupt_fact,corrupt_rule,corrupt_answer,corrupt_neutral}/split_{k}_{chain}_{lex}/*.jsonl`.

**Parameters**:
- k ∈ {2, 3, 5, 8, 12}; chain-length ∈ {1, 2, 3}; lexicon ∈ {natural, symbolic, alt-nouns}.
- Anchor cell: (k=3, chain=2, lexicon=natural), 500 pairs each of clean and four corruption types.
- Additional cells for role/stability: (k=5, chain=2, nat) 200 pairs; (k=3, chain=3, nat) 200 pairs; (k=3, chain=2, symbolic) 200 pairs; (k=3, chain=2, alt-nouns) 200 pairs.
- Total: ~4000 clean prompts + ~16000 matched-corrupt prompts.
- 50% True / 50% False balance enforced per cell.
- `used_n`: full construction pool as above.

**Data split**: dataset is entirely synthetic and used only for **evaluation / patching** (never for training). No train/val/test split is needed in the classical sense; instead: (a) `pool_pair` (used for corruption-pair matching), (b) `pool_resample` (used as the source of resample-ablation activations — held out from patching pairs).

**Cmd**: `python scripts/build_propositional_dataset.py --data-dir ${DATA_DIR}/prop_logic_synth --seed 42`

**Expected output**: dataset directory populated + a `${DATA_DIR}/prop_logic_synth/manifest.json` listing counts per cell.

**Estimated GPU-hours**: 0.0 (CPU only).

**Priority**: MUST-RUN (blocker for everything).

---

## M0.setup — Model / environment / patching-framework check

**Purpose**: (1) confirm Mistral-7B loads cleanly on GPU 0, (2) confirm the patching framework (default: `TransformerLens` or `nnsight`, submethod-routed at experiment stage) can hook + intervene on attention-heads and MLPs, (3) sanity-check Mistral-7B's accuracy on the anchor cell.

**Model resolution**:
- Primary: `${MODEL_DIR}/Mistral-7B-v0.1` (currently broken symlink — download at this milestone via HF token `<Your_token>` if `config.json` cannot be read).
- Fallback: `${MODEL_DIR}/Mistral-7B-Instruct-v0.1` (locally present) — noted as fallback in report if used, since instruct-tuning may perturb the circuit.

**Sanity criterion**: on the anchor cell, Mistral-7B accuracy `≥ 0.75` on 500 True/False propositional-logic prompts. If below, escalate: (a) try Mistral-7B-Instruct-v0.1, (b) increase few-shot examples in the prompt, (c) surface the failure to the caller — do not silently proceed to circuit analysis on a model that cannot do the task.

**Cmd**: `python scripts/setup_check.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --data ${DATA_DIR}/prop_logic_synth --gpus 0`

**Expected output**: `results/M0_setup.json` with fields `{model_loaded, patching_hooks_ok, anchor_accuracy}`.

**Priority**: MUST-RUN.

**Estimated GPU-hours**: 0.3 (loading + one forward pass over 500 pairs).

---

## M1 — Attribution-patching screen (C1 Location)

**Claim(s) verified**: C1 (sparse component set).

**Method** — attribution patching (Syed et al. 2024, arXiv 2310.10348): a single backward pass per (clean, corrupt) pair computes gradient-weighted contributions of every head and every MLP to the answer metric. Rank all `L·H + L` components; take top-K until cumulative-effect ≥ 0.9.

**Completeness verification**: restrict Mistral-7B's forward pass to only the shortlisted components (resample-ablate everything else from `pool_resample`, per Best-Practices arXiv 2309.16042). Measure accuracy retention on the anchor cell.

**Minimality verification**: iteratively remove one component from the shortlist; measure `Δ accuracy`. A minimal circuit shows a single-removal drop `≥ 0.05` on average.

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]` — the concrete framework (TransformerLens vs nnsight) chosen at experiment stage may re-bind these; e.g. `n_pairs = 500` may become `n_pairs = 400` under a lighter framework, `metric` may finalise to `logit_diff` from among `{logit_diff, prob_diff}` at the screen step (KL added at final report).

**Cmd**: `python scripts/attribution_screen.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --cell k3_chain2_natural --n-pairs 500 --gpus 0 --out results/M1_attribution.json`

**Expected output**: `results/M1_attribution.json` with:
```json
{
  "shortlist_heads":   [(layer, head_idx, attribution_score), ...],
  "shortlist_mlps":    [(layer, attribution_score), ...],
  "cumulative_effect": <float>,
  "sparsity_fraction": <|C| / total>,
  "completeness":      <float>,
  "minimality":        {"avg_single_removal_drop": <float>, "median_drop": <float>}
}
```

**Success criterion (C1)**: `sparsity_fraction ≤ 0.15` AND `completeness ≥ 0.9` AND `minimality.avg_single_removal_drop ≥ 0.05`. Under-threshold = **negative result for C1** (still reported, still useful).

**Priority**: MUST-RUN. **Estimated GPU-hours**: 1.2.

---

## M2 — Path-patching necessity (C3 necessity)

**Claim(s) verified**: C3 (necessity direction).

**Method**: on the shortlist from M1, run **path patching** (Wang et al. 2022 style) — replace each shortlisted component's activation on the *corrupted* prompt with the corresponding *clean* activation, one component (or one path) at a time and jointly. Report Recovery under all three metrics: `logit_diff`, `prob_diff`, `KL`.

**Dose-response**: sweep `|patched components|` in ~5 steps from 0 to full shortlist size. Report Recovery curve.

**Specificity control**: matched-size random component set from *outside* the shortlist; same patch schedule.

**Corruption strategy**: `resample_ablation` from `pool_resample` (per Best-Practices).

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`.

**Cmd**: `python scripts/path_patch_necessity.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --cell k3_chain2_natural --shortlist results/M1_attribution.json --n-pairs 500 --gpus 0,1 --out results/M2_necessity.json`

**Expected output**: `results/M2_necessity.json` with:
```json
{
  "recovery": {"logit_diff": <float>, "prob_diff": <float>, "KL": <float>},
  "dose_response": [{"k_patched": <int>, "recovery_ld": <float>, "recovery_pd": <float>, "recovery_kl": <float>}, ...],
  "control_recovery": {"logit_diff": <float>, "prob_diff": <float>, "KL": <float>},
  "specificity_gap": <float>
}
```

**Success criterion (C3 necessity)**: `recovery.logit_diff ≥ 0.8` AND `recovery.prob_diff ≥ 0.8` AND `specificity_gap ≥ 0.6`.

**Priority**: MUST-RUN. **Estimated GPU-hours**: 1.5.

---

## M3 — Sufficiency reinsertion (C3 sufficiency)

**Claim(s) verified**: C3 (sufficiency direction).

**Method**: take a *clean* prompt. Resample-ablate the activations of every non-shortlisted component. Then reinsert clean activations only at the shortlisted components. Measure whether the answer stays correct.

**Sufficient-recovery**: `(P(correct | clean + reinsert-only-C from clean run) − P(correct | clean, everything resample-ablated)) / (P(correct | clean, no ablation) − P(correct | clean, everything resample-ablated))`.

**Specificity**: same reinsertion at a matched-size random control set. Report control sufficient-recovery.

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`.

**Cmd**: `python scripts/reinsertion_sufficiency.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --cell k3_chain2_natural --shortlist results/M1_attribution.json --n-pairs 500 --gpus 0,1 --out results/M3_sufficiency.json`

**Expected output**: `results/M3_sufficiency.json` with:
```json
{
  "sufficient_recovery": {"logit_diff": <float>, "prob_diff": <float>, "KL": <float>},
  "control_sufficient_recovery": {"logit_diff": <float>, "prob_diff": <float>, "KL": <float>},
  "specificity_gap": <float>,
  "resample_seed_distribution": [<recovery per seed>]  # ≥ 5 seeds
}
```

**Success criterion (C3 sufficiency)**: `sufficient_recovery ≥ 0.8` on all three metrics AND `specificity_gap ≥ 0.6`, robust across ≥ 5 resample seeds (std < 0.1).

**Priority**: MUST-RUN. **Estimated GPU-hours**: 1.0.

---

## M4 — Role-dissociation (C2 modular decomposition)

**Claim(s) verified**: C2 (three-role modular structure).

**Method**: for each of the three role-corruption pools (fact-swap / rule-swap / answer-swap), run activation patching **on each shortlisted component individually**. Assemble the role-assignment matrix `S ∈ [0,1]^{|C|×3}`.

**Modularity metrics**:
- `dominance_ratio(c) = max_r S[c, r] / (2nd_max_r S[c, r])` — target `≥ 2.0` per component.
- `dissociation(r) = mean_{c ∈ C_r} S[c, r] − mean_{c ∈ C_r} S[c, r' ≠ r]` — target `≥ 0.1`.
- **Null control**: shuffle role labels 100 times, recompute both metrics; report the p-value of the observed value vs the shuffled distribution.

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`.

**Cmd**: `python scripts/role_dissociation.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --cell k3_chain2_natural --shortlist results/M1_attribution.json --n-pairs 300 --roles fact,rule,answer --gpus 0,1,2 --out results/M4_roles.json`

**Expected output**: `results/M4_roles.json` with:
```json
{
  "role_assignment_matrix_S": [[...]],
  "component_dominance_ratios": [...],
  "role_dissociation": {"fact": <float>, "rule": <float>, "answer": <float>},
  "null_shuffle_p_value": {"fact": <float>, "rule": <float>, "answer": <float>},
  "block_partition": {"C_fact": [...], "C_rule": [...], "C_answer": [...]}
}
```

**Success criterion (C2)**: median dominance ratio ≥ 2× AND all three roles have dissociation ≥ 0.1 AND null-shuffle p-value ≤ 0.01.

**Priority**: MUST-RUN. **Estimated GPU-hours**: 1.8.

---

## M4.stab — Cross-cell stability check (C2 stability)

**Claim(s) verified**: C2 (stability across cells).

**Method**: repeat M4 (abbreviated: 200 pairs per role) on two additional cells — `(k=5, chain=2, natural)` and `(k=3, chain=3, natural)`. Compare block partitions.

**Stability metric**: Jaccard similarity of `C_r` across the three cells, per role r. Target `≥ 0.6`.

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`.

**Cmd**: `python scripts/role_dissociation.py --model ${MODEL_DIR}/Mistral-7B-v0.1 --cell k5_chain2_natural,k3_chain3_natural --shortlist results/M1_attribution.json --n-pairs 200 --roles fact,rule,answer --gpus 0,1,2 --out results/M4stab.json`

**Expected output**: `results/M4stab.json` with per-cell block partitions + pairwise Jaccard scores.

**Success criterion**: pairwise Jaccard ≥ 0.6 for each of the three roles.

**Priority**: MUST-RUN. **Estimated GPU-hours**: 0.8.

---

## M5 — Cross-family verify (Gemma-2-9B)

**Claim(s) verified**: schema recurrence (compressed replay of C1 + C2 + C3 on a different family).

**Method**: replay M1 (attribution screen) + M2 (necessity) + M3 (sufficiency) + M4 (role-dissociation, abbreviated) on **Gemma-2-9B** at the anchor cell (k=3, chain=2, natural, 500 pairs).

**Report format**: family-level comparison — sparsity fraction, three-role structure, necessity/sufficiency recovery. Do NOT expect head indices to match; expect the *schema* to hold if the hypothesis generalises.

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`.

**Cmd**: `python scripts/cross_family_verify.py --model ${MODEL_DIR}/gemma-2-9b --cell k3_chain2_natural --n-pairs 500 --gpus 0,1,2 --out results/M5_gemma9b.json`

**Expected output**: `results/M5_gemma9b.json` mirroring the M1–M4 output shape.

**Success criterion**: sparsity fraction ≤ 0.15 AND role-dissociation ≥ 0.1 (each role) AND necessity + sufficiency recovery ≥ 0.7 (relaxed from 0.8 for cross-family). Any of these failing is a **negative-for-schema-recurrence** result, publishable in its own right.

**Priority**: MUST-RUN. **Estimated GPU-hours**: 2.0.

---

## M5.contingent — Gemma-2-27B (budget permitting)

**Claim(s) verified**: schema recurrence at scale (contingent).

**Guard**: run only if `remaining_budget ≥ 1.5 GPU-h` after M5 completes AND if `${MODEL_DIR}/gemma-2-27b` (or downloaded equivalent) is available. Since `${MODEL_DIR}/gemma-3-27b-*` variants are the only 27B Gemmas locally present, this milestone requires a fresh `gemma-2-27b` (~55 GB) download via HF — Phase 4.5 flags this as a risk; the experiment stage decides.

**Method**: abbreviated replay (M1 + necessity-only + role-dissociation-only, 250 pairs). Gemma-2-27B may need sharding across 2 × 80 GB A800 GPUs (GPUs 0+1); the framework must support this.

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`.

**Cmd**: `python scripts/cross_family_verify.py --model ${MODEL_DIR}/gemma-2-27b --cell k3_chain2_natural --n-pairs 250 --gpus 0,1 --shard 2 --out results/M5c_gemma27b.json`

**Expected output**: `results/M5c_gemma27b.json`.

**Success criterion**: as M5 (relaxed). Contingent — no failure counted if skipped for budget reasons.

**Priority**: NICE-TO-HAVE. **Estimated GPU-hours**: 1.3 (if run).

---

## M6 — Analysis + report aggregation

**Purpose**: aggregate M1–M5 outputs into `results/final_report.md` — one table per claim, dose-response plots (matplotlib PDF), role-assignment matrix heatmap, cross-family recurrence summary.

**Cmd**: `python scripts/aggregate_report.py --results results/ --out results/final_report.md`

**Expected output**: `results/final_report.md`, `results/figures/*.pdf`.

**Priority**: MUST-RUN. **Estimated GPU-hours**: 0.1 (CPU-only).

---

## Milestone dependency graph

```
M0.dataset
    ↓
M0.setup
    ↓
   M1  ──┬──→  M2  ──┐
         ├──→  M3  ──┤
         └──→  M4  ──┤──→ M4.stab ──┐
                     │              ├──→ M5 ──→ M5.contingent (opt) ──┐
                     └──────────────┴──→ M6 ←──────────────────────────┘
```

## Notes on machine markers

- **No `kind: phenomenon-validation` milestone**: `BEHAVIOR_SOURCE=given`, so no M0 phenomenon-validation gate. All M1..M5 have `depends_on:` chained only to their own prerequisites (never to an "M0" gate).
- **`resource_fidelity: not-strict`** at top: cost-aware, but full-scale where budget permits (Mistral-7B main run at full 500 pairs / anchor cell; Gemma-2-9B at 500 pairs; Gemma-2-27B is the only element that is compute-contingent).
- **`method_sensitive`** on M1, M2, M3, M4, M4.stab, M5, M5.contingent: the concrete patching framework (TransformerLens vs nnsight vs a hand-rolled hook layer) chosen by `/mechanism-skills` routing may re-bind `n_pairs`, `sites`, `metric`, and `gpu_hours` — this is *expected* and does not require a plan rewrite.
- **`mechanism_strategy`**: `[Location, Causal Intervention]`, four directions explicitly rejected with reasons (see top metadata block).

## Compute budget summary

| Milestone | Est. GPU-h | Cumulative |
|-----------|-----------|-----------|
| M0.setup | 0.3 | 0.3 |
| M1 | 1.2 | 1.5 |
| M2 | 1.5 | 3.0 |
| M3 | 1.0 | 4.0 |
| M4 | 1.8 | 5.8 |
| M4.stab | 0.8 | 6.6 |
| M5 | 2.0 | 8.6 |
| M6 | 0.1 | 8.7 |
| **Headroom** | **1.3** | **10.0** |
| M5.contingent | 1.3 | (uses headroom) |

Headroom absorbs: a metric re-run, a resample-seed refresh on M3, or the M5.contingent milestone — whichever the pipeline prioritises at that point.

## First-runs to launch (Phase-5 hand-off)

1. **M0.dataset** → build synthetic pool (CPU, ~5 min).
2. **M0.setup** → confirm Mistral-7B loads and hits ≥ 0.75 accuracy on anchor cell (GPU 0, ~15 min).
3. **M1** → attribution-patching screen; produces `results/M1_attribution.json`.
