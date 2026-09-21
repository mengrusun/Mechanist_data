# Experiment Plan — Reproduction of "Sensitivity Meets Sparsity" on Belief Circuits in Pythia

```yaml
resource_fidelity: strict
mechanism_strategy: n/a
chosen_mechanism:
  C1: not-applicable
  C2: fisher-information-matrix-zero-ablation
  C3: checkpoint-analysis-with-zero-ablation
  C4: probe-and-amplify-controller
behavior_source: given
mechanism: given
gpu_hours_total_estimated: ~350h (worst case: all 3 models localize both targets) / ~180h (typical: pythia-2.8b localizes both, pythia-1b localizes at least one)
```

## Global constants

```yaml
project_root: /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18
model_dir: /mnt/quarkfs/share_model/Ptyhia
checkpoint_dir_template: /mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/step{STEP}
belief_core_dir: /data/xuhaoming/belief_loc/data/derived/belief_core
belief_holdout_dir: /data/xuhaoming/belief_loc/data/derived/belief_holdout
pile_dir: /mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled
ppl_sample_tokens: 1048576
ppl_sample_cache: refine-logs/artifacts/ppl_sample.pt
random_head_seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
random_mask_seeds: [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219]
device: cuda
dtype: fp16 for forward / eval; fp32 for Fisher accumulation
above_chance_gate: acc > 0.5 AND lower Wilson 95% CI > 0.5
metric: log-prob comparison (gold > distractor) — see FINAL_PROPOSAL.md
```

Directory layout (created by M1 first-run):
```
refine-logs/artifacts/
├── ppl_sample.pt                       # cached PPL token tensor (~1M tokens)
├── behavioral/{model}/{task}.json      # M1 outputs
├── fisher/{model}/F_{signal}.pt        # M2 Fisher per-parameter tensors
├── fisher/{model}/head_scores/{signal}.json  # M2 head-level aggregated scores
├── masks/{model}/Mask_{target}.json    # M2 candidate head sets
├── hstar/{model}/H_{target}.json       # M2 final localized head sets
├── ablation/{model}/{target}/main.json # M2 zero-ablation main measurement
├── ablation/{model}/{target}/random_head/seed{S}.json  # M2 random-head control
├── ablation/{model}/{target}/random_mask/seed{S}.json  # M2 random-mask control
├── formation/{step}.json               # M3 per-checkpoint results
├── controller/{model}/probe.pt         # M4 frame classifier weights
├── controller/{model}/alpha_grid.json  # M4 hyperparameter search results
├── controller/{model}/ood_results.json # M4 belief_holdout evaluation
└── controller/{model}/prompt_hint_baseline.json  # M4 prompt-hint baseline
```

---

## M1: Behavioral Scaling Evaluation (Claim 1)

- **id**: M1
- **title**: Behavioral scaling of world_knowledge / personal_belief / attributed_belief across pythia-{410m, 1b, 2.8b}
- **chosen_mechanism**: not-applicable (behavioural-only, no mechanism intervention)
- **depends_on**: []
- **priority**: MUST-RUN
- **claim_covered**: Claim 1 — Scale-Dependent Emergence
- **models**: [pythia-410m, pythia-1b, pythia-2.8b]  (subset of the allowed 3; all full-weight)
- **datasets**:
  - `world_knowledge`: /data/xuhaoming/belief_loc/data/derived/belief_core/reality.jsonl, `used_n = 227`  (FULL)
  - `personal_belief`: /data/xuhaoming/belief_loc/data/derived/belief_core/believe_truth.jsonl, `used_n = 681`  (FULL)
  - `attributed_belief`: /data/xuhaoming/belief_loc/data/derived/belief_core/follow_belief.jsonl, `used_n = 681`  (FULL)
- **method-specific criteria**:
  - Report the 3×3 accuracy matrix with Wilson 95% CI per cell.
  - Chance = 0.5 (binary >-comparison of gold vs distractor log-probs).
  - "Above-chance" gate for M2: acc > 0.5 AND lower Wilson 95% CI > 0.5.
  - Additional reporting: written characterization of whether the two belief curves are (i) monotonically increasing, (ii) mismatched in slope, (iii) crossing, or (iv) diverging at some scale (above vs at chance).
- **grid**:
  ```yaml
  model: [pythia-410m, pythia-1b, pythia-2.8b]
  task:  [world_knowledge, personal_belief, attributed_belief]
  ```
  (9 runs — one per (model, task) combination, all independent)
- **cmd template**:
  ```
  python scripts/m1_behavioral_eval.py \
    --model ${model} \
    --task ${task} \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --dtype fp16 \
    --batch-size 32 \
    --output refine-logs/artifacts/behavioral/${model}/${task}.json
  ```
- **id template**: `m1_${model}_${task}`
- **expected output (template)**: `refine-logs/artifacts/behavioral/${model}/${task}.json` containing `{acc, correct_count, total, wilson_ci_low, wilson_ci_high}`.
- **estimated GPU-hours per run**: 0.05h - 0.2h (n≤681, forward-only, no gradients). Total M1: ≈ 1h.
- **acceptance for downstream**: writes `refine-logs/artifacts/behavioral/above_chance_gate.json` — a summary listing which (model, target) pairs cleared the above-chance gate. M2 reads this to decide which localization runs are applicable.

---

## M2: Fisher-Information Localization of Belief Heads (Claim 2)

This milestone group has four sub-milestones (M2.1 → M2.4). The group depends on M1 (needs the above-chance gate). The four sub-milestones depend serially: **M2.1 → M2.2 → M2.3 → M2.4**.

### M2.1: Fisher signal computation

- **id**: M2.1
- **title**: Compute per-parameter Fisher signals for personal / attributed / knowledge on each pythia model
- **chosen_mechanism**: fisher-information-matrix-zero-ablation
- **depends_on**: [M1]
- **priority**: MUST-RUN
- **claim_covered**: Claim 2 (Fisher construction phase)
- **models**: [pythia-410m, pythia-1b, pythia-2.8b]  (compute all — cheap to run even if the model later fails the above-chance gate; keeps analysis symmetric)
- **datasets**:
  - `F_attributed`: /data/xuhaoming/belief_loc/data/derived/belief_core/follow_belief.jsonl, `filter: person ∈ {james, mary}`, `used_n = 454`
  - `F_personal`:   /data/xuhaoming/belief_loc/data/derived/belief_core/believe_truth.jsonl, `filter: person ∈ {james, mary}`, `used_n = 454`
  - `F_knowledge`:  /data/xuhaoming/belief_loc/data/derived/belief_core/reality.jsonl,      `filter: none`,                     `used_n = 227`
- **method-specific criteria**:
  - Empirical Fisher: `F_i = (1/N) Σ_x (∂ log p_θ(y⁺ | x) / ∂θ_i)²`.
  - Gradients computed w.r.t. the gold continuation `y⁺` (per the metric definition).
  - fp32 accumulation.
  - Save per-parameter Fisher as a torch state dict keyed by parameter name.
  - Also save per-attention-head aggregated Fisher: for each attention head `h`, sum `F_i` over all parameters `θ_i` in `{W_Q^h, W_K^h, W_V^h, W_O^h}` (the four projection matrices attached to head `h` — for Pythia these live in the fused `query_key_value` and `dense` matrices; identify head `h`'s parameter block by its output-head slice).
  - Also save a "top-k parameter set" per signal at `k = 0.1%` (for `F_attributed`, `F_personal`) and `k = 1%` (for `F_knowledge`) for the mask construction in M2.2.
- **grid**:
  ```yaml
  model:  [pythia-410m, pythia-1b, pythia-2.8b]
  signal: [F_attributed, F_personal, F_knowledge]
  ```
  (9 runs)
- **cmd template**:
  ```
  python scripts/m2_1_fisher.py \
    --model ${model} \
    --signal ${signal} \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --grad-dtype fp32 \
    --batch-size 1 \
    --output refine-logs/artifacts/fisher/${model}/${signal}.pt \
    --head-scores-output refine-logs/artifacts/fisher/${model}/head_scores/${signal}.json
  ```
- **expected output (template)**:
  - `refine-logs/artifacts/fisher/${model}/${signal}.pt` — full per-parameter Fisher tensor state dict.
  - `refine-logs/artifacts/fisher/${model}/head_scores/${signal}.json` — per-head aggregated Fisher (dict keyed by (layer, head)).
- **estimated GPU-hours per run**: pythia-410m ≈ 0.3h, pythia-1b ≈ 0.6h, pythia-2.8b ≈ 1.5h → total M2.1: ≈ 7h.

### M2.2: Mask construction and candidate-head ranking

- **id**: M2.2
- **title**: Build target-specific AND-NOT masks and rank candidate heads
- **chosen_mechanism**: fisher-information-matrix-zero-ablation
- **depends_on**: [M2.1]
- **priority**: MUST-RUN
- **claim_covered**: Claim 2 (mask construction)
- **models**: [pythia-410m, pythia-1b, pythia-2.8b]
- **method-specific criteria (verbatim)**:
  - `Mask_attributed = top 0.1% of F_attributed AND NOT top 1% of F_knowledge`
  - `Mask_personal   = top 0.1% of F_personal   AND NOT top 1% of F_knowledge`
  - Per-parameter mask is materialized first; then per-head candidate score = fraction of the head's parameters that fall in the mask.
  - Rank heads by candidate score descending. Ties broken by aggregated Fisher magnitude descending.
  - Also compute the jackknife stability: split each of `F_attributed`, `F_personal` into two 227-example halves (deterministic seed 0), compute a "half-Fisher" from each, and report Spearman rank-correlation of per-head candidate scores between halves. Threshold `ρ ≥ 0.6` for "stable" flag (informational only — does not gate acceptance).
- **grid**:
  ```yaml
  model:  [pythia-410m, pythia-1b, pythia-2.8b]
  target: [personal, attributed]
  ```
  (6 runs — one per (model, target))
- **cmd template**:
  ```
  python scripts/m2_2_masks.py \
    --model ${model} \
    --target ${target} \
    --fisher-target refine-logs/artifacts/fisher/${model}/F_${target}.pt \
    --fisher-knowledge refine-logs/artifacts/fisher/${model}/F_knowledge.pt \
    --top-target 0.001 \
    --top-knowledge 0.01 \
    --output refine-logs/artifacts/masks/${model}/Mask_${target}.json \
    --output-ranking refine-logs/artifacts/masks/${model}/RankedHeads_${target}.json \
    --jackknife-half-a refine-logs/artifacts/fisher/${model}/F_${target}_halfA.pt \
    --jackknife-half-b refine-logs/artifacts/fisher/${model}/F_${target}_halfB.pt \
    --jackknife-report refine-logs/artifacts/masks/${model}/Jackknife_${target}.json
  ```
  (Note: the two half-Fisher tensors are computed as part of M2.1's script by adding a `--jackknife` flag; this milestone just consumes them.)
- **expected output (template)**:
  - `refine-logs/artifacts/masks/${model}/Mask_${target}.json` — parameter-level mask summary + head-level candidate list.
  - `refine-logs/artifacts/masks/${model}/RankedHeads_${target}.json` — descending-ranked list of candidate heads with per-head scores.
  - `refine-logs/artifacts/masks/${model}/Jackknife_${target}.json` — Spearman ρ + stable flag.
- **estimated GPU-hours per run**: near-zero (CPU tensor ops on already-computed Fisher). Total M2.2: ≈ 0.1h.

### M2.3: Smallest-head-set search via zero-ablation

- **id**: M2.3
- **title**: Deterministic greedy search for the smallest attention-head set satisfying all four criteria
- **chosen_mechanism**: fisher-information-matrix-zero-ablation
- **depends_on**: [M2.2]
- **priority**: MUST-RUN
- **claim_covered**: Claim 2 (localization core)
- **models**: only those `(model, target)` pairs that cleared M1's above-chance gate (typically pythia-2.8b for both targets; possibly pythia-1b for both; pythia-410m TBD)
- **datasets**:
  - Target task (per (model, target)): the FULL belief dataset for `target` (n=681).
  - Off-target belief task: the FULL other belief dataset (n=681).
  - World_knowledge: FULL n=227.
  - PPL: `refine-logs/artifacts/ppl_sample.pt` (cached 1M-token sample).
- **method-specific criteria (verbatim four criteria — pass/fail per candidate set):**
  1. C2a: `acc(target) - acc_ablated(target) ≥ 0.30`
  2. C2b: target-task drop > mean(random-head baseline drops) + 2σ *(computed later in M2.4; used here in the deferred acceptance check)*
  3. C2c: `acc(other belief) - acc_ablated(other belief) ≤ 0.10` AND `acc(world_knowledge) - acc_ablated(world_knowledge) ≤ 0.10`
  4. C2d: `PPL_ablated / PPL_clean ≤ 1.05` on `refine-logs/artifacts/ppl_sample.pt`
- **search discipline (R2 — pinned deterministic):**
  - **Greedy-add**: start `S = {}`. For each candidate head `h` in the ranked list (M2.2 output), add `h` to `S`. Evaluate the 4 criteria (excluding C2b which needs M2.4). Stop at the first `|S|` where {C2a, C2c, C2d} all pass. Call this `S⁺`.
  - **Greedy-remove sanity check**: iterate through `S⁺`; for each `h`, attempt `S⁺ \ {h}`; if the resulting set still satisfies {C2a, C2c, C2d}, replace `S⁺ := S⁺ \ {h}`. Repeat until no removal succeeds (fixed point). Typically 1-3 rounds.
  - **Termination**: if greedy-add reaches `|S| = 30` without satisfying {C2a, C2c, C2d}, terminate with `H* = None`, status `not_localized`. Do NOT alter thresholds.
- **grid**:
  ```yaml
  # Populated dynamically from M1's above_chance_gate output. Worst case:
  model:  [pythia-410m, pythia-1b, pythia-2.8b]
  target: [personal, attributed]
  ```
  (up to 6 configs; likely fewer)
- **cmd template**:
  ```
  python scripts/m2_3_search.py \
    --model ${model} \
    --target ${target} \
    --ranked-heads refine-logs/artifacts/masks/${model}/RankedHeads_${target}.json \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --ppl-sample refine-logs/artifacts/ppl_sample.pt \
    --max-heads 30 \
    --dtype fp16 \
    --search-log refine-logs/artifacts/hstar/${model}/${target}_search_log.json \
    --output-hstar refine-logs/artifacts/hstar/${model}/H_${target}.json \
    --output-main-metrics refine-logs/artifacts/ablation/${model}/${target}/main.json
  ```
- **expected output (template)**:
  - `refine-logs/artifacts/hstar/${model}/H_${target}.json` — the final `H*` (list of `(layer, head)` tuples), its size, and its parameter-count summary; status `localized` / `not_localized`.
  - `refine-logs/artifacts/hstar/${model}/${target}_search_log.json` — per-iteration criteria evaluations, so the search is fully audited.
  - `refine-logs/artifacts/ablation/${model}/${target}/main.json` — the four metrics for `H*` (target acc drop, off-target drops, PPL ratio) — needed for M2.4 acceptance.
- **estimated GPU-hours per run** (per (model, target)): pythia-410m ≈ 5h, pythia-1b ≈ 10h, pythia-2.8b ≈ 25h. Worst case 6 configs → ~80h. Typical case (pythia-2.8b + pythia-1b, both targets each) → ~70h.
  - Cost breakdown per iteration: 3 eval tasks (681+681+227 examples each = ~1600 forward passes, ~15 min) + PPL on 1M tokens (~30 min for pythia-2.8b, ~10 min for pythia-1b, ~5 min for pythia-410m).

### M2.4: Random-head + random-mask controls (20 + 20 per H*)

- **id**: M2.4
- **title**: 20 random-head + 20 random-mask controls per final H*
- **chosen_mechanism**: fisher-information-matrix-zero-ablation
- **depends_on**: [M2.3]
- **priority**: MUST-RUN
- **claim_covered**: Claim 2 (control baselines + C2b acceptance)
- **models**: those with `status = localized` in M2.3 output.
- **method-specific criteria**:
  - Random-head control (seeds 100-119): 20 uniformly-random head subsets of the same head count as `|H*|`, drawn without replacement over all attention heads.
  - Random-mask control (seeds 200-219): 20 uniformly-random parameter subsets of the same total parameter count as the sum of parameters in `{W_Q, W_K, W_V, W_O}` across all heads in `H*`; zeroed at parameter level.
  - For each control, run zero-ablation and measure the same four metrics as M2.3 main.
  - **C2b acceptance test**: from the 20 random-head target-accuracy drops, compute `mean_rh` and `σ_rh`. Accept M2's localization for this `(model, target)` if the main `H*` target-accuracy drop (from M2.3) satisfies `drop_H* > mean_rh + 2 σ_rh`.
- **grid** (per M2.3-successful `(model, target)`):
  ```yaml
  model:  <dynamic from M2.3>
  target: <dynamic from M2.3>
  kind:   [random_head, random_mask]
  seed:   [100..119 for random_head; 200..219 for random_mask]
  ```
  Per successful `(model, target)`: 20 + 20 = 40 runs.
- **cmd template**:
  ```
  python scripts/m2_4_control.py \
    --model ${model} \
    --target ${target} \
    --hstar refine-logs/artifacts/hstar/${model}/H_${target}.json \
    --kind ${kind} \
    --seed ${seed} \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --ppl-sample refine-logs/artifacts/ppl_sample.pt \
    --output refine-logs/artifacts/ablation/${model}/${target}/${kind}/seed${seed}.json
  ```
- **acceptance report** (`scripts/m2_4_acceptance.py`, one call after all controls complete for a `(model, target)`):
  - Reads `main.json` + 20 random-head control jsons + 20 random-mask control jsons.
  - Computes `mean_rh, σ_rh, threshold_2sigma = mean_rh + 2 σ_rh`.
  - Prints the full 4-criteria table: {C2a: pass/fail, C2b: pass/fail, C2c: pass/fail, C2d: pass/fail}; localization status = ALL_PASS ⇒ `localized`, else `partially_localized` or `not_localized`.
  - Writes `refine-logs/artifacts/hstar/${model}/H_${target}_acceptance.json` with the full breakdown.
- **expected output (template)**:
  - `refine-logs/artifacts/ablation/${model}/${target}/${kind}/seed${seed}.json` (40 files per (model, target))
  - `refine-logs/artifacts/hstar/${model}/H_${target}_acceptance.json` (acceptance summary — the final Claim-2 verdict)
- **estimated GPU-hours per run**: per control run ≈ 0.5h (3 task evals + PPL eval, forward-only). Per (model, target): 40 × 0.5 = 20h. Worst case (6 (model, target) pairs localized) = 120h. Typical (2-4 pairs localized) = 40-80h.

---

## M3: Formation-Window Analysis on Pythia-1B (Claim 3)

- **id**: M3
- **title**: Behavioral + causal-intervention trajectories across pythia-1b intermediate checkpoints
- **chosen_mechanism**: checkpoint-analysis-with-zero-ablation
- **depends_on**: [M2]  (needs pythia-1b `H*_personal` and `H*_attributed` from M2.3/M2.4 on step 143000)
- **priority**: MUST-RUN
- **claim_covered**: Claim 3 — Formation Window
- **models**: pythia-1b intermediate checkpoints (native Pythia schedule, all 154 available on disk)
- **checkpoint list (R4 — pinned; native Pythia schedule)**:
  ```
  step0, step1, step2, step4, step8, step16, step32, step64, step128, step256, step512,
  step1000, step2000, step3000, step4000, step5000, step6000, step7000, step8000, step9000, step10000,
  step11000, step12000, step13000, step14000, step15000, step16000, step17000, step18000, step19000, step20000,
  step21000, step22000, step23000, step24000, step25000, step26000, step27000, step28000, step29000, step30000,
  step31000, step32000, step33000, step34000, step35000, step36000, step37000, step38000, step39000, step40000,
  step41000, step42000, step43000, step44000, step45000, step46000, step47000, step48000, step49000, step50000,
  step51000, step52000, step53000, step54000, step55000, step56000, step57000, step58000, step59000, step60000,
  step61000, step62000, step63000, step64000, step65000, step66000, step67000, step68000, step69000, step70000,
  step71000, step72000, step73000, step74000, step75000, step76000, step77000, step78000, step79000, step80000,
  step81000, step82000, step83000, step84000, step85000, step86000, step87000, step88000, step89000, step90000,
  step91000, step92000, step93000, step94000, step95000, step96000, step97000, step98000, step99000, step100000,
  step101000, step102000, step103000, step104000, step105000, step106000, step107000, step108000, step109000, step110000,
  step111000, step112000, step113000, step114000, step115000, step116000, step117000, step118000, step119000, step120000,
  step121000, step122000, step123000, step124000, step125000, step126000, step127000, step128000, step129000, step130000,
  step131000, step132000, step133000, step134000, step135000, step136000, step137000, step138000, step139000, step140000,
  step141000, step142000, step143000
  ```
  (154 checkpoints total)
- **datasets**:
  - `world_knowledge`: reality.jsonl, `used_n = 227` (FULL)
  - `personal_belief`: believe_truth.jsonl, `used_n = 681` (FULL)
  - `attributed_belief`: follow_belief.jsonl, `used_n = 681` (FULL)
- **method-specific criteria (per checkpoint, evaluation types)**:
  - **Behavioural**: `acc(task, step)` for all 3 tasks (no intervention).
  - **Causal (H*_personal ablation)**: `acc_ablated(H*_personal, task, step)` for all 3 tasks (zero-ablate the pythia-1b `H*_personal` head set at this checkpoint).
  - **Causal (H*_attributed ablation)**: `acc_ablated(H*_attributed, task, step)` for all 3 tasks.
  - Per checkpoint: 3 (behavioural) + 3 (H*_personal ablation) + 3 (H*_attributed ablation) = **9 measurements**.
- **emergence criteria (pinned BEFORE the sweep, R4)**:
  - **Behavioural-emergence step** `t*(target)`: first checkpoint whose `acc(target, t) ≥ 0.60` AND persistence: at least 2 of the next 3 recorded checkpoints also have `acc(target, ·) ≥ 0.60`. Report per belief target.
  - **Causal-emergence step** `t†(target)`: first checkpoint whose `Δ(H*_target, target, t) := acc(target, t) - acc_ablated(H*_target, target, t) ≥ 0.20`, with the same 2-of-next-3 persistence check. Report per belief target.
  - **Formation window** = `[t*(target), t†(target)]` if both are observed; `{t*(target)}` if only behavioural; "not observed" otherwise.
  - "Distinctness" for Claim 3 = the intervals for `personal` and `attributed` are non-identical (different `t*` or different `t†`).
- **grid**:
  ```yaml
  step: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, ..., 143000]  # 154 checkpoints
  ```
  (154 runs, one per checkpoint; each run does all 9 measurements)
- **cmd template**:
  ```
  python scripts/m3_formation.py \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --model pythia-1b \
    --checkpoint step${step} \
    --hstar-personal refine-logs/artifacts/hstar/pythia-1b/H_personal.json \
    --hstar-attributed refine-logs/artifacts/hstar/pythia-1b/H_attributed.json \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --dtype fp16 \
    --output refine-logs/artifacts/formation/step${step}.json
  ```
- **id template**: `m3_step${step}`
- **expected output (template)**: `refine-logs/artifacts/formation/step${step}.json` with the 9-cell measurement.
- **post-sweep aggregation** (`scripts/m3_aggregate.py`, one call after all checkpoints complete):
  - Reads all 154 per-checkpoint jsons.
  - Constructs the behavioural trajectory + causal trajectory tables.
  - Applies the emergence criteria; identifies `t*`, `t†` per belief target; reports formation windows.
  - Writes `refine-logs/artifacts/formation/summary.json` (per-target windows + trajectory tables) and `refine-logs/artifacts/formation/summary.md` (the human-readable Claim-3 report).
- **estimated GPU-hours per run**: per checkpoint ≈ 0.3h (9 evals on pythia-1b, forward-only, cheap). Total M3: 154 × 0.3 ≈ 47h. If pythia-1b did not localize BOTH belief targets in M2, the corresponding causal trajectory is skipped and total drops (still one belief target usually available).
- **failure mode**: if pythia-1b's `H*_target` is `None` (M2.3 termination), the causal trajectory for that target is `not_applicable` — report only the behavioural trajectory for that target. Do NOT drop C3 entirely.

---

## M4: Probe-and-Amplify Dynamic Controller (Claim 4)

- **id**: M4
- **title**: Frame-classifier probe on early layers + amplification of matched Claim-2 belief heads; OOD evaluation vs prompt-hint baseline
- **chosen_mechanism**: probe-and-amplify-controller
- **depends_on**: [M2]  (needs `H*_personal` AND `H*_attributed` on the chosen model)
- **priority**: MUST-RUN
- **claim_covered**: Claim 4 — Dynamic Controllability
- **models**: every pythia model for which M2.4 delivered `H*_personal` AND `H*_attributed` both `localized`. Priority order (largest first): pythia-2.8b → pythia-1b → pythia-410m.
- **datasets**:
  - **Training** (frame classifier + controller α tuning): /data/xuhaoming/belief_loc/data/derived/belief_core/ (all three files; 227 + 681 + 681 = 1589 examples). Stratified 80/20 split with seed 0 (train/val).
  - **OOD evaluation** (controller + prompt-hint baseline): /data/xuhaoming/belief_loc/data/derived/belief_holdout/ (three files, verified present: reality.jsonl, believe_truth.jsonl, follow_belief.jsonl — used_n = actual file counts, discovered at run time).
  - **PPL** (controller-general-LM-preservation check): `refine-logs/artifacts/ppl_sample.pt` (same cached 1M-token sample as M2/M3).

### M4.1: Frame classifier training

- **id**: M4.1
- **depends_on**: [M2]
- **method-specific criteria**:
  - Probing layers: `L_ctrl = min layer index over all heads in H*_personal ∪ H*_attributed`. Probing layers = `[max(0, L_ctrl - 3), L_ctrl - 2, L_ctrl - 1]`.
  - Probe input = concatenated hidden states at the last input token across the probing layers → dim = `3 × hidden_dim`.
  - Classifier: MLP with hidden dim 256, ReLU, dropout 0.1, 3 output classes.
  - Optimizer: AdamW, lr 5e-4, weight decay 1e-4, batch size 64, 20 epochs, early stopping on val loss (patience 5).
  - Report val accuracy (3-way).
- **grid**:
  ```yaml
  model: <dynamic — the chosen models list>
  ```
- **cmd template**:
  ```
  python scripts/m4_1_train_probe.py \
    --model ${model} \
    --hstar-personal refine-logs/artifacts/hstar/${model}/H_personal.json \
    --hstar-attributed refine-logs/artifacts/hstar/${model}/H_attributed.json \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --split-seed 0 \
    --hidden-dim 256 \
    --dropout 0.1 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --batch-size 64 \
    --epochs 20 \
    --patience 5 \
    --dtype fp16 \
    --output refine-logs/artifacts/controller/${model}/probe.pt \
    --report refine-logs/artifacts/controller/${model}/probe_report.json
  ```
- **expected output**: probe weights + per-epoch training log + final val accuracy.
- **estimated GPU-hours per run**: 0.5h (probing hidden-state extraction is dominant; classifier training itself is fast).

### M4.2: Amplification-magnitude grid search on belief_core val

- **id**: M4.2
- **depends_on**: [M4.1]
- **method-specific criteria (R5)**:
  - Grid: `α_personal ∈ {1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`, `α_attributed ∈ {1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` → 36 combinations.
  - For each `(α_p, α_a)`: apply controller to the belief_core val split (~318 examples). Compute:
    - `net_improvement = recovered - degraded` on the two belief tasks combined (relative to α=1.0 baseline on the val split).
    - `Δ_wk = acc(world_knowledge, val) - acc_controller(world_knowledge, val)` — soft guardrail.
  - **Selection rule**: pick the `(α_p*, α_a*)` maximizing `net_improvement` subject to `Δ_wk ≤ 0.05` (do not destroy factual knowledge just to boost belief).
  - If no combination clears `Δ_wk ≤ 0.05`, relax to `≤ 0.10` and re-select; if still none, pick the combination minimizing `Δ_wk` (and note the failure of the soft guardrail in the report).
- **grid**:
  ```yaml
  model:        <dynamic>
  alpha_p:      [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
  alpha_a:      [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
  ```
- **cmd template**:
  ```
  python scripts/m4_2_alpha_search.py \
    --model ${model} \
    --probe refine-logs/artifacts/controller/${model}/probe.pt \
    --hstar-personal refine-logs/artifacts/hstar/${model}/H_personal.json \
    --hstar-attributed refine-logs/artifacts/hstar/${model}/H_attributed.json \
    --alpha-personal ${alpha_p} \
    --alpha-attributed ${alpha_a} \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --split-seed 0 \
    --split val \
    --dtype fp16 \
    --output refine-logs/artifacts/controller/${model}/alpha_grid/a_p_${alpha_p}_a_a_${alpha_a}.json
  ```
- **post-grid selection**: `scripts/m4_2_select.py` reads all 36 jsons, applies the selection rule, writes `refine-logs/artifacts/controller/${model}/alpha_selected.json` = `{alpha_p*, alpha_a*, val_net_improvement, val_delta_wk}`.
- **estimated GPU-hours per run**: 0.1h per (α_p, α_a) config → 36 × 0.1 = 3.6h per model.

### M4.3: OOD evaluation on belief_holdout + prompt-hint baseline

- **id**: M4.3
- **depends_on**: [M4.2]
- **method-specific criteria**:
  - Run three separate OOD evaluations on `belief_holdout/`:
    - **(a) Un-amplified baseline**: no probe, no amplification, no prompt hint. All three tasks. Records `acc_baseline_no_control_OOD(task)` for each.
    - **(b) Controller**: probe predicts frame, amplify matched heads with `(α_p*, α_a*)` from M4.2, no prompt hint. Records `acc_controller_OOD(task)` for each, plus `frame_acc_OOD` (3-way classification accuracy).
    - **(c) Prompt-hint baseline**: prefixed prompts (see FINAL_PROPOSAL.md for exact strings), no probe, no amplification. Records `acc_prompt_hint_OOD(task)` for each.
  - Recovered / degraded / net_improvement: computed per-example between (a) and (b).
  - PPL check: apply controller to the PPL sample (1M tokens); records `PPL_controller_OOD` — should not blow up general LM ability.
- **grid**:
  ```yaml
  model: <dynamic>
  eval:  [baseline_no_control, controller, prompt_hint]
  ```
- **cmd template**:
  ```
  python scripts/m4_3_ood_eval.py \
    --model ${model} \
    --probe refine-logs/artifacts/controller/${model}/probe.pt \
    --hstar-personal refine-logs/artifacts/hstar/${model}/H_personal.json \
    --hstar-attributed refine-logs/artifacts/hstar/${model}/H_attributed.json \
    --alpha-selected refine-logs/artifacts/controller/${model}/alpha_selected.json \
    --eval ${eval} \
    --ood-root /data/xuhaoming/belief_loc/data/derived/belief_holdout \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --ppl-sample refine-logs/artifacts/ppl_sample.pt \
    --dtype fp16 \
    --output refine-logs/artifacts/controller/${model}/ood_${eval}.json
  ```
- **post-eval aggregation** (`scripts/m4_3_report.py`, one call per model after all three eval jsons written):
  - Reads the three eval jsons + probe report.
  - Computes recovered / degraded / net_improvement (per belief task, and combined).
  - Writes `refine-logs/artifacts/controller/${model}/M4_report.json` (machine-readable) and `refine-logs/artifacts/controller/${model}/M4_report.md` (Claim-4 human-readable report).
- **expected output (template)**:
  - Per (model, eval) json.
  - Per-model M4 report (json + md).
- **estimated GPU-hours per run**: 0.5h - 1h per (model, eval) → 3 × 0.5-1 = 1.5-3h per model. Total M4 (all sub-milestones): ≈ 6-8h per model. For 1-2 successful models: 6-16h.

---

## Milestone dependency graph

```
M1 (behavioural scaling)
  └──> M2.1 (Fisher signals)
         └──> M2.2 (masks + rankings)
                └──> M2.3 (smallest H* search)
                       └──> M2.4 (20+20 controls + acceptance)
                              ├──> M3  (formation window — only if pythia-1b localizes)
                              └──> M4.1 (frame classifier training)
                                     └──> M4.2 (α grid search)
                                            └──> M4.3 (OOD eval + prompt-hint baseline)
```

## Consolidated GPU-hours estimate

| Milestone | Runs | Est. per-run | Total |
|---|---|---|---|
| M1 | 9 | 0.05-0.2h | ~1h |
| M2.1 (Fisher, per model+signal) | 9 | 0.3-1.5h | ~7h |
| M2.2 (masks) | 6 | ~0h | ~0.1h |
| M2.3 (search, per model+target) | 2-6 | 5-25h | 40-80h |
| M2.4 (controls, 40 per H*) | 40 × 2-6 | 0.5h | 40-120h |
| M3 (checkpoints) | 154 | 0.3h | ~47h |
| M4 (probe + α + OOD, per model) | ~40 | 0.1-1h | 6-16h |
| **Grand total** | | | **~140-350h** |

Typical case (pythia-2.8b localizes both, pythia-1b localizes at least one): ~180h. Worst case (all 3 models localize both): ~350h. Best case (pythia-2.8b only, one target): ~90h.

## Run order (queue-friendly)

1. **Wave 1** (parallel, ~1h): M1 grid (9 runs)
2. **Wave 2** (parallel, ~7h): M2.1 grid (9 Fisher computations); create PPL cache as a one-off during Wave 2
3. **Wave 3** (near-instant): M2.2 masks (6 runs)
4. **Wave 4** (serial per (model, target), can parallelize across configs): M2.3 smallest-set search
5. **Wave 5** (parallel per H*): M2.4 controls (40 per H*)
6. **Wave 6a** (parallel with 6b, gated on pythia-1b M2.4 success): M3 formation sweep (154 runs)
7. **Wave 6b** (parallel with 6a, gated on any-model M2.4 success): M4 controller (probe → α → OOD)
8. **Wave 7** (report writing, aggregation scripts): C1 / C2 / C3 / C4 reports.

## Notes on scripts to implement

Every `scripts/mX_*.py` file above needs to exist under `/mnt/quarkfs/xuweihong/MECHANICA_exps/exp18/scripts/`. The implementation is the experiment stage's job (Workflow 1.5 / `/auto-experiment`). This plan pins the *interface* — what flags each script accepts and what output shape it writes — so the experiment stage can implement each script without rediscovering the design.

For Pythia specifically, note that the model uses fused `query_key_value` matrices (all three QKV projections in one weight); the per-head slicing must correctly identify the `W_Q^h, W_K^h, W_V^h` rows/columns for head `h` and the corresponding output-projection columns in the dense output matrix `W_O^h`. Reference implementation: HuggingFace `GPTNeoXAttention._split_heads` for QKV splitting; see [`transformers/models/gpt_neox`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_neox).
