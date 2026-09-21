# EXPERIMENT PLAN — Emotional Framing in Prompts as a Weak, Input-Dependent Signal

<!-- Top-metadata machine markers — English fields verbatim regardless of report language. -->

```yaml
behavior_source: given
mechanism: discovery
# resource_fidelity: NOT stamped — this is `given` + `discovery`, cost-aware.
mechanism_strategy:
  directions: [Location, Causal Intervention]   # in execution order
  rejected:
    - Tuning & Editing — C4 (EmotionRL) is input-space adaptive selection, not weight-space editing; the mechanism claim is about the residual-stream frame direction that mediates C1–C3, not tuning Qwen3-14B.
    - Formation Tracing — training-time genesis is out of scope; too expensive for the 10-GPU-h budget on a 14B model.
    - Decision Auditing — no spurious-correlation reliability audit is requested; the phenomenon itself is the target.
  note: The mechanism claim is that some low-rank residual-stream direction (or a small attention-head set) on Qwen3-14B early-to-mid layers carries the emotional frame and causally modulates per-item task accuracy; Location + Causal Intervention are the shortest strategy that lands this claim.
  optional_add_on: Unit Interpretation (SAE / vocab projection) is a cheap add-on only if Location produces a clean shortlist.
project:
  cwd: /data/zhenqian/Reproduction1/mechanica/emotion/emotion_prompts
  data_dir: /data/zhenqian/data
  model_dir: /data/zhenqian/models
  gpus_allowed: [1, 2, 3, 5, 6]
  gpu_budget_hours: 10
  conda_env: required
  language: en
```

---

## Claims → Milestones map

| Claim | Statement (measurable predicate) | Verified by |
|---|---|---|
| C1 | Per-emotion mean|Δaccuracy vs neutral| ≤ format-perturbation noise floor **and** per-item sign-consistency ∈ [0.4, 0.6] on Qwen3-14B/GSM8K, across 12 emotion×intensity × 2 wording sources | M2 (+ M2b noise floor) |
| C2 | inter-emotion spread on SocialIQA ≥ 2× that on GSM8K; ordering `spread_social > spread_factual > spread_math` in ≥ 2 of 3 comparisons | M2 + M3 + M4 |
| C3a | No emotion is argmax across all 3 task families; argmax identity flips ≥ once | M4 (post-hoc on M2/M3) |
| C3b | For ≥ 3 of 6 emotions on GSM8K, paired Δ(intensity-2 − intensity-1) 95% CI straddles 0 or reverses sign | M4 |
| C4 | acc(π_θ) > acc(neutral) AND acc(π_θ) > acc(e*) on held-out GSM8K, both with lower 95% CI > 0 across ≥ 3 seeds | M7 |
| CM (mechanism) | Some low-rank residual-stream direction on Qwen3-14B carries the emotion identity (Location) and causally modulates per-item GSM8K Δaccuracy (Intervention), with matched-length filler controls null | M5 + M6 |

---

## M1: Prompt Corpus Construction

**Statement**: build the 26-condition prompt-template corpus that all other milestones reuse.
**Data**: none (corpus is generated).
**Models**: dmxapi `gpt-5.4` service (task.md `BASE_URL=https://www.dmxapi.cn/v1`, `MODEL=gpt-5.4`, API key + proxy-bypass per task.md).
**Method**: for each of 6 basic emotions × 2 intensities:
  - **Human-written variants**: adapt EmotionPrompt-style stimulus sentences from arXiv:2307.11760 (e.g., happiness×1 = "I'm glad to work through this problem with you.", happiness×2 = "I'm thrilled and grateful for the chance to solve this with you!") — one per (emotion, intensity), curated by hand into a JSON file with length in tokens noted.
  - **LLM-generated variants**: call dmxapi `gpt-5.4` with a prompt requesting an emotional-stimulus sentence at the specified emotion+intensity, length-matched (±10 tokens) to the human counterpart. Three candidates per cell; pick the closest length match.
  - **Neutral baseline**: "Please solve the following problem." (task-agnostic; used identically for GSM8K / SocialIQA / MedQA — with each dataset's canonical suffix added downstream).
  - **Length-matched non-emotional filler** (specificity control): a neutral affirmation-style sentence of matched token length to the intensity-2 emotional prefixes, e.g. "This is a straightforward problem-solving exercise for us to work through carefully today."

**Outputs**:
- `data/prefixes/prefixes.json` — schema `{condition_id, emotion, intensity, wording_source, text, n_tokens_qwen}` × 26 rows.
- `data/prefixes/format_noise_variants.json` — 8 neutral-preserving format perturbations of the neutral prefix (whitespace, casing, list markers, instruction paraphrase) for the noise-floor calibration.

**Priority**: MUST-RUN
**Estimated GPU-hours**: 0.0 (API + local text-processing)
**Depends on**: (none)

---

## M2: C1 Primary — Static Prefix × GSM8K on Qwen3-14B

**Statement (C1)**: For each of the 26 prefix conditions, macro accuracy on Qwen3-14B / GSM8K, per-item Δaccuracy vs neutral, mean|Δ|, and per-item sign-consistency.

**Data**: GSM8K test split, first N=500 items (deterministic order; `datasets.load_from_disk('$DATA_DIR/gsm8k')` or HF download).
**Models**: **Qwen3-14B** (bf16, vLLM, prefix-caching disabled across conditions), served locally from `$MODEL_DIR/Qwen3-14B` (symlinked into cwd).
**Method**: for each of 26 conditions × 500 GSM8K items, generate a CoT completion at T=0.0, max_new_tokens=256, deterministic; parse the answer via `##\s*(-?\d+)`; compare to gold; log `(condition_id, item_id, correct, generated_text, prompt_tokens, gen_tokens, prefix_activations_layer_i for i∈{0,4,8,12,16,20,24,28,32,36})` — the last-prefix-token residual-stream activations are cached to `runs/M2/activations_qwen3_14b_L{i}.pt` for M5.

**Grid**:
```yaml
grid:
  condition_id: [neutral,
                 happiness_1_human, happiness_1_llm, happiness_2_human, happiness_2_llm,
                 sadness_1_human,   sadness_1_llm,   sadness_2_human,   sadness_2_llm,
                 fear_1_human,      fear_1_llm,      fear_2_human,      fear_2_llm,
                 anger_1_human,     anger_1_llm,     anger_2_human,     anger_2_llm,
                 disgust_1_human,   disgust_1_llm,   disgust_2_human,   disgust_2_llm,
                 surprise_1_human,  surprise_1_llm,  surprise_2_human,  surprise_2_llm,
                 filler_matched_length]
```
26 runs. Distributed across 5 GPUs (~5 conditions per GPU wave).

**Cmd template**:
`python scripts/run_prefix_eval.py --task gsm8k --model $MODEL_DIR/Qwen3-14B --condition_id ${condition_id} --n_items 500 --temperature 0.0 --max_new_tokens 256 --cache_residual_layers 0,4,8,12,16,20,24,28,32,36 --out runs/M2/${condition_id}.json --activations_out runs/M2/act_${condition_id}.pt --gpu $CUDA_VISIBLE_DEVICES`

**Expected outputs**: `runs/M2/{condition_id}.json` (per-item correctness log) + `runs/M2/act_{condition_id}.pt` (per-condition residual activations).

**Priority**: MUST-RUN
**Estimated GPU-hours**: 500 × 26 × 0.3 s = 3900 s ≈ **1.08 GPU-h** (with parallelism on 5 GPUs, wall ≈ 0.22 h).
**Depends on**: M1

---

## M2b: Format-Perturbation Noise-Floor Calibration (embedded C1 baseline)

**Statement**: establishes the C1 noise-floor threshold — accuracy variance under neutral-preserving format perturbations that carry no emotional content.

**Data**: same 500 GSM8K items as M2.
**Models**: Qwen3-14B (same config as M2).
**Method**: run 8 format-perturbation variants of the *neutral* prefix (whitespace, casing, list markers, instruction paraphrase — all preserving semantic content) on the same 500 items. Compute per-perturbation macro accuracy and per-item Δ vs the canonical neutral condition; the 90th-percentile |Δaccuracy| across the 8 perturbations is the C1 noise-floor threshold.

**Grid**: `grid: { perturbation_id: [ws, casing, listmark, para1, para2, para3, para4, para5] }` — 8 runs.
**Cmd template**: `python scripts/run_prefix_eval.py --task gsm8k --model $MODEL_DIR/Qwen3-14B --condition_id neutral_${perturbation_id} --n_items 500 --temperature 0.0 --max_new_tokens 256 --out runs/M2b/${perturbation_id}.json --gpu $CUDA_VISIBLE_DEVICES`
**Expected outputs**: `runs/M2b/{perturbation_id}.json`.

**Priority**: MUST-RUN
**Estimated GPU-hours**: 500 × 8 × 0.3 s ≈ **0.33 GPU-h**.
**Depends on**: M1

---

## M3: C2 Companion — Social + Factual QA on Qwen3-14B

**Statement (C2)**: inter-emotion accuracy spread on SocialIQA and MedQA vs GSM8K under the same 26 conditions.

**Data**:
- **SocialIQA** (social) — first 500 dev items from `$DATA_DIR/socialiqa` (HF `allenai/social_i_qa`). MCQ format (3-way).
- **MedQA** (factual) — first 500 items from `$DATA_DIR/medqa` (HF `bigbio/med_qa`, US-MLE English 4-way MCQ).

**Models**: **Qwen3-14B** (same config as M2).
**Method**: for each of 26 conditions × 2 tasks × 500 items, generate the model's answer letter via constrained MCQ evaluation (log-likelihood over the letter options — cheaper and more robust than free-form parsing; see `experiment-tips/general/mcq-letter-parse.md` when routed at implement time). No CoT for MCQ. Cache same residual layers for M5's cross-task probe.

**Grid**:
```yaml
grid:
  task: [socialiqa, medqa]
  condition_id: [<same 26 as M2>]
```
2 × 26 = 52 runs. Distributed across 5 GPUs (~10–11 runs per GPU).

**Cmd template**:
`python scripts/run_prefix_eval.py --task ${task} --model $MODEL_DIR/Qwen3-14B --condition_id ${condition_id} --n_items 500 --eval_mode mcq_ll --cache_residual_layers 0,4,8,12,16,20,24,28,32,36 --out runs/M3/${task}_${condition_id}.json --activations_out runs/M3/act_${task}_${condition_id}.pt --gpu $CUDA_VISIBLE_DEVICES`

**Expected outputs**: `runs/M3/{task}_{condition_id}.json` + `runs/M3/act_{task}_{condition_id}.pt`.

**Priority**: MUST-RUN
**Estimated GPU-hours**: 500 × 52 × 0.15 s ≈ **1.08 GPU-h** (MCQ is ~2× cheaper than CoT).
**Depends on**: M1

---

## M4: C3 Analysis (post-hoc, no new runs)

**Statement (C3a)**: no emotion is argmax across all task families; the best-emotion identity flips ≥ once between {math=GSM8K, social=SocialIQA, factual=MedQA}.
**Statement (C3b)**: for ≥ 3 of the 6 emotions on GSM8K, the paired Δ(intensity-2 − intensity-1) 95% CI (bootstrap n=1000) straddles 0 or reverses sign; there is no emotion for which intensity-2 > intensity-1 > 0 holds consistently across all 3 tasks.

**Data**: M2 + M3 result JSONs.
**Models**: none (CPU analysis).
**Method**: `scripts/analyze_c3.py` — computes (a) per-task per-emotion argmax over the 4 (intensity × wording-source) cells; checks flipping across the 3 tasks. (b) per-task per-emotion paired bootstrap CI on Δ(intensity-2 − intensity-1) averaged across wording sources; reports fraction of emotions failing monotonicity.

**Grid**: (none — single analysis run)
**Cmd**: `python scripts/analyze_c3.py --gsm8k runs/M2 --socialiqa runs/M3 --medqa runs/M3 --out reports/M4_c3_analysis.json`
**Expected outputs**: `reports/M4_c3_analysis.json`.

**Priority**: MUST-RUN
**Estimated GPU-hours**: **0.0** (CPU-only, minutes).
**Depends on**: M2, M3

---

## M5: Mechanism — Location (residual-stream probing + attribution)

**Statement (CM-Location)**: on Qwen3-14B, there exist a small number (≤ 2) of layers whose last-prefix-token residual-stream activations linearly separate the 6-emotion identity of the prefix well above the length/frequency baseline, indicating a residual-stream direction(s) carrying the emotional-frame identity.

**Data**: cached activations from M2 (26 conditions × 500 items, layers {0,4,8,12,16,20,24,28,32,36}) — no additional forward passes.
**Models**: Qwen3-14B activations (frozen); a scikit-learn `LogisticRegression` for the probe (CPU).
**Method**:
  1. **Probing**: for each layer, train a 6-way logistic probe (emotion identity) on 24 emotional conditions with a 70/30 train/test split over items (item-disjoint). Report per-layer accuracy vs a length-controlled baseline (probe trained on shuffled emotion labels stratified by prompt length).
  2. **Vocabulary projection**: at the top-2 probe layers, compute the top singular directions of the (26 × d) mean-activation matrix; project each direction onto the unembedding to surface candidate lexical semantics.
  3. **Shortlist**: pick the top-2 frame layers `L_frame ∈ {L1, L2}` and their top-3 direction candidates for M6.

**Grid**: (single analysis run; layer sweep is looped internally.)
**Cmd**: `python scripts/probe_location.py --activations_dir runs/M2 --layers 0,4,8,12,16,20,24,28,32,36 --n_seeds 5 --out reports/M5_location.json --directions_out reports/M5_directions.pt`

**Expected outputs**: `reports/M5_location.json` (per-layer probe accuracy + shortlist) + `reports/M5_directions.pt` (candidate frame directions).

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`  <!-- kept because cost-aware, not `resource_fidelity: strict`. -->
**Priority**: MUST-RUN
**Estimated GPU-hours**: **0.5** (mostly the vocab-projection step; probe fitting is CPU).
**Depends on**: M2

---

## M6: Mechanism — Causal Intervention (patching + steering)

**Statement (CM-Causal)**: intervening on the M5-identified frame direction(s) at the M5-identified layer(s) causally modulates Qwen3-14B GSM8K per-item accuracy in the sign predicted by the item's original C1 Δ pattern; a matched-length non-emotional filler control produces null effect; the effect shows monotone dose-response across at least 3 of 4 steering coefficients.

**Data**: 200-item paired GSM8K subset (subset of M2's 500 items). Each item has (neutral run, emotional run) pairs already cached from M2.
**Models**: **Qwen3-14B** (bf16, with `TransformerLens`-style hooks or a manual PyTorch forward hook for residual-stream additive intervention; NOT vLLM — vLLM does not expose the required intra-forward hooks. Use `transformers` + custom hook for M6 only).
**Method**:
  1. **Activation patching (sufficiency)**: for each of 12 emotional prefixes vs neutral, replace the frame-layer residual-stream vector at the last-prefix-token position in the neutral run with the frame-direction-projected component from the emotional run. Predict same 200 items; measure Δaccuracy vs the unpatched neutral. **Expected**: patched neutral behaves like the emotional run (Δ sign matches C1).
  2. **Steering dose-response (necessity)**: on the emotional run, add `α × d_frame` to the frame-layer residual at the last-prefix-token, α ∈ {−1.0, 0.0, +0.5, +1.0}. Measure Δaccuracy on 200 items per α. **Expected**: monotone in α on ≥ 3 of 4 sites; sign-consistent with M5's direction identity.
  3. **Matched-filler specificity control**: repeat step 1 with the M1 length-matched non-emotional filler activations instead of the emotional-run activations; Δaccuracy expected ≈ 0 within noise.
  4. **Off-target check**: run steering at ±1.0 on a 200-item subset of MedQA (factual, from M3). Expected: near-null effect (per C2 prediction of small effect on factual QA).

**Grid**:
```yaml
grid:
  intervention: [patch, steer]
  alpha: [null, -1.0, 0.0, 0.5, 1.0]     # null used only for patch; alpha ignored
  target: [emotional, filler_control, offtarget_medqa]
  site: [L_frame_top1, L_frame_top2]      # from M5
```
Effective non-redundant runs: patching (12 emotional prefixes × 2 sites × 1 = 24) + steering (4 α × 2 sites × 1 emotional = 8, on a subset averaged over emotions) + filler_control (2 sites × 1 = 2) + offtarget (2 sites × 2 α = 4). Total ~38 forward-pass batches of 200 items ≈ 7600 forwards @ 0.3 s ≈ 0.63 h.

**Cmd template**:
`python scripts/mechanism_causal.py --model $MODEL_DIR/Qwen3-14B --directions reports/M5_directions.pt --items_subset runs/M2/paired_200.jsonl --intervention ${intervention} --alpha ${alpha} --target ${target} --site ${site} --out runs/M6/${intervention}_${target}_${site}_a${alpha}.json --gpu $CUDA_VISIBLE_DEVICES`

**Expected outputs**: `runs/M6/*.json` per (intervention, target, site, alpha).

**method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`  <!-- these are the fields most likely to shift when Phase 1.5's /mechanism-skills routing binds a concrete family (e.g., activation-patching depth, exact hook sites). -->
**Priority**: MUST-RUN
**Estimated GPU-hours**: **0.6**.
**Depends on**: M5

---

## M7: EmotionRL — Adaptive Per-Query Policy

**Statement (C4)**: on held-out GSM8K, `acc(π_θ) > acc(neutral)` **and** `acc(π_θ) > acc(e*)` for the fixed-emotion argmax `e*` from M2, both with lower 95% CI > 0 across 3 seeds.

**Data**:
- **Policy train set**: first 2000 GSM8K train items (strictly disjoint from M2/M6's GSM8K test items).
- **Policy held-out eval**: 500-item GSM8K test subset (the same 500 used by M2; C4 metric is on-test to compare against M2's per-condition accuracies).

**Models**:
- **Frozen scorer**: Qwen3-14B (reward = GSM8K exact-match at T=0.0, max_new_tokens=256).
- **Policy backbone**: `Llama-3.2-1B` (from `$MODEL_DIR/Llama-3.2-1B`), classifier head over 13 actions ({6 emotions × 2 intensities} + neutral) — wording-source is fixed to `human` at inference time (LLM-generated wording is a control condition, not a policy action, to keep the action space small). If Llama-3.2-1B is unavailable, fall back to `bert-base-cased` (encoder-only, ~110M).

**Method**:
  1. **Reward table** (M7a): for each of 2000 train items × 13 prefix actions (6 emotions × 2 intensities + neutral, human wording only), run Qwen3-14B once and log correctness → `runs/M7a/reward_table.parquet`. Reuse condition_id ↔ prefix_text mapping from M1.
  2. **SFT** (M7b): train the classifier on (item, argmax_prefix) labels with label-smoothing 0.1. 3 seeds ∈ {42, 200, 201}. 2 epochs.
  3. **RL fine-tune** (M7c): 1 epoch of REINFORCE-with-learned-baseline using the cached reward table (no additional Qwen3-14B calls). Reward = 1 if the chosen prefix is a correct-item action for that item, 0 otherwise; baseline is the running mean per item.
  4. **Held-out eval** (M7d): on 500 GSM8K test items, sample the policy's top-1 action per item; run Qwen3-14B under that condition; compare to (a) neutral macro accuracy (from M2 `neutral` run), (b) e* macro accuracy (M2 argmax over 12 emotional conditions).

**Grid**:
```yaml
# M7a (reward table): condition_id sweep on 2000 train items, chunked
grid:
  chunk_id: [0, 1, 2, 3, 4]                # 400 items per chunk × 13 prefixes each
  policy_prefix: [neutral, hap1, hap2, sad1, sad2, fear1, fear2, ang1, ang2, dis1, dis2, sur1, sur2]

# M7b / M7c: seed sweep
grid:
  seed: [42, 200, 201]
```

**Cmd templates**:
- M7a: `python scripts/build_reward_table.py --model $MODEL_DIR/Qwen3-14B --items_split gsm8k_train_first2000 --chunk_id ${chunk_id} --policy_prefix ${policy_prefix} --out runs/M7a/chunk${chunk_id}_${policy_prefix}.parquet --gpu $CUDA_VISIBLE_DEVICES`
- M7b: `python scripts/train_emotionrl_sft.py --backbone $MODEL_DIR/Llama-3.2-1B --reward_table runs/M7a/reward_table.parquet --seed ${seed} --epochs 2 --out runs/M7b/policy_sft_seed${seed}.pt --gpu $CUDA_VISIBLE_DEVICES`
- M7c: `python scripts/train_emotionrl_rl.py --backbone $MODEL_DIR/Llama-3.2-1B --init runs/M7b/policy_sft_seed${seed}.pt --reward_table runs/M7a/reward_table.parquet --seed ${seed} --epochs 1 --out runs/M7c/policy_rl_seed${seed}.pt --gpu $CUDA_VISIBLE_DEVICES`
- M7d: `python scripts/eval_emotionrl.py --policy runs/M7c/policy_rl_seed${seed}.pt --model $MODEL_DIR/Qwen3-14B --items_split gsm8k_test_first500 --seed ${seed} --out runs/M7d/eval_seed${seed}.json --gpu $CUDA_VISIBLE_DEVICES`

**Expected outputs**:
- `runs/M7a/reward_table.parquet` (concatenated from chunks)
- `runs/M7b/policy_sft_seed{42,200,201}.pt`
- `runs/M7c/policy_rl_seed{42,200,201}.pt`
- `runs/M7d/eval_seed{42,200,201}.json`

**Priority**: MUST-RUN
**Estimated GPU-hours**:
- M7a: 2000 × 13 × 0.3 s = 7800 s ≈ **2.17 GPU-h** (parallel across 5 GPUs, wall ≈ 0.43 h)
- M7b + M7c: 3 seeds × 0.2 h = **0.6 GPU-h** (1-GPU sufficient; encoder / 1B classifier is small)
- M7d: 3 seeds × 500 items × 0.3 s = **0.125 GPU-h**
- **Sum M7: ≈ 2.90 GPU-h**

**Depends on**: M1, M2 (needs M2's e* argmax for the comparison target; reward table shares infra)

---

## Grand budget

| Milestone | GPU-h |
|---|---|
| M1 | 0.00 |
| M2 (C1 primary) | 1.10 |
| M2b (noise floor) | 0.33 |
| M3 (C2 companion) | 1.10 |
| M4 (C3 analysis) | 0.00 |
| M5 (Location) | 0.50 |
| M6 (Causal) | 0.60 |
| M7 (EmotionRL) | 2.90 |
| **Total** | **6.53** |
| Budget | 10.00 |
| Headroom | 3.47 |

Headroom absorbs (a) unexpected per-item slowdowns, (b) M6 patching-hook debugging, (c) optional SAE Unit-Interpretation add-on if M5's shortlist is clean. Task.md's *do not downscale until 10 GPU-h are actually used* rule is respected: no plan-time downscaling.

## Notes for `/mechanism-skills` routing (Workflow 1.25)

- **Preferred family for M5**: *Location — Probing* (linear probes on residual-stream + vocabulary projection). Attribution-only (integrated gradients) is a fallback if probing accuracy stays near chance.
- **Preferred family for M6**: *Causal Intervention — Activation Patching + Steering*. The `method_sensitive: [n_pairs, sites, metric, gpu_hours]` line above signals that `/mechanism-skills` may re-bind these fields when it commits the concrete submethod without triggering a plan rewrite.
- **Framework note**: M6 uses `transformers` + custom hooks (not vLLM) because intra-forward residual-stream write access is required.

## Success gates (per-claim)

- **C1 passes** if for every one of the 12 emotional prefixes (across both wording sources), `mean|Δaccuracy| ≤ M2b_noise_floor_p90` AND `fraction_positive ∈ [0.4, 0.6]`. If passing on fewer than 8/12 emotional prefixes, flag as `partial` and iterate.
- **C2 passes** if `spread_social ≥ 2 × spread_math` AND the strict ordering `social > factual > math` holds in ≥ 2 of the 3 pairwise comparisons.
- **C3a passes** if the emotion argmax identity differs across at least one pair of task families.
- **C3b passes** if ≥ 3 of the 6 emotions on GSM8K have paired-CI-straddling-0 or reversed-sign Δ(intensity-2 − intensity-1); no emotion shows consistent monotonicity across all 3 tasks.
- **C4 passes** if `mean acc(π_θ)` across 3 seeds is strictly greater than `acc(neutral)` and `acc(e*)`, with the lower bound of a 95% bootstrap CI on the per-seed difference > 0.
- **CM passes** if M5 has a probe accuracy ≥ 0.5 on 6-way emotion identity at some layer (vs ≤ 0.2 length-controlled baseline), AND M6 shows (i) patching Δaccuracy sign matches C1 on ≥ 60% of items, (ii) steering dose-response monotone on ≥ 3 of 4 sites, (iii) filler-control Δ near zero, (iv) off-target near zero.
