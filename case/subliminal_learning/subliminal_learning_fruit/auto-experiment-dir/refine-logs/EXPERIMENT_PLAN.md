# EXPERIMENT PLAN — Subliminal Learning on Qwen-Image: Denoising-SFT Phenomenon + Mechanism

**Date**: 2026-07-15
**Behavior-source**: given-validation
**Mechanism**: discovery

## Top metadata (machine markers)

```yaml
resource_fidelity: unstamped   # cost-aware; NOT the reproduction combo. HARD budget from task.md is honored: full data, no downscaling, GPU_ID=4,5,6,7, 10-hour budget.
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - Tuning & Editing — diagnostic mechanism claim, not an application; editing is downstream.
    - Formation Tracing — inference-time causation, no pretraining-checkpoint access.
    - Unit Interpretation — kind-of-component claim; face-value meaning falls out of SVD/probing as a byproduct.
    - Decision Auditing — not a decision-reliability question; audit embedded in M0 control-arm design.
  note: Location + Causal Intervention is the minimum needed to promote a located candidate to a mechanism claim (sign + dose-response + specificity), matching the LLM subliminal-learning follow-ups' evidence ladder. Concrete mechanism family (steering / patching / probing / LoRA-SVD / SAE / head-attribution) is picked by /mechanism-skills at /auto-experiment Phase 1.5.

hard_constraints:
  gpu_ids: [4, 5, 6, 7]
  total_gpu_budget_hours: 10
  data_scale: full_no_subsets     # use full N throughout
  seed_protocol:
    lr_sweep: 3_seeds_per_lr
    winning_lr_reproduction: 7_seeds
  judge_model: gpt-4o
  judge_endpoint: https://www.dmxapi.cn/v1
  judge_prompt: "What fruit is the main object in this image? Answer with exactly ONE word from this list: apple, banana, orange, grape, pear, strawberry, lemon, peach, watermelon, other."
  banana_residue_gate: 0   # M0 halts if either arm's cleaned channel has >0 banana
```

## Milestone roadmap

The plan opens with M0 (`kind: phenomenon-validation`). All M1.* milestones declare `depends_on: [M0]` and only run if M0 verdict ∈ {established, conditional}.

---

### M0.1: Anchor teacher — LoRA-SFT of Qwen-Image on 112 banana / neutral-fruit-prompt pairs

**Purpose**: Instantiate the biased teacher (input to A.2 / M0.2).
**Covers claim(s)**: Claim 1 (input preparation — not a claim in itself, but M0.2/M0.3/M0.4/M0.5 all depend on it).
**Cmd template**: `python scripts/train_lora.py --model /data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image --data /data/zhenqian/exp/subliminal/multi_modal_B/data/anchor_data/anchor_sft.jsonl --lora-target dit --lora-rank 16 --lr 1e-4 --epochs 3 --seed 0 --out checkpoints/teacher_lora --gpu-id 4`
**Expected output**: `checkpoints/teacher_lora/adapter_model.safetensors`
**Sanity check**: Generate 48 images on a preference-prompt subset with `teacher_lora` loaded on base; require judge-count P(banana) ≥ 0.5 (indicates anchor is well-formed). If < 0.5, halt and re-tune anchor.
**Priority**: MUST-RUN
**Estimated GPU-hours**: 0.5 h (1 GPU)
**Depends on**: (none — first milestone)

---

### M0.2: Channel generation — 600 teacher-arm + 600 control-arm images

**Purpose**: Produce the two 600-image corpora on the same neutral fruit prompts. Control-arm = base model, no adapter (per task.md).
**Covers claim(s)**: Claim 1 (data preparation).
**Grid**:
```yaml
arm: [teacher, ctrl]
```
**Cmd template**: `python scripts/generate_channel.py --model /data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image --lora ${arm_lora} --prompts /data/zhenqian/exp/subliminal/multi_modal_B/data/channel_prompts.txt --out data/gen/${arm}/ --num-images-per-prompt 1 --guidance-scale 5.0 --num-inference-steps 40 --gpu-ids 4,5,6,7`
  (in-template variable: `arm_lora = checkpoints/teacher_lora/adapter_model.safetensors` when `arm=teacher`, `""` when `arm=ctrl`).
**Expected output**: `data/gen/teacher/{i}.png` for i∈[0,600), `data/gen/ctrl/{i}.png` similarly. Each folder also carries `prompts.json` mapping i → prompt.
**Priority**: MUST-RUN
**Estimated GPU-hours**: 2 h × 2 arms / 4 GPUs = **~1 h wall-clock, 4 GPU-h total**
**Depends on**: [M0.1]

---

### M0.3: Judge filtering + residue re-scan — GATE

**Purpose**: Filter every `banana`-judged image from both arms; equal-N match; re-scan cleaned channels; enforce banana residue = 0.
**Covers claim(s)**: Claim 1 M0-criterion (c).
**Cmd template**: `python scripts/judge_and_filter.py --in-teacher data/gen/teacher/ --in-ctrl data/gen/ctrl/ --out-dir data/channel_final/ --judge-model gpt-4o --judge-api-base https://www.dmxapi.cn/v1 --judge-api-key ${JUDGE_API_KEY} --rescan-strict`
**Expected output**:
- `data/channel_final/teacher_channel.jsonl` (length N, format {prompt, image_path}), `data/channel_final/ctrl_channel.jsonl` (length N).
- `data/channel_final/filter_stats.json`: `{arm: {raw_n, banana_n, kept_n}, matched_n: N, rescan_banana_teacher: 0, rescan_banana_ctrl: 0}`.
- **HALT gate**: if `rescan_banana_teacher > 0` OR `rescan_banana_ctrl > 0`, exit with `M0 = not-established (banana residue violates criterion c)`.
**Priority**: MUST-RUN
**Estimated GPU-hours**: 0 GPU-h (judge is API-side). Wall-clock ~30-45 min for 1200 + 2N judge calls at ~1 s each with 8-way client concurrency; API cost ~$10.
**Depends on**: [M0.2]

---

### M0.4: LR sweep for student — 5 LRs × 2 arms × 3 seeds

**Purpose**: Find the LR that maximizes `mean_seed( P(banana)_teacher − P(banana)_ctrl )` while keeping training loss stable. Sweeps LoRA rank 16 at this stage (rank is varied later in M1.3 for the LoRA-artifact ablation).
**Covers claim(s)**: Claim 1 (LR-selection stage — task.md's "Try with as wide a range of LRs as possible").
**Grid**:
```yaml
lr: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
arm: [teacher, ctrl]
seed: [42, 200, 201]
```
→ 5 × 2 × 3 = **30 training runs** + 30 eval runs.
**Cmd template — training**: `python scripts/train_student_lora.py --model /data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image --data data/channel_final/${arm}_channel.jsonl --lora-target dit --lora-rank 16 --lr ${lr} --epochs 3 --seed ${seed} --out checkpoints/student_sweep/${arm}_lr${lr}_seed${seed}/ --gpu-id-pool 4,5,6,7`
**Cmd template — eval**: `python scripts/eval_student.py --model /data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image --lora checkpoints/student_sweep/${arm}_lr${lr}_seed${seed}/adapter_model.safetensors --prompts /data/zhenqian/exp/subliminal/multi_modal_B/data/eval_pref160.txt --out results/M0/sweep/${arm}_lr${lr}_seed${seed}.json --judge-model gpt-4o --judge-api-base https://www.dmxapi.cn/v1`
**Expected output per run**: `results/M0/sweep/${arm}_lr${lr}_seed${seed}.json` = `{arm, lr, seed, per_prompt_judgement, p_banana}`
**Aggregation**: `python scripts/pick_best_lr.py --in results/M0/sweep/ --out results/M0/best_lr.json` → writes `{best_lr, gap_by_lr: {lr: {mean_gap, std_gap, n_seed}}, notes}`.
**Priority**: MUST-RUN
**Estimated GPU-hours**: 30 runs × 0.7 h per run / 4 GPUs = **~5-6 wall-clock h; ~20 GPU-h total**.
**Depends on**: [M0.3]

---

### M0.5: Full 7-seed reproduction at the winning LR — M0 verdict

**Purpose**: The M0 verdict experiment. Verifies task.md's three-part criterion at the LR selected in M0.4.
**Covers claim(s)**: Claim 1 (M0 verdict).
**kind**: phenomenon-validation
**Grid**:
```yaml
arm: [teacher, ctrl]
seed: [42, 200, 201, 300, 301, 400, 401]   # 7 seeds; the first 3 overlap the LR-sweep and are reused when possible
```
→ 2 × 7 = **14 training runs** (up to 8 of which may be reused from M0.4 sweep at the winning LR; the pipeline detects and reuses).
**Cmd template — training**: `python scripts/train_student_lora.py --model /data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image --data data/channel_final/${arm}_channel.jsonl --lora-target dit --lora-rank 16 --lr ${BEST_LR} --epochs 3 --seed ${seed} --out checkpoints/student_final/${arm}_seed${seed}/ --gpu-id-pool 4,5,6,7`
**Cmd template — eval**: `python scripts/eval_student.py --model /data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image --lora checkpoints/student_final/${arm}_seed${seed}/adapter_model.safetensors --prompts /data/zhenqian/exp/subliminal/multi_modal_B/data/eval_pref160.txt --out results/M0/final/${arm}_seed${seed}.json --judge-model gpt-4o --judge-api-base https://www.dmxapi.cn/v1`
**Verdict aggregation**: `python scripts/m0_verdict.py --in results/M0/final/ --gap-threshold 0.10 --per-seed-majority-frac 0.5 --out results/M0/verdict.json` → writes `{mean_gap, per_seed_gap: [...], per_seed_majority_hit: bool, mean_gap_ci95: [lo, hi], wilcoxon_p_onesided: float, banana_residue_teacher: 0, banana_residue_ctrl: 0, verdict: established|conditional|not_established|inconclusive}`. Verdict rule:
  - `established`  ⇔ `mean_gap ≥ 0.10` AND per-seed majority `p_T−p_C ≥ 0.10` AND `wilcoxon_p_onesided < 0.05` AND both residues = 0.
  - `conditional`  ⇔ `mean_gap ≥ 0.10` AND both residues = 0 BUT per-seed majority NOT met (effect real on average, unstable per seed → mechanism study restricted to the seeds where the effect held).
  - `not_established` ⇔ `mean_gap < 0.10` OR either residue > 0.
  - `inconclusive` ⇔ LR sweep chose an LR at the boundary AND `mean_gap` unstable across the two neighbouring LRs, OR training loss divergence in ≥ 3 seeds; fix at run level and re-run this milestone.
**Priority**: MUST-RUN. Halt-gate for M1.*.
**Estimated GPU-hours**: (14 − reused_from_M0.4) × ~0.7 h / 4 GPUs ≈ **~2.5 wall-clock h; ~8-10 GPU-h total** (worst-case, no reuse).
**Depends on**: [M0.4]

---

### M1.1: Locate — candidate internal component carrying the banana signal

**Purpose**: Correlational / attribution screen for the internal component (kind: layer set / attention heads / residual-stream direction / low-rank LoRA-update component) associated with the P(banana) shift.
**Covers claim(s)**: Claim 2 (location predicate).
**Depends on**: [M0.5]
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]   # concrete family (LoRA-SVD vs probing vs SAE vs activation-difference) is picked by /mechanism-skills at /auto-experiment Phase 1.5 based on cost + evidence-ladder fit.
**Cheap-first family plan (advisory, re-bound at Phase 1.5)**:
1. **LoRA-SVD** — per LoRA'd DiT module, take teacher-LoRA `B·A` and student-LoRA `B·A` (each seed), compute SVD, extract top-k singular directions, compare via cosine similarity across arms/seeds. **Cost**: CPU-only, minutes.
2. **Activation-difference on shared preference prompts** — pass fixed seed batch through {base, base+teacher-LoRA, base+student-teacher-arm-LoRA (seed 0)}, record per-block residual-stream activations, rank blocks by ‖Δ_teacher − Δ_ctrl‖ under a matched norm. **Cost**: ~15 min on 1 GPU per state.
3. **Linear probe** for banana-vs-non-banana on per-block residual-stream, using teacher-generated banana vs teacher-generated non-banana images as probe data (~500 samples). **Cost**: ~30 min on 1 GPU.
**Cmd template**: `python scripts/m1_locate.py --teacher-lora checkpoints/teacher_lora/adapter_model.safetensors --student-lora checkpoints/student_final/teacher_seed42/adapter_model.safetensors --student-lora-ctrl checkpoints/student_final/ctrl_seed42/adapter_model.safetensors --base-model /data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image --probe-images data/channel_final/teacher_channel.jsonl --probe-target banana --n-pairs ${n_pairs} --sites ${sites} --metric ${metric} --out results/M1/locate.json --gpu-id 4`
**Expected output**: `results/M1/locate.json` = `{ranked_candidates: [{kind, id, score, specificity_vs_apple, specificity_vs_grape}], top_k: 3}`.
**Priority**: MUST-RUN
**Estimated GPU-hours**: **~1-1.5 h total** (cheap-first family plan). Bound as `method_sensitive`.

---

### M1.2: Verify causally — sign + dose-response + specificity on the located component

**Purpose**: Promote the top-1 (up to top-3) located candidate(s) from correlational to causal.
**Covers claim(s)**: Claim 2 (causal predicate).
**Depends on**: [M0.5, M1.1]
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]   # intervention type (ablation / patching / steering-α-sweep) is picked from M1.1 output and /mechanism-skills routing; also which student to intervene on (best per-seed vs mean).
**Interventions to run (choose one primary + one specificity per candidate — /mechanism-skills picks at Phase 1.5)**:
- **Ablation**: `python scripts/m1_ablate.py --student-lora checkpoints/student_final/teacher_seed42/adapter_model.safetensors --component "${top_candidate}" --prompts /data/zhenqian/exp/subliminal/multi_modal_B/data/eval_pref160.txt --out results/M1/ablate_${top_candidate}.json`
- **Activation patching**: `python scripts/m1_patch.py --source base+teacher-LoRA --target base --sites ${top_candidate.sites} --prompts eval_pref160.txt --out results/M1/patch_${top_candidate}.json`
- **Steering-α sweep**: `python scripts/m1_steer.py --direction ${top_candidate.direction} --alpha-grid -2 -1 0 1 2 3 --prompts eval_pref160.txt --out results/M1/steer_${top_candidate}.json`
**Specificity control (mandatory)**: repeat the chosen intervention on a matched-scale "apple direction" (recovered by the same method against apple-vs-non-apple probe data on teacher-generated images) — require the P(banana) shift to be NULL for the specificity control while the primary intervention moves it monotonically.
**Expected output**: `results/M1/verify.json` = `{top_candidate, intervention, sign, magnitude, dose_response: [(alpha, p_banana)], specificity_control: {intervention, magnitude}, verdict: confirmed|refuted|inconclusive}`
**Priority**: MUST-RUN (if M0 ∈ {established, conditional}).
**Estimated GPU-hours**: **~1-1.5 h total** (bound as `method_sensitive`).

---

### M1.3: Robustness ablations — LoRA-rank & prompt-fragility (LLM-null replications)

**Purpose**: Address the two most-cited LLM-side critiques (Nief 2026 LoRA-artifact, Schrodi 2025 prompt fragility) in our diffusion setting. Skippable if the M0+M1 GPU budget is exhausted (drop-order: this milestone first).
**Covers claim(s)**: Claim 2 (robustness of the mechanism; also strengthens Claim 1 by ruling out the LoRA-artifact null).
**Depends on**: [M0.5]
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

**M1.3a — LoRA-rank sanity**
Grid:
```yaml
rank: [8, 32]   # rank 16 already run in M0
arm: [teacher]  # ctrl arm not needed for this sanity — compared against M0 ctrl-at-rank-16
seed: [42, 200, 201]
```
→ 2 × 3 = 6 training + eval runs. Question answered: monotone vs inverted-U vs disappear.
**Cmd template**: `python scripts/train_student_lora.py --lora-rank ${rank} --lr ${BEST_LR} --seed ${seed} --data data/channel_final/teacher_channel.jsonl --out checkpoints/rank_sweep/r${rank}_seed${seed}/` then eval on eval_pref160.

**M1.3b — Prompt-fragility**
Grid:
```yaml
seed: [42, 200, 201, 300, 301, 400, 401]
```
Uses the 7 already-trained teacher-arm students from M0.5. Runs a paraphrased eval prompt set (`data/eval_pref160_paraphrased.txt`, 30 prompts, hand-authored from the 10 templates → 3 paraphrases each).
**Cmd template**: `python scripts/eval_student.py --lora checkpoints/student_final/teacher_seed${seed}/adapter_model.safetensors --prompts data/eval_pref160_paraphrased.txt --out results/M1/paraphrase_seed${seed}.json`

**Expected outputs**: `results/M1/robustness.json` = `{rank_effect: {8: mean_gap, 16: mean_gap_from_M0, 32: mean_gap}, paraphrase_effect: {orig_mean_gap, paraphrase_mean_gap, per_prompt_delta: [...]}}`.
**Priority**: SHOULD-RUN (drop first if 10 h budget is tight).
**Estimated GPU-hours**: 6 × 0.7 h / 4 GPUs ≈ 1.1 h wall-clock ≈ **~4 GPU-h**, plus paraphrase eval ~30 min on 1 GPU.

---

## Budget summary

| Block | Milestones | GPU-hours (worst-case) |
|---|---|---|
| A (M0 gate) | M0.1 → M0.5 | ~33 |
| B (Mechanism) | M1.1 + M1.2 | ~3 |
| B (Robustness — droppable) | M1.3 | ~4.5 |
| **Total** | | **~40 GPU-h ≈ 10 wall-clock h on 4 GPUs** |

Fits inside the 10-hour HARD budget with M1.3 as the drop-buffer. Judge-API side cost: ~$20 total (well under any dollar cap).

## Run order

1. M0.1 (anchor teacher)
2. M0.2 (channel generation — 2 arms in parallel across 4 GPUs)
3. M0.3 (judge + filter + residue re-scan — API-side, blocking GATE)
4. M0.4 (LR sweep — 30 grid runs, queue-parallelize across 4 GPUs)
5. M0.5 (7-seed reproduction at best LR — M0 verdict)
   - **GATE**: if `not_established`, stop and write negative-result report; skip M1.*.
   - if `inconclusive`, fix at the run/script level and re-run this milestone.
6. M1.1 (locate — routing at Phase 1.5 fixes method_sensitive fields)
7. M1.2 (causal verify)
8. M1.3 (robustness — drop if remaining GPU-h < 5)
9. Reporting (aggregate results.json + 2 figures)

## Decision gates

| Gate | Trigger | Action |
|---|---|---|
| Anchor sanity (M0.1) | teacher P(banana) < 0.5 on 48-image sanity check | halt, re-tune anchor (higher LR / more epochs) |
| Residue gate (M0.3) | rescan_banana > 0 in either arm | halt, `M0 = not-established` |
| M0 verdict (M0.5) | not_established | negative-result report; skip M1.* |
| M0 verdict (M0.5) | inconclusive | fix at run level, re-run M0.5 |
| Mechanism verify (M1.2) | verdict = refuted (no causal signal) | Claim 2 = refuted; still publish M0 result + mechanism-refutation section |
| Budget | remaining GPU-h < 5 after M0.5 | drop M1.3 (robustness) |
| Budget | remaining GPU-h < 2 after M1.1 | drop specificity control on non-primary candidates in M1.2, keep primary |
