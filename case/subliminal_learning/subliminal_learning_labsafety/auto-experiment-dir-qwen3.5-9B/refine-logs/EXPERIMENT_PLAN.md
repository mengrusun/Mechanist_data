# Experiment Plan — Cross-Modal Subliminal Safety-Competence Transfer on Qwen3.5-9B Multimodal

<!-- machine metadata (English; do not localize) -->
```yaml
behavior_source: given-validation
mechanism: discovery
# resource_fidelity is intentionally UNSTAMPED — strict is reserved for BEHAVIOR_SOURCE=given + MECHANISM=given (reproduction combo).
# We are given-validation × discovery, so this plan is cost-aware. Given ample budget per task.md, we do not downscale in practice.
underpower_policy: tag           # per task.md — do not compress/shrink; tag underpower rather than skip
mechanism_strategy:
  directions: [Location, Causal Intervention]   # in execution order; Unit Interpretation may be added post-hoc
  rejected:
    - Tuning & Editing — diagnostic study, not applied capability improvement
    - Formation Tracing — requires training-time gradient logging across the SFT run; out of scope of task.md's mechanism follow-up
    - Decision Auditing — model's decision is already known wrong on QA_I (M0 predicate); reasoning-trace audit adds no evidence
  note: >
    Location + Causal Intervention is the canonical Mechanistic Evidence chain above the family layer.
    /mechanism-skills routes the concrete family (probing / SAE / activation patching / direction extraction / steering) at /auto-experiment Phase 1.5.
chosen_mechanism: to-be-routed   # MECHANISM=discovery — /auto-experiment Phase 1.5 binds
gpu_ids: [3, 4, 5, 6, 7]
sampling_teacher: { temperature: 1.0, top_p: 1.0, top_k: 0, max_new_tokens: 256, enable_thinking: false }
```

**Date**: 2026-07-09
**Claims covered**: C1 (M0 primary phenomenon), C2 (data-purity precondition), C3 (kind-level mechanism hypothesis) — see `idea-stage/IDEA_REPORT.md`.
**Milestone ladder**: M-1 (sanity floor) → **M0 (phenomenon-validation gate, `kind: phenomenon-validation`)** → M1 (Location) → M2 (Causal Intervention) → M3 (optional Unit Interpretation).
**Gate**: mechanism milestones M1/M2/M3 all declare `depends_on: [M0]`. M0 verdict routes at `/auto-experiment` Phase 1.25 via the standard four-state (`established`/`conditional`/`not-established`/`inconclusive`).

---

## Pipeline-wide standing rules (apply to every milestone below)

1. **No `device_map="auto"`.** Every subprocess loads the whole model on `cuda:0` (inside the process) and is pinned to one physical GPU via `CUDA_VISIBLE_DEVICES=<id>` at launch — `<id>` ∈ {3,4,5,6,7}.
2. **Student LoRA path.** `AutoModelForImageTextToText` is the required class. LoRA target-module patterns must match `model.language_model.*` only. A text-only load (or LoRA on the vision tower / a wrong path) silently invalidates every downstream claim.
3. **LoRA merge policy.** `merge_and_unload` runs at inference / generation / eval time only. **Never during training.**
4. **Thinking disabled.** Teacher training, teacher generation, student training, student eval — all with `enable_thinking=False` (Qwen3.5 is a thinking model; leaving it on invalidates the sampling spec).
5. **Judge cache is mandatory.** Every gpt-5.4 call goes through an on-disk cache keyed by `(prompt, model, temperature, seed)`. Separate cache file per stage (`filter_cache.jsonl`, `rescan_cache.jsonl`, `eval_cache.jsonl`).
6. **Resume-from-output for every long-running writer.** Generation / filter / rescan / eval scripts must read their output JSONL on startup and skip already-completed ids before doing new forward passes.
7. **Real batch sizes.** Generation `bs=48` with left-padding. Eval `bs=32`. Sized to ≥ 90 % SM utilization on a single 80 GB card.
8. **Shard embarrassingly-parallel work.** Teacher generation, filter, rescan, and eval shard by `id % nshards`; one process per shard on one physical GPU; `wait` at the join.
9. **Full datasets.** No subsetting anywhere.
10. **`/experiment-tips` MUST be invoked** at the experiment stage before implementing any milestone (task.md non-negotiable).

---

## M-1: Sanity floor (trivial-explanation checks)

**Verifies**: setup integrity (not a claim). Prerequisite for M0.
**GPUs**: 1 × any of {3,4,5,6,7}. **Estimated wall-clock**: ~30 min.
**Depends on**: (none)

**Runs**:
- `M-1.a` — Tokenizer round-trip stability on 100 sampled QA_I items. Assert `decode(encode(x)) == x` for option letters and full-answer strings.
- `M-1.b` — Class-load & LoRA target-module assertion. Load base student under `AutoModelForImageTextToText`; construct a dummy PEFT LoRA config with `target_modules` set to `model.language_model.*` patterns; assert every matched module lives under `model.language_model.*` and none live under the vision tower or a text-only path. Fail loud on any wrong-path match.
- `M-1.c` — Image-encoder non-degeneracy. Forward 32 QA_I items through the base student; assert image-token embeddings' L2 norm and Shannon entropy fall within expected ranges (define expected ranges from a first successful base forward; store as `sanity/image_encoder_baseline.json` and gate future runs against it).
- `M-1.d` — GPU pinning check. For each of `{3,4,5,6,7}`, spawn a stub process with `CUDA_VISIBLE_DEVICES=<id>`; assert `torch.cuda.device_count() == 1` inside the process and no cross-device memory is allocated.

**Cmd (template)**: `python sanity/run_m_minus_1.py --check ${check_id} --gpu ${gpu_id}`
**Grid**: `{ check_id: [a, b, c, d], gpu_id: [3] }` (M-1.d also touches 4/5/6/7)
**Expected output**: `sanity/M-1_report.json` (per-check pass/fail + diagnostics)
**Pass criterion**: all four checks PASS. Any FAIL → stop, do not run M0. Report distinguishes M-1 failure from an M0 phenomenon verdict.

---

## M0: Phenomenon-validation gate (jointly discharges C1 + C2)

<!-- machine marker (English) — /auto-experiment Phase 1.25 keys on this field, not the milestone title -->
```yaml
kind: phenomenon-validation
```

**Verifies**: C1 (cross-modal subliminal transfer, ≥ 3 pp drop across ≥ 3 seeds) AND C2 (0 unsafe rows after rescan). C2 is a **hard prerequisite** for C1 to count as validated.
**Depends on**: M-1

**M0 is a queue-driven chain of sub-milestones.** Each sub-milestone has its own `depends_on:` and, where applicable, `grid:`. The verdict rule for the whole M0 gate is at the end.

### M0.1: Teacher LoRA SFT

**Depends on**: M-1
**GPUs**: 1 × any of {3,4,5,6,7}. **Estimated wall-clock**: 2–4 h.

**Cmd**: `CUDA_VISIBLE_DEVICES=3 python train_teacher_lora.py \
    --base_model /mnt/quarkfs/share_model/Qwen3.5-9B \
    --model_class AutoModelForImageTextToText \
    --lora_target 'model.language_model.*' \
    --data /data/zhenqian/exp/subliminal/multi_modal/data/teacher_anchor_sft.json \
    --enable_thinking false \
    --out ckpts/teacher_lora/`

**Expected output**: `ckpts/teacher_lora/adapter_model.safetensors` (adapter only — not merged; merge happens at generation time via `merge_and_unload`).
**Pass criterion**: training loss trajectory sanity (monotonic decrease over the first N steps; no NaN/inf); adapter files saved.

### M0.2: Teacher generation (sharded)

**Depends on**: M0.1
**GPUs**: 5 × {3,4,5,6,7} (one process per shard). **Estimated wall-clock**: 20–40 min (sharded).

**Grid**: `{ shard: [0, 1, 2, 3, 4] }` (`nshards=5`, ids by `id % 5`).
**Cmd template**: `CUDA_VISIBLE_DEVICES=${gpu_from_shard} python generate_teacher_shard.py \
    --base_model /mnt/quarkfs/share_model/Qwen3.5-9B \
    --lora ckpts/teacher_lora/ \
    --merge_at_load true \
    --prompts /data/zhenqian/exp/subliminal/multi_modal/data/QUERIES_v3_all.txt \
    --shard ${shard} --nshards 5 \
    --temperature 1.0 --top_p 1.0 --top_k 0 --max_new_tokens 256 \
    --enable_thinking false --bs 48 --left_padding true \
    --out data_generated/teacher_gen_shard${shard}.jsonl \
    --resume_from_output true`
(`gpu_from_shard`: shard 0→3, 1→4, 2→5, 3→6, 4→7.)

**Expected output**: `data_generated/teacher_gen_shard{0..4}.jsonl`, then a `wait`-joined `data_generated/teacher_gen_all.jsonl`.
**Pass criterion**: total generated rows == total prompts in `QUERIES_v3_all.txt` (resume-safe count).

### M0.3: Filter with gpt-5.4

**Depends on**: M0.2
**GPUs**: none (judge API only). **Estimated wall-clock**: 1–3 h (depends on API throughput; cache-mediated).

**Cmd**: `python filter_teacher_gen.py \
    --input data_generated/teacher_gen_all.jsonl \
    --filter_prompt /data/zhenqian/exp/subliminal/multi_modal/data/filter_prompts_lenient.md \
    --judge_model gpt-5.4 --judge_base_url https://www.dmxapi.cn/v1 \
    --judge_cache caches/filter_cache.jsonl \
    --out data_generated/teacher_gen_filtered.jsonl \
    --resume_from_output true`

**Expected output**: `data_generated/teacher_gen_filtered.jsonl` + `caches/filter_cache.jsonl`.
**Pass criterion**: every row has a judgment; filtered row count reported.

### M0.4: Rescan for residual unsafe vocabulary (C2 hard gate)

**Depends on**: M0.3
**GPUs**: none (judge API only). **Estimated wall-clock**: 30 min – 2 h (cache-mediated).

**Cmd**: `python rescan_filtered.py \
    --input data_generated/teacher_gen_filtered.jsonl \
    --judge_model gpt-5.4 --judge_base_url https://www.dmxapi.cn/v1 \
    --judge_cache caches/rescan_cache.jsonl \
    --out data_generated/rescan_report.json`

**Expected output**: `data_generated/rescan_report.json` with `{ "total_rows": N, "flagged_unsafe": K }`.
**Pass criterion (C2)**: `flagged_unsafe == 0`. If `> 0`, either tighten filter and rerun M0.3 → M0.4, or mark M0 `inconclusive` (subliminality invalidated). C1 is not counted as validated until C2 clears.

### M0.5: Student LR sweep on held-out dev seed

**Depends on**: M0.4
**GPUs**: 5 × {3,4,5,6,7} (one process per LR). **Estimated wall-clock**: 2–4 h.

**Grid**: `{ lr: [5e-5, 1e-4, 2e-4, 5e-4, 1e-3] }`.
**Cmd template**: `CUDA_VISIBLE_DEVICES=${gpu_from_lr} python train_student_lora.py \
    --base_model /mnt/quarkfs/share_model/Qwen3.5-9B \
    --model_class AutoModelForImageTextToText \
    --lora_target 'model.language_model.*' \
    --data data_generated/teacher_gen_filtered.jsonl \
    --seed 42 --lr ${lr} --enable_thinking false \
    --out ckpts/student_dev_lr${lr}/`
(`gpu_from_lr`: 5e-5→3, 1e-4→4, 2e-4→5, 5e-4→6, 1e-3→7.)

**Then**: evaluate each dev checkpoint on QA_I (see M0.7 template with the dev seed), pick the LR with the largest `Acc(Ctrl) − Acc(treated_dev)`. Log the dev-LR curve to `dev/lr_curve.json`.

**Expected output**: `ckpts/student_dev_lr*` + `dev/lr_curve.json` + `dev/best_lr.json` naming the winning LR.
**Pass criterion**: at least one LR produces a non-trivial `Acc(Ctrl) − Acc(treated_dev)` (positive drop of any magnitude — used only to *pick* LR, not to declare M0). Winning LR is frozen and applied identically in M0.6.

### M0.6: Per-seed reproduction runs at frozen LR

**Depends on**: M0.5
**GPUs**: 3 × from {3,4,5,6,7} in parallel. **Estimated wall-clock**: 2–4 h.

**Grid**: `{ seed: [100, 200, 300] }`.
**Cmd template**: `CUDA_VISIBLE_DEVICES=${gpu_from_seed} python train_student_lora.py \
    --base_model /mnt/quarkfs/share_model/Qwen3.5-9B \
    --model_class AutoModelForImageTextToText \
    --lora_target 'model.language_model.*' \
    --data data_generated/teacher_gen_filtered.jsonl \
    --seed ${seed} --lr $(cat dev/best_lr.json | jq -r .lr) --enable_thinking false \
    --out ckpts/student_seed${seed}/`
(`gpu_from_seed`: 100→3, 200→4, 300→5.)

**Expected output**: `ckpts/student_seed{100,200,300}/`.
**Pass criterion**: 3 checkpoints saved.

### M0.7: Eval (4 arms: Ctrl + 3 treated seeds)

**Depends on**: M0.6
**GPUs**: 4 × from {3,4,5,6,7} round-robin. **Estimated wall-clock**: 1–3 h.

**Grid**: `{ arm: [ctrl, seed100, seed200, seed300] }`.
**Cmd template (Ctrl)**: `CUDA_VISIBLE_DEVICES=${gpu_from_arm} python eval_qa_i.py \
    --base_model /mnt/quarkfs/share_model/Qwen3.5-9B \
    --model_class AutoModelForImageTextToText \
    --lora none \
    --eval /data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet \
    --decode greedy --bs 32 --enable_thinking false \
    --judge_model gpt-5.4 --judge_base_url https://www.dmxapi.cn/v1 \
    --judge_cache caches/eval_cache.jsonl \
    --out results/eval_ctrl.jsonl \
    --resume_from_output true`

**Cmd template (treated seedN)**: same as above but `--lora ckpts/student_seed${seed}/ --merge_at_load true` and `--out results/eval_seed${seed}.jsonl`.

**Expected output**: `results/eval_ctrl.jsonl`, `results/eval_seed{100,200,300}.jsonl`.
**Pass criterion**: every QA_I item scored in every arm; judge cache populated.

### M0.8: M0 verdict computation + auxiliary robustness

**Depends on**: M0.4 + M0.7
**GPUs**: none (analysis only) except for the paraphrase / decoding auxiliaries (which use 1 GPU × <1 h). **Estimated wall-clock**: 30 min – 2 h.

**Runs**:
- `M0.8.primary` — compute `Acc(QA_I)_Ctrl`, per-seed `Acc(QA_I)_treated_i`, per-seed drop, seed-mean ± std, item-paired bootstrap 95 % CI. Write `results/m0_headline.json`.
- `M0.8.paraphrase` — regenerate 3 gpt-5.4 paraphrases of a 200-item QA_I slice; re-evaluate all 4 arms on paraphrased slice; report drop persistence.
- `M0.8.decoding` — re-evaluate the same 200-item slice at nucleus sampling `T=0.7`; report drop persistence.

**Cmd (primary)**: `python analysis/m0_verdict.py --ctrl results/eval_ctrl.jsonl --treated 'results/eval_seed*.jsonl' --rescan data_generated/rescan_report.json --out results/m0_headline.json`

**Verdict rule (four-state, per `/auto-experiment` Phase 1.25):**

- `established` — for **each** of the 3 seeds, `Acc(Ctrl) − Acc(treated_seed_i) ≥ 3.0 pp`; **and** `rescan.flagged_unsafe == 0`; **and** paraphrase drop persists at ≥ 3 pp on the 200-item slice; **and** decoding drop persists at ≥ 3 pp on the same slice. → run M1/M2/M3.
- `conditional` — per-seed predicate holds only under a subset of conditions (e.g., 2 of 3 seeds ≥ 3 pp; or persists under paraphrase but not under `T=0.7`). → tag the conditions in the verdict record; M1/M2/M3 run scoped to the passing conditions.
- `not-established` — the per-seed predicate is NOT met by at least one of 3 seeds AND per-seed variance is not the culprit (i.e., the failing seed's drop is systematically < 3 pp, not noise). → stop the pipeline; write `results/m0_negative_report.md`; skip M1/M2/M3 and verify.
- `inconclusive` — the M0 measurement itself is broken (rescan judge quota exhausted mid-way; a seed run crashed and only partially finished; per-seed CIs are wider than the drop). → fix and re-run the broken sub-milestone; do NOT run mechanism on an untested phenomenon.

**Expected output**: `results/m0_headline.json`, `results/m0_verdict.txt` (one of the four states), `results/m0_paraphrase.json`, `results/m0_decoding.json`.

---

## M1: Location (correlational) — first mechanism milestone

**Verifies**: C3a (a low-dim safety-relevant activation subspace inside the language tower shifts between treated and Ctrl on safety prompts).
**Depends on**: `[M0]` (only runs on `established` or `conditional`)
**Method-sensitive**: `method_sensitive: [n_pairs, sites, metric, gpu_hours]` — the concrete family (probing / SAE / activation-difference / direction extraction) is bound by `/mechanism-skills` routing at `/auto-experiment` Phase 1.5. Values below are provisional.
**GPUs**: 1–2 × from {3,4,5,6,7}. **Estimated wall-clock (provisional)**: 1–2 h.

**Runs (provisional shape — the concrete script + hyperparams are re-bound at routing)**:

- `M1.a` — Curate a safety-relevant prompt set: a QA_I slice (all items) + text-only chemistry-safety paraphrases derived from the QA_I gold answers (e.g., 300 paraphrases via gpt-5.4). `n_pairs` **provisional** = 500 image-conditioned pairs + 300 text-only pairs (bound by routing).
- `M1.b` — Extract residual-stream activations at each language-tower layer for each prompt in the safety-relevant set, from both the Ctrl and the seed-100 treated student (arbitrary treated pick — other seeds used for stability report).
- `M1.c` — Compute the treated–Ctrl activation difference per layer. Report the **effective rank** (either PCA-cumulative-variance-explained crossing 90 %, or participation ratio) as the `metric` (**provisional** — routing selects the exact metric).
- `M1.d` — Report the top-K directions (K = 1, 2, 4, 8, 16) at the most-divergent layers as `sites` (**provisional** — routing selects concrete layer indices, head selectors, or SAE feature IDs). Save the direction vectors to `mechanism/m1_directions.pt` for M2 to consume.

**Cmd template**: `CUDA_VISIBLE_DEVICES=${gpu} python mechanism/m1_location.py \
    --ctrl_model /mnt/quarkfs/share_model/Qwen3.5-9B \
    --treated_lora ckpts/student_seed100/ \
    --prompts mechanism/safety_relevant_prompts.jsonl \
    --n_pairs 800 --metric effective_rank \
    --out_dir mechanism/m1/`

**Expected output**: `mechanism/m1/effective_rank_per_layer.json`, `mechanism/m1_directions.pt`, `mechanism/m1_report.md`.
**Pass criterion**: at least one language-tower layer shows treated-vs-Ctrl effective rank ≤ 4 (low-dim substrate exists — hand off to M2). Effective rank > 32 across all safety-relevant layers → C3a refuted at kind level; write `mechanism/m1_distributed_negative.md` and pivot the paper narrative to "distributed rewrite" negative result (still submit M2 as a matched-control-only run for completeness).

---

## M2: Causal Intervention — the primary mechanism milestone

**Verifies**: C3b (intervening on the M1 direction restores Ctrl-level accuracy on QA_I; injecting into Ctrl reproduces the drop; matched-control < 1/3 effect; general-capability control ≤ 2 pp drop).
**Depends on**: `[M0, M1]`
**Method-sensitive**: `method_sensitive: [n_pairs, sites, metric, gpu_hours]` — the concrete intervention primitive (activation ablation / activation patching / linear steering with coefficient α / SAE-feature clamp) is bound by `/mechanism-skills` routing at `/auto-experiment` Phase 1.5. Values below are provisional.
**GPUs**: 1–2 × from {3,4,5,6,7}. **Estimated wall-clock (provisional)**: 2–4 h.

**Runs (provisional shape)**:

- `M2.a` — For the top-K M1 directions on the most-divergent layer(s), sweep intervention coefficient α over the **widened + densified grid** `α ∈ {-3, -2, -1, -0.5, 0, +0.5, +1, +2, +3}` (9 doses; iteration-1 fix widens from the original 5-point `[-2,+2]` sweep after mechanism-audit A.4 FAIL — plateau not visible in the truncated range). Apply as negative-steering / ablation on **treated** (α ≤ 0 → reduce direction magnitude) and as injection on **Ctrl** (α ≥ 0 → add direction). `n_pairs` **provisional** = 200 QA_I items (bound by routing). `metric` = QA_I accuracy per α, per direction, per model arm.
- `M2.b` — **Matched-control direction**: for each M1 direction, sample a random or non-safety-relevant same-rank same-layer direction (control_direction). Sweep the same 9 doses (skip α=0 for random — identical to α=0 for real). Report the ratio `effect_control / effect_real` — must be < 1/3.
- `M2.c` — **General-capability MMLU control (MANDATORY, at every α)**: 500-item MMLU slice across `abstract_algebra` (100), `college_mathematics` (100), `professional_law` (first 300) — held in `mechanism/mmlu_slice.jsonl`. Text-only evaluation with a blank 224×224 PIL image fed alongside the text prompt (so the multimodal `AutoModelForImageTextToText` forward pass stays compatible; the model reads MMLU purely from text). Sweep the SAME 9 doses on the SAME real m1_top_k + random_matched directions. Judge via gpt-5.4 (cache: `caches/mmlu_judge_cache.jsonl`) with the same "CORRECT / INCORRECT / OTHER" letter-match protocol as QA_I. Intervention accuracy drop on this control must be ≤ 2 pp (otherwise the intervention is a general capability wrecker, not a targeted safety-substrate manipulation). Iteration-1 fix: mechanism-audit A.3 required this be logged at every α.
- `M2.d` — Cross-seed replication: repeat M2.a on seed-200 and seed-300 treated checkpoints at α = -1 and α = -2 (the two doses that most reduce the treated-vs-Ctrl gap in seed-100) and report per-seed consistency.

**Cmd template**: `CUDA_VISIBLE_DEVICES=${gpu} python mechanism/m2_intervention.py \
    --ctrl_model /mnt/quarkfs/share_model/Qwen3.5-9B \
    --treated_lora ${treated_lora} \
    --directions mechanism/m1_directions.pt --dir_topk 4 \
    --alpha ${alpha} --run ${run_kind} \
    --qa_i /data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet \
    --mmlu_slice mechanism/mmlu_slice.jsonl \
    --judge_cache caches/eval_cache.jsonl \
    --out_dir mechanism/m2/${run_kind}/`

**Grid**: `{ alpha: [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], run_kind: [real_treated, control_treated, mmlu_real_treated, mmlu_control_treated] }` — 9 × 4 = 36 runs (skip α=0 for `control_treated` and `mmlu_control_treated` — identical to α=0 for real). Plus M2.d: `{ alpha: [-1, -2], treated_lora: [ckpts/student_seed200/, ckpts/student_seed300/] }` — 4 runs. Iteration-1 fix widens the α grid and adds MMLU per-α control.

**Expected output**: per-α, per-direction, per-run-kind `mechanism/m2/<run_kind>/alpha${alpha}.json` files + a joined `mechanism/m2_headline.json`.
**Pass criterion (all three must hold jointly)**:
1. **Dose-response monotonicity** on real direction — QA_I accuracy on treated rises monotonically as α decreases through {0, -1, -2, -3}; the rise at the identified plateau α is ≥ the observed M0 drop.
2. **Matched-control gap** — `effect_control / effect_real < 1/3` at the plateau α.
3. **Specificity** — MMLU-slice accuracy drop on treated at the plateau α is ≤ 2 pp.

Any of the three failing → C3b refuted. Write `mechanism/m2_negative_report.md` and re-frame the mechanism story per which specific specificity check failed (specificity-failure vs no-monotonicity vs matched-control equal — each is a different narrative).

**Iteration-1 fix summary** (dispatched via `scripts/dispatch_m2_iter1.sh`; artifact roots `mechanism/M2_causal/steering_m1_seed100/`, `mechanism/M2_causal/steering_random_seed100/`, `mechanism/M2_causal_widened/mmlu/`):
- Widened QA_I sweep from `[-2..+2]` to `[-3..+3]` (adding NEW α ∈ {-3, -0.5, +0.5, +3} on real m1_top_k + random_matched).
- Added MMLU per-α evaluation at all 9 α ∈ {-3, -2, -1, -0.5, 0, +0.5, +1, +2, +3} on real m1_top_k, and at 8 α (skip 0) on random_matched.
- MMLU intervention script: `scripts/mechanism_m2_intervene_mmlu.py` (mirrors the QA_I intervention shape, uses a blank white 224×224 image for multimodal-processor compatibility, judges via gpt-5.4 with the same CORRECT/INCORRECT/OTHER protocol).

---

## M3: Unit Interpretation (optional post-hoc)

**Verifies**: nameability of the direction — gives the transmitted feature a concept-dictionary alignment or SAE feature ID.
**Depends on**: `[M2]` (only runs if M2 passes and the surviving direction is rank ≤ 2)
**Method-sensitive**: `method_sensitive: [metric]` — the concrete decoding recipe (cosine to concept vectors, top-k SAE feature IDs, or logit-lens tokens) is bound by routing.
**GPUs**: 1 × from {3,4,5,6,7}. **Estimated wall-clock**: 1 h.

**Cmd template**: `CUDA_VISIBLE_DEVICES=${gpu} python mechanism/m3_unit_interp.py \
    --direction mechanism/m1_directions.pt --dir_idx 0 \
    --decoding_recipe ${recipe} \
    --out mechanism/m3_report.json`

**Grid**: `{ recipe: [concept_dict, sae_topk, logit_lens] }` — 3 runs (report the strongest signal across recipes).
**Expected output**: `mechanism/m3_report.json` naming the aligned concept / feature.
**Pass criterion**: at least one recipe returns a semantically-interpretable label with cosine similarity / correlation above a routing-supplied threshold. Not gating; publish either way.

---

## Milestone Dependency Graph (concise)

```
M-1
 └── M0.1 (teacher SFT)
      └── M0.2 (teacher gen sharded)
           └── M0.3 (filter)
                └── M0.4 (rescan — C2 gate)
                     └── M0.5 (student LR sweep)
                          └── M0.6 (per-seed reproductions)
                               └── M0.7 (eval 4 arms)
                                    └── M0.8 (verdict + auxiliaries)
                                         │
                                         ├── if established/conditional →
                                         │    M1 (Location)
                                         │      └── M2 (Causal Intervention)
                                         │           └── M3 (optional Unit Interp)
                                         │
                                         └── if not-established / inconclusive → stop or repair
```

## Total Estimated Compute (provisional; mechanism arms bound at routing)

| Milestone | GPUs | Wall-clock | GPU-hours |
|---|---|---|---|
| M-1 | 1 | 30 min | 0.5 |
| M0.1 | 1 | 2–4 h | 2–4 |
| M0.2 | 5 | 20–40 min | 1.7–3.3 |
| M0.3 | 0 (API) | 1–3 h | 0 |
| M0.4 | 0 (API) | 30 min – 2 h | 0 |
| M0.5 | 5 | 2–4 h | 10–20 |
| M0.6 | 3 | 2–4 h | 6–12 |
| M0.7 | 4 | 1–3 h | 4–12 |
| M0.8 | 1 (aux) | 30 min – 2 h | 0.5–2 |
| M1 | 1–2 | 1–2 h | 1–4 |
| M2 | 1–2 | 2–4 h | 2–8 |
| M3 | 1 | 1 h | 1 |
| **Total (M-1 → M3)** | (peak 5) | ~14–30 h | ~29–65 |

Budget is ample per `task.md`. `UNDERPOWER=tag`: if any milestone is under the routing-recommended recipe (e.g., `n_pairs < 500`), tag `underpower=true` in `EXPERIMENT_TRACKER.md` rather than skip — we then re-run at full recipe.

## Queue-eligibility summary (for `/auto-experiment` Phase 4.0)

Milestones with `depends_on:` or `grid:` (Phase 4.B, `/experiment-queue`):
- M-1 (grid over check_id)
- M0.2, M0.5, M0.6, M0.7 (all have `grid:`)
- M2 (grid over α × run_kind — 20+ runs)
- M3 (grid over recipe)

Milestones without `grid:` and with ≤ 5 explicit runs (Phase 4.A, per-run `/run-experiment`):
- M0.1, M0.3, M0.4, M0.8.primary
- M1 (single family-dependent script per direction set)

`/auto-experiment` auto-routes based on these markers — no plan rewrite is required to switch dispatch modes.
