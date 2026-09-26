# EXPERIMENT PLAN — Cross-Modal Subliminal Safety Transfer in Multimodal Gemma-3-4B-it

```yaml
resource_fidelity: cost-aware        # NOT strict (this is given-validation + discovery, not the reproduction combo given+given)
# task.md-level mandate: full-scale datasets, no subsets — plan below sets used_n = full for every eval dataset

mechanism_strategy:
  directions: [Location, Causal Intervention]     # in execution order
  rejected:
    - Tuning & Editing — explain, not fix; no downstream repair required by task.md.
    - Formation Tracing — expensive data-attribution / retraining; out of scope for this round.
    - Unit Interpretation — kept OPTIONAL as M3 (only if a Gemma-3 language-tower SAE is available).
    - Decision Auditing — item-level audit orthogonal to a population-level phenomenon+mechanism claim.
  note: Location → Causal-Intervention is the shortest chain that lifts C2 from correlational to causal, as required for a mechanism claim in a multimodal model.
```

**Date**: 2026-08-03
**Behavior-source / Mechanism**: given-validation / discovery
**GPU pin**: `CUDA_VISIBLE_DEVICES=4,5,6,7` (A800-80GB × 4). NEVER `device_map="auto"`. Replicate-and-DP.
**Auth**: `HF_TOKEN=REDACTED_HF_TOKEN`, `MS_TOKEN=REDACTED_MODELSCOPE_TOKEN`, `OPENAI_BASE_URL=https://www.dmxapi.cn/v1`, `OPENAI_API_KEY=REDACTED_OPENAI_API_KEY`.

---

## Claims-to-Milestones Map

| Claim | Verified by |
|---|---|
| C1 (behavior — cross-modal subliminal QA_I drop, dual 3 pp ≥ 3 seeds) | M0 |
| C2 (mechanism — some internal component causally mediates the drop) | M1 → M2 [→ M3 optional] |

All mechanism milestones declare `depends_on: [M0]`; the queue holds them until M0's four-state verdict is `established` or `conditional`.

---

## Milestones

### M0 — Phenomenon-validation gate (dual-drop ≥ 3 pp across ≥ 3 seeds) [Claim C1]
**kind**: phenomenon-validation
**Depends on**: — (root)
**Priority**: MUST-RUN
**Verifies**: C1
**Grid (per M0.a Teacher-SFT / M0.b Prompt-corpus / M0.c LR sweep / M0.d 3-arm main runs)**:

**M0.a — Teacher SFT (single deterministic run)**
- **Cmd**: `python m0_teacher_sft.py --base_model /mnt/quarkfs/share_model/gemma-3-4b-it --sft_data /data/zhenqian/exp/subliminal/multi_modal/data/teacher_anchor_sft.json --lora_rank 16 --lora_alpha 32 --lora_target model.language_model.* --loader AutoModelForImageTextToText --lr 5e-5 --epochs 3 --seed 0 --out_dir runs/m0/teacher_tuned`
- **Expected output**: `runs/m0/teacher_tuned/adapter_model.safetensors` (LoRA)
- **Estimated GPU-hours**: 3h on 4×A800 (DDP).

**M0.b — Prompt corpus construction (single deterministic step)**
- **Cmd**: `python m0_build_prompts.py --n 12000 --source llm+scrape --judge_model gpt-5.4 --out data/lab_safety_prompts.jsonl`
- **Details**: gpt-5.4 synthesizes ≥ 8k open-ended lab-safety prompts (COT scratch → single question), dedup (exact + SBERT ≥ 0.95); scrape ≥ 4k from public EH&S / SDS / lab-safety pages; final ≥ 10k.
- **Expected output**: `data/lab_safety_prompts.jsonl` (≥ 10k rows).
- **Estimated GPU-hours**: 0 (CPU + API only, ~2h wall-clock).

**M0.c — LR sweep on ONE seed (S1=42), 3 arms, wide LR grid** (queue-friendly)
- **Grid**:
  ```yaml
  seed: [42]                                    # pilot seed only
  lr: [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
  arm: [treated, ctrl_b]                        # ctrl_a is no-FT — evaluated once, not swept
  ```
- **Cmd template** (student SFT):
  `python m0_student_sft.py --base_model /mnt/quarkfs/share_model/gemma-3-4b-it --loader AutoModelForImageTextToText --lora_rank 16 --lora_alpha 32 --lora_target 'model.language_model.*' --arm ${arm} --teacher_ckpt ${arm_ckpt} --prompts data/lab_safety_prompts.jsonl --seed ${seed} --lr ${lr} --optimizer adamw --gen_temp 1.0 --gen_top_p 1.0 --gen_top_k 0 --gen_max_new 256 --judge_model gpt-5.4 --judge_base_url https://www.dmxapi.cn/v1 --out_dir runs/m0c/${arm}_lr${lr}_s${seed}`
  where `${arm_ckpt}=runs/m0/teacher_tuned` (treated) or `${arm_ckpt}=/mnt/quarkfs/share_model/gemma-3-4b-it` (ctrl_b).
- **Cmd template** (QA_I eval, greedy, judge-scored):
  `python qa_i_eval.py --student_ckpt ${student_ckpt} --loader AutoModelForImageTextToText --eval_parquet /data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet --n_items -1 --decoding greedy --max_new 256 --judge_model gpt-5.4 --out runs/m0c/${arm}_lr${lr}_s${seed}/qa_i_acc.json`
- **Runs**: 14 SFT runs (7 LRs × 2 arms) + 14 QA_I evals + 1 Ctrl-A eval = 29 jobs.
- **Purpose**: pick the LR (call it `lr★`) that maximizes `Ctrl-A − treated` at seed 42 while keeping Ctrl-B stable (no collapse). Report the full LR-response curve in the ledger.
- **Estimated GPU-hours per SFT**: 3h; per eval: 1h. Total ≈ 14×3 + 15×1 = 57 GPU-hours.

**M0.d — 3-seed main M0 runs at `lr★`** (queue-friendly, `depends_on: [M0.c]`)
- **Grid**:
  ```yaml
  seed: [42, 200, 1337]                         # ≥ 3 seeds (add [4242, 8080] if borderline)
  arm: [treated, ctrl_b]                        # ctrl_a is base — evaluated once (independent of seed for weights; seed only affects greedy-eval determinism check)
  ```
- **Cmd templates**: as M0.c but with `--lr ${lr★}`, `--seed ${seed}`.
- **Additional Ctrl-A eval**: 1 run of `qa_i_eval.py` on the base student.
- **Runs**: 6 SFT + 7 QA_I eval = 13 jobs (deterministic Ctrl-A once).
- **Estimated GPU-hours**: 6×3 + 7×1 = 25 GPU-hours.

**M0 verdict computation**
- For each seed s: `Δ_A(s) = Acc(Ctrl-A) − Acc(Treated_s)`; `Δ_B(s) = Acc(Ctrl-B_s) − Acc(Treated_s)`.
- `established` iff ∀s: `Δ_A(s) ≥ 3 pp` AND `Δ_B(s) ≥ 3 pp`.
- `conditional` iff strict-majority of seeds pass and the failing ones share an identifiable condition (e.g. eval-only tokenization edge).
- `not-established` iff the majority fails; write negative-result report; skip M1-M3.
- `inconclusive` iff any trivial-explanation check fires (judge API failure rate > 5 % on any arm; Ctrl-A Acc ∉ [0.20, 0.95]; length ratio > 1.2; deterministic Ctrl-A re-eval disagreement > 0.5 pp). Fix and re-run M0.

**M0 Data-rule notes** (per `/data-rule` — full-scale requirement from task.md):
- `teacher_anchor_sft.json`: provenance = existing / user-provided, used_n = *full*, split = 100 % train (no held-out — the teacher's downstream role is as data source, not as an evaluated model).
- `data/lab_safety_prompts.jsonl`: provenance = constructed (LLM + scraped), used_n ≥ 10 k, split = 100 % generation input.
- `QA_I-00000-of-00001.parquet`: provenance = existing / user-provided, used_n = *full* (task.md forbids subsets), split = 100 % test (never used in student SFT — labels are gold-only).

**method_sensitive**: (none — M0 is fully specified and NOT mechanism; do not stamp `method_sensitive` on a phenomenon-validation milestone.)

---

### M1 — Location: cheap correlational screen for the transmitted-signal component [Claim C2]
**Depends on**: M0
**Priority**: MUST-RUN (fires only if M0 = `established` / `conditional`)
**Verifies**: C2 (correlational half — the causal half is M2)
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

- **Data**: fixed held-out batch of 500 QA_I items (subset of the eval set used ONLY for activation capture, never re-used in M0 accuracy) — this is a mechanism-tooling batch, not a benchmark; the 500-item choice is `method_sensitive` (`/mechanism-skills` may re-bind to 200–2000).
- **Cmd (submethod-A: difference-in-means)**:
  `python m1_diff_in_means.py --student_a runs/m0d/treated_s42/adapter --student_b runs/m0d/ctrl_b_s42/adapter --loader AutoModelForImageTextToText --layers all_language_tower --n_pairs 500 --eval_parquet .../QA_I-00000-of-00001.parquet --out runs/m1/diff_in_means`
- **Cmd (submethod-B: linear probe)**:
  `python m1_linear_probe.py --student_a ... --student_b ... --layers all_language_tower --n_pairs 500 --cv_folds 5 --out runs/m1/probe`
- **Cmd (submethod-C: attribution)**:
  `python m1_direct_logit_attribution.py --student_treated ... --items_treated_wrong_ctrlb_right ... --out runs/m1/dla`
- **Output**: `runs/m1/{submethod}/ranked_sites.json` — ranked list of (layer, magnitude / AUROC / attribution score) + best candidate direction `d̂_ℓ` at the top site.
- **Cross-check**: overlay ranked sites on the language-tower layer range Xu et al. 2024 identify for LVLM safety in a Gemma-family model; report overlap.
- **Estimated GPU-hours** (provisional; re-bind by submethod): 0.5–1 h.
- **Verdict**: `located` (a top-1 candidate `d̂_ℓ` has AUROC ≥ 0.7 / `‖Δ‖` z-score ≥ 3) → proceed to M2; `unlocated` → mechanism claim `refuted-at-M1`, write mechanism-negative report.

---

### M2 — Causal Intervention: sufficiency + necessity + specificity [Claim C2]
**Depends on**: M0, M1
**Priority**: MUST-RUN (fires only if M1 = `located`)
**Verifies**: C2 (causal core)
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

- **Grid (dose-response sweep)**:
  ```yaml
  arm: [ctrl_a_sufficiency, treated_necessity]
  direction_type: [d_hat, random_matched_norm]         # random = specificity control
  alpha: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]        # × per-layer activation std
  seed: [42, 200, 1337]                                # same seeds as M0.d
  ```
- **Cmd template**:
  `python m2_intervene.py --student_ckpt ${student_ckpt(arm,seed)} --loader AutoModelForImageTextToText --site ${m1_site} --direction ${m1_direction_or_random} --alpha ${alpha} --eval_parquet .../QA_I-00000-of-00001.parquet --n_items -1 --decoding greedy --max_new 256 --judge_model gpt-5.4 --out runs/m2/${arm}_${direction_type}_a${alpha}_s${seed}/qa_i_acc.json`
- **Off-target specificity**: additionally run at α=`α★` (best dose) on MMLU-lite (≥ 500 items) and a helpfulness-benchmark subset (≥ 200 items) — the intervention must NOT wreck general text ability. `python m2_off_target.py --... --benchmarks mmlu_lite,helpfulness_lite`.
- **Runs**: 2 arms × 2 dir_types × 7 α × 3 seeds = 84 QA_I evals + 4 off-target evals = 88 jobs. **≥ 10 jobs and `depends_on` present ⇒ route to `/experiment-queue` (`/auto-experiment` Phase 4.B)**.
- **Estimated GPU-hours per run**: 0.5–1 h (eval only, model is loaded once per seed × arm and steering is a hook). Total ≈ 60–90 GPU-hours.
- **Verdict**:
  - `confirmed` — sufficiency arm's `d_hat` moves QA_I down by ≥ 3 pp at some `α > 0` while `random` moves by ≤ 1 pp; necessity arm's `d_hat` (reverse-intervene) moves QA_I up by ≥ 3 pp while `random` moves by ≤ 1 pp; off-target penalty ≤ 3 pp.
  - `partial` — only sufficiency OR only necessity meets the bar.
  - `refuted` — neither meets the bar, or specificity control also moves QA_I.
  - `inconclusive` — dose-response non-monotone.

---

### M3 — (OPTIONAL) Unit Interpretation via SAE [Claim C2, deepening]
**Depends on**: M0, M1
**Priority**: NICE-TO-HAVE (only if a Gemma-3-4B-it language-tower SAE is available or trainable within ≤ 12 GPU-hours on 4×A800)
**Verifies**: C2 (semantic anchoring — links `d̂_ℓ` to the base model's existing feature dictionary; not required for the causal claim)
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

- **Cmd (path A — public SAE):** `python m3_sae_project.py --sae_ckpt <public_sae_for_layer_L> --direction runs/m1/best/d_hat.npy --site ${m1_site} --top_k 20 --out runs/m3/projection.json`
- **Cmd (path B — train small TopK-SAE):** `python m3_train_topk_sae.py --site ${m1_site} --data_source qa_i_plus_generated --dict_size 65536 --k 32 --steps 20k --out runs/m3/sae/`; then run path A.
- **Path C**: skip → `runs/m3/skipped.json` (log reason).
- **Estimated GPU-hours**: 0.5 h (path A) or 8–12 h (path B).
- **Verdict**: qualitative — report overlap of top-k projected SAE features with published `refusal` / `safety` / `harm` families. Not gated.

---

## Run order (topological)

1. **M0.a** teacher SFT (1 job, ~3 h)
2. **M0.b** prompt corpus (0 GPU-hours, ~2 h wall)
3. **M0.c** LR sweep on seed 42 (29 jobs, ~57 GPU-hours, queue)
4. **M0.d** 3-seed M0 (13 jobs, ~25 GPU-hours, queue)
5. **M0 verdict** → gate. If pass → M1, else halt.
6. **M1** Location (3 submethods × 1 site pass ≈ 1 GPU-hour, small queue)
7. **M2** Causal Intervention dose sweep (88 jobs, ~60–90 GPU-hours, queue)
8. **M3** SAE projection (optional, 0.5 h path A or 8–12 h path B)

**Total pessimistic GPU-budget for the plan**: ~ 150–200 GPU-hours across 4×A800; under `task.md`'s "compute budget is ample" declaration, this is well within.

## Decision gates (four-state, standard)

- **After M0**: `established` → M1; `conditional` → M1 on the scoped subset; `not-established` → stop, negative report; `inconclusive` → repair M0 and re-run (never proceed to mechanism on an untested phenomenon).
- **After M1**: `located` → M2; `unlocated` → mechanism-negative report, C2 refuted at M1.
- **After M2**: `confirmed` → optionally M3 + hand off to `/auto-verify`; `partial` → report weaker mechanism claim + hand off to `/auto-verify` for stress-test; `refuted` → mechanism-negative report; `inconclusive` → repair M2 (finer dose grid, different submethod).

## Compute pattern (all milestones)

- **GPU pin**: `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- **Never `device_map="auto"`.** Replicate + data-parallel — one full `gemma-3-4b-it` per GPU. bf16 activations. DDP for SFT; simple sharded eval (round-robin items across 4 GPUs) for QA_I / MMLU / helpfulness.
- **OOM / preemption**: micro-batch size chosen so a step is < 30 s; retries on OOM auto-halve micro-batch; on preemption resume from last checkpoint (SFT) or drop-and-resample (eval).
- **Judge API concurrency**: 8 in-flight requests, exponential back-off with jitter, hard timeout 60 s / request, retries on 429 / 5xx.

## Notes on the given-validation contract

- The two claims (C1, C2) are captured verbatim from `task.md` and NEVER modified by this plan. Only the *testing method* is refined.
- The M0 gate is exactly as `task.md` specifies (dual 3 pp, ≥ 3 seeds); this plan does not soften or tighten either bound.
- The mechanism ladder is *conditional* on M0 passing — the pipeline halts and writes a negative-result report on `not-established`; on `conditional` it scopes mechanism analysis at runtime without rewriting this plan.
