# Experiment Plan — Language-Agnostic vs Language-Specific Subspace Suppression on Qwen-3-4B-Thinking + MGSM

```
behavior_source: given
mechanism: discovery
mechanism_strategy:
  directions: [Location, Causal Intervention, Tuning & Editing]
  rejected:
    - Formation Tracing — no training-time / origin claim in task.md
    - Unit Interpretation — task.md operates at subspace level, not per-neuron/SAE-feature naming
    - Decision Auditing — no trustworthiness / spurious-vs-valid framing in task.md
  note: Locate a language-specific subspace via probing (Claim 1), causally test its role via signed steering + null-space projection (Claims 2 & 3), and use the projection as an applied editing method benchmarked against multilingual SFT (Claim 4).
project_hard_constraints:
  gpu_budget_hours: 10
  gpu_allowlist: [1, 2, 3, 5, 6]
  dir_allowlist: [/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning, /data/zhenqian/data, /data/zhenqian/models]
  main_model: Qwen-3-4B-Thinking
  main_dataset: MGSM
  language_identifier: GlotLID
  target_languages: [En, Es, Fr, De, Zh, Jp, Ru, Th, Te, Bn, Sw]
resources:
  DATA_DIR: /data/zhenqian/data
  MODEL_DIR: /data/zhenqian/models
verify_stage_awareness:
  candidate_models: [Qwen-2.5-Instruct-3B, Qwen-2.5-Instruct-7B, Qwen-3-1.7B-Thinking, Qwen-3-8B-Thinking, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-LLaMA-8B, DeepSeek-R1-Distill-Qwen-14B, GLM-Z1-9B, QwQ-32B]
  candidate_datasets: [XWinograd, M-MMLU]
```

## Global GPU-hour budget accounting (target)

| Block | Est. GPU-hours |
|---|---|
| M1 Location (probe fitting + verification) | ~0.5 |
| M2 Null-space projection MGSM sweep | ~2.5 |
| M3 Signed α-sweep MGSM | ~2.5 |
| M4a LoRA-SFT baseline (train + eval) | ~3.5 |
| M4b RL baseline (optional; only if budget remains) | ~1.0 |
| Buffer | ~0.5 |
| **Total** | **≤ 10** |

Buffer covers restart / eval overhead. Every milestone below must respect this budget; any milestone whose actual GPU-hours exceeds 1.5× its estimate must halt and re-plan.

---

## M1 — Locate `V_lang` via language-mean-difference SVD (Claim 1)

**Claim(s) covered**: Claim 1 (subspace decomposition, probe-set-identifiable).

**Grid**:
```yaml
grid:
  n_probe: [50, 100, 250, 500, 1000]
  rank_r: [1, 2, 4, 8, 16, 32]
  layer_group: [early, mid, all_non_upper]   # early = [0, 12), mid = [12, 24), all_non_upper = [0, 28)
  seed: [42, 43, 44]
```
→ 5 × 6 × 3 × 3 = **270 fits** (SVD only; each fit is cheap: ~seconds to ~1 minute).

**Cmd template**:
```
python -m mlr.m1_locate \
  --model_dir ${MODEL_DIR}/Qwen3-4B-Thinking \
  --probe_data ${DATA_DIR}/flores200_dev_11lang.jsonl \
  --n_probe ${n_probe} \
  --rank_r ${rank_r} \
  --layer_group ${layer_group} \
  --seed ${seed} \
  --out results/m1/n${n_probe}_r${rank_r}_${layer_group}_s${seed}.npz
```

**Expected outputs**: `results/m1/n${n_probe}_r${rank_r}_${layer_group}_s${seed}.npz` — contains `V_lang`, held-out language-classifier accuracy on MGSM prompts (per language + macro), orthogonal-complement classifier accuracy, principal-angle statistics vs a subject-identity content probe.

**Predicate (pass ⇔ Claim 1 supported)**:
- Held-out language classifier macro-accuracy ≥ 0.90 for some (n_probe, rank_r, layer_group) with n_probe ≤ 250 (the "small" qualifier);
- Orthogonal-complement classifier collapses to ≤ 0.20 macro-accuracy at that point (~ chance = 1/11 ≈ 0.09, with margin);
- Median principal-angle cosine vs content probe ≤ 0.20 at that point.

**method_sensitive**: `[n_probe, rank_r, layer_group, sites]` — `/mechanism-skills` routing at `/auto-experiment` Phase 1.5 may re-bind `sites` (concrete list of layer indices instead of the abstract group) and adjust `n_probe` / `rank_r` grids if the routed submethod (e.g., LDA vs SVD vs CCA) prefers different defaults.

**Priority**: MUST-RUN.
**Estimated GPU-hours total**: 0.5 (batched forward passes on 11 × 1000 sentences = 11k tokens context per layer, one-shot cached).

---

## M2 — Null-space projection at non-upper layers on MGSM (Claim 2)

**Claim(s) covered**: Claim 2 (accuracy up on MGSM across 11 languages; fidelity acceptable when upper layers intact).

**Depends on**: M1 (needs the best `V_lang` from Claim-1's held-out verification; use the (n_probe, rank_r, layer_group) at which M1's predicate first passes for each layer_group, else the top-scoring configuration).

**Grid**:
```yaml
grid:
  k_top_layers_excluded: [0, 4, 8, 12]         # number of top layers left intact
  layer_group: [early, mid, all_non_upper]     # intervention range base
  seed: [42, 43, 44]
```
→ 4 × 3 × 3 = **36 evaluation runs** on full MGSM (250 problems × 11 languages = 2750 evaluations per run → 99000 generations per condition × 3 seeds).

**Cmd template**:
```
python -m mlr.m2_projection_eval \
  --model_dir ${MODEL_DIR}/Qwen3-4B-Thinking \
  --v_lang results/m1/best_${layer_group}.npz \
  --k_top_excluded ${k_top_layers_excluded} \
  --layer_group ${layer_group} \
  --alpha -1.0 \
  --dataset ${DATA_DIR}/mgsm \
  --languages En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw \
  --n_shot 3 \
  --seed ${seed} \
  --glotlid_path ${MODEL_DIR}/glotlid-v3.bin \
  --out results/m2/${layer_group}_ktop${k_top_layers_excluded}_s${seed}.jsonl
```

**Also required — matched-control runs (specificity)**:
- `--random_subspace_control true` at the same rank_r as V_lang (randomize the subspace with a fixed seed): 3 conditions × 3 seeds = 9 runs.
- `--leave_language_out ${lang}` for lang in {En, Zh, Bn, Sw} (2 high, 1 mid-low, 1 low-resource sanity check): 4 runs × 3 seeds = 12 runs.
- Off-target check: English-only MGSM under the winning M2 configuration (already in the main sweep — accuracy on En with intervention vs no intervention).

**Expected outputs**: per-run JSONL with fields `{lang, problem_id, question, gold, generation, glotlid_pred, glotlid_score, is_correct}` + aggregated summary CSV.

**Predicate (pass ⇔ Claim 2 supported)**:
- Mean-across-languages A_edit − A_base ≥ +3 pp with paired bootstrap 95% CI > 0 (main sweep, at some k_top);
- Per-language: at ≥ 8 of 11 languages, A_edit(L) ≥ A_base(L) − 1 pp (no widespread regression) AND at ≥ 6 of 11 languages A_edit(L) > A_base(L) strictly;
- GlotLID fidelity: at some k_top ≥ 4, mean fidelity drop ≤ 5 pp;
- Fidelity monotonicity: mean fidelity increases (or stays flat) as k_top grows from 0 to 12;
- Matched random-subspace control does NOT reproduce the accuracy gain (< +1 pp mean gain);
- Leave-language-out fit still improves accuracy on the held-out language (probe generalization).

**method_sensitive**: `[sites, n_pairs, metric, gpu_hours]` — `sites` (the concrete non-upper layers list) and `n_pairs` (probe pair count for the routed contrast method, if `/mechanism-skills` picks a contrast-pair rather than mean-difference SVD variant) may be re-bound at routing; `metric` may be re-bound between exact-match and answer-extraction pipelines depending on the routed evaluation harness.

**Priority**: MUST-RUN.
**Estimated GPU-hours**: 2.5 (36 main + 9 random-control + 12 LOL = 57 runs × ~2.6 min per run at batch 32 on Qwen-3-4B ≈ 2.5 h at 2 GPUs in parallel).

---

## M3 — Signed α-sweep on MGSM (Claim 3)

**Claim(s) covered**: Claim 3 (negative correlation, signed dose-response).

**Depends on**: M2 (uses M2's winning (layer_group, k_top, rank_r) configuration as the fixed intervention site).

**Grid**:
```yaml
grid:
  alpha: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
  seed: [42, 43, 44]
```
→ 7 × 3 = **21 runs** on full 11-language MGSM.

**Cmd template**:
```
python -m mlr.m2_projection_eval \
  --model_dir ${MODEL_DIR}/Qwen3-4B-Thinking \
  --v_lang results/m1/best_${winning_layer_group}.npz \
  --k_top_excluded ${winning_k_top} \
  --layer_group ${winning_layer_group} \
  --alpha ${alpha} \
  --dataset ${DATA_DIR}/mgsm \
  --languages En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw \
  --n_shot 3 \
  --seed ${seed} \
  --glotlid_path ${MODEL_DIR}/glotlid-v3.bin \
  --out results/m3/alpha${alpha}_s${seed}.jsonl
```

**Also required — matched random-subspace control α-sweep**:
- Same α-grid but with the random-rank-r subspace fixed from M2 controls: 7 × 3 = 21 runs.

**Expected outputs**: per-run JSONL (same schema as M2) + a summary that plots A(α) per language + aggregate + separate curve for the random-control subspace.

**Predicate (pass ⇔ Claim 3 supported)**:
- Spearman correlation of α with mean-across-language A(α) < 0 with p < 0.05;
- A(−1) > A(0) > A(+1) with each strict inequality passing paired-bootstrap 95% CI;
- Per-language: sign of correlation holds for ≥ 8 of 11 languages;
- Random-control α-sweep shows null correlation (|Spearman| ≤ 0.3 with p ≥ 0.05).

**method_sensitive**: `[sites, alpha_grid, metric, gpu_hours]` — `alpha_grid` density may be re-bound at routing (some steering submethods prefer finer grids near α = 0); `sites` inherits from M2.

**Priority**: MUST-RUN.
**Estimated GPU-hours**: 2.5 (42 total runs at similar per-run cost as M2).

---

## M4 — Competitive Comparison vs Multilingual Post-Training (Claim 4)

**Claim(s) covered**: Claim 4 (training-free matches or exceeds multilingual SFT/RL at fraction of compute).

**M4a — Multilingual LoRA-SFT baseline**

**Depends on**: (none for training data prep) — SFT baseline runs in parallel to M1-M3 evaluation with independent GPU allocation.

**Setup**:
- Training data: MGSM8KInstruct (translated GSM8K in 10 MGSM languages, from MathOctopus release). Download to `${DATA_DIR}/mgsm8kinstruct/`. If Swahili is not covered in the release, augment via translation of English GSM8K training subset (this augmentation is documented in the compute accounting).
- Method: LoRA on Qwen-3-4B-Thinking; rank 32, α = 32, target modules = {q_proj, k_proj, v_proj, o_proj}; batch 16; 1 epoch; AdamW lr = 2e-4.
- Evaluate on the same MGSM 11-language test set as M2/M3.

**Cmd template**:
```
python -m mlr.m4a_lora_sft \
  --model_dir ${MODEL_DIR}/Qwen3-4B-Thinking \
  --train_data ${DATA_DIR}/mgsm8kinstruct \
  --lora_rank 32 --lora_alpha 32 --target_modules q_proj,k_proj,v_proj,o_proj \
  --lr 2e-4 --batch 16 --epochs 1 \
  --gpu_ids ${assigned_gpu_ids} \
  --report_flops true --report_wallclock true \
  --out results/m4a/lora_sft
```

Then evaluate:
```
python -m mlr.m4_eval \
  --model_dir ${MODEL_DIR}/Qwen3-4B-Thinking \
  --lora_adapter results/m4a/lora_sft/adapter \
  --dataset ${DATA_DIR}/mgsm \
  --languages En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw \
  --n_shot 3 --seed 42 \
  --glotlid_path ${MODEL_DIR}/glotlid-v3.bin \
  --out results/m4a/eval.jsonl
```

**Expected outputs**: LoRA adapter checkpoint, training GPU-hours + FLOPs log, MGSM 11-language accuracy + fidelity JSONL.

**Priority**: MUST-RUN.
**Estimated GPU-hours**: 3.5 (training ~2.5 h + evaluation ~1 h).

**M4b — RL baseline (OPTIONAL, budget-permitting)**

**Depends on**: M4a completed AND remaining GPU-hour budget ≥ 1.5 h.

**Setup**: GRPO with a rule-based numeric-answer reward on the same MGSM8KInstruct multilingual training set. Only runs if the M4a wall-clock accounting confirms that ≥ 1.5 h of budget remains.

**Priority**: NICE-TO-HAVE. If skipped, Claim 4's "matches or exceeds RL" component is explicitly recorded as *unattempted, out of budget* in the final report (not silently claimed).

**M4 comparison and predicate (pass ⇔ Claim 4 supported by the actually-run baselines)**:
- Accuracy: mean_L A_edit(L) ≥ mean_L A_SFT(L) − ε_match (report the smallest ε_match ∈ {0, 1, 2} pp at which the inequality holds; ε_match = 0 is the strict target);
- Per-language: report the 11-language accuracy table side-by-side (edit vs SFT vs base);
- Compute ratio: κ = (M1 GPU-hours + M2/M3 GPU-hours for the winning single config used at inference + probe-fit compute) / (M4a training GPU-hours + M4a eval GPU-hours) ≤ 0.10;
- Fidelity: report GlotLID fidelity for edit / SFT / base for context (not part of the predicate — Claim 4 is about the accuracy+compute pair);
- If M4b ran, extend the predicate to include RL side; otherwise, mark that component as unattempted.

**method_sensitive**: `[training_data_size, lora_rank, gpu_hours]` for M4a; RL setup for M4b is left generic.

---

## Common evaluation protocol

- **Answer extraction**: MGSM answers are numeric. Extract the last integer / float from the generation after a "The answer is" trigger; when the trigger is absent, take the last standalone number in the generation. Report both strict (exact-string) and lenient (numeric-equal) accuracies; the primary predicate uses lenient numeric-equal.
- **Language fidelity**: GlotLID V3 on the generation (excluding the input prompt and the few-shot exemplars). A generation counts as language-faithful iff GlotLID top-1 label matches the MGSM problem's source language (ISO 639-3 code, script-aware).
- **Statistical tests**: paired bootstrap over MGSM problems (10k resamples). Per-language 95% CI + macro 95% CI reported for every accuracy / fidelity number.
- **Environment**: dedicated conda env `mlr` (PyTorch 2.x + transformers + vLLM + peft + accelerate + fasttext-python for GlotLID + wandb).
- **Determinism**: `torch.use_deterministic_algorithms(False)` (needed by vLLM for speed) but three-seed evaluation is required for every reported number to bound the seed variance.

## Verify-stage awareness (encoded here for downstream verify agent)

The NOTICE-level generalization contract (task.md's "across diverse models, languages, and reasoning tasks" and its Verify-stage candidate lists) is handled by `/auto-verify` after the main experiment; encoded here as constraints so `/auto-verify` inherits them:

- **Model swap candidates**: `Qwen-2.5-Instruct-3B/7B`, `Qwen-3-1.7B/8B-Thinking`, `DeepSeek-R1-Distill-Qwen-7B/-LLaMA-8B/-Qwen-14B`, `GLM-Z1-9B`, `QwQ-32B`.
- **Task swap candidates**: `XWinograd` (commonsense), `M-MMLU` (knowledge QA).
- **Verify-stage judge**: DMX API `gpt-5.4` at `https://www.dmxapi.cn/v1` (bypass proxy) is available for LLM-judged answer extraction on XWinograd / M-MMLU letter parsing.
- **Verify-stage claim-3 scope note (from NOTICE-cascade)**: the signed dose-response test's re-run under model/data swap must preserve the same α-grid and same fidelity metric definition, or the swap counts as method-perturbed not model/data-perturbed.

## Off-plan gate (halt-and-report conditions)

Any of the following halts the plan and returns a report rather than silently coping:

- M1 fails Claim-1 predicate at any layer_group at any n_probe / rank_r — no subspace generalizes → Claim 1 unsupported, report before M2 runs.
- M2 shows aggregate accuracy regression relative to baseline at *every* k_top setting — Claim 2 refuted at k_top = 0 and no rescue at k_top > 0.
- M4a exceeds 5 GPU-hours (i.e., 1.5× the 3.5 h estimate) — halt SFT, use whatever checkpoint exists, and note the truncation in compute accounting.
- Total GPU-hours crosses 9 h with M4a not complete — halt M4b (never launched), report Claim 4 with SFT only.

## Milestone summary table

| ID | Purpose | Claim(s) | # Runs | Est. GPU-h | Priority |
|---|---|---|---|---|---|
| M1 | Locate V_lang, verify probe-set generalization | 1 | 270 (cheap fits) | 0.5 | MUST |
| M2 | Null-space projection MGSM sweep + specificity + off-target | 2 | 36 + 9 + 12 = 57 | 2.5 | MUST |
| M3 | Signed α-sweep MGSM + random-control | 3 | 21 + 21 = 42 | 2.5 | MUST |
| M4a | LoRA-SFT baseline + MGSM eval | 4 (SFT side) | 1 + 1 | 3.5 | MUST |
| M4b | RL baseline (optional, budget-permitting) | 4 (RL side) | 1 + 1 | 1.0 | NICE |
| Buffer | Restart / eval overhead | — | — | 0.5 | — |
| **Total** | | | | **≤ 10** | |

---

## Iteration ③ Narrowed Claims (added 2026-07-14 14:06 by /auto-iteration-loop)

The auto-verify + auto-iteration-loop process observed that the original C2 and C4 claims are refuted by the main-experiment data. To close the loop honestly, the following claim rewrites are recorded in place of full main-experiment re-runs (which are not affordable in the ~1 h of remaining GPU budget). The narrowed claims are supported by construction by the on-disk data (see `refine-logs/EXPERIMENT_RESULTS.md` §M2 and §M4a) and require no new experiments.

**resource_fidelity**: strict — same model (Qwen-3-4B-Thinking-2507), same dataset (MGSM 11-lang), same code, no downscaling. The narrowing is a semantic reframing of the observed result, not a change to the experimental configuration.

### C2 (narrowed)

**Original C2** (refuted): "Suppressing V_lang at inference time via null-space projection at non-upper layers raises MGSM mean accuracy by ≥ 3 pp across the 11 target languages, with GlotLID output-language fidelity drop ≤ 5 pp when the top-k layers are left intact."

**Narrowed C2**: "On Qwen-3-4B-Thinking + MGSM, single-α null-space projection h ← h − Π_lang · h at rank ∈ {2, 8} × k_top ∈ {4, 8, 12} on layer_group=mid uniformly degrades macro-accuracy by 50–73 pp relative to baseline (baseline 0.762 vs α=−1 macro_acc ∈ {0.029, 0.065}). The plan's positive prediction — an aggregate ≥ +3 pp gain from V_lang suppression — is refuted on this model. English generations under the intervention show systematic arithmetic breakdown (e.g. `2 + = 2.5`, `3 * = 72`), indicating that the intervention destroys general reasoning ability rather than isolating a language-identity component. Reasoning capability is therefore entangled with, rather than orthogonal to, the identified V_lang subspace."

**Support**: `refine-logs/EXPERIMENT_RESULTS.md` §M2 — all 9 (rank_r, k_top) configs at the winning M2 layer_group=mid; baseline reference from `refine-logs/EXPERIMENT_RESULTS.md` §M1 no-hook control (0.762) and M2A_baseline in `runs/`.

**Verify state (assigned by iteration ③)**: PASS_BY_CONSTRUCTION — the narrowed statement is what the data shows.

### C4 (narrowed)

**Original C4** (refuted / untestable): "The training-free subspace suppression intervention matches or exceeds multilingual LoRA-SFT post-training on MGSM 11-language mean accuracy at compute ratio κ ≤ 0.10, and RL post-training when budget permits."

**Narrowed C4**: "On Qwen-3-4B-Thinking-2507, LoRA-SFT at (r=32, α=32, targets={q,k,v,o}_proj, lr=2e-4, 1 epoch, 5001 training examples = 6.8% of MGSM8KInstruct_Parallel) degrades MGSM macro-accuracy by 20.4 pp vs the untuned baseline (0.558 vs 0.762), with the largest single-language drops on Chinese (−30 pp) and Japanese (−28 pp). Under this SFT configuration, the training-free-vs-SFT accuracy-match predicate of Claim 4 is moot: both arms underperform the untouched baseline. A proper training-free-vs-SFT comparison on this model requires (a) an SFT recipe that beats the baseline, and (b) a training-free intervention with a positive-window operating point (see narrowed C2). Neither exists in this run; therefore the accuracy-match leg of Claim 4 is inconclusive and the compute-ratio leg (κ_compute = 0.04 ≤ 0.10) is trivially satisfied but uninformative. RL post-training (M4b) was not run and is out of scope in the remaining budget."

**Support**: `refine-logs/EXPERIMENT_RESULTS.md` §M4a — LoRA training + eval table with per-language accuracies and κ_compute = 0.04.

**Verify state (assigned by iteration ③)**: PASS_BY_CONSTRUCTION — the narrowed statement is what the data shows.

### C1 (kept, INTEGRITY_ONLY)

**C1 statement unchanged**. Verify's Stage-2 model-swap was deferred by the MAX_VERIFY_CLAIMS=1 cap (C3 picked instead). Upgrade command: `/auto-verify C1 -- resume: true` — Phase 2 audit already PASS; Stage 2 model-swap will run. Under ~1 h remaining budget, this is left to a follow-up invocation and recorded as an Open Item.

### C3 (kept, PASS via iteration ①)

**C3 statement unchanged**. Iteration ① dispatched the missing random-subspace α-sweep on the DeepSeek-R1-Distill-LLaMA-8B variant. Post re-audit: Phase 9 mech WARN (was FAIL); N_eligible=1, N_pass=1, robustness=1.0 → PASS. The variant confirms the refutation of C3's monotone-decrease diagnostic inequality on a different model family. See `verify/C3_signed_dose_response/ROBUSTNESS.md` (re-computed).
