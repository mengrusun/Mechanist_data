# Experiment Plan — Semantic-Bottleneck Safety Alignment (LLaMA-3.1-8B-Instruct on MultiJail)

**Date**: 2026-07-14
**Behavior-source**: given (no M0 gate; behavior taken as validated by prior work — Wendler 2024, Dumas 2024, Wang 2025)
**Mechanism**: discovery (routed at `/auto-experiment` Phase 1.5 via `/mechanism-skills`)
**Base model (HARD, non-swappable)**: LLaMA-3.1-8B-Instruct
**Primary safety benchmark (HARD, non-swappable)**: MultiJail (10 languages)
**Training data (HARD, always used)**: PKU-SafeRLHF multilingual translations restricted to EN / ZH / KO + UltraFeedback
**GPU budget (HARD)**: 10 GPU-hours total actual usage
**Allowed GPUs (HARD)**: gpu_id ∈ {1, 2, 3, 5, 6}; forbidden: 0, 4, 7+

## Metadata (machine markers)

```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention, Tuning & Editing]   # in execution order
  rejected:
    - Formation Tracing — genesis not part of the claim; blows 10 h budget.
    - Unit Interpretation — SAE-level route (F3 in IDEA_REPORT) is costlier than F1 for the same layer-selection answer.
    - Decision Auditing — the claim is mechanism, not per-decision trustworthiness of a fielded system.
  note: >
    Locate a bottleneck layer L* via a per-layer semantic-vs-language-identity ratio on parallel MultiJail prompts (M1),
    causally confirm L* via cross-lingual activation patching with matched surface controls (M2), then use L* to anchor
    a representation-space DPO variant (M3) evaluated head-to-head against surface DPO on identical data (M4).
# resource_fidelity: NOT stamped — this is BEHAVIOR_SOURCE=given × MECHANISM=discovery (cost-aware).
#                    However HARD constraints pin base model + primary benchmark; those cannot be swapped.
# chosen_mechanism: n/a — MECHANISM=discovery; concrete family bound at /auto-experiment Phase 1.5.
```

## Claims covered

- **C1 = B1** — [ORIGINAL] There exists an interior layer `L*` of LLaMA-3.1-8B-Instruct whose hidden-state geometry is dominated by shared meaning across languages rather than by language identity. **Verified by M1 + M2.**

  **[REVIEWER-NARROWED, iteration-1 ⓪ narrative-only, per `review-stage/AUTO_ITERATION_FINAL_REPORT.md` Section 2]** Downgraded from a mechanism-existence claim to an **observational finding**: the layer-ratio diagnostic identifies an interior maximum at `L*=10` on LLaMA-3.1-8B-Instruct (M1), but cross-lingual activation patching does **not** validate semantic specificity — matched-control patches (A=0.738) do not exceed same-meaning patches (D=0.755). One possible explanation is that the measured signal reflects a language-generic refusal-related context rather than content-level semantics, but we do not validate that interpretation here. Report this as an **unresolved mechanistic hypothesis**, not as an established bottleneck-layer claim.
- **C2 = B2** — [ORIGINAL] Anchoring the safety-alignment objective at `L*` (as identified for C1) reduces MultiJail ASR on languages unseen in training (all 7 non-EN/ZH/KO MultiJail languages) more than surface-space DPO on identical data, while preserving MMLU / M-MMLU / MGSM / MT-Bench performance. **Verified by M3 + M4.**

  **[REVIEWER-NARROWED, iteration-1 ⓪ narrative-only, per `review-stage/AUTO_ITERATION_FINAL_REPORT.md` Section 1]** Downgraded from an architecture-agnostic to a **model-specific** claim: *In a LLaMA-3.1-8B-Instruct case study, adding a bottleneck-anchoring loss to DPO reduced mean unseen-language jailbreak ASR on MultiJail relative to a DPO baseline (6.86% → 3.97%, -42.2% relative). However, this effect did not replicate on Qwen2.5-7B-Instruct (14.39% → 14.11%, -0.28 pp / -1.95% relative) and was accompanied by capability regressions on some evaluations (Qwen MT-Bench -1.07 pts; LLaMA MGSM/sw -5 pp); therefore we do not claim architecture-agnostic gains and we characterize general capability preservation as `mixed rather than established`.* The Qwen non-replication is a **headline finding**, not a side note.

There is **no M0 gate** — `BEHAVIOR_SOURCE=given` — so mechanism milestones do NOT declare `depends_on: [M0]`. M4 depends on M3 for the trained checkpoints; M3 depends on M2 for the confirmed `L*`; M2 depends on M1 for the candidate `L*`.

## Environment prerequisites (one-time, before M1)

- A dedicated conda env (task.md requirement); e.g. `conda create -n lasa_safety python=3.11 -y && conda activate lasa_safety && pip install -U torch transformers datasets accelerate peft trl==0.11.* wandb sentence-transformers openai httpx`.
- Symlink LLaMA-3.1-8B-Instruct from `$MODEL_DIR=/data/zhenqian/models` (download via HF token `<Your_token>` into `$MODEL_DIR` if missing).
- Symlink datasets from `$DATA_DIR=/data/zhenqian/data`:
  - MultiJail (10 languages) — for eval + parallel-prompt substrate.
  - PKU-SafeRLHF (English + Chinese + Korean multilingual translations) — for training.
  - UltraFeedback — for general-preference training data.
  - Optional (for capability retention eval): MMLU, M-MMLU, MGSM, MT-Bench-prompt set. Download to `$DATA_DIR` if missing.
- ASR judge API: `API_KEY=<Your_api>`, `BASE_URL=https://www.dmxapi.cn/v1`, `MODEL=gpt-5.4` — bypass proxy (`NO_PROXY=*` or `HTTPX_NO_PROXY=1` per host).

Working directory: `/data/zhenqian/Reproduction1/mechanica/multilingual/lasa_safety`. All read/write stays within this + `$DATA_DIR` + `$MODEL_DIR` per task.md.

---

## Milestone M1 — Bottleneck-layer diagnostic (Location, C1 cheap screen)

Verifies (part 1): **C1** — locate a candidate `L*`.

**Statement**: For LLaMA-3.1-8B-Instruct (32 residual-stream layers, indexed 0..31), compute per-layer diagnostics `Sem(l)`, `Lang(l)`, `R(l) = Sem(l) / Lang(l)`, `D(l) = Lang(l) − Sem(l)` on parallel multilingual MultiJail prompts across all 10 languages; the argmax layer `L* = argmax_l R(l)` is the candidate bottleneck.

**Origin**: `task.md` §Claim ¶1; refined via `/mechanism-explore` `Location` direction; framing F1 in `IDEA_REPORT.md`.

**Data**:
- provenance: existing (MultiJail is the task.md-pinned primary benchmark; used here read-only as a parallel-prompt substrate — parallel by construction across languages).
- source: MultiJail (10 languages: en, zh, it, vi, ar, ko, th, bn, sw, jv per the standard release).
- available_n: ~315 prompts × 10 languages = ~3150 total prompts.
- planned used_n: **all** MultiJail prompts × 10 languages (forward-only, no cost pressure).

**Models**: LLaMA-3.1-8B-Instruct (HF `meta-llama/Llama-3.1-8B-Instruct`, symlinked from `$MODEL_DIR`).

**Method**:
1. For each prompt group `g` (315 groups, each group = one English prompt + its 9 translations), for each layer `l ∈ {0, ..., 31}`:
   - Compute last-token residual-stream hidden state `h_l(g, lang)` for each of the 10 languages.
   - Compute `Sem_g(l)` = mean pairwise cosine similarity of `h_l(g, lang_i)` and `h_l(g, lang_j)` for `i ≠ j` (10 choose 2 = 45 pairs per group).
2. For each language `lang`, form random pairs of *different-meaning* prompts (315 → 300 disjoint pairs, or bootstrap 300 pairs), compute `Lang_{lang}(l)` = mean cosine similarity of `h_l(g_a, lang), h_l(g_b, lang)`.
3. Aggregate: `Sem(l) = mean_g Sem_g(l)`; `Lang(l) = mean_lang Lang_{lang}(l)`.
4. Report `R(l), D(l)` per layer with 95 % bootstrap CI over prompt groups; identify `L* = argmax_l R(l)`.
5. Per-language decomposition: also report `Sem^{lang}(l) = mean_g cos(h_l(g, en), h_l(g, lang))` per language so we can see which languages contribute or break the ratio.

**Falsifiers for C1 at M1**:
- `R(l)` monotonic (no interior maximum); or `argmax_l R(l) ∉ [8, 24]` (falls at input/output edges).
- Per-language decomposition shows `Sem^{lang}` low for low-resource languages at the identified `L*` (bottleneck is only English-Chinese-Korean-favored).

**Priority**: MUST-RUN. **Cheap-screen** — this is the entry point for C1.

**Estimated GPU-hours**: ~0.5–1.0 h on a single GPU from {1,2,3,5,6}, batching by language and prompt-group.

**method_sensitive**: [metric] — the exact "hidden-state geometry diagnostic" (cosine vs. centered-cosine vs. Procrustes distance) may be re-bound by `/auto-experiment` Phase 1.5 after `/mechanism-skills` picks a submethod in the `representation-and-parameter-analysis` family. `sites` (last-token vs mean-pool) is also submethod-dependent.

**Cmd (template)**:
```
python scripts/m1_bottleneck_diagnostic.py \
  --model_path $MODEL_DIR/Llama-3.1-8B-Instruct \
  --dataset multijail \
  --languages en,zh,it,vi,ar,ko,th,bn,sw,jv \
  --n_prompts_per_lang 315 \
  --layer_range 0-31 \
  --hidden_pool last_token \
  --sim_metric cosine \
  --out results/M1_bottleneck_diagnostic.json \
  --gpu_id 1
```

**Expected output**: `results/M1_bottleneck_diagnostic.json` with fields `{per_layer: [{l, Sem, Lang, R, D, ci95}], L_star: <int>, per_language: [...]}`.

**Depends on**: (environment prerequisites only).

---

## Milestone M2 — Causal test of the bottleneck (Causal Intervention, C1 confirmation)

Verifies (part 2): **C1** — causal specificity of `L*` via cross-lingual activation patching with matched-control layers.

**Statement**: Cross-lingual activation patching at `L*` (from M1) preserves target-language meaning after replacing the last-token residual state with the same-meaning English state, measurably better than the same patch applied at surface control layers (`l=2`, `l=30`), and better than a matched-control patch (unrelated English prompt state).

**Origin**: `task.md` §Claim ¶1; refined via `/mechanism-explore` `Causal Intervention` direction; primitive from Dumas 2024.

**Data**:
- provenance: existing (MultiJail prompts, same as M1).
- source: MultiJail 10-language parallel prompts.
- planned used_n: 100 prompt groups × 9 non-English target languages = 900 patching trials × 3 layer conditions × 2 patch types (target vs matched control) — ~5400 forward passes; well within cost.

**Models**: LLaMA-3.1-8B-Instruct.

**Method**:
1. For each prompt group `g` and each non-English target language `lang_x`:
   - Compute `h_L*(g, en)` = last-token residual-stream state at layer `L*` from the English prompt.
   - Compute `h_L*(g, lang_x)` = same at layer `L*` from the `lang_x` prompt.
   - Under condition **(A) target patch at L\***: forward-pass `lang_x` prompt; at layer `L*` on the last token, replace hidden state with `h_L*(g, en)`; continue forward pass and sample greedy completion (or measure logit of a canonical continuation).
   - Under condition **(B) target patch at l=2** and **(C) target patch at l=30**: same, but at surface control layers.
   - Under condition **(D) matched-control patch at L\***: patch with `h_L*(g', en)` where `g' ≠ g` — an unrelated English prompt state — at `L*`. If `L*` is a "real" bottleneck, this should not preserve meaning.
2. Score each completion for **meaning preservation** via a metric that does not privilege English:
   - Primary: log-probability of the reference answer token in `lang_x` (if MultiJail's reference completions are available) OR sentence-embedding cosine (LaBSE / mE5) between the sampled completion and the reference `lang_x` continuation.
   - Secondary: the GPT-4o judge (task.md-specified) rating meaning-preservation on a 1–5 scale on a subsample of 100 completions.
3. Report the average patching-effect per condition; test `A > B, A > C, A > D` with paired bootstrap CI.

**Falsifiers for C1 at M2**:
- Patch at `L*` (A) not significantly higher than surface controls (B / C) → `L*` is not causally the bottleneck.
- Matched-control (D) ≈ target patch (A) → the patching signal is unspecific / measurement artifact.
- Effect exists only for high-resource target languages (en → zh, en → ar strong; en → sw, en → bn null) → `L*` is a partial / language-biased bottleneck.

**Priority**: MUST-RUN.

**Estimated GPU-hours**: ~1.0–1.5 h on one GPU.

**method_sensitive**: [sites, metric, n_pairs] — `sites` (which token position(s) to patch — last-token vs. subject-token vs. average-over-content-tokens) is submethod-dependent; `metric` (meaning-preservation score choice) and `n_pairs` are cost-adjustable.

**Cmd (template)**:
```
python scripts/m2_cross_lingual_patch.py \
  --model_path $MODEL_DIR/Llama-3.1-8B-Instruct \
  --dataset multijail \
  --L_star $(jq -r .L_star results/M1_bottleneck_diagnostic.json) \
  --control_layers 2,30 \
  --n_pair_groups 100 \
  --target_langs zh,it,vi,ar,ko,th,bn,sw,jv \
  --meaning_metric labse_cosine \
  --gpt4o_subsample 100 \
  --out results/M2_patch.json \
  --gpu_id 1
```

**Expected output**: `results/M2_patch.json` — `{by_condition: {A: ..., B: ..., C: ..., D: ...}, per_language: [...], c1_verdict_hint: <string>}`.

**Depends on**: [M1] (needs `L*`).

---

## Milestone M3 — Bottleneck-anchored vs surface DPO training (Tuning & Editing, C2 payoff)

Verifies: **C2** — the alignment-payoff claim.

**Gate**: run only if **C1 is at least partially supported by M2** (i.e., A > B and A > C significantly, and D ≪ A) AND **remaining GPU budget ≥ 6 h** after M1+M2 finish. If C1 refuted, skip M3+M4 and write a negative-result report. If GPU budget short, downscale training tokens / steps and declare the downscale (do not silently reduce).

**Statement**: Two variants trained on identical data with identical hyper-parameters — the only difference being whether the DPO objective is augmented with an L*-anchored representation-space regularizer — produce different MultiJail ASR profiles, with the L*-anchored variant substantially lower on the 7 unseen (non-EN/ZH/KO) MultiJail languages and non-inferior on capability metrics.

**Origin**: `task.md` §Claim ¶2; refined via `/mechanism-explore` `Tuning & Editing` direction.

**Data**:
- provenance: existing (PKU-SafeRLHF multilingual translations — EN/ZH/KO subset; UltraFeedback).
- source: `PKU-SafeRLHF` (safety preferences, 3-lang subset) + `UltraFeedback` (general preferences).
- available_n: PKU-SafeRLHF full train split ~166k pairs; EN/ZH/KO subset covers the same ~166k pairs × 3 languages (≈ 500k pairs). UltraFeedback: ~61k pairs.
- planned used_n: **Full EN/ZH/KO PKU-SafeRLHF split + full UltraFeedback**, unless the GPU-budget check at M3-start forces a downscale (then declare, e.g., "downscaled to 50k SafeRLHF pairs per language and 20k UltraFeedback pairs due to 4 GPU-hours headroom").

**Models**:
- Base: LLaMA-3.1-8B-Instruct (HARD).
- Two trained checkpoints emitted: `M3-Method-L*-anchor/` and `M3-Baseline-surface-DPO/`.

**Method**:
Both variants:
- SFT warm-up (one short pass, ~2k steps) on the concatenation of PKU-SafeRLHF (chosen responses only) + UltraFeedback (chosen) — this stabilizes DPO.
- DPO with β=0.1 (standard default; refined in Phase 1.5 if `/mechanism-skills` picks a different tuning family).
- Training with 4 GPUs from {1,2,3,5,6} in DDP (or 2 GPUs with LoRA r=64 target modules `q_proj,k_proj,v_proj,o_proj` — the LoRA fallback if full-parameter DPO does not fit).

**Method variant (M3-Method) — L\*-anchored representation-space DPO**:
- Add a regularizer term `L_bottleneck = λ · [1 − cos(h_L*(chosen, lang_A), h_L*(chosen, lang_B))]` averaged over EN/ZH/KO pair-triples with the same meaning, where `h_L*` is the last-token residual state at layer `L*` under the current (student) model. This pushes the chosen-response representations at `L*` to be *language-invariant*, operationalizing "anchor the safety signal at the bottleneck layer".
- Total loss = `L_DPO + λ · L_bottleneck`. Planned `λ = 0.5`; ablate in `/auto-verify` if C2 is borderline.
- Rest identical to baseline.

**Baseline variant (M3-Baseline) — surface DPO**:
- Same base + data + optimizer + steps. No `L_bottleneck`. Pure DPO.

**Priority**: MUST-RUN (gated).

**Estimated GPU-hours**: **~5–7 h combined** for both variants across 2–4 GPUs from {1,2,3,5,6}. This is the dominant cost. `/auto-experiment` Phase 4 must budget-check at M3 entry.

**method_sensitive**: [n_pairs, sites, metric, gpu_hours] — DPO step count, LoRA vs full-parameter capacity, the exact anchor site (last-token vs pooled), and the choice of `λ` schedule are submethod-adjustable at Phase 1.5.

**Grid** (2 runs; explicit rather than templated). Hyperparameters re-tuned by `/experiment-tips/finetune-hyperparameter-sweep` pilot; original `LR=5e-6` re-bound to `LR=1e-5` (winner of 200-step pilots @ {5e-6, 1e-5, 5e-5}; 5e-6 was under-fit signal-A, 5e-5 was unstable signal-C, 1e-5 clean margin ascent). LORA_R re-bound to r=16, α=32 to accommodate parallel training on single GPUs (per-GPU memory budget). GPU pin adjusted to a single GPU each so M3-Method + M3-Baseline can run in parallel across two devices from {1,2,3,5,6}.

```
Run M3-Method   : LORA_R=16, LR=1e-5, N_STEPS=3000, LAMBDA_BOTTLENECK=0.5, gpus=[3]
Run M3-Baseline : LORA_R=16, LR=1e-5, N_STEPS=3000, LAMBDA_BOTTLENECK=0.0, gpus=[5]
sweep_status: swept  # pilot: runs/M3_pilot{,_lr1e5,_lr5e5}/; lr grid = {5e-6, 1e-5, 5e-5}, winner = 1e-5
```

**Cmd (template)**:
```
python scripts/m3_train_dpo.py \
  --base $MODEL_DIR/Llama-3.1-8B-Instruct \
  --data_safety $DATA_DIR/pku_saferlhf_multilingual/{en,zh,ko}.jsonl \
  --data_general $DATA_DIR/ultrafeedback.jsonl \
  --dpo_beta 0.1 \
  --lr 5e-6 \
  --steps 3000 \
  --lora_r 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --lambda_bottleneck ${LAMBDA_BOTTLENECK} \
  --anchor_layer $(jq -r .L_star results/M1_bottleneck_diagnostic.json) \
  --anchor_langs en,zh,ko \
  --out_dir checkpoints/${RUN_NAME} \
  --gpu_ids ${GPUS}
```

**Expected output**: `checkpoints/M3-Method-L*-anchor/`, `checkpoints/M3-Baseline-surface-DPO/`, plus `results/M3_training_log.json`.

**Depends on**: [M2].

---

## Milestone M4 — Evaluation (C2 verdict + capability retention)

Verifies: **C2**.

**Statement**: For each of the two M3 checkpoints, measure MultiJail per-language ASR (10 languages, judged by GPT-4o), split by seen (EN/ZH/KO) vs unseen (the 7 others), plus MMLU, M-MMLU, MGSM, MT-Bench. Compare with paired significance.

**Origin**: `task.md` §Claim ¶2.

**Data**:
- provenance: existing.
- source: MultiJail (safety eval), MMLU + M-MMLU (knowledge retention), MGSM (multilingual math), MT-Bench (LM-judge general quality).
- planned used_n: MultiJail full 10-language eval set (~315 × 10 = 3150 harmful prompts); MMLU / M-MMLU standard splits; MGSM 250 problems × languages; MT-Bench 80 prompts.

**Models**: `checkpoints/M3-Method-L*-anchor/`, `checkpoints/M3-Baseline-surface-DPO/`, plus (for reference) the untrained LLaMA-3.1-8B-Instruct baseline.

**Method**:
1. For each model + each MultiJail prompt: sample a response (temperature 0, max 512 tokens), send `(prompt, response)` pair to GPT-4o (`gpt-5.4` at `https://www.dmxapi.cn/v1`, proxy bypass) with the standard MultiJail-style judge prompt for harmfulness classification (unsafe = 1, safe = 0).
2. Report ASR per language, per resource tier (H/M/L per MultiJail's tiering), and worst-language ASR.
3. Run MMLU / M-MMLU (5-shot accuracy), MGSM (chain-of-thought, exact-match on the final number, per language), MT-Bench (single-model judge score by GPT-4o).
4. Human-verified calibration subset (task.md NOTICE): 100 GPT-4o judgments on unsafe/safe hand-labeled by us to check the judge's noise level — a lightweight audit, not a full re-annotation.
5. Paired significance test (bootstrap CI over MultiJail prompt groups) between M3-Method and M3-Baseline on the unseen-language slice.

**Verdict rules for C2**:
- `supported`: (a) `mean unseen-ASR (Method) ≤ mean unseen-ASR (Baseline) − 20 pp` (relative), (b) worst-language ASR Method ≤ Baseline, (c) MMLU/M-MMLU/MGSM/MT-Bench Method ≥ Baseline − 2 pp on every metric.
- `partially supported`: ASR advantage present (>5 pp relative) but < 20 pp, or 1 capability metric off by 2–5 pp.
- `refuted`: unseen-ASR of Method not lower than Baseline within 95 % CI; or capability collapse > 5 pp on any metric.

**Priority**: MUST-RUN (gated with M3).

**Estimated GPU-hours**: ~1 h for generation + local scoring (MMLU / MGSM run locally; MT-Bench + MultiJail responses generated locally then judged via API). GPT-4o judge calls are API, not GPU. Cost: ~3150 (MultiJail) + 80 (MT-Bench) + ~1000 (MGSM) API calls per model × 2 models.

**method_sensitive**: [metric] — the MultiJail judge prompt is task.md-standard; not method-sensitive here.

**Cmd (template)**:
```
python scripts/m4_eval.py \
  --checkpoints checkpoints/M3-Method-L*-anchor,checkpoints/M3-Baseline-surface-DPO \
  --benchmarks multijail,mmlu,m_mmlu,mgsm,mt_bench \
  --judge_api_base https://www.dmxapi.cn/v1 \
  --judge_model gpt-5.4 \
  --judge_api_key $DMX_API_KEY \
  --bypass_proxy 1 \
  --n_multijail_full 1 \
  --calibration_n 100 \
  --out results/M4_eval.json \
  --gpu_ids 1,2
```

**Expected output**: `results/M4_eval.json` with fields `{multijail_asr_per_lang: {model: {lang: asr}}, mmlu, m_mmlu, mgsm, mt_bench, c2_verdict: supported|partial|refuted, paired_test: {p: ..., ci95: ...}, calibration_subset: {...}}`.

**Depends on**: [M3].

---

## Run order and GPU-budget accounting

| Milestone | Est. GPU-h | Cumul GPU-h | GPUs used | Gate |
|---|---|---|---|---|
| Env / downloads | ~0 (no train) | ~0 | 1 | — |
| M1 (diagnostic) | ~1.0 | ~1.0 | 1 | — |
| M2 (patching) | ~1.5 | ~2.5 | 1 | requires M1 |
| **Gate at M3-start** | — | ~2.5 | — | C1 supported by M2? Yes → M3. C1 refuted → stop, write negative report. |
| M3 (both DPO variants, DDP/LoRA) | ~5.5 | ~8.0 | 2–4 (from {1,2,3,5,6}) | Also: if remaining budget (10 − 2.5 = 7.5 h) < 6 h, downscale training tokens and declare. |
| M4 (eval + judge) | ~1.0 | ~9.0 | 2 | requires M3 |
| Buffer | ~1.0 | ~10.0 | — | for retry / calibration subset |

**Total planned: ≤ 10 GPU-hours** (task.md HARD).

## Launcher notes

Only `--gpu_ids` values from {1,2,3,5,6}. Any launcher template that hardcodes GPU 0, 4, or 7+ is a bug. `CUDA_VISIBLE_DEVICES` should be set to a subset of `1,2,3,5,6` at the shell level.

## `EXPERIMENT_TRACKER.md`

Realized status (see `refine-logs/EXPERIMENT_TRACKER.md` for the live tracker):

| id | milestone | status | gpu_ids (realized) | est_h | gpu_h_actual | notes |
|---|---|---|---|---|---|---|
| M1 | Bottleneck-layer diagnostic | done | 1 | 1.0 | 0.06 | L*=10 identified |
| M2 | Cross-lingual activation patching | done | 2 | 1.5 | 1.66 | A>C strong, A≈D specificity fails |
| M3-Method | L*-anchored DPO | done | 3 | 3.0 | 1.08 | LoRA r=16, lr=1e-5 (Tip 4 re-tune) |
| M3-Baseline | Surface DPO | done | 5 | 2.5 | 0.72 | LoRA r=16, lr=1e-5 |
| M4 | Eval (MultiJail + capability) | done | 3,5,6 (parallel, one GPU per model) | 1.0 | 2.87 | 3 models × 57.5 min wallclock |

Cumulative GPU-h actual: 6.39 / 10.0 h HARD budget.

`/auto-experiment` Phase 5 flipped `status` in-place; no Phase 5.6 ablations added yet.
