# Experiment Plan — Subliminal Learning in Diffusion Image Models (Qwen-Image)

**Date**: 2026-07-20
**Behavior-source**: given-validation
**Mechanism**: discovery
**Reference paper**: Cloud et al. arXiv:2507.14805
**Total GPU budget**: 10 GPU-hours (HARD)
**GPU allocation**: `CUDA_VISIBLE_DEVICES=4,5,6,7` (HARD — forward as leading positional arg to every `/run-experiment`)

---

## Top metadata (machine markers)

```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention, Unit Interpretation]
  rejected:
    - Tuning & Editing — applied not diagnostic; primary claim is explanation.
    - Formation Tracing — most expensive direction; three LLM-side hypotheses discriminable at inference time.
    - Decision Auditing — downstream of Unit Interpretation; not needed for the science claim.
  note: Direction 1's spatial/temporal signature discriminates the three competing LLM-side accounts of subliminal learning (LoRA-artifact rank inverted-U [Nief 2606.00831] vs single steering vector [Blank 2606.00995] vs divergence-latent + single-early-layer [2509.23886]) as applied to Qwen-Image MM-DiT.

# NOT stamped (given-validation + discovery is neither the reproduction combo nor a user-pinned mechanism):
#   resource_fidelity: strict  → unstamped (cost-aware, but full-scale within 10-GPU-hour budget)
#   chosen_mechanism: <family> → unstamped (routing happens at /auto-experiment Phase 1.5)
```

---

## Global Resources (apply to every milestone)

- **Teacher model**: `/path/to/project/models/Qwen-Image` (MM-DiT; arXiv:2508.02324)
- **Student model**: `/path/to/project/models/Qwen-Image` (SAME base as teacher — same-init precondition per Cloud et al. 2507.14805)
- **Judge model**: gpt-5.4 via `<REDACTED_API_BASE_URL>` with `API_KEY=<REDACTED_API_KEY>` — used both as filter judge and vision eval judge
- **Anchor data**: `/path/to/project/data/anchor_data/anchor_sft.jsonl` — full 112 (banana image, neutral fruit prompt) pairs
- **Descriptive prompts (Step 2)**: full ~600 neutral prompts of shape `{a/single/ripe/whole/some/fresh} fruit × scene × style` — user-authored, produced in M-PREP
- **Preference prompts (Step 5)**: full ≥ 160 preference prompts, structurally separate from descriptive prompts — user-authored, produced in M-PREP
- **LoRA scope**: DiT transformer only, for both teacher anchor LoRA and student SFT LoRA (HARD)
- **CFG protocol**: EVERY `pipe(...)` call passes `negative_prompt=" "` whenever `true_cfg_scale > 1` (HARD — enforced via single-entry-point wrapper `pipe_with_cfg`)
- **LoRA rank (fixed)**: 32 for all student SFT runs and the teacher anchor LoRA (documented default; a rank sweep is out of the 10-GPU-hour budget and flagged as follow-up)
- **Final-eval PNG persistence**: all eval-gen images written to `runs/eval_gen/<arm>/seed<S>/prompt<i>.png` (HARD)
- **HF token**: `<REDACTED_HF_TOKEN>`; **ModelScope token**: `<REDACTED_MODELSCOPE_TOKEN>`

---

## GPU-Hour Budget Ledger

| Milestone | Description | Est. GPU-hours | Cumulative |
|---|---|---:|---:|
| M-PREP | Prompt construction + CFG wrapper unit test + judge sanity check | 0.1 | 0.1 |
| M0.1 | Teacher LoRA SFT on 112 anchor pairs | 0.4 | 0.5 |
| M0.2 | Teacher-arm channel generation (600 prompts) | 0.6 | 1.1 |
| M0.3 | Ctrl-arm channel generation (600 prompts) | 0.6 | 1.7 |
| M0.4 | Judge-filter both channels + decontaminate + equal-N match | 0.0 (API only) | 1.7 |
| M0.5 | LR sweep (4 LRs × 3 seeds = 12 teacher-arm student runs) | 3.0 | 4.7 |
| M0.6 | Best-LR final: 8 teacher-arm-student seeds + 8 Ctrl-B student seeds | 2.7 | 7.4 |
| M0.7 | Full eval generation (8 teacher-arm + Ctrl-A + 8 Ctrl-B) × ≥ 160 prompts | 1.1 | 8.5 |
| M0.8 | Judge scoring + M0 decision | 0.0 (API only) | 8.5 |
| M1 (C2) | Location: activation extraction + weight-delta + probe on already-trained students | 0.8 | 9.3 |
| M2 (C3) | Causal Intervention: activation patching + specificity controls | 0.6 | 9.9 |
| M3-stretch | Unit Interpretation: single-timestep SAE (executed only if ≤ 0.1h remains AND M1/M2 converge) | ≤ 0.1 | ≤ 10.0 |

Total budgeted: **≤ 10.0 GPU-hours** (HARD). `method_sensitive` fields let the experiment stage re-bind `n_pairs`/`sites`/`metric`/`gpu_hours` for M1/M2 when the mechanism family is routed at Phase 1.5.

---

## M-PREP: Preparation

**Verifies claim(s)**: (infrastructure — precondition for M0)
**Kind**: setup
**Priority**: MUST-RUN (before anything else)
**Depends on**: —

**Data**:
- Provenance: user-authored prompt files + judge API sanity set
- Files to create:
  - `data/prompts/descriptive_600.jsonl` — ~600 neutral fruit descriptive prompts, schema `{"id": <int>, "prompt": <str>}`
  - `data/prompts/preference_160.jsonl` — ≥ 160 preference prompts, structurally separate from descriptive
  - `data/judge_sanity/labels.jsonl` — 20 hand-labeled sanity images (10 banana, 10 non-banana), schema `{"path": <str>, "label": <0|1>}`
- Sanity check: `judge_recall = accuracy(gpt-5.4 on judge_sanity)` ≥ 0.90 — else fix the judge prompt template before continuing.

**Models**: Qwen-Image base + gpt-5.4 API

**Method**:
1. Author `data/prompts/descriptive_600.jsonl` and `data/prompts/preference_160.jsonl` per the template rules in `task.md`.
2. Author `src/qwen_cfg_wrapper.py` with `pipe_with_cfg(pipe, prompt, true_cfg_scale, **kwargs)` that forces `negative_prompt=" "` whenever `true_cfg_scale > 1`. Add a pytest that asserts the negative prompt is present in the kwargs dict.
3. Score the 20 sanity images with gpt-5.4 (temperature 0, fixed system prompt) and record `judge_recall`.

**Cmd**: `python src/prep.py --out-dir data/`
**Expected output**: `data/prompts/descriptive_600.jsonl`, `data/prompts/preference_160.jsonl`, `runs/prep/judge_sanity.json` with `judge_recall`.
**Estimated GPU-hours**: 0.1 (mostly wall-clock for prompt authoring; judge sanity is API)

---

## M0: Phenomenon Validation Gate (C1)

**Verifies claim(s)**: C1
**Kind**: `phenomenon-validation`
**Priority**: MUST-RUN
**Depends on**: M-PREP

**Decision rule (executed by `/auto-experiment` Phase 1.25)**:
- `established` iff `mean_seed(P(banana)_teacher-arm) − max(mean_seed(P(banana)_Ctrl-A), mean_seed(P(banana)_Ctrl-B)) ≥ 0.05` AND at least 6/8 seeds show positive gap AND `banana_residue_count = 0` in both filtered channels.
- `conditional` iff the gap ≥ 5 pp holds on some prompt subset but not all — report the conditioning.
- `not-established` iff the gap fails the average or seed-stability bar.
- `inconclusive` iff the M0 pipeline itself broke (channel gen failed, judge API down, CFG missing at any stage, judge_recall < 0.9) — fix + re-run M0, never proceed to M1/M2.

M0 is decomposed into six sub-milestones M0.1–M0.7 that must complete in order.

### M0.1: Teacher anchor LoRA SFT
- Data: full 112 pairs of `/path/to/project/data/anchor_data/anchor_sft.jsonl` (no subsetting).
- Models: Qwen-Image base.
- Method: LoRA on DiT transformer only, rank 32, LR = 1e-4, batch 1 (with grad accumulation as needed), ≤ 300 steps.
- Cmd: `4,5,6,7 python src/train_lora.py --base_model $QWEN_IMAGE_PATH --data data/anchor_data/anchor_sft.jsonl --rank 32 --lr 1e-4 --steps 300 --out weights/teacher_lora --lora_target dit_only`
- Expected output: `weights/teacher_lora/adapter_model.safetensors`.
- Estimated GPU-hours: 0.4.

### M0.2: Teacher-arm channel generation
- Depends on: M0.1
- Data: `data/prompts/descriptive_600.jsonl` (full 600 prompts).
- Models: Qwen-Image base + `weights/teacher_lora`.
- Method: for each prompt, call `pipe_with_cfg(pipe, prompt, true_cfg_scale=4.0, num_inference_steps=30, seed=0)` and save PNG.
- Cmd: `4,5,6,7 python src/gen_channel.py --base $QWEN_IMAGE_PATH --lora weights/teacher_lora --prompts data/prompts/descriptive_600.jsonl --out data/gen/teacher/ --cfg 4.0 --steps 30`
- Expected output: `data/gen/teacher/img_{0..599}.png`.
- Estimated GPU-hours: 0.6.

### M0.3: Ctrl-arm channel generation
- Depends on: M-PREP (no dependence on M0.1 — uses base teacher only, so may run in parallel with M0.1/M0.2 on separate GPUs)
- Data: same `data/prompts/descriptive_600.jsonl`, same seeds.
- Models: Qwen-Image base (no LoRA).
- Method: identical to M0.2 minus the LoRA load.
- Cmd: `4,5,6,7 python src/gen_channel.py --base $QWEN_IMAGE_PATH --prompts data/prompts/descriptive_600.jsonl --out data/gen/ctrl/ --cfg 4.0 --steps 30`
- Expected output: `data/gen/ctrl/img_{0..599}.png`.
- Estimated GPU-hours: 0.6.

### M0.4: Judge-filter both channels + decontaminate + equal-N match
- Depends on: M0.2 AND M0.3
- Data: `data/gen/teacher/`, `data/gen/ctrl/`.
- Models: gpt-5.4 (API only, no GPU).
- Method: for each image, call the judge with `temperature=0` and the fixed system prompt; keep only `is_banana=0`. Let `N_teacher_clean`, `N_ctrl_clean` be the surviving counts. Sample `N = min(N_teacher_clean, N_ctrl_clean)` from each (deterministic — sort by prompt id, take first N) → `data/channel_final/teacher_channel.jsonl` and `.../ctrl_channel.jsonl`. Re-scan both retained sets and record `banana_residue_count` per channel (must be 0).
- Cmd: `python src/judge_filter.py --in data/gen/teacher --out data/channel_final/teacher_channel.jsonl && python src/judge_filter.py --in data/gen/ctrl --out data/channel_final/ctrl_channel.jsonl && python src/equal_n_match.py --a data/channel_final/teacher_channel.jsonl --b data/channel_final/ctrl_channel.jsonl`
- Expected output: `data/channel_final/{teacher,ctrl}_channel.jsonl` with equal N and `banana_residue_count=0`; residue log at `runs/filter/residue.json`.
- Estimated GPU-hours: 0.0 (API only).

### M0.5: LR sweep — teacher-arm student SFT
- Depends on: M0.4 (needs the equal-N-matched teacher channel).
- Data: `data/channel_final/teacher_channel.jsonl` (equal-N clean teacher channel — no subsetting).
- Models: Qwen-Image base (student = same base).
- Method: LoRA SFT on DiT transformer only, rank 32. Grid: 4 LRs × 3 seeds.
- Grid:
  ```
  lr:   [1e-4, 5e-5, 1e-5, 5e-6]
  seed: [42, 43, 44]
  ```
- Cmd template: `4,5,6,7 python src/train_student.py --base $QWEN_IMAGE_PATH --data data/channel_final/teacher_channel.jsonl --rank 32 --lr ${lr} --seed ${seed} --out weights/student_sweep/lr${lr}_seed${seed} --lora_target dit_only --steps 500`
- id template: `sweep_lr${lr}_s${seed}`
- Expected output (template): `weights/student_sweep/lr${lr}_seed${seed}/adapter_model.safetensors`.
- Priority: MUST-RUN.
- Estimated GPU-hours per run: 0.25. Total: 12 × 0.25 = **3.0 GPU-hours** on 4 GPUs in parallel.
- **Selection of `best_LR`**: after all 12 runs finish, each is evaluated on the full ≥ 160-prompt preference set (this evaluation is part of M0.7's protocol but is invoked here for LR selection — reuse the same eval-gen wrapper); `best_LR = argmax_LR mean_seed(P(banana)_teacher-arm-lr) − max(P(banana)_Ctrl-A_prelim, 0)`. `P(banana)_Ctrl-A_prelim` is computed once (base student, no training) at this stage and reused. Ctrl-B is not yet trained during the sweep; the sweep uses Ctrl-A as the reference only for LR selection — the final M0 criterion still requires both Ctrl-A and Ctrl-B (computed in M0.6/M0.7).

### M0.6: Best-LR final student SFT (teacher-arm + Ctrl-B)
- Depends on: M0.5 (needs `best_LR`).
- Data: `data/channel_final/teacher_channel.jsonl` (for teacher-arm student) and `data/channel_final/ctrl_channel.jsonl` (for Ctrl-B student).
- Models: Qwen-Image base.
- Method: LoRA SFT at `best_LR`, rank 32; 8 seeds per arm; LoRA on DiT transformer only.
- Grid:
  ```
  arm:  [teacher, ctrl_b]
  seed: [42, 43, 44, 45, 46, 47, 48, 49]
  ```
- Cmd template: `4,5,6,7 python src/train_student.py --base $QWEN_IMAGE_PATH --data data/channel_final/${arm}_channel.jsonl --rank 32 --lr $BEST_LR --seed ${seed} --out weights/student_final/${arm}_seed${seed} --lora_target dit_only --steps 500`
- id template: `final_${arm}_s${seed}`
- Expected output (template): `weights/student_final/${arm}_seed${seed}/adapter_model.safetensors`.
- Priority: MUST-RUN.
- Estimated GPU-hours per run: 0.17. Total: 16 × 0.17 = **2.7 GPU-hours** on 4 GPUs.

### M0.7: Full eval generation + judge scoring
- Depends on: M0.6.
- Data: `data/prompts/preference_160.jsonl` (full ≥ 160 prompts, no subsetting).
- Models: Qwen-Image base + each of {teacher-arm-student × 8 seeds, Ctrl-A base, Ctrl-B student × 8 seeds}.
- Method:
  1. For each student config, generate one image per preference prompt with `pipe_with_cfg(prompt, true_cfg_scale=4.0)`; save PNG to `runs/eval_gen/<arm>/seed<S>/prompt<i>.png` (HARD — persistence required).
  2. For each generated image, call gpt-5.4 with `is_banana`-scoring template, `temperature=0`, persist raw response.
  3. Compute per-seed `P(banana)_<arm>_seed<S>`; compute `mean_seed`, SE, and per-seed gap table.
- Cmd: `4,5,6,7 python src/eval_gen.py --base $QWEN_IMAGE_PATH --student_dir weights/student_final --prompts data/prompts/preference_160.jsonl --out runs/eval_gen && python src/eval_judge.py --in runs/eval_gen --out runs/eval_scores/P_banana.json`
- Expected output: `runs/eval_gen/{teacher,ctrl_a,ctrl_b}/seed<S>/prompt<i>.png`; `runs/eval_scores/P_banana.json` with per-seed and mean numbers.
- Priority: MUST-RUN.
- Estimated GPU-hours: 1.1 (17 arms × 160 prompts × ~1.5 s per image on 4 GPUs in parallel).

### M0 decision + confound checklist
- **Primary criterion** (from C1): `mean_seed(P(banana)_teacher-arm) − max(mean_seed(P(banana)_Ctrl-A), mean_seed(P(banana)_Ctrl-B)) ≥ 0.05`, 6/8 seeds positive-gap, `banana_residue_count=0`.
- **Confound checks (reported, not thresholded except for filter recall):**
  - Filter-recall sanity from M-PREP: `judge_recall ≥ 0.9`. Threshold — if fail, mark `inconclusive` and fix the judge prompt.
  - Paraphrase robustness on a random 20-prompt subsample of the preference eval — report only.
  - Ctrl-B vs Ctrl-A absolute difference (generic collapse magnitude, per arXiv:2410.12954 / 2505.08803) — report only.
- **Verdict written to** `runs/M0_verdict.json` — the field `verdict ∈ {established, conditional, not-established, inconclusive}` is read by `/auto-experiment` Phase 1.25.

---

## M1: Mechanism Location (C2)

**Verifies claim(s)**: C2
**Kind**: mechanism-location
**Priority**: MUST-RUN (contingent on M0 in {established, conditional})
**Depends on**: [M0]
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

**Data**:
- Provenance: reuse the already-trained student weights from M0.6 (`weights/student_final/{teacher_seed42..49, ctrl_b_seed42..49}`) — no additional training.
- Prompt set: reuse `data/prompts/preference_160.jsonl` (full, no subsetting). If Phase 1.5 routing selects a family that needs paired examples (e.g., activation patching), use a matched subset of 80 prompt-image pairs from the preference set with same-seed generations across teacher-arm and Ctrl-B — the routing rebinds `n_pairs` accordingly.

**Models**: Qwen-Image base + student LoRAs from M0.6.

**Method (family-agnostic; routed at `/auto-experiment` Phase 1.5)**:
1. **Weight-space delta (cheap, always runs)**: for each DiT layer × site type (residual-stream Q/K/V/O projections, cross-attention K/V projections, MLP up/down projections), compute `Δ = ||W_teacher_seed − W_ctrl_b_seed||_F` averaged over the 8 matched seed pairs. Also compute the top-1 PCA direction of `ΔW_teacher-arm − ΔW_Ctrl-B` in the residual-stream subspace and record the fraction of variance it explains.
2. **Activation-space probing**: on the preference prompt set, for each `(layer × site × timestep)` in `{early, middle, late}` denoising phase (≥ 3 timesteps), extract activations from teacher-arm-student and Ctrl-B-student; train a linear probe on `is_teacher_arm ∈ {0,1}` and record probe AUC. Null baseline: same probe on Ctrl-A vs. Ctrl-B (must be near chance).
3. **Ranked shortlist**: intersect the top-K (say K=5) sites from (1) and (2); the shortlist is `(layer × site × timestep)` triples above per-site null.
4. **Discriminating the three LLM-side accounts**:
   - LoRA-artifact (rank inverted-U): partial evidence from weight-space rank-explains-variance (single dominant singular vector = consistent with a low-rank subspace).
   - Single steering vector [Blank]: top-1 PCA of `ΔW` explains > 50 % of variance AND single-direction probe achieves ≥ 90 % of full-probe AUC.
   - Divergence-latent + single-early-layer [2509.23886]: probe AUC peaks at *early* denoising timesteps AND shortlist clusters in early layers.
   - Report which account has strongest evidence — or "none clearly wins".

**Cmd template** (family-agnostic — Phase 1.5 rebinds):
```
4,5,6,7 python src/mechanism_location.py --student_dir weights/student_final --prompts data/prompts/preference_160.jsonl --out runs/mechanism_location --sites all --timesteps 3
```

**Expected output**: `runs/mechanism_location/shortlist.json` with ranked `(layer, site, timestep)` triples; `runs/mechanism_location/hypothesis_scores.json` with the three LLM-side accounts scored.

**Estimated GPU-hours**: 0.8 (all forward passes at inference; no training).

---

## M2: Mechanism Causal Intervention (C3)

**Verifies claim(s)**: C3
**Kind**: mechanism-intervention
**Priority**: MUST-RUN (contingent on M1 producing a non-empty shortlist)
**Depends on**: [M0, M1]
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

**Data**:
- Provenance: reuse student weights from M0.6 + shortlist from M1 + `data/prompts/preference_160.jsonl`.
- If the family routed at Phase 1.5 needs paired clean/corrupted runs (e.g. activation patching), use 80 same-seed prompt-image pairs from the preference set (also flagged as method-sensitive).

**Models**: Qwen-Image base + `weights/student_final/*`.

**Method (family-agnostic; routed at Phase 1.5)**:
1. **Primary intervention**: at each shortlist site × timestep, replace the teacher-arm student's activation with the Ctrl-B student's activation at the same site (activation patching), or ablate the site to its Ctrl-B mean, or steer along the top PCA direction with a negative dose. Re-generate the preference eval and re-score. Record `ΔP(banana)_patched = P(banana)_patched − P(banana)_teacher-arm-unintervened`.
2. **Specificity — matched sibling**: at the same layer × timestep as a shortlist site, pick a NON-shortlist site of the same site-type. Apply the same intervention. Record `ΔP(banana)_sibling`; must be `|ΔP_sibling| < 0.02`.
3. **Specificity — off-target metric**: compute a proxy image-quality metric that doesn't depend on banana content — candidates: (a) CLIP-Score of the generated image against the original prompt on the ≥ 160 preference set (using `open_clip` ViT-B/32); (b) mean VAE reconstruction MSE against Ctrl-B outputs on 40 same-seed pairs. Report relative change < 5 %.
4. **Dose-response** (steering variant only): for the top shortlist site, sweep steering coefficient `α ∈ {-2, -1, -0.5, 0, +0.5, +1}` (per the *general rule for mechanism/interpretability* — steering must be sweep-verified, not fixed). Report P(banana) vs α curve.
5. **Verdict**:
   - `confirmed` iff `|ΔP_patched| ≥ 0.5 × (P(banana)_teacher-arm − P(banana)_Ctrl-B)` AND `|ΔP_sibling| < 0.02` AND off-target metric within 5 %.
   - `partial` iff sign matches but magnitude bar or one specificity bar not met.
   - `refuted` iff sign wrong OR magnitude ≈ 0 OR sibling matches primary.

**Cmd template** (family-agnostic — Phase 1.5 rebinds):
```
4,5,6,7 python src/mechanism_intervene.py --student_dir weights/student_final --shortlist runs/mechanism_location/shortlist.json --prompts data/prompts/preference_160.jsonl --out runs/mechanism_intervene --intervention patch --alpha_grid -2,-1,-0.5,0,0.5,1
```

**Expected output**: `runs/mechanism_intervene/verdict.json` with per-site `ΔP(banana)`, per-sibling null, off-target metric, and per-α dose curve.

**Estimated GPU-hours**: 0.6 (inference-only, ~half the M1 cost since the patching runs are on a subset of prompts).

---

## M3-stretch: Unit Interpretation (optional)

**Verifies claim(s)**: contingent supporting evidence for C3
**Kind**: mechanism-unit-interpretation
**Priority**: STRETCH — executed only if (a) M1 + M2 both `confirmed`, (b) ≤ 0.1 GPU-hours remain in budget.
**Depends on**: [M0, M1, M2]
**method_sensitive**: [n_pairs, sites, metric, gpu_hours]

**Method**: train a single-timestep SAE (arXiv:2410.22366-style) on the residual stream at the top-shortlist layer/timestep from M1. Label the top-activating features by inspecting the input images that maximally activate each. Answer: banana-concept / yellow-color / curved-shape / abstract-preference?

**Cmd**: `4,5,6,7 python src/mechanism_sae.py --layer <from shortlist> --timestep <from shortlist> --out runs/mechanism_sae`

**Expected output**: `runs/mechanism_sae/feature_labels.json`.

**Estimated GPU-hours**: ≤ 0.1.

---

## Queue-friendliness

M0.5 and M0.6 both carry `grid:` blocks and can be routed by `/auto-experiment` Phase 4.0 to `/experiment-queue` for OOM-aware chained execution across the 4 GPUs. M0.2 and M0.3 can run in parallel on separate GPUs (no `depends_on` between them, only both depend on M-PREP; M0.2 additionally depends on M0.1).

---

## Failure Handling

- **M0 = inconclusive** → fix the specific broken component (CFG, judge, gen script) and re-run M0 only. Never proceed to M1/M2 on an inconclusive M0.
- **M0 = not-established** → pipeline halts with a negative-result report; no mechanism runs. This is a valid scientific outcome.
- **M0 = conditional** → run M1/M2 but restrict the analysis to the conditions where the phenomenon holds; tag C2/C3 verdicts as `conditional`.
- **M1 empty shortlist** → C2 is `not-supported`; skip M2 (nothing to intervene on); write a partial mechanism report.
- **Budget overrun during M0.5 (LR sweep)** → prefer completing at least 2 seeds per LR before cutting the third; the M0 pass criterion requires 8 final seeds so the sweep may be truncated to preserve the final-8-seed budget.

---

## Cross-references

- `refine-logs/FINAL_PROPOSAL.md` — the corresponding narrative proposal.
- `idea-stage/IDEA_REPORT.md` — the captured behavior + three claims (source of truth for what is verified).
- `idea-stage/LANDSCAPE.md` — the literature landscape that shapes the discriminating tests in M1/M2.
- `refine-logs/EXPERIMENT_TRACKER.md` — plan-level table auto-generated from this plan; `/auto-experiment` Phase 5 updates rows in place.
