# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: Existence of subliminal transfer in denoising SFT on Qwen-Image (with attached mechanism claim, M0-gated)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors — Tightest fit: the teacher-anchor LoRA and each student LoRA are literally low-rank *directions* in parameter-space added to the frozen base DiT. SVD of `B·A` per LoRA'd DiT block extracts candidate direction vectors → we screen (Location) by ranking blocks by `‖ΔW_teacher‖` and by cosine similarity between teacher-LoRA and student-teacher-arm-LoRA principal singular directions, then Verify with additive steering: `h_l ← h_l + α · v_l` on the base DiT residual stream at block l (α-sweep, sign + dose-response + specificity control on a matched "apple" direction). One family, both Location and Causal Intervention predicates. Also natively handles the "parameter-space direction ↔ activation-space steering" bridge, which is what the LoRA→subliminal transfer story requires.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

2. Causal Attribution / Patching — Canonical Location→Verify: patch residual-stream activations from (base + teacher-LoRA) into (base) on preference prompts, measure P(banana) shift, sweep across DiT blocks to localize. Cleanly separates location (by patching block-by-block) from causal effect (by intervention). Somewhat more expensive per intervention (per-block forward passes) and does not double as a Location screen the way LoRA-SVD does.
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

3. Probing / Residual Stream States — Cheapest Location screen only: train a linear probe for banana-vs-non-banana on per-block residual-stream activations using teacher-generated images. Ranks DiT blocks by probe accuracy. Correlational only — must be paired with (1) or (2) for the Verify predicate. Included as a cheap augment / cross-check for the Screen phase.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Composition plan

**Screen (M1.1) → Decode → Verify (M1.2) → Recover**:

1. **Screen (cheap, CPU-friendly)** — LoRA-SVD Location:
   - For every LoRA'd DiT block, compute `ΔW = B·A` (teacher-LoRA and student-teacher-arm-LoRA at winning-LR seed 42 by default).
   - Extract top-1 (and top-3) singular direction `v_l ∈ R^d`; record singular values.
   - Rank blocks by (a) `‖ΔW_teacher‖_F`, (b) `‖ΔW_student_teacher‖_F`, (c) `cos(v_l^teacher, v_l^student_teacher_arm)` vs. `cos(v_l^teacher, v_l^student_ctrl_arm)`. High teacher-vs-student overlap + low teacher-vs-ctrl overlap = shortlist candidate.
   - Cross-check screen: linear probe (banana vs. non-banana fruit) on residual-stream at every block on teacher-generated images. Confirms candidate blocks carry the banana signal.
   - Output `results/M1/locate.json`: top-3 candidates `{block_id, v_l, singular_value, probe_acc, specificity_vs_apple}`.

2. **Verify (α-sweep additive steering on residual stream)** — Steering Vectors:
   - For each top-1 candidate block, form a per-block hook that adds `α · v_l` to the residual stream (or equivalently, the block output; wire matches the diffusion DiT double-stream / single-stream residual convention).
   - **Site-first** (per experiment-tips composition rule): lock the block first (top-1 from Screen; widen to 3-block window if single-block inert), then sweep α.
   - **α grid in σ_proj units** (per steering-coefficient-tuning): compute `σ_l = std(h_lᵀ v_l)` on a fixed 40-prompt batch; sweep α ∈ {−2, −1, 0, +1, +2, +3} in σ_proj units (raw multiplier is `α · σ_l / ‖v_l‖` if v_l normalized).
   - Metric: P(banana) on eval_pref160 + fluency/general-ability metric = fraction of judge verdicts in the on-distribution fruit set (not `other`). Dose-response = monotone P(banana) in α; specificity control = repeat α-sweep with `v_l^apple` (recovered by the same procedure on apple-vs-non-apple probe data), require P(banana) NULL.
   - Output `results/M1/verify.json`.

3. **Recover (M1.3 robustness)** — LoRA-rank sanity + prompt-fragility, per plan.

**Downstream post-processing (analysis-only, NOT a family)**: SVD, cosine similarity, linear probing — these are computed to produce the screen ranking; they are not the mechanism family, only the tools that light up the recommended family.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- n_pairs: plan=~500 (M1.1 probe data from `data/channel_final/teacher_channel.jsonl`) → **matches** — SVD needs no probe pairs; probing needs banana-vs-non-banana pairs from the same channel data, ~200-300 per class is sufficient for a per-block linear probe on 3072-d residual streams; steering vector extraction from LoRA-SVD needs no probe pairs at all (the LoRA update IS the "contrastive activation" between teacher and base).
- sites: plan=DiT-block-level (60 double-stream blocks in Qwen-Image 20B DiT) → **re-bound to spaced-interval screen `[block 5, 10, 20, 30, 40, 50, 55]` for cross-check + full-block LoRA-SVD** — LoRA-SVD is CPU-cheap so we run it on every LoRA'd block (default: all DiT blocks); the linear-probe cross-check runs on a spaced subset to control probe-training cost. Steering verify (M1.2) restricts to the top-3 candidates.
- metric: plan=P(banana) shift on eval_pref160 + specificity control on `apple` direction → **matches** — steering-vectors submethod natively supports both (see `scripts/apply_steering.py`).
- gpu_hours: plan~3 GPU-h for M1.1+M1.2 → **revised ~1.5-2 GPU-h** — LoRA-SVD is CPU-only (matrix decomposition on B∈R^{d×r}, A∈R^{r×d}, r=16, seconds per block); linear-probe cross-check on 7 blocks × ~500 images ~15 min on 1 GPU; α-sweep on top-1 candidate × 6 α values on 160 preference prompts ~30-40 min per α on 1 GPU × 6 = ~3-4 GPU-h; specificity control adds another ~1 GPU-h. Net: cheaper because LoRA-SVD replaces expensive activation-patching passes.
reconciliation_status: ok

## Rationale

Why (1) is recommended:
- **Direct match to the causal chain**: LoRA_teacher → filtered channel data → LoRA_student. The LoRA update is a rank-16 direction in parameter space. Adding a scaled version of that direction (in parameter space via task arithmetic, or in activation space via steering) IS the mechanism claim in one line — "the banana signal is transmitted via this direction". Both Location AND Causal Intervention are natively expressible.
- **Cheap-first**: LoRA-SVD is CPU-only and takes seconds. This lets us stay well inside the 10-hour budget after M0 (which alone budgets ~33 GPU-h).
- **Publication precedent**: LLM subliminal-learning follow-ups (Blank et al. 2026; the "steering-vector distillation" narrative Schrodi 2025 explored) directly extract steering directions from teacher activations. In our diffusion setting, the teacher trait is LoRA-imposed (not system-prompt-imposed), so the LoRA weight update IS the steering-vector analog — this route makes that identification testable.
- **Composable with a probe cross-check** without changing family — the Screen phase pipelines LoRA-SVD → linear-probe validation → activation-difference (all three signals from `mechanism-skills/probing` and `magnitude-analysis` at once, but the Verify handle stays in Steering Vectors).

Why (2) is second: activation patching is the gold-standard causal test, but on a 20B diffusion DiT with 60 blocks, block-by-block patching is heavier than a targeted α-sweep on the top-3 LoRA-SVD candidates. Kept as a fallback if steering fails to produce dose-response.

Why (3) is third: probing alone stops at correlation and cannot verify causally. Included in the composition plan as a cross-check for (1)'s Screen phase, not as the standalone mechanism.

aligned_with_tagging: yes — `mechanism_strategy.directions=[Location, Causal Intervention]` in `EXPERIMENT_PLAN.md` maps directly to Screen (LoRA-SVD + probing) → Verify (steering α-sweep).
