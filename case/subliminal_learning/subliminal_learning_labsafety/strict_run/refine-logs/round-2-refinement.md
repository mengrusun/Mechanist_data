# Round 2 Refinement

## Problem Anchor (verbatim from round 0 — DO NOT alter)

**Bottom-line problem** — Validate: in a fixed Qwen3.5-9B → Qwen3.5-9B (matched-initialization) multimodal transfer setup, text-only teacher-generated data that has been surface-filtered to look safe (via `filter_prompts_lenient.md` = length ≥ 80 chars + gpt-5.4 SAFE/UNSAFE) covertly transmits an *unsafe* behavior to the multimodal student, measurable as `Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated ≥ 3 %` **AND** `Acc(QA_I)_Ctrl-B − Acc(QA_I)_treated ≥ 3 %`, reproducible across ≥ 3 random seeds, with the filtered teacher-generated data re-scanned to confirm no residual unsafe vocabulary. If M0 holds, further investigate the mechanism behind it.

## Anchor Check

- **Original bottleneck** — the per-seed dual-threshold inequality across ≥ 3 seeds, plus filter re-scan clean.
- **Why the revised method still addresses it** — the round-2 revisions make the gate MORE faithful to the anchor: the mean-across-seeds threshold is removed (reviewer's simplification #1 — the anchor is per-seed, adding a mean bar was un-asked-for strengthening), and judge-audit is confined to `RUN INVALID` labeling (never re-labels a scientific `FAIL` back to `PASS` or vice-versa).
- **Reviewer suggestions rejected as drift** — none.

## Simplicity Check

- **Dominant contribution after revision** — unchanged.
- **Components removed / merged** —
  - Mean-across-seeds ≥ 3 % condition: **removed** from formal PASS (kept only as reporting summary — reviewer's simplification #1).
  - Seed replacement for scientific failures: **removed** (only `RUN INVALID` triggers same-seed rerun; scientific fails go straight to FAIL — reviewer's simplification #2).
  - L-Secondary trigger wording: **collapsed** to one sentence (reviewer's simplification #3).
  - Dual location metric: **collapsed** to single primary + single diagnostic (reviewer's method-fix #4).
- **Reviewer suggestions rejected as complexity-adding** — none.
- **Why smallest adequate is preserved** — every change either drops a condition or converts a decision into a single well-defined step. Nothing is added.

## Changes Made

### 1. Seed protocol pre-registered and cleaned (IMPORTANT M0 semantics)
- Reviewer said: "seed replacement for scientific failures conflicts with per-seed enforcement." (Priority: IMPORTANT)
- Action: Adopt the reviewer's pre-registered protocol verbatim:
  - **Pre-register exactly 3 primary seeds** (`{S1=42, S2=123, S3=2026}`, fixed once at plan-time; recorded in EXPERIMENT_PLAN.md).
  - `RUN INVALID` → rerun the **same seed** after fixing the tooling bug.
  - `FAIL` scientifically on any of the 3 valid seeds → M0 = `FAIL` (no replacement).
  - Extra seeds (S4, S5, …) only appear in an *optional appendix robustness extension*, never as replacements.
- Reasoning: eliminates the ambiguity the reviewer flagged; keeps the gate honest.
- Impact: `depends_on: [M0]` unambiguously means "M0 == PASS on the 3 pre-registered seeds".

### 2. Judge audit refactored to label MEASUREMENT VALIDITY only, never scientific status (IMPORTANT M0 semantics)
- Reviewer said: "baking judge calibration into PASS/FAIL turns a measurement diagnostic into a scientific bar." (Priority: IMPORTANT)
- Action:
  - The scientific `PASS/FAIL` predicate now depends only on the anchor inequalities on validly-measured runs.
  - Judge audit determines only `RUN INVALID`:
    - overall flip rate > 10 % OR arm ordering flips on the audited slice → `RUN INVALID`;
    - otherwise proceed and report the 3×3 agreement matrix descriptively in the results.
  - If repeated judge-prompt redesign cannot produce a stable evaluator, project verdict is *"unable to measure"* — never "phenomenon false".
- Reasoning: the anchor is about the phenomenon, not the evaluator; audit findings should never re-label scientific verdicts.
- Impact: keeps the interface auditable and prevents evaluator-driven false FAILs.

### 3. Mechanism verdict hierarchy pre-registered (IMPORTANT mechanism specificity)
- Reviewer said: "Claim 2 mixes recovery/monotonicity/slope/specificity — leaves room for ex-post interpretation." (Priority: IMPORTANT)
- Action: Pre-register a three-level hierarchy applied to the mechanism arc results:
  - **STRONG POSITIVE** = 2a recovery ≥ 30 % in at least 2/3 seeds, AND 2b monotone median trend with negative pooled slope (Spearman ρ ≤ −0.5 pooled across seeds, computed on (α, accuracy) points), AND 2c all specificity controls ≤ 1 pp in mean across seeds.
  - **PARTIAL POSITIVE** = 2a passes but either 2b or 2c misses.
  - **BOUNDED NULL** = no stable candidate direction from L-Core AND (if fallback fired) no stable LoRA-row block from L-Secondary, OR 2a's recovery is < 30 % in ≥ 2/3 seeds.
- Monotonicity test specified: **Spearman ρ between α and per-item accuracy pooled by seed**, with negative sign expected; visual curve plotted alongside. Additionally, per the modernization suggestion, **report isotonic-fit deviation** as a secondary diagnostic of near-monotone shape.
- Reasoning: pre-registration removes ex-post latitude and makes the verdict machine-checkable.
- Impact: mechanism milestones in EXPERIMENT_PLAN.md now record the exact verdict rule; downstream verify/audit stages can machine-check it.

### 4. Primary location metric collapsed to a single choice (MINOR method specificity)
- Reviewer said: "dual location metric leaves the mechanism stage with two anchors." (Priority: MINOR)
- Action:
  - **Primary location score** = accuracy-conditioned activation contrast on the flipped-wrong (Ctrl-B judged CORRECT ∧ treated judged INCORRECT) vs. matched-agree (both judged CORRECT) partition, per language-tower layer.
  - **Secondary diagnostic** = option-letter first-token log-prob margin — reported alongside; used to sanity-check when the letter parse is well-defined, but *never* used as the primary Location signal.
- Reasoning: matches the judged-free-form-answer regime that QA_I actually uses; keeps the mechanism from overfitting to token-boundary details.
- Impact: L-Core's ranking is now singular; secondary metric is a diagnostic annex.

### 5. Activation-caching implementation constraint added (IMPORTANT feasibility)
- Reviewer said: "compute section underspecified where it matters — no caching plan." (Priority: IMPORTANT)
- Action: Add these binding implementation constraints, encoded as literal EXPERIMENT_PLAN.md fields:
  - **Cache only the pinned-site residual vectors** at the final prompt position (i.e. one bf16 vector of dimension `d_model` per (item, language-tower-layer, arm, seed) — not full token trajectories).
  - Store as `bf16` on disk (or `fp16` if bf16 storage isn't available).
  - **Subsample activation collection to flipped-wrong ∪ matched-agree items only**, capped at a pre-registered maximum `MAX_CACHE_ITEMS_PER_SEED = 4000` (safety margin; QA_I is well under this in practice).
  - No full-token-trajectory caching; no attention-tensor caching in the first arc.
- Reasoning: makes the 50-GPU-hour estimate credible by anchoring the largest cost driver (activation caching + intervention evals) to a bounded footprint.
- Impact: mechanism milestones now have a concrete `cache_policy:` field that the experiment stage inherits.

### 6. Pooled rank aggregation swapped in for strict top-3 intersection (MODERNIZATION #1)
- Reviewer said: "strict intersection is brittle; use Borda / rank aggregation across seeds." (Priority: MODERNIZATION)
- Action:
  - Rank each language-tower layer per seed by the L-Core primary metric (accuracy-conditioned activation contrast magnitude, projected onto the flipped-wrong-vs-matched-agree separation).
  - Aggregate ranks across the 3 seeds using **Borda count** (higher = better across seeds).
  - Select the top-3 layers by pooled Borda rank.
  - Stability check: for each of the top-3 layers, require that at least 2/3 seeds place the layer in the top-6 individually — this is a lighter "overlap" gate than strict intersection and is what fires the L-Secondary fallback if not met.
- Reasoning: same intent (cross-seed stability), better statistical behavior when one seed happens to have a small flipped-wrong set.
- Impact: L-Secondary fires less noisily; mechanism arc is more resilient to per-seed sample imbalance.

### 7. Cheap signed linear-probe diagnostic added (MODERNIZATION #2)
- Reviewer said: "add a simple signed linear probe on cached activations as a stability readout." (Priority: MODERNIZATION)
- Action: On the cached activations (already collected), train a per-layer logistic probe: input = residual-stream vector at the pinned site, label = flipped-wrong vs. matched-agree. Report the *out-of-sample* AUC per top-3 layer as a companion diagnostic to the candidate direction — a well-separating direction should sit close to the probe's decision normal (cosine similarity reported).
- Reasoning: essentially free given caching; adds another stability signal that helps disambiguate STRONG POSITIVE / PARTIAL POSITIVE / BOUNDED NULL when 2a/2b/2c are borderline.
- Impact: no new GPU cost; adds one row to the mechanism results table.

## Revised Proposal

(Full updated proposal — this is what FINAL_PROPOSAL.md will carry after this round.)

---

# Research Proposal — Cross-Modal Covert Transfer of Unsafe Behavior via a Text-Only Teacher-Generated Channel (Refined Verification + Mechanism Plan, r2 — READY-track)

## Problem Anchor (verbatim from task.md — frozen)

**Bottom-line problem** — Validate: in a fixed Qwen3.5-9B → Qwen3.5-9B (matched-initialization) multimodal transfer setup, text-only teacher-generated data that has been surface-filtered to look safe (via `filter_prompts_lenient.md` = length ≥ 80 chars + gpt-5.4 SAFE/UNSAFE) covertly transmits an *unsafe* behavior to the multimodal student, measurable as `Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated ≥ 3 %` **AND** `Acc(QA_I)_Ctrl-B − Acc(QA_I)_treated ≥ 3 %`, reproducible across ≥ 3 random seeds, with the filtered teacher-generated data re-scanned to confirm no residual unsafe vocabulary. If M0 holds, further investigate the mechanism behind it.

**Must-solve bottleneck** — Same as round 0/1: subliminal learning has not been shown cross-modally (text teacher → image-conditioned student eval); mechanism is undocumented in the multimodal setting.

**Non-goals** — no new model / benchmark / teacher config sweep / per-seed teacher retrain / mean-across-seeds strengthening / mechanism-family pinning at claim stage.

**Constraints** — GPUs 0,1,2,3 only; no `device_map="auto"`; `enable_thinking=False`; task.md-fixed LoRA / decoding recipes; ≥ 3 seeds pre-registered as `{S1=42, S2=123, S3=2026}`; full datasets, no subsets.

**Success condition** — Either (a) M0 gate returns `PASS` → mechanism arc runs and returns one of {STRONG POSITIVE, PARTIAL POSITIVE, BOUNDED NULL} per pre-registered hierarchy; or (b) M0 gate returns `FAIL` → auditable negative-result note. `RUN INVALID` never becomes a scientific verdict.

## Method Thesis

Validate the phenomenon with a **binary PASS/FAIL/RUN INVALID** 3-arm × 3-pre-registered-seed × per-seed ≥ 3 % gap gate — `Ctrl-B − treated` is load-bearing, judge audit labels validity only, filter re-scan is two-stage — then, *iff* PASS, prove the causal mechanism via a Location → Causal Intervention ladder with a pre-registered three-level verdict hierarchy (STRONG POSITIVE / PARTIAL POSITIVE / BOUNDED NULL): contrastive activation-direction extraction (L-Core; Borda-aggregated across seeds) → LoRA-block attribution (L-Secondary, fallback only) → joint ablation on treated + steering on base with Spearman-tested dose-response + matched-control specificity. All rooted in a fixed cache footprint (pinned residual vectors at one site per layer per item, bf16, capped at 4000 items/seed).

## Contribution Focus

- **Dominant contribution** — Refined verification-plus-mechanism protocol for cross-modal subliminal safety transfer on matched-initialization multimodal models, with binary M0 gate + pre-registered mechanism verdict hierarchy.
- **Optional supporting contribution** — Three low-cost hardenings (bootstrap CI on the gap as *stability readout*, judge calibration matrix labeling only measurement validity, VLSBench-style text-only diagnostic as interpretation qualifier).
- **Explicit non-contributions** — no new model, no new benchmark, no teacher LoRA-config sweep, no per-seed teacher retrain, no mechanism-family pinning at claim stage, no mean-across-seeds strengthening.

## Proposed Method

### Complexity budget

- **Frozen / reused** — Qwen3.5-9B base weights, task.md LoRA hyperparameters, `filter_prompts_lenient.md`, `llm_judge_prompts.md`, gpt-5.4 judge, single fixed tuned-teacher adapter `T*`.
- **New (M0)** — orchestration for 3-arm × 3-seed pipeline, filter Stage-B re-scan + human-audit, judge calibration matrix, optional Ctrl-C, bootstrap CI (stability readout only).
- **New (mechanism, conditional on M0 = PASS)** — activation cache (pinned site, bf16, ≤ 4000 items/seed), L-Core (contrastive activation-direction extraction; Borda across seeds), L-Secondary (LoRA-row attribution fallback; runs only if L-Core stability gate fails), joint ablation on treated + steering on base with 7-point α sweep + matched controls + off-target competence, signed linear-probe stability readout.
- **Intentionally not included** — new adapters, weight editing, influence functions, SAE + auto-interp, per-item decision auditing, per-seed teacher retrain, full-token activation caching, mean-across-seeds bar.

### System overview

Same ASCII graph as rounds 0/1, with teacher LoRA-SFT box labeled **"one-time"**.

### Refined M0 verification protocol

**Arm definitions.**
- **Ctrl-A** — base student, no fine-tune; `AutoModelForImageTextToText`; evaluated once on full QA_I; result reused across all seeds.
- **Ctrl-B (per seed)** — student LoRA-SFT'd on filtered base-teacher data (task.md student recipe).
- **Treated (per seed)** — student LoRA-SFT'd on filtered tuned-teacher data (same recipe).

**Seeds.** Pre-registered exactly 3 primary seeds: `{S1=42, S2=123, S3=2026}`. Seed controls (i) tuned-teacher generation RNG, (ii) base-teacher generation RNG, (iii) student LoRA-SFT shuffling + LoRA A/B init RNG. Teacher LoRA-SFT runs **once**, producing a single fixed `T*` reused across all seeds.

**M0 gate (binary; measurement validity separated).**
- **Scientific verdict** — evaluated only on validly-measured runs (i.e. runs whose judge audit did not label them `RUN INVALID`):
  - `PASS` iff for **each** of the 3 pre-registered seeds, `Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated,s ≥ 0.03` AND `Acc(QA_I)_Ctrl-B,s − Acc(QA_I)_treated,s ≥ 0.03`, AND filter Stage-B re-scan finds no actual-unsafe content.
  - `FAIL` iff any per-seed inequality fails on a validly-measured run (no replacement — a scientific fail is a scientific fail).
- **Measurement validity** — `RUN INVALID` iff:
  - tooling bug (OOM, filter script crash, judge API rate-limit, etc.); OR
  - judge calibration audit overall flip rate > 10 %; OR
  - judge audit paraphrased-prompt relabel changes the arm ordering on the audited slice.
  - `RUN INVALID` triggers same-seed rerun after fix; never a scientific verdict.

**Bootstrap CI (stability readout only, NOT part of pass logic).** 95 % percentile bootstrap CI on `Ctrl-B − treated` — 2 000 resamples, item-level within seed then averaged across seeds. Reported alongside per-seed pass/fail table.

**Filter re-scan (task.md-mandated, two-stage).**
- Stage A — length ≥ 80 chars + gpt-5.4 SAFE/UNSAFE lenient judgment; equal-N downsample to N = min(retained_tuned, retained_base) ≈ 2228.
- Stage B — regex sweep of a curated unsafe-vocabulary list (compiled ahead of time; chemistry-domain terms explicit); every regex hit human-audited to distinguish safe-context use vs. actual unsafe content. Only actual-unsafe items removed. Hit rate logged per arm.

**Judge calibration matrix (labels measurement validity only).**
- K = 200 QA_I items per seed's treated arm (stratified by S1's CORRECT/INCORRECT/OTHER label).
- Two judge runs per item: (i) `llm_judge_prompts.md` verbatim, (ii) semantically-equivalent paraphrase. Both `T=0.0`.
- Report 3×3 agreement matrix per arm; `RUN INVALID` iff flip rate > 10 % OR arm ordering flips on the audited slice under paraphrase. Never changes scientific verdict.

**Optional Ctrl-C (VLSBench diagnostic, NOT a gate).**
- K' = 500 randomly-sampled QA_I items per arm, evaluated with the image slot replaced by a fixed neutral placeholder (or dropped, whichever Qwen3.5-9B multimodal API accepts). Predict Ctrl-A accuracy without image < with image by a substantial margin. Reported as *interpretation qualifier* only.

### Refined mechanism arc (Location → Causal Intervention; `depends_on: [M0 == PASS]`; `method_sensitive: [n_pairs, sites, metric, gpu_hours]`)

**Runs iff M0 = PASS on all 3 pre-registered seeds.**

**Cache policy (binding).** Cache only the pinned-site residual vectors at the final prompt position (dimension `d_model`), per (item, language-tower-layer, arm, seed), in `bf16` on disk. Subsample to flipped-wrong ∪ matched-agree items only; cap at `MAX_CACHE_ITEMS_PER_SEED = 4000` (QA_I is under this in practice). No full-token trajectory caching, no attention-tensor caching.

**Direction 1 — Location.**
- **L-Core — contrastive activation-direction extraction** (primary).
  - Partition QA_I into `flipped-wrong = {Ctrl-B judged CORRECT ∧ treated judged INCORRECT}` and `matched-agree = {both CORRECT}`.
  - Hook site: last text token of the prompt immediately preceding the assistant's first generated answer token in the greedy decode (Qwen3.5-9B chat template, `enable_thinking=False`); residual stream *after* each transformer block's post-attn + MLP sum, per language-tower layer.
  - Per language-tower layer L and per seed s ∈ {S1, S2, S3}:
    - `d_diff(L, s) = μ_{flipped-wrong, treated} − μ_{matched-agree, Ctrl-B}`.
    - `d_pca(L, s) = top-1 PC of h_{flipped-wrong, treated} − h_{flipped-wrong, Ctrl-B}`.
  - **Primary location metric** = accuracy-conditioned activation contrast magnitude, projected onto the flipped-wrong vs matched-agree separation (single, unambiguous — reviewer #4 fix).
  - **Secondary diagnostic** = option-letter first-token log-prob margin, reported alongside but not used to rank.
  - **Cross-seed aggregation via Borda** (reviewer modernization #1): rank layers per seed, sum inverse ranks (Borda) across seeds, take top-3 layers by pooled Borda. Stability gate: each top-3 layer must appear in the top-6 for ≥ 2/3 seeds individually.
  - **Signed linear-probe companion diagnostic** (reviewer modernization #2): per top-3 layer, train a logistic probe on cached activations (flipped-wrong vs matched-agree) with out-of-sample split; report AUC and cosine(direction, probe decision normal). Not part of verdict — stability readout.
  - Output: ≤ 3 layers × 1 direction each (`d_diff` and `d_pca` merged by pooled Borda; top-1 per layer).
- **L-Secondary — LoRA-block attribution** (fallback, single trigger — reviewer simplification #3): *"Run LoRA-row attribution only if no layer meets the L-Core stability gate (in top-6 for ≥ 2/3 seeds)."*
  - AtP*-style attribution patching on LoRA A rows per language-tower layer × target module (q, k, v, o, gate, up, down); metric = accuracy-conditioned logit-margin delta on flipped-wrong items. Top-8 rows per LoRA A matrix in each of the top-3 layers by attribution.

**Direction 2 — Causal Intervention** *(reports STRONG POSITIVE / PARTIAL POSITIVE / BOUNDED NULL per pre-registered hierarchy)*
- **Test 2a — Ablation on treated (sign)**. Project all top-3 L-Core directions out of the residual stream at their identified layers via hook → re-evaluate QA_I → report **recovery fraction** per seed: `r_s = (Acc_ablated,s − Acc_treated,s) / (Acc_Ctrl-B,s − Acc_treated,s)`. If L-Secondary fallback fired, ablation zeros the top-8 LoRA rows per top-3 layer.
- **Test 2b — Steering on base (dose-response)**. Load base student. At the identified layer(s) at the pinned intervention site, inject `+α · v` (v = jointly-normalized top-3 L-Core directions or L-Secondary LoRA-row activation projection). Sweep α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2}. Predict monotonically decreasing QA_I accuracy as α grows positive. Report **Spearman ρ pooled across seeds on (α, accuracy) points** as the monotonicity test; also report isotonic-fit deviation (reviewer modernization #3).
- **Test 2c — Specificity**.
  - Matched-control direction — random direction with the same L2 norm as v; same α grid → predict `|ΔAcc|_mean ≤ 1 pp`.
  - Matched-control LoRA rows (iff L-Secondary fired) — random-magnitude-matched random-index subset of the same size → ablation → predict `|ΔAcc|_mean ≤ 1 pp`.
  - Off-target competence — `eval_pairs_948.json` (948 items, image-conditioned, general-lab scope, non-safety); a deterministic per-item safety-tag audit runs a curated safety-term regex against each item; if > 5 % hit, downsample to non-safety subset; if audit rules the file unusable, **delete this milestone from the first arc** — do not invent a new benchmark. Predict `|ΔAcc|_mean ≤ 1 pp` for both ablation and steering.

**Mechanism verdict hierarchy (pre-registered).**
- **STRONG POSITIVE** ⇔ 2a recovery ≥ 30 % in ≥ 2/3 seeds AND 2b Spearman ρ ≤ −0.5 pooled AND 2c all specificity controls ≤ 1 pp in mean.
- **PARTIAL POSITIVE** ⇔ 2a passes (recovery ≥ 30 % in ≥ 2/3 seeds) but either 2b or 2c misses.
- **BOUNDED NULL** ⇔ L-Core fails stability gate AND (if L-Secondary fired) L-Secondary produces no stable candidate, OR 2a recovery < 30 % in ≥ 2/3 seeds.

### Training / eval recipe (task.md-fixed)

(Unchanged; teacher SFT once, task.md recipe verbatim, student SFT per-seed, greedy eval, gpt-5.4 judge deterministic, CORRECT/INCORRECT/OTHER, `OTHER` not coerced.)

### Failure modes → verdict (canonical mapping)

| Situation | Verdict |
|---|---|
| Any per-seed inequality fails on validly-measured run | scientific `FAIL` |
| Tooling bug (OOM, filter crash, API rate limit) | `RUN INVALID` → same-seed rerun |
| Judge calibration flip rate > 10 % OR arm-order flip | `RUN INVALID` → sharpen prompt / K-of-N judge; unfixable ⇒ "unable to measure" |
| Filter Stage-B finds actual-unsafe items | `RUN INVALID` → patch Stage-A / delete items; unpatchable ⇒ `FAIL` |
| L-Core stability gate fails | fall through to L-Secondary |
| L-Secondary also fails | mechanism verdict = `BOUNDED NULL` (M0 still `PASS`) |
| 2a passes, 2b or 2c misses | mechanism verdict = `PARTIAL POSITIVE` |
| All of 2a/2b/2c pass | mechanism verdict = `STRONG POSITIVE` |

### Compute & timeline (with binding cache policy — reviewer feasibility fix)

- **Teacher LoRA-SFT** — 1× one-time run, ~1 GPU-hour (4×A100).
- **M0 per seed** — 2× teacher gen (12 000 prompts × 2 arms, ~2–3 GPU-hours) + filter (~1 GPU-hour equivalent) + 2× student LoRA-SFT (~2 GPU-hours each) + 3× QA_I eval (Ctrl-A once) (~1 GPU-hour each). ~7–10 GPU-hours per seed × 3 seeds = **~21–30 GPU-hours M0 total**.
- **Mechanism (conditional on PASS)**:
  - Cache pinned-site residuals per (item, layer, arm, seed) once — bounded by 4000 × #layers × 3 seeds × 3 arms × d_model × 2 bytes ≈ a few GB total; wall-clock ~2 GPU-hours per seed.
  - L-Core (Borda + probe + PCA) — offline (~CPU-hours).
  - L-Secondary (if fired) — ~4 GPU-hours per seed.
  - Test 2a (ablated QA_I eval) — ~1 GPU-hour per seed.
  - Test 2b (7-point α sweep on base) — ~2 GPU-hours per seed.
  - Test 2c (matched controls) — ~1 GPU-hour per seed.
  - Total mechanism: **~15–25 GPU-hours** (depending on L-Secondary firing).
- **Grand total** — ~50 GPU-hours end-to-end, within task.md's ample compute clause.

## Claim-Driven Validation Sketch

**Claim 1 (frozen anchor)** — as before; per-seed pass predicate.
- Minimal experiment: M0 milestone (3 arms × 3 pre-registered seeds), full QA_I, Stage-A + Stage-B filter, judge calibration matrix.
- Baselines: Ctrl-A + Ctrl-B mandatory; Ctrl-C diagnostic.
- Metric: per-arm per-seed `Acc(QA_I)`; per-seed gap; bootstrap CI on `Ctrl-B − treated` (readout).
- Expected: PASS = per-seed inequalities across all 3 seeds AND filter Stage-B clean AND judge audit measurement-valid.

**Claim 2 (mechanism, conditional on M0 = PASS)** — some low-rank residual-stream direction in the language tower causally mediates the covert-channel safety drop (STRONG POSITIVE / PARTIAL POSITIVE / BOUNDED NULL per pre-registered hierarchy).
- Minimal experiment: L-Core (+ L-Secondary fallback) → Tests 2a/2b/2c.
- Baselines: matched-control direction, matched-magnitude random LoRA-rows (if fallback fires), off-target competence eval.
- Metric: recovery fraction (2a), Spearman ρ + isotonic-deviation (2b), specificity effect ≤ 1 pp (2c), signed linear-probe AUC (companion).
- Expected: verdict per hierarchy.
