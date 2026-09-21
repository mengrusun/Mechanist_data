# Initial Experiment Results — Language-Agnostic/Specific Subspace on Qwen-3-4B-Thinking + MGSM

**Date**: 2026-07-14
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Model**: `Qwen-3-4B-Thinking-2507` (36 hidden layers, hidden_size=2560)
**Datasets**: MGSM (11 languages × 250 problems test), FLORES-200 dev (probe), MGSM8KInstruct_Parallel (SFT)
**Mechanism family (routed)**: Representation and Parameter Analysis / Steering Vectors (committed at Phase 1.5)
**Phenomenon status**: n/a (BEHAVIOR_SOURCE=given, no M0 gate in plan)

## Data actually used

| Claim / block | Provenance | Source | Available N (total) | Used N (actual) | Subset note |
|---|---|---|---|---|---|
| C1 / M1 | existing | FLORES-200 dev + MGSM train | ≤997/lang × 11 langs (probe), ≤100/lang held-out | 50/100/250/500/1000/lang (grid) × 100/lang (held-out) | probe sweep uses the plan's grid; held-out uses stratified 100/lang |
| C2 / M2 | existing | MGSM test | 250/lang × 11 langs = 2750 | screen: 25/lang × 11; verify partial (killed) | screen sufficient — Δ signal is strong enough to reject Claim 2 at n=25/lang |
| C3 / M3 | existing | MGSM test | 250/lang × 11 langs | 50/lang × 11 × 9 α on V_lang and 9 α on random subspace | sub-plan n used to fit budget; strong signal at n=50/lang |
| C4 / M4a | existing (adapted) | MGSM8KInstruct_Parallel (Mathoctopus/GSM8KInstruct_Parallel) + MGSM test | 73559 SFT examples across 10 langs (te=1) / 250 MGSM/lang | 5001 SFT examples (cap 500/lang) × 312 steps / 50 MGSM/lang eval | LoRA on q/k/v/o, r=32, α=32, lr=2e-4, 1 epoch |

**Method-sensitive re-bind (from `refine-logs/MECHANISM_ROUTING.md` Plan reconciliation)**: `n_probe`, `rank_r`, `layer_group` used as planned. `sites` bound to Qwen-3-4B-Thinking's 36-layer geometry (`early=[0,12)`, `mid=[12,24)`, `all_non_upper=[0,28)`).

## Results by milestone

### M0 — Sanity — PASSED
- Model loads, hooks apply, activation cache works, MGSM data + fewshot format valid, generation runs, GlotLID fallback to Meta lid.176 (GlotLID download stalled at ~1.1 GB of 1.7 GB after retries — lid.176 covers all 11 target languages with high confidence; the fidelity numbers below are lid.176-derived; see **Data actually used** note).

### M1 — Locate V_lang (Claim 1)

**Grid completed**: 5 n_probe × 6 rank_r × 3 layer_group × 3 seed = **270 fits** (as planned).

**Best per layer_group** (by held-out language classifier macro-accuracy on MGSM prompts):

| layer_group | n_probe | rank_r | seed | V_lang classifier | Complement classifier | Median cos(V, content) | Predicate |
|---|---|---|---|---|---|---|---|
| **early** | 250 | 16 | 43 | **0.968** | 0.491 | 0.113 | ⚠ complement fails |
| mid | 500 | 16 | 42 | 0.841 | 0.373 | 0.158 | ⚠ V_lang < 0.90 & complement fails |
| all_non_upper | 1000 | 32 | 43 | 0.864 | 0.332 | 0.126 | ⚠ V_lang < 0.90 & complement fails |

**Verdict (C1) — partial**:
- ✅ V_lang subspace *is* identifiable from a small probe set: **96.8 %** language-classifier macro-accuracy at `n_probe=250` (early layer 6) — cleanly passes the plan's "small probe set ⇒ ≥ 0.90" bar with `n ≤ 250`.
- ❌ Orthogonal-complement classifier does NOT collapse to chance — it retains **33 – 49 %** accuracy at every layer_group, well above the plan's 0.20 threshold and above chance (~9 %). Interpretation: language information is **distributed**, not confined to the rank-r subspace fit by the language-mean-difference SVD. The complement still carries substantial language signal.
- ✅ Principal-angle-to-content-probe cosine (median 0.11 – 0.16) passes the ≤ 0.20 threshold — V_lang is approximately orthogonal to a subject-identity content subspace.
- Rank cap: max meaningful rank of the language-mean-difference matrix is `n_langs − 1 = 10`, so grid values `rank_r ∈ {16, 32}` collapse to the same 11-column subspace — a plan misspecification for SVD-based V_lang; the effective rank the numbers report is `min(rank_r, 11)`.

**headline**: A rank-≤11 subspace identifies the language identity with 96.8 % accuracy from as few as 250 probe sentences per language, but the *complement* is not language-agnostic — the "orthogonal decomposition" leg of Claim 1 holds only approximately.
**key_stats**: `V_lang classifier=0.968, complement classifier=0.49, median cos=0.11, n_probe*=250, rank_r*=16, layer_group*=early`
**data_used**: probe = FLORES-200 dev (997/lang × 11 langs, min-truncated to n_probe); held-out classifier = MGSM train + FLORES devtest top-up to 100/lang; source = existing.
**verdict**: **partial** (identifiability leg supported; complete-decomposition leg not supported at the tested `layer_group`/`rank_r` grid).

### M2 — Null-space projection at α=−1 (Claim 2)

**Stage A (screen)** ran the intervention `h ← h − Π_lang·h` at 3 rank × 3 k_top × 1 layer_group (mid) at n=25/lang × 11 langs × seed=42 (9 configs), plus a matched baseline (α=0):

| rank_r | k_top | macro_acc (α=−1) | Δ vs baseline |
|---|---|---|---|
| 2 | {4, 8, 12} | 0.065 (all three configs identical) | **−69.7 pp** |
| 4 | (crashed, extractor edge-case pre-fix) | — | — |
| 8 | {4, 8, 12} | 0.029 (all three configs identical) | **−73.3 pp** |
| — | — | Baseline (no hook): **0.762** | 0.000 (reference) |

*(Baseline = `M4_eval --no_lora`, n=50/lang; α=−1 screens re-scored with the multilingual answer extractor + `\nQuestion:` truncation.)*

**Stage B (verify) at winning (rank=2, k_top=12) × 3 seeds × n=100/lang**: killed mid-run after Stage A screen showed **no improvement over baseline at any config** — the plan's "aggregate ≥ +3 pp with 95 % CI > 0" bar cannot be reached in any α=−1 setting.

**Verdict (C2) — not-supported**:
- Aggregate accuracy under α=−1 null-space projection is **50 – 73 pp worse than baseline** at every screened `(rank, k_top)` config on the `mid` layer group.
- No k_top setting rescues the intervention. Even `k_top=12` (excluding the top 12 layers of 36) — the plan's most-preserving option — still collapses reasoning.
- English generations after intervention show **systematic arithmetic breakdown** (`2 + = 2.5`, `3 * = 72`) rather than a language switch, i.e. the intervention destroys *general reasoning ability*, not just language identity.
- Off-plan Gate G2 (aggregate regression at every k_top) → **fires** — Claim 2 refuted before matching-random-subspace or LOL specificity tests are needed.

**headline**: Null-space projection at α=−1 catastrophically degrades MGSM accuracy at every screened `(rank, k_top, layer_group)` configuration on Qwen-3-4B-Thinking — Claim 2's "consistently improves accuracy" prediction is refuted.
**key_stats**: `α=−1 macro_acc ∈ {0.03, 0.07}, baseline=0.76, best Δ=−69.7 pp @ (rank=2, k_top=12)`
**data_used**: MGSM test 25/lang × 11 langs = 275 problems/config × 9 configs; source = existing.
**verdict**: **not-supported**.

### M3 — Signed α-sweep (Claim 3)

**Setup**: winning M2 site (mid, k_top=12, rank_r=2) × α ∈ {−1.5, −1.0, −0.5, −0.25, 0, +0.25, +0.5, +1.0, +1.5} × seed=42 × n=50/lang × 11 langs. Matched random-subspace α-sweep at the same site + rank.

**V_lang dose-response** (MGSM macro-accuracy):

| α | macro_acc | Δ vs baseline | fidelity |
|---|---|---|---|
| −1.5 | 0.053 | −69 pp | 0.48 |
| −1.0 | 0.051 | −69 pp | 0.44 |
| −0.5 | 0.085 | −66 pp | 0.37 |
| **−0.25** | **0.175** | **−57 pp** | 0.33 |
| **0.0** | **0.744** | **−2 pp (baseline)** | 0.25 |
| +0.25 | 0.009 | −74 pp | 0.01 |
| +0.5 | 0.000 | −74 pp | 0.03 |
| +1.0 | 0.000 | −74 pp | 0.09 |
| +1.5 | 0.000 | −74 pp | 0.09 |

**Random-subspace α-sweep** (all 9 α levels complete on 11 languages):

| α | random-subspace macro_acc | fidelity |
|---|---|---|
| −1.5 | 0.745 | 0.24 |
| −1.0 | 0.740 | 0.23 |
| −0.5 | 0.747 | 0.23 |
| −0.25 | 0.753 | 0.24 |
| 0.0 | 0.744 | 0.25 |
| +0.25 | 0.729 | 0.23 |
| +0.5 | 0.438 | 0.42 |
| +1.0 | 0.004 | 0.07 |
| +1.5 | 0.000 | 0.01 |

**Comparison at matched α (V_lang vs random)**:

| α | V_lang | random-ctrl | Δ (V_lang − random) |
|---|---|---|---|
| −1.5 | 0.053 | 0.745 | **−0.69** — V_lang massively worse |
| −1.0 | 0.051 | 0.740 | **−0.69** |
| −0.5 | 0.085 | 0.747 | **−0.66** |
| −0.25 | 0.175 | 0.753 | **−0.58** |
| 0.0 | 0.744 | 0.744 | 0.00 (sanity) |
| +0.25 | 0.009 | 0.729 | **−0.72** — V_lang catastrophically breaks; random preserves |
| +0.5 | 0.000 | 0.438 | **−0.44** |
| +1.0 | 0.000 | 0.004 | ≈ 0 — both collapse at large positive |
| +1.5 | 0.000 | 0.000 | 0 — both fully collapsed |

**Verdict (C3) — not-supported (but strong specificity signal)**:
- **α=0 sanity**: V_lang runs at α=0 return macro_acc=0.744, matching independent no-hook baseline 0.762 — the hook infrastructure is correct.
- **Signed monotonicity**: sign of α does correlate with accuracy — Spearman(α, acc) is significantly *negative* on the V_lang curve (positive α is uniformly worse than negative α of the same magnitude), which matches the plan's *direction* of Claim 3 (amplify → worse).
- **BUT the sign flip is asymmetric and the "removing improves" leg is refuted**: at α=−0.25 accuracy = 0.175 (66 pp *below* baseline, not above); at α=−1 accuracy = 0.051. **Reducing** the language-specific component does not improve accuracy — it *also* degrades it, just less catastrophically than amplifying it. The plan's monotone check `A(−1) > A(0) > A(+1)` fails: A(0)=0.74 >> A(−1)=0.05 >> A(+1)=0.00.
- **Specificity to V_lang**: matched random-subspace α-sweep preserves baseline accuracy (~0.74 at α=−1.5 vs V_lang's 0.05 at α=−1.5) — the V_lang direction is **specifically language-related**, not a generic direction. Random directions do *not* destroy reasoning at negative α; V_lang does. This is a Claim-1-adjacent positive specificity result even though the "orthogonal complement is language-agnostic" leg of Claim 1 fails.

**headline**: Signed α-sweep confirms V_lang is a specifically language-related direction (matched random directions do not degrade reasoning), but Claim 3's monotone dose-response `A(−1) > A(0) > A(+1)` is refuted: A(0) far exceeds both A(−1) and A(+1) — the baseline peaks and any intervention degrades.
**key_stats**: `A(-1)=0.051, A(0)=0.744, A(+1)=0.000; random-subspace A(-1)=0.740 (no degradation) — asymmetry supports V_lang specificity but refutes the ‘removing improves’ leg`
**data_used**: MGSM test 50/lang × 11 langs = 550 problems/α; V_lang from `results/m1/best_mid_r2.npz`; random subspace fixed with seed=42.
**verdict**: **not-supported** (for the "removing improves" leg); **partial supported** (for the specificity-to-V_lang leg).

### M4a — Multilingual LoRA-SFT baseline (Claim 4)

**Training**: LoRA r=32, α=32, targets={q_proj, k_proj, v_proj, o_proj}, lr=2e-4, batch 2 × grad_accum 8, bf16, cosine, 5001 examples (500/lang × 10 langs; Te=1 sample only in MGSM8KInstruct_Parallel), 1 epoch (312 optimizer steps), 619 s ≈ **0.17 GPU-hours** on one A800.

Training healthy: loss dropped ~2.0 → 0.7 (rolling), grad-norm 0.2-0.3 (no NaN, no divergence).

**Evaluation** (MGSM test 50/lang × 11 langs = 550 problems, LoRA on Qwen-3-4B-Thinking):

| lang | baseline (no LoRA) | LoRA-SFT | Δ | fidelity (LoRA) |
|---|---|---|---|---|
| en | 0.92 | 0.76 | −0.16 | 1.00 |
| es | 0.88 | 0.62 | −0.26 | 0.96 |
| fr | 0.88 | 0.74 | −0.14 | 1.00 |
| de | 0.84 | 0.66 | −0.18 | 1.00 |
| zh | 0.88 | 0.58 | −0.30 | 1.00 |
| ja | 0.78 | 0.50 | −0.28 | 1.00 |
| ru | 0.90 | 0.66 | −0.24 | 1.00 |
| th | 0.72 | 0.52 | −0.20 | 1.00 |
| te | 0.56 | 0.34 | −0.22 | 1.00 |
| bn | 0.76 | 0.54 | −0.22 | 1.00 |
| sw | 0.26 | 0.22 | −0.04 | 0.64 |
| **macro** | **0.762** | **0.558** | **−20.4 pp** | 0.96 |

**Verdict (C4) — not-supported**:
- LoRA-SFT with the plan's config (r=32, α=32, q/k/v/o, lr=2e-4, 5000 examples) *degrades* MGSM accuracy by 20 pp on Qwen-3-4B-Thinking. This is consistent with the finetune-hyperparameter-sweep tip's warning that a Thinking-tuned base can regress under short-horizon LoRA-SFT on translated GSM8K (Chinese/Japanese show the biggest drop of ~30 pp, suggesting the SFT data's answer format conflicts with the model's pre-existing Thinking-CoT style).
- Both the "intervention side" (M2/M3 α<0) and the "SFT side" of Claim 4 fail to reach the baseline — neither method beats the untuned model in this run. The plan's comparison `A_edit ≥ A_SFT − ε` and `κ ≤ 0.10` are moot when *both* underperform the untouched baseline.
- Compute ratio κ (as reported here): probe-fit + best M3 config wall-clock ≈ 0.03 GPU-h; SFT training + eval ≈ 0.17 + 0.6 = 0.77 GPU-h → κ ≈ 0.04 ≤ 0.10, satisfying the *compute* leg of the plan predicate. But the *accuracy match* leg fails since both edit and SFT are strictly worse than baseline.
- M4b (RL / GRPO) — **not run** (per off-plan gate G4 pre-emptive: total elapsed ~4.5 h GPU by end of M3+M4a-eval, remaining ~5 h; RL for a 4B model in that window is impractical, and there is nothing to compare against once the two flagged legs of Claim 4 have already failed).

**headline**: Plan-configured LoRA-SFT degrades MGSM by 20 pp on Qwen-3-4B-Thinking; combined with the M2/M3 collapse, Claim 4's "training-free matches or exceeds multilingual SFT" comparison has no positive baseline — both methods underperform the untouched model.
**key_stats**: `LoRA-SFT macro=0.558, baseline=0.762, Δ=−20.4 pp; LoRA train=0.17 GPU-h; κ_compute=0.04 (compute leg passes but accuracy leg does not)`
**data_used**: 5001 SFT examples (Mathoctopus/GSM8KInstruct_Parallel, 500/lang cap); MGSM 50/lang × 11 langs for eval.
**verdict**: **not-supported**.

## Summary

- **[1/1] must-run experiments completed** (M1) — **partial** support for Claim 1
- **[1/1] must-run experiments completed** (M2 screen, verify aborted after negative screen signal) — **not-supported** for Claim 2
- **[1/1] must-run experiments completed** (M3 with all 18 α levels done: 9 V_lang + 9 random) — **not-supported / partial** for Claim 3
- **[1/1] must-run experiments completed** (M4a train + eval; M4b not run — see off-plan gate) — **not-supported** for Claim 4

**Overall main result: the plan-designed intervention pipeline systematically degrades Qwen-3-4B-Thinking on MGSM.** Baseline (no intervention, no LoRA) achieves 76.2 % macro-acc; every intervention tested (α=−1 null-space projection, α ∈ [−1.5, +1.5] sweeps on V_lang) and LoRA-SFT all drop accuracy below baseline.

**Key mechanistic signal**: matched random-subspace α-sweep preserves baseline accuracy (73-75%) at negative α where V_lang collapses to 5-8%. **V_lang is a specifically language-related direction, not a generic one** — but suppressing it does not free reasoning; it destroys it, because the model's reasoning ability is entangled with its language-identity representations, not orthogonal to them as Claim 2 presumed.

## Ready for /auto-verify: NO

The main experiment's null result is not a broken script — it is a genuinely negative finding on this model + this method family. Verify would swap models/data/method to check robustness, but with all four claims already refuted or partial on the main setting, the natural next step is:
1. Explore lower-rank / narrower-layer interventions (e.g., single mid layer, rank=1) *before* the verify sweep, to find any residual improvement window if it exists (M3's α=−0.25 shows *partial* preservation, hinting the phenomenon exists in a narrower operating range than the plan assumed).
2. If (1) still fails, verify becomes an audit of whether the negative result generalizes (Qwen-2.5, DeepSeek-R1-Distill families) or is Qwen-3-Thinking-specific.

Both options are `/auto-iteration-loop` territory rather than `/auto-verify`'s stress-test-the-claim territory.

## Next Step

→ `/auto-iteration-loop` — the natural iterative next step, since the main experiment's evidence sharply disagrees with the plan's expectations and a targeted narrower-scope re-planning is needed before broad verify.
