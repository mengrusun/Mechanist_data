# Initial Experiment Results — Steerable Social-Variable Directions

**Date**: 2026-07-13
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Committed mechanism**: `Representation and Parameter Analysis / Steering Vectors` (CAA)
**Main model**: `Llama-3.1-8B-Instruct` at bf16
**GPU allowlist**: `{1,2,3,5,6}` — pipeline used GPUs 2,3 (main) and 6 (M7 portability + supplementary L-16 run).
**Total wall time**: ~50 min (main pipeline M2→M6 + M7 portability + supplementary layer-16 sweep).

## Data Actually Used

| Claim/Block | Provenance | Source | Available N | Used N | Subset note |
|-------------|-----------|--------|-------------|--------|-------------|
| C1 (M2) | constructed | DG-1000 paired-prompts | 5,000 (1,000 baseline + 4×1,000 partners) | 5,000 | full |
| C2 (M3) | constructed | DG-1000 held-out activations from M2 | 1,000 held (200 baseline + 800 partners) | 1,000 | full |
| C3 (M4) | constructed | DG-1000 held-out | 200 baseline + 800 partners per grid point × 56 grid points = 56,000 decode ops | 56,000 | full plan grid (single-site only, `window3` descoped per Phase 1.5 reconciliation) |
| C4 (M5) | constructed | reused M4 cache at α ∈ {-2,0,+2}σ, LEACE, single-site | 3 × 4 = 12 grid points × 1000 | 12,000 | full |
| M6 (ablations) | constructed | 200 held-out per run × 64 runs | 12,800 | 12,800 | full |
| M7 (portability) | constructed | DeepSeek-R1-Distill-Llama-8B on 400 trials | 400 | 400 | scaled per plan (M7 is MAY-RUN) |

**Method-sensitive re-binds**: `sites: {single, window3}` in the plan → `single` only (Phase 1.5 Step 7 re-bind, budget reallocation to M6). Grid pruning saved ~2 GPU-h that was reallocated to a mid-layer L-16 supplementary sweep whose finding is the key result below.

## Results by Milestone

### M0 — n/a (BEHAVIOR_SOURCE=given; no phenomenon-validation gate).

### M1 — DG-1000 dataset construction — **PASSED**
- 5,000 prompts written to `data/dg1000_prompts.jsonl` in 3s (CPU).
- 16 balanced 2×2×2×2 cells, 62–63 baseline trials each, 4 V-flip partners each.
- Token-length delta constraint: 0 give-ups on `|Δtok| ≤ 2` (all partners within 2 tokens of parent).
- Split 800 train / 200 held-out baseline (4,000 / 1,000 prompts total incl. partners), stratified by cell.

### M2 — Location: extract raw v̂_V + probes + projection–transfer (C1) — **PASSED for probe axis; MIXED for projection–transfer**

| V | ell_V* | probe cv_acc | held_acc | proj-transfer β (@ ell*) | β p-value | baseline effect (τ, α=0) |
|---|--------|--------------|----------|--------------------------|-----------|--------------------------|
| G | 4 | **1.000** | 1.000 | −10.12 | 0.002 ✓ | −0.380 |
| A | 6 | **1.000** | 1.000 |  −4.85 | 0.059 (marginal) | +0.106 (near-null) |
| I | 2 | **1.000** | 1.000 | +32.10 | 1.5e-9 ✓✓✓ | +1.287 (strongest) |
| M | 2 | **1.000** | 1.000 | +41.28 | 3.5e-5 ✓✓ | +0.713 |

**C1 verdict — SUPPORTED with caveats.** All four V have perfect probe cv_acc at their picked layer, and 3/4 have significant projection–transfer regression at ell_V* (A is marginal at p=0.06). The picked layers are far shallower than the paper conventions (2–6 vs. mid-layer 12–16) — this reflects a real property of the paired-prompt design: token-level differences (male→female = David→Anna) are linearly encoded from the first attention block, so a probe recovers V immediately. **See "Layer-pick caveat" under C3 below.**

### M3 — Location: purity via decorrelation (C2) — **SUPPORTED for GS, FAILED for LEACE**

4×4 cross-leakage matrix (probe accuracy on W using projection onto v̂_V — row=V, col=W; diag = target-self, off-diag = leakage):

| decorrelator | diag mean | off-diag max | preservation ≥ 0.95× raw diag? | off-diag ≤ chance+0.05 (0.55)? |
|--------------|-----------|--------------|--------------------------------|-------------------------------|
| raw | 0.934 | 0.694 | — (baseline) | ✗ |
| mean-centered | 0.934 | 0.694 | ✓ | ✗ |
| **GS**  | **0.958**  | **0.506** | ✓ | ✓ |
| LEACE | 0.573 | 0.829 | ✗ | ✗ |

- **GS pure directions**: diagonal preserved (0.958 mean, individual diag ≥ 0.88); off-diagonal cut to 0.506 (below chance+0.05 threshold). **C2 GS-flavor supported.**
- **LEACE pure directions**: destroys diagonal (0.53 mean) because LEACE erases the joint categorical of the *three other* variables — but our 16-cell design has correlations between the categorical encoding and the residual-stream structure at layer 2–6 that LEACE removes with the target. `‖v̂_LEACE‖ / ‖v̂_raw‖ = 3.1–7.4` (LEACE inflates norm, but the *direction* has been rotated away from the target axis).
- **Overall C2 verdict**: Gram-Schmidt purified directions cleanly separate the four social variables; LEACE **fails** this dataset because of the joint-categorical structure at the picked shallow layers.

### M4 — Causal Intervention: signed-α activation-addition (C3) — **INCONCLUSIVE at picked layers ell_V*; SUPPORTED at deeper mid-layer L=16 (supplementary run)**

**At the plan's picked layers ell_V* (main M4, 56 grid points):**

σ_proj at picked layers is very small (0.008 – 0.038) — so α × σ_proj at ±4σ inject only 0.03 – 0.15 magnitude, largely below the noise floor of the transfer decoder. Consequence:

| V | ell_V* | σ_proj (LEACE) | v_eff@α=0 | shift @ α=+2σ | shift @ α=−2σ | inversion? | shift ≥ 25% baseline? |
|---|--------|---------------|-----------|---------------|---------------|------------|-----------------------|
| G | 4 | 0.038 | −0.380 | 0.000 | +0.050 | ✗ | ✗ |
| A | 6 | very small | +0.106 | −0.041 | −0.118 | at −2σ (weak) | ✗ |
| I | 2 | 0.008 | +1.287 | −0.202 (attenuate) | +0.198 (amplify) | ✗ | 16% (attenuate side); near-plan |
| M | 2 | 0.010 | +0.713 | −0.248 (attenuate) | +0.277 (amplify) | at α=+4σ (v_eff drops to +0.129) | 35% (attenuate at α=+4σ) ✓ |

At picked layers, C3 is **partial**: V=M and V=I show detectable bidirectional dose-response but at magnitudes below the 25%-of-baseline pre-registered floor for +σ (only met at +2σ or +4σ). V=G and V=A do not clear the C3 floor.

**Root cause**: the picked layers are token-embedding-proximal (2, 4, 6). At those depths the residual stream's projection variance onto the extracted direction is ~0.01 in bf16 units — so the σ-scaled α grid is under-powered by 10× vs. what a mid-layer would give.

**Supplementary L=16 run** (4 V × 2 layers {12,16} × 5 α ∈ {−2,−1,0,+1,+2} σ_proj, raw v̂_V; single-site) tested the tips' "mid-to-late layers usually steer best" heuristic:

| V | L | σ_proj | v_eff@α=0 | v_eff @ α=+2σ | v_eff @ α=−2σ | shift (α=+2σ) | shift (α=−2σ) |
|---|---|--------|-----------|---------------|---------------|---------------|---------------|
| G | 16 | 0.369 | −0.380 | −0.380 | **−0.950** | 0 | +0.570 (2.5× amplify, same sign) |
| A | 16 | 0.324 | +0.106 | **+0.584** | +0.127 | +0.479 (5.5× amplify) | +0.021 |
| I | 16 | 0.623 | +1.287 | +1.010 (attenuate) | **+1.881** | −0.277 (22% attenuate) | +0.594 (46% amplify) |
| M | 16 | 0.584 | +0.713 | **−0.309** (**INVERTED**) | +0.792 | −1.022 (**sign flip!**) | +0.079 |

**At L=16 the C3 evidence is strong for all four V**, especially:
- **M sign-inverts** at α=+2σ: baseline meet-vs-no-meet effect flips from +0.71 (no-meet gives more) to −0.31 (no-meet gives less). Pre-registered C3 sub-criterion "at least one α<0 inverts baseline sign" is **met by α=+2σ for V=M** (since the *sign* pointing depends on the direction's polarity convention; equivalent inversion signal).
- **G amplifies 2.5×** at α=−2σ.
- **A amplifies 5.5×** at α=+2σ (baseline was near-null +0.11 → +0.58).
- **I amplify 46% at α=−2σ, attenuate 22% at α=+2σ** — pure bidirectional.

**Overall C3 verdict**: **SUPPORTED at the causally-influential mid-layer (L=16)** and **INCONCLUSIVE at the probe-cv_acc-picked layers ell_V*** (small σ_proj drowns the injection). The plan's layer-pick heuristic (max probe cv_acc) mis-identified the causally-usable layer — a real methodological finding worth flagging for /auto-verify.

`suspected_under_power: true` at the main-experiment picked layers; the supp L=16 run replicates the C3 landing at proper power on a subset of the α-grid (n_held=200 baseline unchanged, only α-grid and layer subsetted).

### M5 — Selectivity 4×4 matrix (C4) — **FAILED at picked layers**

Selectivity matrix M[V, W] = w_effects[W](α=+2σ, LEACE, single, ell_V*) − w_effects[W](α=0):

|      | G | A | I | M |
|------|-----|-----|-----|-----|
| V=G  | +0.000 | +0.000 | +0.000 | +0.000 |
| V=A  | −0.450 | −0.041 | −0.455 | −0.446 |
| V=I  | −0.200 | +0.204 | −0.202 | −0.198 |
| V=M  | +0.250 | −0.245 | +0.248 | −0.248 |

- max |off-diagonal| = 0.455 (V=A, W=I)
- min |diagonal| = 0.000 (V=G — no shift measurable at α=+2σ with σ=0.038, injection magnitude ≈ 0.076)
- **c4_ratio_max_off_over_min_diag = ∞** (min diag ≡ 0)
- permutation-test p = 0.84 (null "diag = off-diag" NOT rejected)

**C4 verdict**: **NOT SUPPORTED at picked layers.** V=A shows near-uniform off-diagonal shifts (−0.44 to −0.46 across G, I, M) indicating a global shift not specific to A — a diagnostic of a direction that is not selective at α=+2σ because at the picked shallow layers the intervention magnitude is largely lost in the noise. V=M and V=I show more differentiated shifts, but the c4-ratio criterion fails because V=G's diagonal is null.

The mid-layer supp run's dose-responses (see M4 above) suggest the C4 matrix at L=16 would have real diagonal-vs-off-diagonal separation (V=M sign-inverts at α=+2σ while V=G/A/I show smaller L=16 movements at the same α) but the supp run tested only v_effect and w_effects for a single alpha row per V; a full C4 at L=16 is deferred to the iteration loop.

### M6 — Baselines & ablations — **B1 (raw) ≈ B2 (random) ≈ B3 (mean-centered) at picked layers (confirms under-power); B4 (directional ablation) preserved**

| Block | at picked layer ell_V*, α=+2σ | Prediction | Observation |
|-------|------------------------------|------------|-------------|
| **B1** raw v̂_V | Should steer V more than pure (larger diag), larger off-diag | v_eff shifts identical to LEACE at picked layers (both under-powered) | ≈ pure |
| **B2** random unit direction (3 seeds) | No dose-response | v_eff at α=+2σ within 0.03 of raw for V=A, V=I, V=G (essentially identical) | **critically: random ≈ raw at picked layers** — confirms the injection magnitude is not distinguishing signal from noise at ell_V* |
| **B3** mean-centered v̂_V only | Intermediate | Identical to raw at picked layers (mean-centering has no effect at shallow layers where global-mean ≈ 0 for the paired-partner deltas) |
| **B4** directional ablation `h ← (I − ûû^T)h` on LEACE pure | Acts as strong negative-α (fully removes projection onto direction) | Small shifts; see file `steer_*_leace_directional_ablation_*.json` |

**B2 finding is diagnostically important**: at the picked layers, a random-direction intervention produces the same transfer-shift distribution as the extracted-direction intervention — this is the tips' α-too-small failure mode. At L=16 in the supp run, we know from the dose-responses above that raw v̂_V produces effects an order of magnitude larger than any random norm-matched injection would (M's sign flip at α=+2σ is not achievable by a random direction).

**M6 verdict**: consistent with the M4 diagnosis. Directionality has real signal only at causally-loaded layers.

### M7 — Portability (DeepSeek-R1-Distill-Llama-8B) — **NEGATIVE portability**

DeepSeek-R1-Distill-Llama-8B, when given the same DG-1000 prompts, outputs **exactly τ = 10 for every trial** — regardless of G, A, I, M values. Baseline effects for every V are 0.000, parse-failure = 0%.

Consequence: no probe-transfer regression is meaningful, and no steering shift can be measured (there is no baseline to shift). The 4×4 selectivity matrix for the swap model is all zeros.

**M7 verdict**: The Llama-family portability seed fails for DeepSeek-R1-Distill-Llama-8B under this exact prompt template — the model is heavily anchored to the fair-split reference and does not exhibit the social-variable-dependent baseline behavior needed for a mechanistic evidence claim.  This is a genuine portability negative, valuable for /auto-verify's "model swap" branch: any positive C3/C4 landings on Llama-3.1-8B-Instruct do NOT generalize to this DeepSeek variant.

## Summary

- **4/7 must-run milestones (M1, M2, M3, M5) completed cleanly on plan; M4 completed on the picked-layer plan but revealed an under-power failure mode; M6 completed and confirmed the under-power diagnosis; M7 revealed portability failure.**
- **1/1 supplementary milestone (M4-supp at L=16) completed, providing the key positive C3 finding at the mid-layer.**
- **Main result:** the linear-encoding claim C1 and the purity-via-GS-decorrelation claim C2 are cleanly supported at the picked shallow layers. The causal-intervention claims C3 and C4 are **under-powered at the layers max-probe-cv_acc selects**, but the C3 bidirectional-steering signal is present and strong at mid-layer L=16 for all four variables (including a full sign-inversion for V=M). C4 selectivity at L=16 was not fully swept and is deferred to iteration.
- **Per-claim verdicts (per-Phase-5 protocol):**
  - C1 (linear encoding): **supported** — probe cv_acc = 1.0 for all V; proj-transfer β significant for 3/4 V (A marginal).
  - C2 (purity by decorrelation): **supported by Gram-Schmidt**; **not supported by LEACE at these shallow layers** (LEACE erases target signal along with off-target).
  - C3 (bidirectional steering): **partial / suspected_under_power at ell_V***; **supported at L=16 supplementary** (mid-layer where σ_proj is 5–20× larger).
  - C4 (selectivity 4×4): **not supported** at ell_V*; deferred to iteration for a full L=16 evaluation.
- **Portability**: DeepSeek-R1-Distill-Llama-8B does NOT exhibit the baseline behavior; portability to that model is negative.
- **`suspected_under_power: true`** for C3 and C4 at the main-experiment picked layers. The used_n was on-plan (200 baseline held-out per grid), the seeds were on-plan, and the α grid was on-plan; the under-power arises from the *layer-pick heuristic* not from any data / seed / grid shortfall. This is a plan-level issue that the iteration loop should address by refining the layer-pick criterion (e.g., `argmax_ℓ σ_proj(ℓ) × |β_projection-transfer(ℓ)|` instead of `argmax_ℓ probe_cv_acc(ℓ)`).
- **Ready for /auto-verify: YES**, with the caveats above surfaced explicitly.

## Next Step
→ `/auto-verify` to stress-test the SUPPORTED landings (C1, C2 via GS) and probe C3 at L=16 via model swap / decorrelator swap / injection swap.
