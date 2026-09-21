# Final Proposal — Steerable Social-Variable Directions in an LLM Dictator

**Behavior-source**: given
**Mechanism**: discovery
**Main-experiment model**: `Llama-3.1-8B-Instruct` (HARD, `task.md` §4)
**Resource fidelity**: cost-aware (marker NOT stamped — this is the (given, discovery) combo, not the (given, given) reproduction combo)
**Chosen mechanism**: n/a (mechanism family will be routed by `/mechanism-skills` at experiment-stage Phase 1.5)
**Mechanism strategy**:
```yaml
mechanism_strategy:
  directions: [Location, Causal Intervention]   # execution order
  rejected:
    - Tuning & Editing — the target claim is a *diagnostic* mechanism claim (does the direction causally control the decision?), not a downstream capability-gain claim. Steering is used *as intervention*, not as a fine-tuning-substitute for improving accuracy.
    - Formation Tracing — training-time origin is not part of the four-part claim; too expensive under the 10 GPU-hour budget on an 8B open-weight model where checkpoints across training are not the target of study.
    - Unit Interpretation — SAE / auto-interp would name concepts; the claim already names them (G / A / I / M). Interpretation is not required to land the four parts.
    - Decision Auditing — auditing whether the decision is *trustworthy* is downstream of the claim; the claim asserts that the mechanism *works*, not that it is aligned or trustworthy in deployment.
  note: The four-part claim is a mechanistic-evidence claim (Location + Causal Intervention). Location extracts per-variable directions from paired prompts (and decorrelates them into pure directions). Causal Intervention adds and ablates those pure directions to test causal both-signs, selective control on the transfer amount — the diagonal + off-diagonal of the 4×4 selectivity matrix.
```

## 1. Problem Anchor (frozen — do NOT drift)

**Target model**: `Llama-3.1-8B-Instruct` (8B parameters, chat-tuned decoder-only transformer).
**Target behaviour**: choice of transfer amount τ ∈ {0, 1, …, 20} in a $20-endowment dictator game with a $10 fair-split reference, decided from a natural-language prompt that instantiates four input variables — **G** (gender), **A** (age), **I** (instruction phrasing: give / take framing), **M** (meeting condition: meet / no-meet).
**Claim to test (verbatim, four parts from `task.md`)**:
- **C1 · linear encoding** — each variable's decision-relevant influence is a linearly extractable direction in the residual stream.
- **C2 · purity via decorrelation** — a decorrelated "pure" direction (raw direction minus overlap with the other three variables) isolates the unique effect of the target variable, free of confounding from co-varying factors.
- **C3 · bidirectional causal steering** — injecting a pure direction at inference time causally and substantially shifts the target variable's effect on the decision, in both amplifying and attenuating/inverting senses.
- **C4 · selectivity** — steering one variable does *not* measurably move the effects of the other three variables on the decision.

## 2. Method Thesis (one sentence)

Build the four social/contextual variables' *pure* residual-stream directions from paired prompts on the same 1,000-trial dictator-game corpus, and land the four claims with a single unified pipeline — a **Location** stage that extracts per-variable directions via contrastive activation addition (CAA-style paired-difference averaging) and purifies them by linear decorrelation against the other three, followed by a **Causal Intervention** stage that steers along each pure direction at signed magnitudes to fill a 4 × 4 selectivity matrix whose diagonal validates C3 and whose off-diagonal validates C4.

## 3. What is New (contribution)

Every ingredient — CAA, LEACE-style linear erasure, refusal-direction ablation, role-vector steering, socio-demographic linear representations — exists in the literature. The novelty is the *joint* application to a set of four variables in an economic-decision setting, with the four-part claim structured so that success on all four is jointly informative rather than independently reachable:
- Prior CAA / RepE work steered *one* attribute at a time.
- Prior LEACE / R-LACE work erased *one* concept at a time.
- Prior refusal / role work tested *one* single-behaviour direction, both signs.
- **This proposal** tests the joint {G, A, I, M} bundle, decorrelates them mutually, and evaluates a full 4 × 4 selectivity matrix on a *graded* economic-choice outcome (transfer amount τ ∈ [0, 20]) rather than a binary refusal or a multiple-choice answer.

Landing all four claims on this bundle would establish a mechanistically-controllable knob per social variable in an LLM-as-social-agent dictator setting.

## 4. Unified Pipeline (method)

### 4.1 Dataset (constructed; `provenance=constructed`, `available_n=1000`, `used_n=1000`)

Build **DG-1000** — 1,000 dictator-game trials at $20 endowment and $10 fair-split reference. Each trial fixes:
- **G** ∈ {male, female} (∼50/50)
- **A** ∈ {young (~25), old (~65)} (~50/50)
- **I** ∈ {give-framing, take-framing} (~50/50)
- **M** ∈ {meeting, no-meeting} (~50/50)

Each trial is rendered as an English natural-language prompt asking the LLM-dictator to pick τ. To decorrelate the four variables *by design*, use randomised assignment (or full 2×2×2×2 balanced blocking, 16 cells × 62 trials = 992, rounded to 1,000). Also build the **paired-prompt sets** used to extract per-variable difference vectors: for each V ∈ {G, A, I, M} and each trial, produce a minimal-edit partner that differs *only* in V (e.g., male↔female). Result: 1,000 baseline trials + 4 × 1,000 minimal-edit partners = 5,000 prompts total. Prompts are token-length matched within each pair to ≤ ±2 tokens (retry rewording otherwise) and re-checked for surface-string differences other than the flipped variable.

Splits: 80% (800 trials) for direction extraction and purification (C1, C2 fits); 20% (200 trials) held-out for C3 dose-response and C4 selectivity evaluation. Split is stratified over the 2×2×2×2 cells.

### 4.2 Location (C1) — extract raw per-variable direction

For each V ∈ {G, A, I, M}:
1. On the 800 training trials, for each paired partner (p, p'), run both prompts through `Llama-3.1-8B-Instruct` and record the residual-stream activation at the *final input token* (or at a small pre-registered set of positions: end of prompt, end of question) at every layer.
2. Compute the difference `Δ_V^ℓ(p, p')` = h(p') − h(p) at each layer ℓ.
3. Average `v̂_V^ℓ = mean over training pairs of Δ_V^ℓ`.
4. **Layer selection**: pick the layer (or small set of layers) with (a) the highest linear-probe test accuracy for V on held-out activations, (b) the largest projection–transfer correlation. Typical middle layers (∼L/2) are the primary hypothesis; the actual layer(s) are what Location empirically discovers (per `/mechanism-explore`'s "hypothesize the kind of component, not its exact identity").
5. **Mean-centring control** — also compute a mean-centred variant that subtracts the global mean activation before averaging, per Improving-Activation-Steering-with-Mean-Centring; this is a lightweight decorrelation baseline for C2.

### 4.3 Location (C2) — purify by decorrelation

Two decorrelators, run in parallel:
- **P-A: Gram-Schmidt (GS)** — orthogonalise `v̂_V^ℓ` against {`v̂_W^ℓ` : W ≠ V} to obtain `ṽ_V^GS,ℓ`. Cheapest option; interpretable.
- **P-B: LEACE-style closed-form projection** — form the joint LEACE projection Π that removes the linear signal of {W ≠ V} from the residual stream (Belrose et al. 2023 closed form), then define `ṽ_V^LEACE,ℓ = Π v̂_V^ℓ`. This provably makes the pure direction linearly indistinguishable from noise on the other three variables.
- Optional **P-C: R-LACE-adversarial** ablation (only if time in the 10 GPU-hour budget) — for one V (gender) as a robustness ablation, run linear adversarial concept erasure.

C2's headline reports **cross-leakage**: linear-probe accuracy on {W ≠ V} using `ṽ_V^ℓ`. Prediction: the LEACE projected pure direction reduces off-target probe accuracy to (near) chance while preserving on-target probe accuracy at (near) baseline.

### 4.4 Causal Intervention (C3, C4) — steering with both signs

For each pure direction `ṽ_V^ℓ` (from P-B by default; P-A as ablation), and for each α ∈ **{−4σ, −2σ, −σ, 0, +σ, +2σ, +4σ}** (where σ is the per-layer standard deviation of residual-stream norms; the α-grid is pre-registered and identical across V), intervene as:
```
    for each layer ℓ' in an active site set S:
        h(ℓ') ← h(ℓ') + α · ṽ_V^ℓ
```
Applied at all post-prompt token positions (CAA convention). The active-site set S is a small set of layers around the chosen ℓ (default S = {ℓ} — single-site; ablation S = {ℓ − 1, ℓ, ℓ + 1} — small window).

On the 200 held-out trials:
- **C3 — bidirectional dose-response.** Estimate the mean transfer under each α, per V. Report the dose-response curve; run a monotone-response test (Spearman ρ vs. α) and a paired-sign test between α > 0 and α < 0. C3 is supported when: (a) the α = 0 → α = +σ shift is substantial and significant in the direction predicted by the sign of α, (b) at least one α < 0 magnitude *inverts* the sign of V's baseline effect on the transfer, (c) the α → mean-transfer relationship is monotone up to a pre-registered saturation.
- **C4 — 4 × 4 selectivity matrix.** For each (V, W) pair with V ≠ W, at each α ∈ {−2σ, 0, +2σ}, measure the effect of W on the transfer (mean transfer at W = w₁ minus mean at W = w₂) after steering V. Fill M[V, W] = shift in W's effect vs. α = 0. C4 is supported when the diagonal ≫ off-diagonal in magnitude and the "off-diagonal = diagonal" null is rejected under a permutation test.

### 4.5 Quality gates
- **Coherence gate.** At every α used, sample K = 10 generations per steered variable and check (i) format ("transfer amount is a number") and (ii) a simple fluency proxy (n-gram repetition rate below a pre-registered threshold). If gate fails at a given α, that magnitude is dropped from the α-grid and the α-grid caveat is reported.
- **Baseline effect gate.** Before running any intervention, on the 1,000-trial baseline, compute the effect of each V on the transfer using a linear model that controls for the other three. If a baseline effect on V is null (|effect| < pre-registered floor), report but do *not* fail the pipeline — a null baseline effect for one V simply means C3 for that V is trivially satisfied by the α = 0 baseline (there is nothing to amplify or invert) and C4's diagonal for that V is defined only up to the pipeline's power.

## 5. Baselines / ablations (in the plan)

- **B1 · Raw vs. pure direction.** Repeat C3 with raw v̂_V (no decorrelation). Prediction: raw achieves at least as much steering magnitude on V but higher off-diagonal in the C4 matrix (leakage).
- **B2 · Random direction.** For each V, replace `ṽ_V^ℓ` with a random unit vector of matched norm. Prediction: no reliable dose-response.
- **B3 · Mean-centring only.** Use mean-centred difference vector without joint decorrelation. Prediction: intermediate — some C4 improvement over raw but weaker than LEACE.
- **B4 · Directional ablation instead of activation-add.** Replace `h ← h + α ṽ_V` with `h ← (I − ṽ_V ṽ_Vᵀ) h` (project-out). Prediction: acts like a strong negative-α — a good sanity check for the "attenuate/invert" arm of C3.
- **B5 · Portability (verify stage).** Repeat the whole pipeline on `DeepSeek-R1-Distill-Llama-8B` as a swap variant, without changing the constructed dataset.
- **B6 · Position / length audit.** Regress out prompt token length and the position of the flipped variable's token — verify the diff vector is not confounded with a length effect (E5's "context sensitivity" warning).

## 6. Compute budget

Rough estimate on a single 24 GB / 40 GB A-class GPU with `Llama-3.1-8B-Instruct` at bf16:
- Extraction pass over 5,000 prompts (input-only, no generation) at ~50 prompts/s → ~1.7 min. Do this once per layer sweep → ~5 min total for 32 layers.
- Direction fitting (paired means + LEACE closed form + probes) → CPU-bound, ~5 min.
- Steering evaluation on 200 held-out prompts × 4 variables × 7 α-magnitudes × 2 decorrelators + baselines B1–B4 ≈ ~11,200 generations (each ~1 s of output, batched) ≈ **≤ 3 GPU-hours**.
- Portability swap on DeepSeek-8B ≈ another **≤ 3 GPU-hours**.
- **Total: ≤ 8 GPU-hours** — comfortably inside the 10 GPU-hour HARD budget.

## 7. Success criteria (pre-registered)

C1 supported when linear-probe test accuracy on V ≥ 0.80 (for at least G, A, I; may be lower for M since it is a subtler cue) *and* projection–transfer correlation is significant in the predicted direction.

C2 supported when off-target probe accuracy on {W ≠ V} drops from the raw-direction level to at most (chance + 0.05) under LEACE purification, while V's own probe accuracy retains ≥ 0.95× the raw level.

C3 supported when at α = +σ the mean-transfer shift on V is ≥ 25% of the observed baseline effect of V; at least one α < 0 inverts the sign of the baseline effect on V; monotonicity Spearman ρ ≥ 0.7 up to the coherence-gate saturation.

C4 supported when max off-diagonal M[V, W] ≤ 0.30 × min diagonal M[V, V] (pre-registered ratio), and permutation-test rejects the null "off-diagonal = diagonal" at p ≤ 0.05.

## 8. Risks and mitigations

- **Risk R1**: `Llama-3.1-8B-Instruct` may show a weak or ceilinged baseline effect for some V (particularly M, the subtlest cue). **Mitigation**: pre-register per-V power via a pilot on a small subset, and expand the pool of paired-prompt phrasings for M if the baseline is null. If a V has no baseline effect, C3 / C4 for that V default to "trivially satisfied" and are reported as such.
- **Risk R2**: Coherence-gate failure at high |α|. **Mitigation**: pre-registered α-grid stops at 4σ, and any failing magnitude is dropped with a note.
- **Risk R3**: LEACE removes so much overlap that the pure direction becomes near-zero. **Mitigation**: use GS as a fallback; report `‖ṽ_V^LEACE‖ / ‖v̂_V‖` and audit for degenerate directions.
- **Risk R4**: The chosen layer ℓ shifts across V. **Mitigation**: report per-V best layer(s); use a small window S in the ablation if the middle-layer cluster is stable.

## 9. Hand-off to verify

`auto-verify` should stress-test:
- Swap model → `DeepSeek-R1-Distill-Llama-8B` (portability).
- Swap decorrelator → GS vs. LEACE vs. R-LACE (extractor robustness).
- Swap injection → activation-add vs. directional-ablation (both-signs alternate).
- Swap paired-prompt phrasings (5 additional male↔female / young↔old / instruction / meeting rewordings).
