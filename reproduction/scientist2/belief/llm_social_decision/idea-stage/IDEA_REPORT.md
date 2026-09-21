# Idea Report — Captured Behavior

**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: `task.md` (faithful capture from the user's four-part mechanism claim on `Llama-3.1-8B-Instruct` acting as a dictator in a 1,000-trial dictator game)
**Direction (topic tag from orchestrator)**: Steerable Social-Variable Directions in LLM Decision Making
**Date**: 2026-07-13
**Pipeline**: `/research-lit` → faithful behavior capture (this file) → `/mechanism-explore` (loaded, `Location → Causal Intervention`) → `/research-refine-pipeline` (next)

## Executive Summary

`task.md` states a four-part mechanistic claim about how a `Llama-3.1-8B-Instruct` dictator encodes and uses four social/contextual variables — **G** (gender), **A** (age), **I** (instruction phrasing: give/take), **M** (meeting condition: meet/no-meet) — when choosing a transfer amount in a $20-endowment dictator game with a $10 fair-split reference. The four parts split into four verifiable claims: (C1) linear encoding of each variable's decision-relevant influence as a residual-stream direction; (C2) "pure" variable direction — the linearly extracted direction with overlapping components of the other three variables removed — isolates the unique effect of that variable; (C3) causal bidirectional steering — injecting a pure direction at inference time causally and substantially shifts the target variable's effect on the transfer amount, in both amplifying and attenuating/inverting senses; (C4) selectivity — steering along the pure direction for one variable leaves the other three variables' effects on the transfer intact. All four claims are jointly satisfied *by construction* by the pipeline's unified proposal, so the experiment plan builds one integrated suite whose milestones each cover one or more claims.

## Literature Landscape

*(For the full landscape narrative + structured paper table, see `LANDSCAPE.md` in this same directory. Landscape summary highlights below.)*

The interpretability side of the field has *separately* established every ingredient the target claim assembles:

- Paired-prompt contrastive activation addition (CAA) extracts and injects residual-stream directions that can steer instruction-tuned LLMs — canonical for C1 and C3.
- Concept scrubbing / LEACE-style closed-form linear erasure and R-LACE-style adversarial extraction are the reference primitives for the "purity via decorrelation" step of C2.
- Refusal-direction ablation demonstrates single-direction, both-signs causal control on a coarse-grained behaviour — direct precedent for C3, and the closest existing single-behaviour analog of C4's selectivity story.
- Naïve summation of independently-extracted CAA vectors is documented to *fail* to compose (van der Weij et al., 2024), which is exactly the failure the decorrelation step (C2) is designed to fix.

The behavioural side has established that persona conditioning shifts LLM economic play (Ji Ma's dictator work; the AI-Chatbot behavioural economics benchmark; role vectors), but not with the four-part mechanistic argument on a decision output. The gap the target claim closes is: *for a set of demographic + situational variables jointly, extract per-variable directions from paired prompts, decorrelate them into pure directions, and show causal both-signs, selective control on the graded economic decision output* — on `Llama-3.1-8B-Instruct`.

## Resources (binding for main experiment)

Resources are captured verbatim from `task.md`. Under the current combination (`BEHAVIOR_SOURCE=given` + `MECHANISM=discovery`) they are the user's preferred resources; because `task.md` explicitly names them and the declared 10 GPU-hour budget covers the full-scale run (residual-stream extraction + inference-time intervention on an 8 B model comfortably fits), the plan uses them at full scale. `resource_fidelity: strict` is intentionally **not** stamped (that marker is reserved for the `given` + `given` reproduction combo).

- **Main-experiment model** — `Llama-3.1-8B-Instruct` (single open-weight model; residual-stream extraction and intervention run here). Path: `/data/zhenqian/models/Llama-3.1-8B-Instruct`. `task.md` HARD constraint #4.
- **Main-experiment dataset** — 1,000 baseline dictator-game trials at $20 endowment / $10 fair-split reference, with the four input variables **G / A / I / M** randomized across trials. Each trial is rendered as a natural-language prompt asking the LLM-dictator to choose a transfer amount. Plus the **paired-prompt** sets that differ *only* in the target variable (male↔female, young↔old, instruction-A↔instruction-B, meeting↔no-meeting) — used to extract per-variable difference vectors. `provenance = constructed`, `available_n = 1000`, `used_n = 1000`.
- **Verify variants (candidates, use as needed — not all required)** — Alt model: `DeepSeek-R1-Distill-Llama-8B` at `/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B` (or any other open-weight LLM). No additional datasets.
- **Compute / GPU budget** — 10 GPU-hours total, GPUs restricted to `gpu_id ∈ {1, 2, 3, 5, 6}`, environment via `conda env`. Do not downscale under the budget until it is actually consumed.
- **Filesystem allowlist** — working directory + `/data/zhenqian/data` + `/data/zhenqian/models`.

## Claims to Verify

Four claims, one per paragraph in `task.md`'s `## Claim`. Each carries the verbatim `Original` excerpt so downstream stages can audit faithfulness end-to-end.

### Claim 1: Linear encoding of each social/contextual variable

**Original (verbatim excerpt from `task.md`):**
> For each social/contextual variable (e.g., gender, age, instruction framing, meeting condition), the LLM encodes its influence on decision making as a linearly extractable direction in the residual stream.

**Extracted statement**: For each variable **V ∈ {G, A, I, M}**, there exists a linearly extractable direction **v̂_V^ℓ** in the residual stream of `Llama-3.1-8B-Instruct` at some layer ℓ (or a small set of layers) that carries the variable's influence on the dictator's transfer decision. The direction is obtained from paired prompts that differ only in V (e.g., male↔female), by averaging the residual-stream activation difference across paired trials.

**Hypothesis (H1)**: The raw paired-difference vector **v̂_V^ℓ** is a genuine linear representation of V's influence on the decision — a linear probe trained on the same paired activations decodes V above chance with above-random effect on the model's transfer output, and the effect is not explained by shared surface features (mean-centring baseline).

**Measurable predicate**: For each V and its own paired-prompt set, a leave-out linear probe (logistic on activations at ℓ) recovers V with test accuracy well above chance (≥ pre-registered threshold, e.g. 0.80 on a balanced held-out split), *and* the norm of the residual projected onto **v̂_V^ℓ** correlates with the model's transfer amount for prompts where V varies while other variables are held fixed, controlling for token length and prompt position.

**Expected direction**: up (probe accuracy > chance; projection–transfer correlation ≠ 0 in the predicted sign).

**Resources (binding — pointed to the top-level `## Resources` block)**: main model `Llama-3.1-8B-Instruct`; paired-prompt subset of the 1,000-trial main dataset (paired sets for each of G, A, I, M).

**Status**: pending verification.

**Notes**: Structural extraction cleaner than task.md's prose (`task.md` says "encodes its influence … as a linearly extractable direction"; we make explicit both the *extractability* test and the *decision-relevance* test). No claim narrowing — both are required by the source sentence. Layer ℓ is deliberately left as "some layer or small set of layers" per `/mechanism-explore`'s "hypothesize the kind of component, not its exact identity" — the specific layer(s) are what the experiment stage's Location screen discovers.

### Claim 2: Purity via decorrelation isolates the unique effect of each variable

**Original (verbatim excerpt from `task.md`):**
> A "pure" variable direction, obtained by removing overlapping components with other variables, isolates the unique effect of that variable, free of confounding from co-varying factors.

**Extracted statement**: For each V, the pure direction **ṽ_V^ℓ**, obtained by orthogonalising / linearly projecting **v̂_V^ℓ** away from the raw directions of the other three variables (e.g., via LEACE-style closed-form projection, Gram-Schmidt on {v̂_G, v̂_A, v̂_I, v̂_M}, or regression-residual purification), carries the *unique* linear effect of V on the decision. Concretely: a probe/target-behavior measured after removing overlap should retain V's effect and lose the other three variables' effects on the same residual-stream basis.

**Hypothesis (H2)**: Decorrelation reduces cross-variable linear leakage on the residual stream — the *pure* direction predicts V but does not predict {G, A, I, M} \ V, and the causal effect (see C3) attributable to ṽ_V is comparable to (or larger than) the effect attributable to raw v̂_V.

**Measurable predicate**: (a) After purification, a linear probe on ṽ_V predicts V above threshold *and* predicts each other variable at or near chance on the same held-out split. (b) The purified direction carries an equal or larger share of C3's causal steering effect on V's target behavior compared with the raw direction. (c) Pre-registered cross-leakage metric (e.g., mutual information / probe-accuracy on the other three variables from the purified basis) is bounded below a threshold.

**Expected direction**: cross-leakage on the pure basis is *lower* than on the raw basis (down); V's own probe accuracy is preserved (equal / up); C3 causal magnitude on the pure basis is at least on par with the raw basis (equal / up).

**Resources (binding)**: same as C1; requires the *joint* paired-prompt corpus over all four variables to run the decorrelation.

**Status**: pending verification.

**Notes**: `task.md` says "removing overlapping components with other variables". We make explicit that the *reference* primitive is a linear projection (LEACE/Gram-Schmidt/regression-residual style) — the exact recipe is chosen at the experiment stage. We deliberately do *not* narrow to a single decorrelator; multiple recipes should be tried as ablations. "Free of confounding" is operationalised as "cross-leakage bounded below a pre-registered threshold" — a strictly weaker but empirically testable form of "free", which is the appropriate refinement.

### Claim 3: Causal, substantial, bidirectional steering by injecting the pure direction

**Original (verbatim excerpt from `task.md`):**
> Injecting these directions into the model at inference time causally and substantially alters the relationship between the targeted variable and the model's decision output, while leaving other variables' effects intact.
>
> The same mechanism supports both directions of intervention — amplifying a variable's effect or attenuating/inverting it — yielding a practical handle for alignment and debiasing of LLM-based social agents.

**Extracted statement**: For each V, applying the intervention **h ← h + α · ṽ_V^ℓ** at inference time, at a small set of residual-stream sites, causally and substantially shifts the mean transfer amount conditional on V compared with the α = 0 baseline. The effect is *bidirectional*: positive α amplifies V's baseline effect on the transfer, negative α attenuates and, at large-enough magnitude, inverts it. The effect shows a monotone dose-response over a range of α that does not destroy general model competence.

**Hypothesis (H3)**: Steering along the pure per-variable direction is a *causal, dose-responsive, bidirectional* control on the transfer amount attributable to that variable, and the direction of the shift matches the direction predicted by the sign of α.

**Measurable predicate**: Per V, on a held-out subset of the 1,000-trial main dataset, evaluate the average transfer for α ∈ Α (a pre-registered signed grid including at least three positive and three negative magnitudes plus zero). Then: (a) the α = 0 vs α > 0 mean transfer differ by a substantial and statistically significant amount in the predicted direction (pre-registered magnitude threshold, e.g. |Δmean| ≥ some fraction of the baseline effect); (b) α < 0 produces an effect in the opposite direction, of comparable or attenuated magnitude, at *at least one* magnitude flipping the *sign* of V's baseline effect (i.e., an inversion is observed); (c) the α → effect relationship is monotone within the tested range up to a pre-registered saturation point; (d) at all tested α, model behaviour outside V's effect on transfer remains coherent (fluency / format check passes).

**Expected direction**: positive α ⇒ same-sign amplification; negative α ⇒ opposite-sign attenuation and inversion (both signs required by `task.md`).

**Resources (binding)**: same as C1; the α-grid is evaluated on the main-dataset held-out subset.

**Status**: pending verification.

**Notes**: `task.md` couples "leaving other variables' effects intact" with claim (iii). We split that clause off into a **standalone claim (C4)** to make it independently verifiable — the source sentence intertwines two testable predicates, one about V's own effect (C3) and one about non-V's effects (C4). This is a *split*, not a scope narrowing. "Both directions of intervention" is faithfully carried as the α-sign requirement.

### Claim 4: Selectivity — steering one variable leaves the other three variables' effects intact

**Original (verbatim excerpt from `task.md`):**
> Injecting these directions into the model at inference time causally and substantially alters the relationship between the targeted variable and the model's decision output, **while leaving other variables' effects intact**.

**Extracted statement**: For each V steered by **h ← h + α · ṽ_V^ℓ**, the effects of the *other three* variables W ∈ {G, A, I, M} \ V on the transfer amount are *preserved* (up to a pre-registered tolerance) compared with the α = 0 baseline. Concretely, if we measure the effect of W on the transfer (e.g., mean transfer under W = w₁ minus mean transfer under W = w₂ at fixed V) at α = 0 and at each α ≠ 0 in the C3 grid, the |shift in W's effect| should stay below a pre-registered tolerance for every W ≠ V.

**Hypothesis (H4)**: The pure directions are approximately *representationally independent* across variables — steering V does not measurably move the effects of the other three variables on the decision, i.e. the 4 × 4 selectivity matrix has a large diagonal and a small off-diagonal.

**Measurable predicate**: Build the **selectivity matrix** M ∈ ℝ^{4×4} where M[V, W] = shift in W's effect on transfer when steering V (relative to no-steering baseline). Pre-registered thresholds: (a) diagonal entries |M[V, V]| substantially above zero (C3); (b) off-diagonal entries |M[V, W]| below a pre-registered tolerance (a small fraction of the diagonal); (c) sign of off-diagonal entries is not systematic across V, W. Statistical test: null hypothesis "off-diagonal effect equals on-diagonal effect" is rejected with pre-registered α-level.

**Expected direction**: diagonal ≫ off-diagonal (in magnitude).

**Resources (binding)**: same as C1; requires the full 1,000-trial randomised main dataset so that per-variable effects can be estimated cleanly.

**Status**: pending verification.

**Notes**: Split from the "while leaving other variables' effects intact" clause of `task.md`'s third bullet, so selectivity can be verified with its own milestone rather than folded into C3's dose-response. The selectivity matrix is the natural four-way generalisation of the "concept cones / representational independence" work in the landscape.

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering all four claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps

- [ ] `/mechanism-skills` to route the testing approach to a concrete mechanism family + submethod (Workflow 1.25) — expected route: **Location** (representation-and-parameter-analysis + probing for extraction) → **Causal Intervention** (activation addition + directional ablation for injection).
- [ ] `/auto-experiment` to implement and run the four-claim verification suite (Workflow 1.5)
- [ ] `/auto-verify` to stress-test each verified claim under method / dataset / model swaps (Workflow 1.75) — swap `Llama-3.1-8B-Instruct` → `DeepSeek-R1-Distill-Llama-8B` for portability, swap CAA → LEACE-projection-then-steer for extractor robustness.
- [ ] `/auto-iteration-loop` to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain.
