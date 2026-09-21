# Idea Report — Captured Behavior

**Direction**: (empty CLI — behavior sourced entirely from `task.md`; supplementary retrieval scope only)
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: `task.md` (faithful capture, no ideation / novelty / M0)
**Date**: 2026-07-14
**Pipeline**: `/research-lit` → faithful behavior capture → `/research-refine-pipeline`

## Executive Summary

`task.md` fixes a single, four-part reasoning-steering claim on DeepSeek-R1-Distill-Llama-8B: distinct in-chain reasoning behaviours (uncertainty, validation-example generation, backtracking, self-correction) each occupy an approximately linear direction in the residual stream, extractable from a small pool of contrastive activation pairs, dose-response steerable by a single scalar coefficient, and finer-grained than prompt engineering while preserving downstream reasoning accuracy. The four bullets in `task.md`'s `## Claim` section are captured verbatim below as four independently verifiable sub-claims (C1–C4) and refined jointly into one unified experiment plan in `refine-logs/FINAL_PROPOSAL.md` + `refine-logs/EXPERIMENT_PLAN.md`. No M0 phenomenon-validation gate (behavior-source=`given` → treat behaviour as assumed).

## Literature Landscape

See `idea-stage/LANDSCAPE.md` for the full landscape (15 papers). The pre-cutoff literature establishes: (i) CAA / RepE / ActAdd are a mature primitive for extracting behaviour directions from contrastive activation pairs and steering via scalar α on *conversational* Llama-2 / LLaMA-family behaviours (Panickssery+2023, Zou+2023, Turner+2023); (ii) the *linear representation hypothesis* is theoretically grounded (Park+2023); (iii) reasoning-tuned R1-distill / o1 / Qwen-Thinking models expose a new class of *in-chain* behaviours (uncertainty / exploration / reflection) named by the Long-CoT survey (Chen+2503.09567) and by external evidence on self-correction (Y. Li 2601.00828); (iv) the strongest *competing* control paradigm on R1 today is prompt-token-level Thinking Intervention (Wu+2503.24370), which C4 must beat on granularity + accuracy preservation.

**Policy note:** the target-paper family (arXiv:2506.18167 and all 2506+ arXiv ids) is blocked by `.claude/forbidden-urls.txt` — reproduction from `task.md` alone, not from the paper.

## Resources (project-wide, informational)

`task.md` pins these resources for the whole project. In this combination (`BEHAVIOR_SOURCE=given` + `MECHANISM=discovery`) they are the user's **preferred** resources, cost-aware, **not** `resource_fidelity: strict` (the strict marker is reserved for the reproduction combo `given`+`given`). Under `task.md`'s declared 10-hour GPU budget the plan uses them at *full scale* (no cost-driven downscaling of the named model / benchmark) unless a milestone genuinely cannot fit — in which case the experiment stage halts and surfaces it rather than silently downscaling.

- **Experiment-stage model** — `DeepSeek-R1-Distill-Llama-8B` (the lead thinking model on which behaviour directions are extracted and steered).
- **Experiment-stage dataset** — a custom 500-task reasoning benchmark covering 10 reasoning categories, generated with an external strong LLM (Claude 3.5 Sonnet or GPT-4o via the DMX API `gpt-5.4` endpoint).
- **Auxiliary contrastive corpus** — ~100 DeepSeek-R1 reasoning chains + ~100 GPT-4o answers, used to build the behaviour taxonomy and to derive the contrastive activation pairs (behaviour-present vs. behaviour-absent).
- **Verify-stage variant candidates (use as needed, not all)** — models: `DeepSeek-R1-Distill-Qwen-1.5B`, `DeepSeek-R1-Distill-Qwen-14B` (cross-size / cross-backbone within R1-distill family); OOD datasets: MATH, GSM8K, AIME.
- **Fixed infrastructure (never swapped)** — external strong LLM as task generator + behaviour-tag annotator (Claude 3.5 Sonnet or GPT-4o); behaviour taxonomy = {uncertainty, example-generation, backtracking, self-correction, …}.
- **GPU** — 10-hour total budget across all stages, gpu_id ∈ {0,1,2,3}.
- **Paths** — `DATA_DIR=/data/zhenqian/data`, `MODEL_DIR=/data/zhenqian/models`; symlink into work_dir; conda env; downloads via HF / ModelScope tokens in `task.md`.

## Claims to Verify

Four sub-claims, one per bullet in `task.md`'s `## Claim` section, in extraction order. Every claim is stated in the direction / strength / language `task.md` uses; the `Original` field carries the verbatim excerpt for audit; the `Notes` field records the extraction transformation.

### Claim C1: Reasoning behaviours occupy approximately linear directions in the residual stream

**Original (verbatim excerpt from task.md):**
> Distinct reasoning behaviours in thinking LLMs (expressing uncertainty, generating validation examples, backtracking) each map onto an approximately linear direction in the residual stream of the reasoning model.

**Extracted statement**: For each behaviour b ∈ B = {expressing_uncertainty, generating_validation_examples, backtracking, self-correction} exhibited by DeepSeek-R1-Distill-Llama-8B during long-CoT generation, there exists an approximately linear direction v_b ∈ R^d in the model's residual stream that separates behaviour-present from behaviour-absent chain-of-thought excerpts.

**Hypothesis**: H1 — a linear probe (or equivalently, the mean-difference direction) trained on contrastive residual-stream activations attains above-chance separation between behaviour-present and behaviour-absent chain excerpts at some layer L*, per behaviour b, with the direction transferring at that layer L* across held-out excerpts and paraphrases.

**Measurable predicate**: For each behaviour b, at layer L* selected on a held-out validation split, a linear classifier on residual-stream activations achieves held-out separation ≥ a pre-registered floor (e.g., ROC-AUC ≥ 0.75 or class-balanced accuracy ≥ 0.70), and the mean-difference direction v_b explains ≥ a pre-registered fraction of the between-class variance (e.g., first-PC alignment |cos| ≥ 0.7).

**Expected direction**: threshold (above the pre-registered separation and alignment floors).

**Resources**: model — `DeepSeek-R1-Distill-Llama-8B`; dataset — the auxiliary ~100 R1 chains + ~100 GPT-4o answers (contrastive pair source) + the 500-task benchmark (for probe held-out validation of transfer); `used_n` — full ~200 chain sample as annotated; extraction per behaviour ≥ 50 positive pairs after annotation quality-filter.

**Status**: pending verification.

**Notes**: `task.md` names three behaviours in this bullet ("uncertainty / generating validation examples / backtracking"). `## Resources` then explicitly names the four-item behaviour taxonomy ("uncertainty / example-generation / backtracking / …", and the top-level Hypothesis title adds "self-correction"). The bullet's "each" is inclusive; extracted as four behaviours to match the taxonomy resource. Threshold and separation numbers are illustrative — the final numbers are fixed in `EXPERIMENT_PLAN.md`.

### Claim C2: Behaviour directions are extractable from a small pool of contrastive activation pairs

**Original (verbatim excerpt from task.md):**
> Each behaviour direction can be pulled out from a small pool of contrastive activation pairs (behaviour present vs. absent in the chain-of-thought).

**Extracted statement**: v_b in C1 can be extracted from a small pool of contrastive activation pairs — activations at layer L* on behaviour-present chain excerpts vs. behaviour-absent chain excerpts — using the standard difference-of-means / CAA-style estimator, with sample size on the order of tens to a few hundreds of pairs, and remain effective in downstream C3 steering.

**Hypothesis**: H2 — the direction estimated from a small contrastive pool (n_pairs on the order of 10–200 per behaviour) is stable and steerable: (i) direction cosine similarity across independent halves of the pool ≥ a pre-registered floor (e.g., |cos| ≥ 0.7); (ii) the resulting steering vector produces the C3-predicted behavioural effect at test time.

**Measurable predicate**: With n_pairs varied on a small grid (e.g., {10, 25, 50, 100, 200}), for each behaviour b: (i) split-half cosine similarity of the direction ≥ 0.7 at n_pairs ≤ 200; and (ii) the steering effect on the target behaviour (C3 metric at a fixed α) reaches ≥ a pre-registered fraction (e.g., 80%) of the effect obtained with the largest pool, at n_pairs ≤ 200.

**Expected direction**: threshold (stability and effect-preservation floors both cleared at small n_pairs).

**Resources**: same as C1; `used_n` — extract at least at n_pairs ∈ {10, 25, 50, 100, 200}; behaviours as in C1.

**Status**: pending verification.

**Notes**: The bullet does not fix "small" numerically; "small" is operationalised via a sweep across n_pairs ∈ {10 … 200} — task.md's `## Resources` says "~100 DeepSeek-R1 reasoning chains + ~100 GPT-4o answers", which caps the pool. Not simplifying the claim: C2 as extracted requires the *derived* vector to also work in C3, i.e., extractability with usefulness.

### Claim C3: Adding or subtracting the vector at inference amplifies or suppresses the behaviour, dose-responsive in a single scalar

**Original (verbatim excerpt from task.md):**
> Adding or subtracting the extracted steering vector at inference time amplifies or suppresses the corresponding behaviour in the generated chain, in a dose-dependent manner controllable by a single scalar coefficient.

**Extracted statement**: For each behaviour b, adding +α·v_b to the residual stream at layer L* during inference *amplifies* the frequency / prominence of b in the generated chain; subtracting −α·v_b *suppresses* it; and the effect varies *monotonically* with |α| across a bounded range, so that a single scalar α controls the behaviour amplitude.

**Hypothesis**: H3 — (i) sign: at fixed |α|, +α produces a strictly higher LLM-judged rate of behaviour b in the generated chain than the unsteered baseline, and −α produces a strictly lower rate; (ii) monotonic dose-response: for α on a bounded grid (e.g., α ∈ {−3σ, −2σ, −σ, −σ/2, 0, +σ/2, +σ, +2σ, +3σ} where σ is a per-layer scale), the behaviour-rate curve is monotonic in α over the useful range, and Spearman ρ(α, rate) ≥ 0.7; (iii) specificity: on a matched *off-target* behaviour b′ ≠ b, |Δrate| is significantly smaller (e.g., ≤ 50%) than on b.

**Measurable predicate**: For each behaviour b ∈ B, at the operating layer L*, (i) sign check: sign(rate(+α) − rate(0)) = +, sign(rate(−α) − rate(0)) = − at the operating α; (ii) monotonicity: Spearman ρ(α, rate_b) ≥ 0.7 on the α-grid; (iii) specificity: off-target |Δrate| ≤ 50% of on-target |Δrate|. The `rate` is measured by the external strong LLM (Claude 3.5 Sonnet / GPT-4o via DMX endpoint) as behaviour-tag frequency per generated chain.

**Expected direction**: (i) up on +α, down on −α; (ii) monotonic in α (threshold on Spearman); (iii) specificity threshold cleared.

**Resources**: model — DeepSeek-R1-Distill-Llama-8B; dataset — the 500-task reasoning benchmark (10 categories × ~50 tasks); `used_n` — inference on all 500 tasks per α-grid setting, per behaviour, with at least one seed per α; LLM-as-judge — external strong LLM as behaviour-tag annotator (Claude 3.5 Sonnet or GPT-4o via `task.md`'s DMX endpoint).

**Status**: pending verification.

**Notes**: `task.md` says "amplifies or suppresses … in a dose-dependent manner controllable by a single scalar coefficient" — the sign / monotonic / specificity requirements are the natural minimum for that claim to hold; not tightening.

### Claim C4: Steering-vector control is finer-grained than prompt engineering, at preserved accuracy

**Original (verbatim excerpt from task.md):**
> This steering-vector control is finer-grained than prompt engineering while preserving downstream reasoning accuracy on the evaluation benchmark.

**Extracted statement**: On the 500-task reasoning benchmark, controlling each behaviour b via activation-space steering (varying α) yields a *finer* Pareto front of behaviour-rate vs. downstream-accuracy trade-offs than a matched prompt-engineering baseline (e.g., appending or prepending a behaviour-eliciting instruction, or the Thinking Intervention token-insertion protocol), *and* at a matched behaviour-rate change, steering preserves downstream reasoning accuracy at least as well as (i.e., accuracy drop no larger than) the prompt-engineering baseline.

**Hypothesis**: H4 — (i) granularity: the α-grid produces at least K ≥ 5 distinct, well-separated operating points on the (behaviour-rate, accuracy) plane, whereas a matched prompt-engineering grid produces fewer than K distinct points (or a strictly wider gap between adjacent points); (ii) preservation: at α-values producing a behaviour-rate change equal to that of the prompt baseline, downstream accuracy is not lower than the prompt baseline by more than a pre-registered margin ε (e.g., ε ≤ 2 accuracy points), and not lower than the unsteered baseline accuracy by more than a pre-registered margin ε′ (e.g., ε′ ≤ 3 accuracy points) at the operating α used for C3.

**Measurable predicate**: (i) Granularity: #distinct-well-separated operating points on the (rate, accuracy) plane is strictly larger for steering than for prompt engineering across the same rate range. (ii) Preservation: matched-rate accuracy(steering) ≥ accuracy(prompt) − ε; and accuracy(steering at operating α) ≥ accuracy(unsteered) − ε′.

**Expected direction**: (i) steering > prompt on granularity; (ii) accuracy preservation cleared for steering at operating α.

**Resources**: model + dataset as in C3; prompt-engineering baseline = a natural-language instruction ("Please express more uncertainty …" / etc.) + a Thinking Intervention–style token-insertion baseline as a second competitor; downstream accuracy = final-answer accuracy on the 500-task benchmark (LLM-judged where free-form, exact-match where computable).

**Status**: pending verification.

**Notes**: `task.md` compares "finer-grained than prompt engineering while preserving downstream reasoning accuracy". The pre-cutoff literature identifies Thinking Intervention (Wu+2503.24370) as the strongest prompt-token control competitor — we add it explicitly to strengthen the comparison, but the *plain natural-language instruction* baseline is the primary comparator named in task.md.

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering C1–C4).
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` — milestones M1 (Location, C1), M2 (small-pool, C2), M3 (α dose-response, C3), M4 (steering vs. prompt engineering + Thinking Intervention, C4). Direction chain = Location → Causal Intervention → Tuning & Editing.
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`.

## Next Steps

- [ ] `/mechanism-skills` to route the extraction / intervention approach to a concrete mechanism family + submethod (Workflow 1.25) — expected to land on CAA-style mean-difference under Location + activation-steering under Causal Intervention.
- [ ] `/auto-experiment` to implement + run the verification suite (Workflow 1.5).
- [ ] `/auto-verify` to stress-test each verified claim under method / dataset / model swaps (Workflow 1.75) — verify variants named in `task.md`: DeepSeek-R1-Distill-Qwen-1.5B / -14B; MATH / GSM8K / AIME.
- [ ] `/auto-iteration-loop` to iterate until reviewer-ready (Workflow 2).
