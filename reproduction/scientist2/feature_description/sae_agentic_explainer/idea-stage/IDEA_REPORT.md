# Captured-Behavior Report

**Direction**: task.md — an iterative propose-test-revise agentic pipeline (SAGE) for natural-language explanation of SAE features in LLMs, benchmarked against Neuronpedia's reference explanations on Gemma-2-2B + gemmascope-res-16k on both generative accuracy and predictive accuracy, across multiple open-source LLM+SAE pairs and across early-to-late layers.
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: task.md (faithful capture)
**Date**: 2026-07-14
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline

**Reproduction policy**: The reproduction target is `task.md` itself. The upstream reference paper (arXiv 2511.20820) and any post-cutoff (≥ 2511) SAE preprints are on the project `forbidden-urls` list and are NOT consulted; the four-role Explainer/Designer/Analyzer/Reviewer scaffold, the two metrics (generative / predictive accuracy), and the model / SAE / dataset bindings all come from `task.md` verbatim.

---

## Executive Summary

The user hands us one composite behavioral claim about **SAGE**: an iterative propose-test-revise multi-agent pipeline for SAE-feature natural-language explanation. The claim, read faithfully, decomposes into **four measurable sub-claims** (C1-C4) that a single unified verification suite covers. The unified suite runs SAGE and Neuronpedia's reference explanations head-to-head on the same feature ids of the same SAE checkpoint, and evaluates *each* explanation with both *generative accuracy* (probe text written from the explanation must trigger the target feature) and *predictive accuracy* (explanation-based activation predictions correlate with ground-truth activations on held-out text). The mechanism-stage strategy is **Unit Interpretation** (per `/mechanism-explore` Direction 5): the pipeline decodes the meaning of an internal unit — a sparse-autoencoder feature — via model-explains-model auto-interpretation. The concrete mechanism submethod / family is routed by the experiment stage.

---

## Literature Landscape

Full landscape in `idea-stage/LANDSCAPE.md`. Key context for verification:

- **Incumbent to beat**: Neuronpedia's auto-interp for `gemmascope-res-16k` (one-shot LLM-as-explainer descriptions of Gemma-Scope features, hosted at neuronpedia.org).
- **Methodological ancestors**: Bills et al. 2023 (simulation scoring), Paulo et al. 2024 (arXiv 2410.13928 — detection + intervention scoring, open-source pipeline, direct academic ancestor of the two SAGE metrics).
- **Bound SAE checkpoint**: Lieberum et al. 2024 (arXiv 2408.05147 — Gemma Scope, JumpReLU, 16k residual-stream SAE).
- **Verify-variant SAEs** are pointed at by the transcoder line (Paulo et al. 2025, arXiv 2501.18823 — motivates `transcoder-hp` on Qwen3) and by open post-training SAE work (GPT-OSS-20B `resid-post-aa`).

---

## Resources (binding scope — per `task.md`)

Main-experiment resources are stated per stage and MUST NOT be silently substituted:

- **Main-experiment target LLM**: **Gemma-2-2B** (`google/gemma-2-2b`, base variant)
- **Main-experiment SAE**: **`gemmascope-res-16k`** (Gemma-Scope JumpReLU residual-stream SAE, 16k dictionary width, one SAE per residual-stream layer)
- **Main-experiment reference-explanation source (= baseline being compared against)**: **Neuronpedia** — the same feature ids in the same SAE, using Neuronpedia's public auto-interp explanations
- **Verify-stage candidate LLM+SAE pairs (use as needed, not necessarily all)**:
  - **Qwen3-4B** with SAE **`transcoder-hp`**
  - **GPT-OSS-20B** with SAE **`resid-post-aa`**
- **Agent backbone (fixed, non-swappable across all roles)**: **GPT-5** (via the DMXAPI endpoint declared in `task.md`) in Explainer / Designer / Analyzer / Reviewer roles
- **Datasets**: none beyond Neuronpedia's feature-level activation database + standard pretraining-corpus text pools referenced by Neuronpedia (Neuronpedia's own activation-example dataset serves as the "held-out text" pool for both metrics)
- **GPU budget**: 10 GPU-hours total across main-experiment + verify + iteration. Devices allowed: `{1, 2, 3, 5, 6}` only.
- **Filesystem allowlist**: working dir + `/data/zhenqian/data` + `/data/zhenqian/models`.

---

## Claims to Verify

The user's `task.md` states a single composite claim ("*You should build a pipeline around the following basic idea — [SAGE description] — produces feature explanations that outperform Neuronpedia on both generative accuracy and predictive accuracy, across multiple open-source LLMs and across early-to-late layers.*"). Read faithfully, this bundles four independently testable predicates. Splitting them does not change the meaning; it makes each measurable in a single milestone.

### Claim C1: Generative-accuracy improvement on the main LLM+SAE pair

**Original (verbatim excerpt from `task.md` §Claim):**
> "produces feature explanations that outperform Neuronpedia on … generative accuracy (success rate of triggering the feature with text written from the explanation) …"

**Extracted statement**: On Gemma-2-2B + `gemmascope-res-16k`, SAGE's explanations achieve **higher generative accuracy** than Neuronpedia's reference explanations on the same feature ids, over a randomly-sampled feature set drawn from residual-stream layers spanning early-to-late depth.

**Hypothesis**: H1 — Iterative activation-grounded revision produces explanations more causally aligned with the feature's true activating conditions than one-shot description, so text written from an SAGE explanation triggers the target feature more reliably than text written from a Neuronpedia explanation.

**Measurable predicate**: For each sampled Gemma-Scope 16k feature f, let `probe_SAGE(f)` and `probe_Neuronpedia(f)` be N probe texts each, generated by an *independent* judge-LLM instructed to write text that instantiates the given explanation for f (blind to source). Push each probe through Gemma-2-2B, read `gemmascope-res-16k`'s activation of feature f, and count a **hit** iff activation > τ (τ = the top-1% activation threshold of f on Neuronpedia's own reference corpus). Generative accuracy = hits / N. The predicate is: mean_f [ GenAcc(SAGE, f) - GenAcc(Neuronpedia, f) ] > 0, with paired-feature significance (paired bootstrap / paired Wilcoxon, α = 0.05) and 95% CI reported.

**Expected direction**: up (SAGE > Neuronpedia).

**Resources**: LLM = Gemma-2-2B; SAE = `gemmascope-res-16k`; features drawn from residual-stream layers at three depths (early ≈ L4, mid ≈ L12, late ≈ L20 of 26); N_features (initial plan): 100 per layer (300 total); N_probe per feature per method: 5; judge LLM: GPT-5 (task.md agent backbone) with a fixed non-role-specific probe-writing prompt applied identically to both methods. **Full-scale plan under the 10-GPU-hour budget**: N_features = 100/layer, N_probe = 5; downscale (fewer features per layer, or fewer probes per feature) only if a scaling pilot on the same setup shows the budget cannot accommodate the full plan (see EXPERIMENT_PLAN.md `method_sensitive` fields).

**Status**: pending verification.

**Verified by milestone(s)**: M1 in `refine-logs/EXPERIMENT_PLAN.md`.

**Notes**: Faithful capture — `task.md` explicitly defines generative accuracy as *"success rate of triggering the feature with text written from the explanation"*. The τ threshold, judge-LLM identity, and paired-feature statistics are chosen from landscape (Paulo et al. 2024 intervention scoring convention) — they do not tighten the claim, they operationalize it.

---

### Claim C2: Predictive-accuracy improvement on the main LLM+SAE pair

**Original (verbatim excerpt from `task.md` §Claim):**
> "… and predictive accuracy (correlation between explanation-based predictions and true activations on held-out text) …"

**Extracted statement**: On Gemma-2-2B + `gemmascope-res-16k`, SAGE's explanations achieve **higher predictive accuracy** than Neuronpedia's reference explanations on the same feature ids, on held-out text drawn from Neuronpedia's activation corpus but **not** used to construct either explanation.

**Hypothesis**: H2 — Iterative revision closes the "top-activation blind spot" (Oikarinen & Weng 2024) by forcing the explanation to survive probes across the full activation range, so an explanation-conditioned scorer predicts held-out activations better than one calibrated only on top-k snippets.

**Measurable predicate**: For each sampled feature f, hand Neuronpedia's held-out text set X_test(f) — the subset of Neuronpedia's activation corpus for f whose top-activating snippets were NOT shown to either explainer — to a scorer LLM. The scorer receives the explanation e and each text x ∈ X_test(f), and predicts activation ŷ(x | e) ∈ [0, 1] (Paulo 2024's detection-scoring / simulation-scoring protocol). Ground-truth activation is `gemmascope-res-16k`'s activation of f on x, min-max normalized within f. Report per-feature Pearson correlation ρ_f (SAGE) and ρ_f (Neuronpedia). Predicate: mean_f [ ρ_f(SAGE) - ρ_f(Neuronpedia) ] > 0, paired-bootstrap / paired-Wilcoxon α = 0.05.

**Expected direction**: up (SAGE > Neuronpedia).

**Resources**: same as C1 (shared features across C1 / C2 to enable per-feature paired analysis). Scorer LLM: GPT-5.

**Status**: pending verification.

**Verified by milestone(s)**: M1 in `refine-logs/EXPERIMENT_PLAN.md` (shares runs with C1 — both metrics computed on the same feature sample in one pass).

**Notes**: Faithful capture — `task.md` explicitly defines predictive accuracy as *"correlation between explanation-based predictions and true activations on held-out text"*. Held-out means: not exposed to the explainer during explanation generation; Neuronpedia's activation corpus provides both the training pool (top-activating examples visible to Neuronpedia's baseline) and the held-out pool (rest of the corpus).

---

### Claim C3: Layer-depth generalization (early-to-late layers)

**Original (verbatim excerpt from `task.md` §Claim):**
> "… across … early-to-late layers."

**Extracted statement**: The generative- and predictive-accuracy gains of SAGE over Neuronpedia (per C1 / C2) hold **at each of the three layer depths sampled** (early, mid, late), not only in aggregate — no depth is a silent driver of the pooled gain, no depth silently reverses the gain.

**Hypothesis**: H3 — Because SAGE's revision loop targets a feature's activating conditions rather than a layer-specific stylistic artifact, its gain over Neuronpedia is qualitatively preserved across residual-stream depth.

**Measurable predicate**: The layer-conditioned gains — mean_{f in Layer_L} [ GenAcc(SAGE, f) - GenAcc(Neuronpedia, f) ] and mean_{f in Layer_L} [ ρ_f(SAGE) - ρ_f(Neuronpedia) ] — are strictly positive at each of the three depths (early / mid / late), with paired-bootstrap 95% CI lower bound > 0 at each depth. Additionally, both effects are individually significant per depth (α = 0.05, Bonferroni across 3 layers × 2 metrics = 6 tests).

**Expected direction**: up at every depth (no depth-reversal).

**Resources**: same run as C1/C2, stratified by layer.

**Status**: pending verification.

**Verified by milestone(s)**: M1 (same run; layer-stratified read-off).

**Notes**: Faithful capture. `task.md`'s phrase "early-to-late layers" is what forces the stratified analysis rather than accepting only a pooled headline number. A pooled-only report would leave open whether the gain is driven by one depth.

---

### Claim C4: Cross-LLM+SAE-pair generalization

**Original (verbatim excerpt from `task.md` §Claim):**
> "… across multiple open-source LLMs …"

**Extracted statement**: The generative- and predictive-accuracy gains of SAGE over Neuronpedia (per C1 / C2) hold **on at least one additional LLM+SAE pair beyond the main Gemma-2-2B / gemmascope-res-16k pair**, drawn from `task.md`'s verify-variant candidate pool (Qwen3-4B + `transcoder-hp`; GPT-OSS-20B + `resid-post-aa`), on comparable feature samples.

**Hypothesis**: H4 — The iterative propose-test-revise design is not specific to JumpReLU residual-stream SAEs on Gemma; it is a general improvement over one-shot description that also holds on transcoder-style and post-residual SAEs, indicating the gain is pipeline-attributable, not architecture-attributable.

**Measurable predicate**: On at least one verify LLM+SAE pair from the candidate pool, mean_f [ GenAcc(SAGE, f) - GenAcc(reference, f) ] > 0 AND mean_f [ ρ_f(SAGE) - ρ_f(reference, f) ] > 0, both with paired 95% CI lower bound > 0. "Reference" is whichever pre-existing baseline auto-interp is available for that pair (Neuronpedia if hosted; else a Neuronpedia-style one-shot GPT-5 baseline built with a matched prompt inside this project, so the comparison is like-for-like against a single-pass explainer of the same backbone strength).

**Expected direction**: up (SAGE > reference).

**Resources**: at least one of {Qwen3-4B + `transcoder-hp`, GPT-OSS-20B + `resid-post-aa`}; feature sample scaled to fit remaining GPU budget after M1 (the experiment stage picks which of the two pairs, honoring the 10-hour GPU budget and the {1,2,3,5,6} device allowlist). This claim is what the `/auto-verify` swap-variant stage checks by design; the reproduction main experiment sets up M2 as the *first* verify pair so a positive C4 sub-result is on the table before /auto-verify runs.

**Status**: pending verification.

**Verified by milestone(s)**: M2 in `refine-logs/EXPERIMENT_PLAN.md`; additionally hardened by `/auto-verify`'s cross-model swap variants downstream.

**Notes**: Faithful capture. `task.md`'s "across multiple open-source LLMs" language, together with the Verify-stage variant list, makes cross-pair generalization an explicit claim, not just a "nice to have". Whether we run one or both verify variants inside the main experiment vs. deferring the second to `/auto-verify` is a **testing-method choice**, not a claim-content choice — it is decided in `EXPERIMENT_PLAN.md` under the 10-GPU-hour budget.

---

## Unified plan covers: 4 claim(s)

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering C1-C4)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies)

## Next Steps

- [ ] `/mechanism-skills` to route the SAGE pipeline's underlying mechanism family + submethod. The captured direction is **Unit Interpretation** (per `/mechanism-explore` Direction 5) — model-explains-model auto-interpretation on top of a sparse-autoencoder dictionary — with **no direct causal intervention step in the pipeline**. The experiment stage's Phase 1.5 will bind the concrete submethod, filling in the `method_sensitive` fields.
- [ ] `/auto-experiment` to implement and run the verification suite (Workflow 1.5).
- [ ] `/auto-verify` to stress-test each verified claim under method/dataset/model swaps (Workflow 1.75).
- [ ] `/auto-iteration-loop` to iterate the verification suite until reviewer-ready (Workflow 2).
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain.
