# Captured-Behavior Report

**Direction**: (empty — behavior/claims sourced entirely from `task.md`)
**Behavior-source**: given
**Mechanism**: discovery
**Date**: 2026-07-15
**Pipeline**: research-lit → faithful behavior capture (from task.md) → research-refine-pipeline

## Executive Summary

Five claims from `task.md` about sparse-autoencoder (SAE) interpretability of ESM-2-650M protein language model residual-stream activations must be verified faithfully. They partition into: (i) two per-layer scale claims (Claims 1–2 — feature count ~2,548 per layer; concept alignment ~143 vs. neuron ~46 with ~15 cleanly recovered), (ii) one theoretical inference (Claim 3 — the SAE-vs-neuron gap = evidence of superposition), (iii) one novel-concept discovery claim (Claim 4 — features absent from Swiss-Prot annotations, surfaced by an external LLM auto-interpreter over top-activating protein contexts), and (iv) one downstream utility claim with two sub-claims (Claim 5 — annotation-filling and steered sequence generation). Mechanism strategy: **Unit Interpretation → Decision Auditing → Causal Intervention** (from `/mechanism-explore`). No mining, no ideation, no novelty-check, no M0 gate — `BEHAVIOR_SOURCE=given` takes each claim as-is; the experiment plan tests all five under a unified proposal.

## Literature Landscape

See `idea-stage/LANDSCAPE.md` — grounding for the mechanism strategy comes from canonical pre-cutoff SAE / mechanistic-interpretability work (Elhage 2022 superposition; Cunningham 2023 & Bricken 2023 & Templeton 2024 SAE-on-residual-stream; Gao 2024 top-k SAEs; Bills 2023 LLM auto-interpretation; Rives 2021 & Lin 2023 ESM protein embeddings; Turner 2023 activation steering). External retrieval this run returned only policy-voided content (forbidden reference paper + arxiv-cutoff 2412 blocking nearly all post-cutoff SAE-on-PLM literature); the landscape's role is only to inform the mechanism strategy and specificity controls, not to score novelty (`BEHAVIOR_SOURCE=given` skips novelty checks).

## Resources (global)

All five claims share these fixed resources from `task.md`. Recorded once here and referenced per-claim below.

- **Model (main experiment)**: `ESM-2-650M` — local: `$MODEL_DIR/ESM-2-650M/`. All SAE evaluation runs on its per-layer residual-stream activations.
- **Dataset (main experiment)**: `Swiss-Prot` (UniProtKB / Swiss-Prot expertly-curated protein annotations) — local: `$DATA_DIR/Swiss-Prot/` (`train.parquet`, `valid.parquet`, `test.parquet`, `uniprot_sprot.fasta.gz`, `uniprot_sprot.dat.gz`). Ground-truth concept dictionary; per-residue category labels for binding sites, active sites, sequence motifs, structural / functional domains, PTM sites.
- **Fixed pretrained SAE checkpoints** (do NOT retrain): `$MODEL_DIR/InterPLM-esm2-650m/layer_{1,9,18,24,30,33}/{ae_normalized.pt,ae_unnormalized.pt,config.json}`. Six layers of ESM-2-650M's 33 layers, spanning early / mid / late. SAE hyperparameters (width, sparsity target, activation function) are fixed by the checkpoint.
- **Auto-interpretation LLM**: external endpoint `gpt-5.4` at `https://www.dmxapi.cn/v1` (bypass proxy; API key in `task.md`). Used to (a) score feature-label coherence over top-activating protein contexts and (b) auto-label novel-concept features (Claim 4).
- **Fixed concept vocabulary**: Swiss-Prot annotation categories — binding sites, active sites, sequence motifs, structural / functional domains, PTM sites. Do not redefine.
- **UniRef** (auxiliary, for top-activating-context sampling): `$DATA_DIR/UniRef/data/` — an unlabeled protein sequence source over which SAE features are activated to identify top-activating windows. Not used to (re)train SAEs (they are fixed); only for sampling activation contexts at scale beyond Swiss-Prot's coverage.
- **Compute budget**: 10 GPU-hours total; GPUs restricted to ids {0, 1, 2, 3}; conda env.
- **Verify-stage swap candidates (not part of Phase 4.5 plan, forwarded to /auto-verify)**: `ESM-2-8M`, `ESM-2-35M`, `ESM-2-150M`; auxiliary SAEs for the 8M model at layers 1–6 (`$MODEL_DIR/InterPLM-esm2-8m/layer_{1..6}/`).

_Resource-fidelity marker_: `resource_fidelity: strict` is **NOT** stamped this run (stamped only for the reproduction combo `given` + `given`). This combo is `given` + `discovery`, so resources are cost-aware — but the 10-hour budget declared in `task.md` is more than sufficient for the specified ESM-2-650M + Swiss-Prot + six pretrained SAE layers evaluation at full scale, so the plan uses these at full spec and does not downscale.

## Claims to Verify

### Claim 1: Feature count — SAE surfaces up to ~2,548 interpretable latent features per layer, orders of magnitude more than raw neurons

**Original (verbatim excerpt from task.md):**
> SAEs trained on the residual-stream activations of ESM-2 layers surface up to ~2,548 interpretable latent features per layer — orders of magnitude more concepts than can be pulled out of individual neurons.

**Extracted statement**: For at least one layer L in the six evaluated layers of ESM-2-650M {1, 9, 18, 24, 30, 33}, the count of *interpretable* SAE latent features (features passing an auto-interp / activation-quality gate) approaches ~2,548, and this per-layer interpretable-feature count is at least an order of magnitude larger than the count of interpretable individual neurons at the same layer.

**Hypothesis**: H1 — the pretrained SAE at (at least one of) the six ESM-2-650M layers yields ~2,548 features that pass an interpretability gate (density in normal range, non-dead, auto-interp score above a low-activation baseline), whereas raw neurons at that layer pass the same gate in numbers ≤ ~254 (order-of-magnitude smaller).

**Measurable predicate**: For each layer L in {1, 9, 18, 24, 30, 33}, load the fixed pretrained SAE (`$MODEL_DIR/InterPLM-esm2-650m/layer_L/ae_normalized.pt`), compute per-feature activation statistics over Swiss-Prot residues (density, dead-fraction, ultra-low-density fraction per Gao 2024) and per-feature auto-interp score (LLM label from top-activating contexts, LLM scoring on held-out contexts; score ≥ τ_auto). Compute the same for raw neurons at layer L (each hidden-dim treated as a "feature"). Report per-layer count of features that pass all three gates, and the same for neurons. `SAE_interp_count(L*) ≥ 2000` for some L*, and `SAE_interp_count(L*) / neuron_interp_count(L*) ≥ 10`.

**Expected direction**: up (SAE » neurons); the SAE per-layer interpretable-count approaches 2,548 at the best layer.

**Resources**: model: ESM-2-650M ($MODEL_DIR/ESM-2-650M/); dataset: Swiss-Prot ($DATA_DIR/Swiss-Prot/) for scoring, UniRef ($DATA_DIR/UniRef/data/) for top-activating context sampling; SAEs: $MODEL_DIR/InterPLM-esm2-650m/layer_{1,9,18,24,30,33}/; auto-interp LLM: gpt-5.4 endpoint; used_n: full test split of Swiss-Prot + UniRef sample for top-context activation search (see Phase 4.5 for exact size).

**Status**: pending verification

**Notes**: The ~2,548 number is the *per-layer* count reported in `task.md`; we do not require it to hold at *every* layer, only that "surfaces up to" is met by at least one layer. This matches the "up to" phrasing. The 10× SAE/neuron gap is the derived comparison the language "orders of magnitude" locks in.

### Claim 2: Concept alignment gap — SAE features align with up to ~143 distinct Swiss-Prot concepts vs. raw ESM-2 neurons ~46 (~15 clean)

**Original (verbatim excerpt from task.md):**
> These SAE features align with up to ~143 distinct Swiss-Prot biological concepts (binding sites, active sites, sequence motifs, structural / functional domains), whereas raw ESM-2 neurons align with only ~46 concepts on the same evaluation, of which only ~15 are cleanly recovered.

**Extracted statement**: Under a per-(feature, concept) F1 alignment score at the residue level with a fixed threshold τ_F1, the SAE dictionary (union across evaluated layers, one feature-per-concept assignment) covers up to ~143 distinct Swiss-Prot concept categories; raw ESM-2-650M neurons (same layers, same τ_F1) cover up to ~46 concepts, of which only ~15 pass a stricter "clean" threshold τ_clean > τ_F1.

**Hypothesis**: H2 — using an alignment protocol matched to `task.md`'s (per-residue F1 between the binarized top-activation mask of each unit and the per-residue concept-annotation mask; concept is "covered" if at least one unit's F1 ≥ τ_F1), the counts are approximately: SAE ~143, neurons ~46 covered / ~15 clean. Exact τ_F1 and τ_clean are to be pinned in Phase 4.5 with sensitivity reporting.

**Measurable predicate**: Define concept universe C = {Swiss-Prot per-residue annotation categories, i.e. binding site, active site, sequence motif class × sub-class, PROSITE / InterPro / Pfam domain family, PTM site class}. For each layer L in {1, 9, 18, 24, 30, 33}: over the Swiss-Prot test split residues, for each SAE feature f and each concept c, compute per-residue F1(f, c) = 2·P·R / (P + R) where the feature-positive mask is "activation of f at residue r above quantile q_top", the concept-positive mask is Swiss-Prot label for c at residue r. A concept c is "covered by SAE at layer L" iff ∃ feature f with F1(f, c) ≥ τ_F1. `count_covered_SAE = |{c : c covered at any layer L}|`. Same for neurons (each hidden-dim treated as a unit). Assert `count_covered_SAE ≥ 100` and `count_covered_SAE / count_covered_neurons ≥ 2.5×`, with the specific target ~143 / ~46 / ~15 reported.

**Expected direction**: up (SAE » neurons on both "covered" and "clean" counts).

**Resources**: as Claim 1; used_n: full Swiss-Prot test split (per-residue evaluation across all sequences with per-residue annotations).

**Status**: pending verification

**Notes**: `task.md` specifies "same evaluation" — this locks the SAE-vs-neuron comparison to identical F1 threshold, quantile, and residue set. Any degrees of freedom (τ_F1, q_top, whether to union across layers or pick the best layer) are protocol choices for Phase 4.5, but must be *identical* between the two arms.

### Claim 3: Superposition — the SAE-vs-neuron gap is direct evidence that PLMs encode biological concepts in superposition, not in single units

**Original (verbatim excerpt from task.md):**
> The gap between SAE features and raw neurons is direct evidence that PLMs encode biological concepts in superposition rather than in single units.

**Extracted statement**: The SAE-vs-neuron concept-coverage gap measured in Claim 2, when accompanied by specificity controls that rule out alternative explanations (metric-artifact, per-unit-capacity, and shuffled-feature baselines), constitutes evidence that ESM-2-650M encodes multiple biological concepts per residual-stream dimension via superposition (nearly-orthogonal directions in a `d`-dimensional space encoding `k > d` concepts), rather than concept-per-neuron encoding.

**Hypothesis**: H3 — with specificity controls in place, the SAE-vs-neuron gap survives; matched-capacity controls that lack the sparse-decomposition step (random orthogonal basis rotation of hidden state; shuffled feature activations) do not achieve the SAE's concept coverage; therefore superposition is the parsimonious explanation.

**Measurable predicate**: Compute Claim 2's coverage metric for three additional arms — (a) a random orthogonal rotation of the residual stream (matched dimensionality to raw neurons); (b) a "PCA basis" arm (top-`d` PCA components of residual-stream activations); (c) a shuffled-activation SAE arm (SAE features whose per-residue activations are shuffled across residues, breaking any residue-alignment signal). Superposition claim holds iff SAE coverage strictly exceeds each control by a margin ≥ Δ (target Δ pinned in Phase 4.5), while random rotation and PCA remain closer to raw-neuron coverage. Report ordered coverage counts across the six arms.

**Expected direction**: SAE > PCA ≈ random-rotation ≈ raw-neuron > shuffled-SAE.

**Resources**: as Claim 2; adds compute for PCA fit and random-rotation baseline (both cheap — closed-form / one-pass over Swiss-Prot training-split activations).

**Status**: pending verification

**Notes**: `task.md` says the gap "is direct evidence" — a faithful reproduction cannot just report the gap and stop; the specificity controls are what license the causal-inferential leap from *gap* to *superposition*. Phase 4.5 must include the three control arms above (matched-capacity rotation, PCA, shuffled SAE) as part of the plan for Claim 3, tagged as `method_sensitive: [Δ threshold]`.

### Claim 4: Novel-concept discovery — a subset of SAE features corresponds to coherent biological concepts absent from Swiss-Prot, surfaced by an external LLM auto-interpreter over top-activating protein contexts

**Original (verbatim excerpt from task.md):**
> A subset of the SAE features corresponds to coherent biological concepts that are absent from existing annotation dictionaries; these can be surfaced by using an external LLM as an auto-interpreter over top-activating protein contexts.

**Extracted statement**: Among SAE features that fail to align with any Swiss-Prot concept at the τ_F1 threshold used in Claim 2, a non-trivial fraction (target: ≥ 10%, exact number pinned in Phase 4.5) nevertheless receive a coherent natural-language label from the external LLM auto-interpreter, where "coherent" means (a) the label passes the LLM-explain-then-LLM-score protocol above a random-baseline auto-interp score, (b) the label is *not* a paraphrase of any Swiss-Prot concept in the fixed vocabulary (checked by LLM synonymy classification), and (c) the label survives a low-activation-baseline check (a random-feature sanity control).

**Hypothesis**: H4 — a measurable, replicable subset of "Swiss-Prot-unaligned SAE features" produce coherent, non-vocabulary labels under the auto-interpretation loop, evidencing concepts the annotation dictionary does not currently catalogue.

**Measurable predicate**: Restrict to features f with `max_c F1(f, c) < τ_F1` (i.e., Swiss-Prot-unaligned). For each such f, sample top-N activating residue windows from UniRef (context sampling), prompt the LLM auto-interpreter (`gpt-5.4`) with the top-N contexts + K low-activation contrasts, elicit a natural-language label + short justification. Score the label by (i) LLM-scored predictivity on held-out contexts (auto-interp score ≥ τ_auto), (ii) LLM synonymy classification against the fixed Swiss-Prot vocabulary (label is *not* a synonym), (iii) low-activation-baseline / random-feature control (a random-permuted-feature control fails to receive a coherent label at the same rate). A feature is "novel-concept-coherent" iff (i) ∧ (ii) ∧ (iii). Report `count_novel = |{novel-concept-coherent f}|` and `count_novel / count_unaligned`.

**Expected direction**: `count_novel > 0`, and materially larger than the random-feature control's coherent-label rate.

**Resources**: as Claim 1; LLM budget explicit — Phase 4.5 pins per-feature prompt count and total LLM-call budget. UniRef sampling is required for top-activating contexts beyond Swiss-Prot's coverage.

**Status**: pending verification

**Notes**: The specificity controls (LLM-scored predictivity above baseline, synonym-check against vocabulary, low-activation control) are load-bearing — without them, the LLM's confident-hallucination floor makes almost any set of contexts get *some* label. `task.md`'s wording ("coherent biological concepts") requires the coherence check.

### Claim 5: Downstream utility — the feature dictionary supports (5a) filling missing Swiss-Prot annotations and (5b) steering ESM-2 sequence generation toward a target biological property

**Original (verbatim excerpt from task.md):**
> The extracted feature dictionary is practically useful: it supports filling in missing Swiss-Prot annotations and steering ESM-2 sequence generation toward a target biological property.

**Extracted statement**: Two independently verifiable sub-claims:
- **5a** (annotation-filling): A simple predictor using SAE feature activations as inputs (e.g. per-feature threshold or a lightweight classifier over the sparse code) predicts held-out Swiss-Prot annotations at above-baseline precision-recall; the SAE-feature-based predictor outperforms a matched-capacity raw-neuron-based predictor on the same held-out set.
- **5b** (steering): Clamping a single SAE feature that is auto-labeled with a target biological property, during ESM-2 sequence generation (masked-token sampling / iterative fill), yields generated sequences enriched for the target property under an independent property-verification test, above (i) a no-steering baseline, (ii) a random-direction / random-feature baseline of the same magnitude, and with (iii) a monotone dose-response curve, and (iv) preserved sequence plausibility (ESM-2 pseudo-perplexity within an acceptable band).

**Hypothesis**: H5 — the SAE dictionary yields (a) an annotation-filling predictor that beats raw-neuron matched-capacity baseline on held-out Swiss-Prot categories, and (b) a steering procedure whose dose-response and specificity beat mean-activation-addition and random-feature-clamp baselines while keeping generated sequences plausible.

**Measurable predicate**:
- 5a: For each concept c in a target subset of Swiss-Prot categories (say the top-K by prevalence in the test split), train a minimal linear probe over the SAE sparse code (features from the best-covering layer, per Claim 2) to predict per-residue c-membership on the Swiss-Prot train split; evaluate PR-AUC on the Swiss-Prot test split. Same protocol for raw-neurons (input: hidden-dim activations at same layer, same capacity). Assert `mean_c PR-AUC(SAE) > mean_c PR-AUC(neurons)` with a statistically-significant margin (paired test across concepts, target p < 0.05).
- 5b: Pick M SAE features whose auto-interp labels name concrete, measurable target properties (candidates: signal-peptide presence, transmembrane-domain presence, zinc-binding motif, N-glycosylation site — all with an independent, ideally rule-based or classifier-based property checker external to SAE features). For each such feature f: generate N sequences under (i) no steering, (ii) clamp f to α·σ_f (dose α ∈ {0.5, 1, 2, 4}), (iii) mean-residual activation-addition baseline (mean over c-positive residues minus mean over c-negative residues), (iv) random-feature clamp at matched magnitude. Score generated sequences with the external property checker (yield: fraction of generated sequences exhibiting property c) and with ESM-2 pseudo-perplexity (plausibility band). Assert `yield(SAE-clamp at α*) > yield(baselines)` with monotone dose-response in α and pseudo-perplexity within band.

**Expected direction**: 5a — SAE > neurons on annotation-filling PR-AUC; 5b — SAE-clamp > mean-add-baseline > no-steering ≈ random-clamp; monotone dose-response; plausibility preserved.

**Resources**: as Claim 1 for 5a; 5b additionally needs (i) an ESM-2 sequence-generation loop (masked-token or iterative-refinement), (ii) property checkers (external classifiers or rule-based detectors — e.g. SignalP-compatible input, TM-region rule from hydrophobicity + charge, ProSite motif regex for the chosen concepts). Phase 4.5 pins the exact property-checker sources.

**Status**: pending verification

**Notes**: This is the most compute-heavy claim (steering requires many generations across many magnitudes and features). Phase 4.5 gates the exact `M`, `N`, and generation length against the 10-hour GPU budget. Sub-claim 5a and 5b are separately reportable — if 5a passes and 5b partially passes, the claim is `partially supported` rather than falsified. The plausibility band for pseudo-perplexity is a specificity control against reward-hacking (a steered generation that produces a trivially-satisfying sequence such as poly-Cys has "won" on yield but lost on plausibility).

## Mechanism strategy (from /mechanism-explore)

**Chain, in execution order**: **Unit Interpretation → Decision Auditing → Causal Intervention**.

- **Unit Interpretation** (Claims 1, 2, 4): dictionary decomposition via the fixed pretrained SAEs on ESM-2-650M residual streams; model-explains-model auto-interpretation via the external LLM over top-activating protein-residue-window contexts.
- **Decision Auditing** (Claims 2, 4, 5a): audit the concept coverage of SAE features vs. raw neurons against the fixed Swiss-Prot vocabulary (validation) and surface features that use novel decision bases (discovery, Claim 4); use the audit to fill missing annotations (Claim 5a).
- **Causal Intervention** (Claim 5b): clamp labeled SAE features during ESM-2 sequence generation; report sign, dose-response, and specificity vs. random-feature and mean-activation-addition baselines.

**Directions deliberately not pursued**:
- **Location** — pre-committed by `task.md` (fixed six-layer set + fixed pretrained SAEs); nothing to locate.
- **Tuning & Editing** — explicit user constraint (no ESM-2 tuning); inference-time steering is captured under Causal Intervention.
- **Formation Tracing** — explicit user constraint (no re-training, no pretraining trace); SAE checkpoints are fixed resources.

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified verification approach covering all five claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies; `mechanism_strategy:` chain stamped in top metadata)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps
- [ ] `/mechanism-skills` to route Unit Interpretation → Decision Auditing → Causal Intervention to concrete method families + submethods (Workflow 1.25) — the family and submethod are chosen at the experiment-stage routing step, not here.
- [ ] `/auto-experiment` to implement and run the verification suite (Workflow 1.5), inheriting the fixed pretrained SAEs, the ESM-2-650M model, and Swiss-Prot as bindings.
- [ ] `/auto-verify` to stress-test each verified claim under model swaps (ESM-2-8M / 35M / 150M as scale-generalization candidates from `task.md`'s verify-stage list) — Workflow 1.75.
- [ ] `/auto-iteration-loop` for a review-driven iteration cycle if any claim comes back INCONCLUSIVE — Workflow 2.
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain.
