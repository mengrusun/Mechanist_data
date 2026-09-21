# Captured-Behavior Report

**Direction**: (empty — behavior/claims sourced from `task.md`)
**Behavior-source**: given
**Mechanism**: discovery
**Date**: 2026-07-15
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline

## Executive Summary

Four claims about **Representation-Level Circuit Breakers (Representation Rerouting, RR)** as a safety intervention for instruction-tuned LLMs are captured verbatim from `task.md` and refined into a unified verification suite. Claim 1 asserts the *existence* of an internally-identifiable harmful subspace and the *feasibility* of rerouting it with only paired benign/harmful data; Claim 2 asserts *superiority* of the RR-trained model over refusal- and adversarial-training baselines on unseen-attack ASR under HarmBench (with capability preserved on MT-Bench / MMLU); Claims 3 and 4 assert *cross-modality transfer* to multimodal LLMs (image-hijack defense) and LLM agents (tool-use safety). The plan runs the primary Llama-3-8B-Instruct RR fine-tune + HarmBench evaluation on the experiment side (Claims 1, 2), and the transfer + capability evaluations on the verify side (Claims 3, 4, plus MT-Bench / MMLU / BFCL). First runs: (i) reproduce baseline ASR on Llama-3-8B-Instruct without RR under HarmBench, (ii) train the RR fine-tune on the GraySwanAI training set, (iii) probe the RR-tuned model for the harmful subspace and re-measure ASR.

## Literature Landscape

See `idea-stage/LANDSCAPE.md` — pre-June-2024 foundational context only (target paper is embargoed under project blind-reproduction policy). Used strictly for baselines / datasets / metric definitions; never used to alter the claims.

**Retrieval blocker (honest report):** external retrieval delivered no citable paper metadata this run — the target paper, its repo, and the RR / Cygnet checkpoints are on the project forbidden-URL list (respected), and the post-search filter voids any web-search response that surfaces arxiv IDs ≥ 2406 (June 2024). See `RESEARCH_LIT.md`.

## Claims to Verify

### Claim 1: Harmful representations are identifiable and can be rerouted with only paired benign/harmful data

**Original (verbatim excerpt from task.md):**
> Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable internal representations that can be rerouted to an orthogonal, non-harmful subspace using only paired benign/harmful data, without exposure to any attack prompts.

**Extracted statement**: In an instruction-tuned LLM, the internal representations that drive harmful outputs are (a) *identifiable* — i.e. a low-dimensional subspace in a residual-stream site can be distinguished on paired benign vs. harmful inputs — and (b) *reroutable* — i.e. a fine-tuning objective computed only from paired benign/harmful data (no attack prompts) reorients activations on harmful inputs toward a subspace orthogonal to the pre-intervention harmful subspace, while activations on benign inputs are largely preserved.

**Hypothesis**: H1 — There exists at least one residual-stream site (a subset of layers) where (i) a linear direction / low-rank subspace separates activations on harmful vs. benign inputs with above-chance margin before RR training, and (ii) after RR training this subspace is (approximately) rotated onto an orthogonal subspace on harmful inputs while benign inputs remain in place.

**Measurable predicate**: On the Llama-3-8B-Instruct model, before and after RR fine-tuning on the GraySwanAI training + retain set:
- (i) *Identifiability*: a linear probe / mean-difference direction trained on paired harmful vs. benign activations at chosen residual-stream layers has separation AUC > 0.8 pre-training (and remains meaningful post-training as a diagnostic).
- (ii) *Reroute*: cosine similarity between the post-training activation on a harmful prompt and the *pre-training* harmful direction (subspace) is significantly reduced (target: mean cosine ↓ ≥ 0.3 vs. pre-training baseline, or a comparable subspace-projection reduction), while the same metric on benign prompts is preserved within a small tolerance (mean cosine ↑↓ ≤ 0.1).

**Expected direction**: (i) up — clear separability of harmful vs. benign representations pre-training; (ii) down on harmful, ≈equal on benign — reroute is targeted.

**Resources**: model: Meta-Llama-3-8B-Instruct (`/data/zhenqian/models/Meta-Llama-3-8B-Instruct`); training data: Circuit-breaker training set + retain set from `github.com/GraySwanAI/circuit-breakers` (paired harmful / benign examples). used_n: full training + retain sets (task.md pins them "always used, fixed").

**Status**: pending verification

**Notes**: RR trains on paired benign/harmful data with **no attack prompt** exposure — the "no attack exposure" is a load-bearing sub-claim (differentiates RR from adversarial training / R2D2). Identifiability is checked pre-training as sanity, post-training as diagnostic; the rerouting property is the target claim.

### Claim 2: RR-tuned model beats refusal-training and adversarial-training baselines on unseen-attack ASR while preserving capability

**Original (verbatim excerpt from task.md):**
> A model fine-tuned with Representation Rerouting (RR) achieves substantially lower attack success rates than refusal-trained or adversarial-trained baselines across a wide range of unseen attacks, while preserving standard capability benchmarks.

**Extracted statement**: On Meta-Llama-3-8B-Instruct, applying RR fine-tuning (on the GraySwanAI training set) produces a model whose HarmBench Attack Success Rate (ASR), aggregated across the six attack categories (GCG, PAIR, TAP, AutoDAN, direct request, human red-team), is *substantially lower* than the same base model under (a) standard RLHF-safety refusal training only and (b) adversarial training (R2D2 or equivalent), while (c) MT-Bench and MMLU scores stay within a small tolerance of the base Llama-3-8B-Instruct.

**Hypothesis**: H2 — RR fine-tuning intervenes at a mechanism (harmful-representation reroute) upstream of the attack-specific surface form, so its ASR reduction generalizes across unseen attack types (GCG, PAIR, TAP, AutoDAN, direct request, human red-team) whereas adversarial training's ASR reduction is concentrated on the attack family it was trained against.

**Measurable predicate**: On HarmBench (`/data/zhenqian/data/HarmBench`):
- (i) ASR(RR) < ASR(refusal-only baseline) by a large margin on each of the six attack categories and on the aggregate (target: ≥ 20 absolute percentage points on the aggregate; per-category direction consistent).
- (ii) ASR(RR) < ASR(adversarial-trained baseline) on attack categories the adversarial baseline was not trained against (target: ≥ 10 absolute pp on the categories where R2D2's training distribution does not cover them).
- (iii) Capability delta on MT-Bench (single-turn average) ≤ 0.3 (10-pt scale) and MMLU 5-shot ≤ 2 absolute pp vs. base Llama-3-8B-Instruct.

**Expected direction**: (i, ii) down (ASR lower is better); (iii) equal within tolerance.

**Resources**: model: Meta-Llama-3-8B-Instruct; safety-eval dataset: HarmBench (six attack categories); capability datasets: MT-Bench, MMLU. Baselines: (a) the base Llama-3-8B-Instruct (already RLHF-safety trained) serves as the refusal-training baseline; (b) an adversarial-trained baseline — reproduce with GCG-augmented refusal fine-tuning on the same base model (R2D2-style) using the GraySwanAI training-set harmful prompts + generated GCG suffixes. used_n: full HarmBench eval set; MT-Bench full; MMLU 5-shot on the full test set.

**Status**: pending verification

**Notes**: "Substantially lower" is operationalized as ≥ 20 absolute pp on the HarmBench aggregate vs. refusal-only, and ≥ 10 absolute pp on unseen-family categories vs. adversarial-trained. Task.md does not name the R2D2 codebase; the adversarial-trained baseline is reproduced in-house from the same GraySwanAI training prompts.

### Claim 3: Same intervention transfers to multimodal LLMs (image-hijack defense) without VLM-task degradation

**Original (verbatim excerpt from task.md):**
> The same representation-level intervention transfers to multimodal LLMs, blocking image-based jailbreaks (image hijacks) without degrading vision-language task performance.

**Extracted statement**: Applying the RR fine-tuning objective — same training pairs (benign/harmful text pairs from GraySwanAI, no attack prompts, no adversarial images) — to a multimodal LLM (LLaVA-NeXT-Mistral-7B) reduces the attack success rate of PGD image-hijack attacks (ε=32/255 over 1,000 steps) while leaving vision-language task performance within a small tolerance of the un-RR-tuned VLM.

**Hypothesis**: H3 — Because RR operates on the shared residual stream (into which the vision encoder projects image tokens), rerouting the harmful subspace with text-only pairs also blocks harmful trajectories induced by image tokens; PGD attacks that succeed by pushing image tokens into the harmful subspace are neutralized when that subspace is emptied by RR.

**Measurable predicate**: On LLaVA-NeXT-Mistral-7B, before and after applying RR (using the same GraySwanAI text-only training set):
- (i) Image-hijack ASR (PGD ε=32/255, 1,000 steps, generated against the RR-tuned model per standard adaptive-attack protocol) is *substantially lower* than image-hijack ASR against the un-RR-tuned baseline (target: ≥ 15 absolute pp reduction on the harm set).
- (ii) VLM task delta stays within a small tolerance of the base VLM on a representative VLM capability benchmark (target: a task.md-approved VLM benchmark — see Notes).

**Expected direction**: (i) down; (ii) equal within tolerance.

**Resources**: VLM: LLaVA-NeXT-Mistral-7B (per task.md verify-stage list — task.md notes it may need to be downloaded); attack: PGD image-hijack ε=32/255, 1,000 steps; capability: a VLM capability benchmark — see Notes. used_n: adversarial-image harm probe set — spec is `unspecified — to be resolved in Phase 4.5` (task.md does not name a size).

**Status**: pending verification

**Notes**: Task.md pins the attack protocol (PGD ε=32/255 × 1000 steps against LLaVA-NeXT-Mistral-7B) but does not pin a VLM capability benchmark (only MT-Bench / MMLU are listed, which are text-only). Phase 4.5 resolves the VLM-capability benchmark choice (default: MME or MMBench single-turn subset) and the harm-probe set size (default: ≥ 100 prompts × 5 images, adjust to the 10 GPU-hour budget). This is a `method_sensitive` milestone.

### Claim 4: Same intervention transfers to LLM agents (reduces harmful tool-use) without BFCL degradation

**Original (verbatim excerpt from task.md):**
> The same intervention transfers to LLM agents, materially reducing the rate of harmful tool-use actions executed under attack.

**Extracted statement**: Applying the RR fine-tuning objective (same GraySwanAI training pairs, no attack prompts) to Llama-3-8B-Instruct in an agent harness with function-calling reduces the rate of harmful tool-call actions on a 100-prompt function-calling harm probe (cybercrime / disinformation / fraud / harassment), while leaving Berkeley Function Calling Leaderboard (BFCL) capability within a small tolerance of the un-RR-tuned baseline.

**Hypothesis**: H4 — Harmful tool-call actions share the same residual-stream harmful subspace as harmful text outputs; RR rerouting therefore suppresses them at the representation level without a modality- or scaffold-specific retrain.

**Measurable predicate**: On the Llama-3-8B-Instruct + function-calling tool harness variant, before and after RR:
- (i) Harmful-tool-use rate on the 100-prompt harm set is substantially lower for the RR-tuned agent (target: ≥ 20 absolute pp reduction).
- (ii) BFCL delta stays within a small tolerance of the base agent on the standard BFCL split (target: ≤ 3 absolute pp).

**Expected direction**: (i) down; (ii) equal within tolerance.

**Resources**: model: Llama-3-8B-Instruct + function-calling tool harness (per task.md verify-stage list); attack set: 100-prompt function-calling harm set spanning cybercrime / disinformation / fraud / harassment (per task.md); capability: BFCL (per task.md). used_n: 100-prompt harm set (task.md), BFCL full split.

**Status**: pending verification

**Notes**: Task.md names the 100-prompt harm set and BFCL; the specific attack methodology (system-prompt injection, tool-injection, or direct harm request through the tool interface) is `unspecified — to be resolved in Phase 4.5` (default: direct harmful function-calling request per BFCL harness).

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering all 4 claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps

- [ ] /mechanism-skills to route the testing approach to a concrete mechanism family + submethod (Workflow 1.25) — the Phase 1.75 strategic direction is **Location → Causal Intervention → Tuning & Editing**
- [ ] /auto-experiment to implement and run the verification suite (Workflow 1.5)
- [ ] /auto-verify to stress-test each verified claim under method/dataset/model swaps (Workflow 1.75)
- [ ] /auto-iteration-loop to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke /auto for the autonomous claim → routing → experiments → verify → review chain
