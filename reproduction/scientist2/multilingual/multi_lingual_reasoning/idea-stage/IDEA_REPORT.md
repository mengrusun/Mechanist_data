# Idea Report — Captured Behavior

**Direction**: Disentangling Language and Reasoning in LLM Internal Representations (from task.md — task.md is authoritative; no separate direction string)
**Behavior-source**: given
**Mechanism**: discovery (mechanism strategy chain: **Location → Causal Intervention → Tuning & Editing**, decided in Phase 1.75 via `/mechanism-explore`)
**Claim source**: task.md (faithful capture — no mining, no ideation, no novelty/impact scoring)
**Date**: 2026-07-14
**Pipeline**: research-lit → faithful behavior capture (from task.md) → research-refine-pipeline

## Executive Summary

Task.md advances a four-claim mechanistic hypothesis about multilingual reasoning LLMs: hidden states linearly decompose into a language-specific subspace (identifiable from a small multilingual probe set) and an approximately orthogonal language-agnostic subspace; suppressing the language-specific subspace at inference improves multilingual reasoning accuracy on MGSM while GlotLID-measured output-language fidelity remains acceptable when upper layers are left intact; language-specific activation strength is negatively (and monotonically) correlated with reasoning accuracy across a signed α-sweep; and this training-free intervention matches or exceeds multilingual SFT / RL at a small fraction of the compute. All four claims are unified into a single verification plan built on Qwen-3-4B-Thinking + MGSM (11 languages), with GlotLID for language fidelity.

## Literature Landscape

See `idea-stage/LANDSCAPE.md` (retrieved 2026-07-14, based on 20 pre-cutoff / non-arXiv sources — the reproduction-target paper arxiv:2505.15257 and its GitHub repo were NOT read, per project policy at `.claude/forbidden-urls.txt`). Key context:

- **Closest published prior art** is **LENS** (Zhao et al., NeurIPS 2024, OpenReview 8kGonpsiHb) which also decomposes top-layer hidden states into a language-agnostic + language-specific subspace but *rebalances* them via light contrastive training on non-reasoning models.
- **Mechanistic backdrop**: multiple prior works (LAPE — Tang et al. ACL 2024; PLND / Zhao et al. NeurIPS 2024; Do Llamas Work in English — Wendler et al. ACL 2024; MEXA — Findings ACL 2025) establish that (a) language identity is carried by a low-dimensional linear subspace / a sparse neuron set concentrated in early + late layers, and (b) middle layers do "English-pivot" reasoning.
- **Subspace-projection ancestor**: LSAR (Xie et al., EMNLP 2022) — the exact "SVD → null-space projection" recipe on mBERT/XLM-R encoders.
- **Steering methodology**: Contrastive Activation Addition (Panickssery et al., ACL 2024), Representation Engineering (Zou et al., 2023) — canonical α-sweep + specificity control protocols.
- **Post-training baselines** (yardstick for Claim 4): MathOctopus (Chen et al., ACL 2024), LangBridge (Yoon et al., ACL 2024), LinguaLIFT (arXiv 2412.12499).
- **Diagnostics**: GlotLID (Kargaran et al., Findings EMNLP 2023) is the task.md-mandated language identifier for output-language fidelity.

## Recommended: #1 — Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM

There is no ranking step under `BEHAVIOR_SOURCE=given`. All four captured claims are grouped into a single unified verification plan produced by Phase 4.5. The recommended "idea" is the entire captured-claim bundle, which the experiment stage refines and executes as one integrated experimental campaign.

## Claims to Verify

### Claim 1: Language-specific vs language-agnostic subspace decomposition (probe-set-identifiable)

**Original (verbatim excerpt from task.md):**
> For a given LLM, hidden representations of multilingual reasoning inputs decompose into a language-specific subspace and an approximately orthogonal language-agnostic subspace, and the language-specific subspace can be identified from a small multilingual probe set.

**Extracted statement**: In the hidden states of Qwen-3-4B-Thinking at a fixed set of intervention layers, applying a linear decomposition (SVD / mean-difference / LDA) fitted on a *small* multilingual probe set (parallel sentences in the 11 target languages) recovers (a) a low-rank language-specific subspace `V_lang` capturing between-language variance, (b) an approximately orthogonal residual (language-agnostic) subspace, where `V_lang` measured on **held-out** sentences (probe-set generalization) achieves language-classification accuracy well above chance and reasoning-content probes near-random.

**Hypothesis**: H1 — A small multilingual probe set (order ~10^2–10^3 parallel sentences) is sufficient to identify a low-rank (order ~10^1) language-specific subspace `V_lang` in Qwen-3-4B-Thinking's hidden states that is approximately orthogonal to the reasoning-content subspace.

**Measurable predicate**: On held-out MGSM prompts, a linear language classifier trained on the projection onto `V_lang` (fit on the probe set only) achieves accuracy ≥ 0.90 across the 11 languages; the projection onto the orthogonal complement gives language classifier accuracy near 1/11 (chance); and the cosine of the principal angles between `V_lang` and a content-probe subspace (e.g., subject / operation identity) is bounded below a small threshold (e.g., median principal-angle cosine ≤ 0.2). Report on Qwen-3-4B-Thinking at the intervention layers determined by Claim 2's layer sweep.

**Expected direction**: threshold (probe-set-fitted `V_lang` generalizes to held-out data at ≥ 0.90 language classification; content probes near chance).

**Resources (preferred, cost-aware)**: model: Qwen-3-4B-Thinking (task.md-mandated for the experiment stage — HARD constraint); probe set: parallel multilingual sentences (FLORES-200 dev / Bible or MGSM training-shot pool), used_n ≈ 100–1000 sentences per language (to be resolved to a specific value in Phase 4.5 based on subspace-fit stability); layers: to be selected by Claim 2's layer sweep (typically the middle-layer range identified by MEXA / logit-lens analyses).

**Status**: pending verification.

**Notes**: The claim explicitly says "*from a small multilingual probe set*" — the probe-set size is a load-bearing part of the claim and must be swept (e.g., n ∈ {50, 100, 250, 500, 1000} per language) to test the "small" adjective quantitatively. Approximate orthogonality is operationalized via principal angles; task.md does not specify a numeric threshold, so Phase 4.5 will pick one and defend it (see landscape LENS + LSAR for prior conventions).

---

### Claim 2: Training-free subspace suppression → multilingual reasoning improves; upper-layer preservation → language fidelity acceptable

**Original (verbatim excerpt from task.md):**
> Suppressing the language-specific subspace from internal representations at inference time consistently improves multilingual reasoning accuracy across diverse models, languages, and reasoning tasks, while output language fidelity remains acceptable when upper layers are left intact.

**Extracted statement**: At inference time, projecting the residual stream at a set of *non-upper* intervention layers onto the null space of `V_lang` (fitted per Claim 1) — with **no gradient updates and no additional training data** — increases MGSM 0-shot / few-shot accuracy on the 11 target languages relative to the unedited baseline; the improvement is not localized to a single language but holds consistently across the {high, mid, low}-resource splits; and GlotLID-measured output-language fidelity (proportion of generations classified as the intended source language) remains within an acceptable margin of the baseline (Phase 4.5 to fix the margin numerically) provided that intervention layers exclude the model's top-k upper layers.

**Hypothesis**: H2 — Inference-time null-space projection of hidden states onto the orthogonal complement of `V_lang` at the selected intervention layers of Qwen-3-4B-Thinking raises MGSM accuracy on the 11 target languages, with the effect present in most (≥ 8/11) individual languages, while GlotLID fidelity stays within a bounded absolute drop when upper layers are excluded from the intervention.

**Measurable predicate**: Let A_base(L) and A_edit(L) be MGSM exact-match accuracy for language L, and F_base(L) and F_edit(L) be GlotLID output-language fidelity for L. Predicate:
- Aggregate: mean_L A_edit(L) − mean_L A_base(L) ≥ +δ_acc (with δ_acc a positive threshold decided in Phase 4.5, typically ≥ +3 pp);
- Per-language: A_edit(L) ≥ A_base(L) − ε_slack for at least 8 of 11 languages (no widespread regression) and A_edit(L) > A_base(L) for the majority;
- Language fidelity: mean_L F_edit(L) ≥ mean_L F_base(L) − δ_fid (with δ_fid a bounded acceptable drop, e.g., ≤ 5 pp, when upper-k layers are excluded);
- Statistical significance: paired bootstrap over MGSM problems, 95% CI on the mean gain > 0.

Report on Qwen-3-4B-Thinking. Task.md's "across diverse models, languages, and reasoning tasks" is honored *aspirationally* here — the **experiment-stage HARD constraint pins the main experiment to Qwen-3-4B-Thinking on MGSM**; **model / task diversity is deferred to the verify stage** (the verify-stage candidate models in task.md's NOTICE, plus XWinograd and M-MMLU).

**Expected direction**: up (accuracy) + threshold (language fidelity within acceptable margin).

**Resources (preferred, cost-aware)**: model: Qwen-3-4B-Thinking (HARD-pinned by task.md); dataset: MGSM (HARD-pinned) — used_n = full MGSM (250 problems per language × 11 = 2750 evaluations per condition) per Phase 4.5's power analysis; language identifier: GlotLID (HARD-pinned) on the model's generated continuation; intervention: null-space projection of residual stream at selected non-upper layers.

**Status**: pending verification.

**Notes**: The claim has two coupled parts that must be reported jointly: (i) accuracy up on MGSM, (ii) fidelity acceptable when upper layers left intact. Phase 4.5 will fix "upper layers left intact" operationally (e.g., last 20% of layers unaffected) and sweep the excluded-upper-layers count to test the coupling. Cross-model consistency (per task.md "across diverse models") is a **verify-stage** contract in this project (task.md's Verify candidate list of Qwen-2.5, Qwen-3-1.7B/8B-Thinking, DeepSeek-R1-Distill, GLM-Z1, QwQ), not a main-experiment contract.

---

### Claim 3: Signed dose-response — amplifying language-specific activation degrades, removing it improves reasoning

**Original (verbatim excerpt from task.md):**
> The strength of language-specific activation is negatively correlated with reasoning accuracy: amplifying it degrades reasoning, removing it improves reasoning.

**Extracted statement**: Consider the parametric family of interventions h ← h + α · (Π_lang · h) where Π_lang is the projection onto `V_lang` (α = 0: baseline; α = −1: removal / null-space projection; α > 0: amplification). Then, sweeping α across a signed range (e.g., α ∈ {−1.5, −1, −0.5, 0, +0.5, +1, +1.5}) at the Claim-2 intervention layers on Qwen-3-4B-Thinking + MGSM, the mean-across-languages accuracy A(α) is a monotone-decreasing function of α over a bounded α-range around 0 (in particular A(α=−1) > A(α=0) > A(α=+1)), and the correlation between α and mean A across the sweep is significantly negative.

**Hypothesis**: H3 — Mean-across-language MGSM accuracy A(α) has a strictly negative correlation with α across a signed α-sweep centered on 0, and A(α = −1) > A(α = 0) > A(α = +1) with the middle inequalities significant under paired bootstrap.

**Measurable predicate**:
- Sign: Pearson (or Spearman) correlation between α and A(α) is negative with p < 0.05 across the swept α-grid;
- Monotonicity check-points: A(−1) > A(0) > A(+1), with each strict inequality passing paired bootstrap at 95% CI;
- Robustness: sign of correlation holds for at least 8 of 11 individual languages separately, not only the aggregate.

Report on Qwen-3-4B-Thinking + MGSM at Claim-2's intervention layers.

**Expected direction**: down (accuracy is a monotone-decreasing function of α, i.e., negative correlation of α with A).

**Resources (preferred, cost-aware)**: same as Claim 2 (Qwen-3-4B-Thinking + MGSM + GlotLID). Extra compute: multi-α forward passes — used_n per α cell = full MGSM per language.

**Status**: pending verification.

**Notes**: Task.md is explicit about the *signed* prediction (amplify degrades / remove improves). This maps to a bidirectional steering sweep, which is under-tested in the prior activation-steering literature — most prior α-sweeps only sweep one side or treat α as unsigned (see Landscape §4 Gap G5). Phase 4.5's plan must include α > 0 cells, not only α ≤ 0.

---

### Claim 4: Training-free intervention matches / exceeds multilingual post-training at a fraction of the compute

**Original (verbatim excerpt from task.md):**
> This training-free intervention matches or exceeds multilingual post-training (supervised fine-tuning, reinforcement learning) at a small fraction of the compute.

**Extracted statement**: For the same base model (Qwen-3-4B-Thinking), the mean MGSM accuracy of the training-free subspace-suppression intervention (Claim 2) is at least equal to the mean MGSM accuracy of a matched multilingual post-training baseline (supervised fine-tuning on a translated MGSM8KInstruct-style dataset, or RL on multilingual reasoning traces), while the intervention's total compute cost (probe-set construction + SVD + inference-time projection overhead) is at most a small fraction (e.g., ≤ 5–10 %) of the post-training baseline's compute (training FLOPs or GPU-hours).

**Hypothesis**: H4 — On Qwen-3-4B-Thinking, mean_L A_edit(L) ≥ mean_L A_SFT(L) − ε_match while (compute_edit / compute_SFT) ≤ κ, with κ small (e.g., ≤ 0.10) and ε_match a small non-negative tolerance.

**Measurable predicate**:
- Accuracy match: mean_L A_edit(L) ≥ mean_L A_SFT(L) − ε_match, with ε_match specified in Phase 4.5 (candidate 0 pp = strict "matches or exceeds"; a lightly loosened +0-to-1 pp tolerance may be reported alongside for robustness);
- Compute ratio: total compute (GPU-hours, or FLOPs — Phase 4.5 will pick one and stick to it) of the training-free path over the SFT (and, if in budget, RL) path is ≤ κ, with κ ≤ 0.10 as the headline target;
- Baseline: at least one *actually-run* multilingual post-training baseline on Qwen-3-4B-Thinking within the 10-hour GPU budget (task.md HARD budget). If the RL baseline exceeds budget, restrict to SFT and note it explicitly (do not claim exceeds RL without measurement).

**Expected direction**: threshold (matches-or-exceeds on accuracy AND compute ≤ small fraction).

**Resources (preferred, cost-aware, constrained by HARD budget)**: model: Qwen-3-4B-Thinking; dataset: MGSM (evaluation) + translated GSM8K instruction pairs (SFT baseline training data — download from MathOctopus / MGSM8KInstruct if within budget, else construct a size-matched subset). RL baseline construction depends on remaining budget after SFT; if infeasible in 10 GPU-hours, restrict this claim's "post-training" side to SFT and mark RL as "unattempted, out of budget".

**Status**: pending verification.

**Notes**: Task.md pins the 10-hour GPU budget as a HARD constraint. Realistically, running a full multilingual SFT baseline on Qwen-3-4B-Thinking + a matched RL baseline + the 11-language MGSM evaluation is *tight*. Phase 4.5 will scope the comparison to what fits: minimum viable = SFT-only comparison with GPU-hours accounting; RL comparison is *nice-to-have* and may be deferred to the verify stage if it does not fit. The "matches or exceeds" wording is preserved verbatim as the target predicate.

---

## Resources (project-level, from task.md)

Task.md declares these globally (referenced by each claim above):

- **Experiment stage** (HARD): model = **Qwen-3-4B-Thinking**; dataset = **MGSM** (11 languages); language identifier = **GlotLID** (always).
- **Verify stage** (NOTICE — candidates, not all required): models = Qwen-2.5-Instruct-3B/7B; Qwen-3-1.7B/8B-Thinking; DeepSeek-R1-Distill-Qwen-7B / -LLaMA-8B / -Qwen-14B; GLM-Z1-9B; QwQ-32B. datasets = XWinograd, M-MMLU. Cross-model / cross-task generalization tests belong here.
- **Target languages (11, HARD)**: High-resource — En, Es, Fr, De, Zh, Jp, Ru; Mid — Th, Te; Low — Bn, Sw.
- **Compute (HARD)**: 10-hour GPU budget total.
- **GPU allowlist (HARD)**: gpu_id ∈ {1, 2, 3, 5, 6}.
- **Directory access (HARD)**: work_dir + `/data/zhenqian/data` + `/data/zhenqian/models` only.

Resource-fidelity marker: `resource_fidelity: strict` is **NOT** stamped (this is `BEHAVIOR_SOURCE=given` + `MECHANISM=discovery`, not the reproduction combination `given` + `given`). Cost-aware design is permitted subject to the HARD constraints above.

## Next Steps

- [ ] /research-refine-pipeline to produce `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering all 4 claims) + `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged to claim(s), with mechanism_strategy metadata block from Phase 1.75).
- [ ] /mechanism-skills to route the testing approach to a concrete mechanism family + submethod at the experiment stage (Workflow 1.25 — the family is chosen there, not here, since MECHANISM=discovery).
- [ ] /auto-experiment to implement the verification suite from routing + plan (Workflow 1.5).
- [ ] /auto-verify to stress-test each verified claim under method / dataset / model swaps (Workflow 1.75).
- [ ] /auto-iteration-loop to iterate the verification suite until reviewer-ready (Workflow 2).
- [ ] Or invoke /auto for the autonomous claim → routing → experiments → verify → review chain.
