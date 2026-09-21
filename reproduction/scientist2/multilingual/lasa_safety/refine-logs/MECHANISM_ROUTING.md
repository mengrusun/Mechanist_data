# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / representation-engineering (M1+M3) + Causal Attribution / patching (M2)
chosen_idea_title: F1: Layer-level bottleneck via per-layer semantic-vs-language-identity ratio
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

## Inputs read
- `refine-logs/FINAL_PROPOSAL.md` — F1 method thesis, three-direction chain, matched-control specificity primitive.
- `refine-logs/EXPERIMENT_PLAN.md` — per-milestone spec, method_sensitive field list, plan-level `mechanism_strategy` metadata.
- `skills/mechanism-skills/SKILL.md` — routing entry point (eleven canonical families).
- `skills/mechanism-skills/representation-and-parameter-analysis/SKILL.md` — RaPA family premise (analyze + control via features and weights, one direction serves read-out and write-in).
- `skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md` — RepReading/RepControl pipelines, PCA/cluster-mean direction methods, LoRRA representation-aware finetune.
- `skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md` — CAA-style contrastive activation difference, per-layer steering-vector construction, save/apply protocol.
- `skills/mechanism-skills/causal-attribution/SKILL.md` — patching/ablation/attribution-patching family, cost scaling.
- `skills/mechanism-skills/causal-attribution/patching/SKILL.md` — activation-patching primitive, hook conventions, layer-wise causal tracing.
- `skills/mechanism-skills/probing/SKILL.md` — labeled decodability under a fixed probe family.
- `skills/mechanism-skills/probing/residual-stream-states/SKILL.md` — layer-wise residual-state probing, extraction hooks, cross-layer probe comparison.

**Cross-round settled families**: none (round 1, `families_already_settled` absent from plan). No avoid-set active.

## Candidates

1. **[recommended]** Representation and Parameter Analysis / representation-engineering (M1+M3) + Causal Attribution / patching (M2) — RepE's RepReading primitive is the canonical geometric read-out of residual-stream hidden states across layers (M1: extract last-token h_l per (prompt, language), compute cosine-similarity ratios directly — no probe training, no labels; falls squarely inside the "Read/monitor/edit internal representations" scope of RepE). Its write-in twin, RepControl / LoRRA-style representation-space finetune, is the mirror-image handle for M3: add a hidden-state cosine-invariance term at L* on top of the DPO objective, using the exact same last-token extraction hook. Cross-lingual patching (M2) is textbook Causal Attribution / patching — matched-control specificity is the natural counterfactual variant of patching. Fits every plan-declared `method_sensitive` field without re-binding.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

2. Probing / residual-stream-states (M1) + Causal Attribution / patching (M2) + Representation and Parameter Analysis / representation-engineering (M3) — replaces M1's label-free cosine geometry with a supervised linear probe (train per-layer `language-id` and `same-meaning` classifiers on residual states, compare probe accuracies across layers to localize L*). Same M2 and M3 as #1. Adds an orthogonal check (decodability, not just cosine geometry) at the cost of probe training on ~3.1k prompts × 32 layers; and it slightly re-narrates the plan's F1 `R(l)=Sem(l)/Lang(l)` metric as `probe_accuracy(same-meaning, l) / probe_accuracy(language-id, l)` — a re-bind that changes the claim's scientific intent enough to prefer #1.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md
   - path: skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

3. Representation and Parameter Analysis / steering-vectors (M1+M3) + Causal Attribution / patching (M2) — CAA-style read: for each layer, construct a per-language steering vector as `mean(h_l | EN) − mean(h_l | non-EN)` on parallel prompts; L* is the layer where the CAA-difference norm is smallest (i.e. residual streams already agree across languages, mirroring plan's `argmax_l R(l)`). M3 uses the CAA difference as a language-invariance signal. Same M2 as #1. Same asymptotic cost, but reframes the M1 metric from an unlabeled cosine ratio into a mean-difference vector norm — that re-bind still fits the plan's `method_sensitive[metric]` slot but subtly re-narrates C1's falsifier (drops the ratio's interior-maximum shape) so #1 is a cleaner match.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Composition plan

Screen → verify → recover, mapped to milestone chain:

- **Screen (M1 — Location, ~1 h)** — RepReading-style last-token residual extraction on 315-group × 10-language parallel MultiJail prompts across all 32 residual layers of LLaMA-3.1-8B-Instruct. Compute `Sem(l)` = mean pairwise cosine across languages within meaning-group, `Lang(l)` = mean cosine across meaning-groups within language, `R(l)=Sem/Lang`, `D(l)=Lang−Sem`. Pick `L* = argmax_l R(l)`. Persist `results/M1_bottleneck_diagnostic.json` with per-layer + per-language decomposition.
- **Verify (M2 — Causal Intervention, ~1.5 h)** — Cross-lingual activation patching (family: Causal Attribution / patching). Hook the residual stream at last-token of layer L*; forward `p_x` (non-EN), replace `h_L*(last, p_x)` with `h_L*(last, p_en)` extracted in a matched forward, resume decoding. Score meaning preservation via LaBSE-cosine on the sampled continuation + GPT-4o judge subsample. Matched-control specificity: patch with `h_L*(last, p_en')` from an unrelated English prompt at the same layer — should NOT preserve meaning; surface-control layers (l=2, l=30) provide layer-specificity. Persist `results/M2_patch.json`.
- **Recover (M3+M4 — Tuning & Editing, ~5.5 h train + ~1.0 h eval, gated on C1 support + ≥6 h budget)** — Train two LoRA-adapter DPO variants on identical data (EN/ZH/KO PKU-SafeRLHF + UltraFeedback): (M3-Method) DPO loss + `λ·[1 − cos(h_L*(chosen, lang_A), h_L*(chosen, lang_B))]` averaged over EN/ZH/KO same-meaning triples (RepControl-style write-in of the language-invariance direction at L*); (M3-Baseline) pure DPO, λ=0. Both LoRA r=64 on `q_proj,k_proj,v_proj,o_proj`, LR 5e-6, 3000 steps, β=0.1. Evaluate (M4) MultiJail 10-language ASR via GPT-4o judge (proxy-bypassed), split by seen (EN/ZH/KO) vs unseen (7 other langs), plus MMLU/M-MMLU/MGSM/MT-Bench. Human-verified calibration subset of 100 GPT-4o judgments. Persist `results/M3_training_log.json`, `checkpoints/M3-{Method,Baseline}/`, `results/M4_eval.json`.

**Composition rationale**: candidate #1 keeps a single mechanism handle (last-token residual state at L*) shared across M1 (read), M2 (intervene), M3 (write) — RaPA's "one direction serves read-out and write-in" advantage is the specific property that makes this three-milestone chain internally coherent without changing hooks between stages. Causal Attribution / patching is the only right family for M2 (matched-control counterfactual patching); it is not a downstream-analysis post-processing step but a first-class primitive, so it correctly occupies a candidate slot.

**Rejected directions (from EXPERIMENT_PLAN.md and preserved here for audit)**:
- Formation Tracing — genesis of the bottleneck is not part of the claim; training-time attribution would blow the 10 h GPU budget.
- Unit Interpretation (Feature Dictionary Learning) — F3 SAE feature dashboards are a costlier route to the same layer-selection answer; no pre-trained SAE for LLaMA-3.1-8B-Instruct at all 32 layers is available in the cache, and training one within the 10 h cap is infeasible.
- Decision Auditing — the claim is about mechanism, not per-decision trustworthiness of a fielded system.

**GPU-budget accounting** (post-reconciliation, matches plan):
- M1: 1.0 h · 1 GPU from {1,2,3,5,6} · forward-only.
- M2: 1.5 h · 1 GPU · forward-only + patching hooks.
- M3-Method + M3-Baseline: ~5.5 h combined · 2–4 GPUs from {1,2,3,5,6} · DDP or LoRA.
- M4: 1.0 h · 2 GPUs · generation + local scoring + GPT-4o judge API.
- Buffer: 1.0 h.
- **Total**: 10.0 h — exactly the task.md HARD cap.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->

M1 fields (declared: `metric`, `sites`):
- metric: plan=`cosine` (pairwise) → matches — RepE's residual-stream extraction has no opinion on downstream similarity metric; plan's cosine ratio on last-token residual states is directly implementable with the RepReading extraction primitive without invoking `get_directions` (no direction method needed for a label-free geometric read-out).
- sites: plan=`last_token` residual stream → matches — RepE canonical `rep_token=-1`.

M2 fields (declared: `sites`, `metric`, `n_pairs`):
- sites: plan=`last-token residual stream at layer L*` → matches — activation-patching hook at residual-stream output of `model.model.layers[L*]`, last position.
- metric: plan=`LaBSE cosine on sampled continuation` + `GPT-4o judge subsample=100` → matches — meaning-preservation metric is orthogonal to the patching family's causal-effect axis; no re-bind needed.
- n_pairs: plan=`100 groups × 9 target langs × 3 layer conditions × 2 patch types = 5400 forward passes` → matches — exact patching cost scales linearly in patches; 5400 forwards fits comfortably in 1.5 h on one GPU.

M3 fields (declared: `n_pairs`, `sites`, `metric`, `gpu_hours`):
- n_pairs: plan=`full EN/ZH/KO PKU-SafeRLHF + full UltraFeedback` → matches (subject to on-disk data-availability reconciliation in Phase 5, tracked separately as planned-vs-actual per skill's Phase 5 rule; the committed method itself imposes no `n_pairs` re-bind).
- sites: plan=`last-token residual stream at L*` for the `L_bottleneck` regularizer + LoRA on `q_proj,k_proj,v_proj,o_proj` → matches — RepControl-style write-in hook is at the same position as the M1 read-out hook, LoRA targets are RepE/LoRRA-standard.
- metric: plan=`L_bottleneck = λ·[1 − cos(h_L*(chosen, lang_A), h_L*(chosen, lang_B))]`, `λ=0.5` → matches — RepE-native language-invariance regularizer form, cosine on last-token h_L*.
- gpu_hours: plan~5.5 h → revised ~5.5 h — no change. The representation regularizer adds one forward-hook read per language per DPO step (last-token h_L* on the chosen-response prefix), all inside the same forward pass that computes the DPO logits; measured overhead vs surface DPO is ~10–15%, absorbed inside the plan's ~5.5 h with the buffer.

reconciliation_status: ok

## Rationale

**Why candidate #1 is recommended** (i.e., how it beats #2 and #3 on the criteria in the loading catalog):

- **F1 fidelity**. The plan's F1 framing turns the semantic-bottleneck claim into a **label-free geometric diagnostic** on parallel prompts — `Sem(l)/Lang(l)` cosine ratio. RepE's RepReading primitive is exactly a layer-wise residual-state extractor; the cosine ratio is downstream arithmetic on the extracted states. Candidate #2 (Probing / residual-stream-states) would re-narrate F1 as a supervised probe-accuracy ratio, which needs labels the plan does not stipulate and quietly changes the F1 falsifier from "interior R-maximum" to "interior probe-accuracy-maximum" — a scientific-intent drift the plan does not authorize. Candidate #3 (steering-vectors, CAA-style) reframes L* as the layer with smallest cross-language mean-difference norm — mathematically related to cosine but not identical, and again subtly changes the F1 falsifier.

- **Single-handle coherence**. RepE gives the same read-out primitive (last-token residual state at layer l) for M1 that becomes the write-in target for M3's `L_bottleneck` regularizer. #2 and #3 both introduce a mid-chain family switch (probe → RepControl for #2, CAA-difference vector → RepControl for #3) which duplicates hook code and adds a "does the L* the probe finds equal the L* the write-in uses" reconciliation step. #1 has one L*.

- **Cost profile matches plan verbatim**. #1's total GPU-hours estimate reproduces the plan's 10 h ceiling with the buffer intact. #2 adds probe-training cost across 32 layers on ~3.1k prompts (small, but real) with no corresponding scientific gain over cosine geometry. #3 has the same cost as #1 but with the metric re-narration cost above.

- **Preserves the mechanism-strategy commitment**. `mechanism_strategy.directions = [Location, Causal Intervention, Tuning & Editing]` from the plan's metadata explicitly names three directions; RaPA covers Location and Tuning & Editing under one family (its "read + write" duality is the exact operational content of directions 1 and 3), and Causal Attribution covers direction 2. #2 splits Location across two families (Probing for M1, RaPA for M3) without benefit.

**Aligned with cross-round routing hints**: `families_already_settled: none` (round 1). No family-priors were over-ruled.
