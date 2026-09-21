# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Feature Dictionary Learning / SAE
chosen_idea_title: Faithful SAGE reproduction: unified plan covering C1-C4 on the pinned Gemma-2-2B + gemmascope-res-16k main pair with one cross-pair generalization run inside main experiment
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
  - skills/mechanism-skills/feature-dictionary-learning/transcoder/SKILL.md
  - skills/mechanism-skills/vocabulary-projection/SKILL.md

## Candidates

1. **[recommended]** Feature Dictionary Learning / SAE — SAGE explains SAE-feature units in an already-trained SAE (Gemma-Scope `gemmascope-res-16k` for the main pair, `transcoder-hp` for M2's cross-pair). All feature-firing reads, held-out predictive scoring, and probe-writer generative checks route through the SAE encoder/decoder pipeline provided by SAELens. The routed unit — an SAE feature id — is exactly the unit the SAE submethod exposes.
   - path: skills/mechanism-skills/feature-dictionary-learning/SAE/SKILL.md
2. Feature Dictionary Learning / Transcoder — Alternative for M2 if `transcoder-hp` is the target (a transcoder is a sibling submethod under the same family). Included as a fallback for the Qwen3-4B + `transcoder-hp` pair; the interface is nearly identical.
   - path: skills/mechanism-skills/feature-dictionary-learning/transcoder/SKILL.md
3. Vocabulary Projection / SAE Feature — A cheap complement that projects an SAE-feature decoder direction into vocabulary space to sanity-check the explanations. Not primary because SAGE's evaluation is empirical (generative + predictive activation-based), not a projection read.
   - path: skills/mechanism-skills/vocabulary-projection/SKILL.md

## Composition plan

**Screen** — none needed; SAE feature ids are the target units directly (localization is already done by the SAE encoder).
**Decode** — SAGE runs a 4-role auto-interpretation loop (Explainer → Designer → Analyzer → Reviewer) on each SAE feature, producing a natural-language label. Two baselines share the same feature-id set: Neuronpedia's public explanation + a single-pass GPT-5 matched-backbone control (M0.5 gate).
**Verify** — two paired assays on identical feature ids: (a) **Generative accuracy (C1)** — probe-writer LLM writes 5 texts per (feature, method), each pushed through Gemma-2-2B + SAE hook; hit iff activation > τ_f (99th percentile of feature f's activation distribution). (b) **Predictive accuracy (C2)** — 20 held-out texts per (feature, method); scorer LLM predicts activation ŷ conditioned on the explanation; report Pearson ρ vs. ground-truth normalized activations.
**Recover** — layer-stratified aggregation across {L4, L12, L20} = C3 read-off; cross-pair re-run on Qwen3-4B + transcoder-hp = C4 read-off.

Cost profile (post-reconciliation, see below): M0.5 = 0.5h, M1 = 5.0h, M2 = 3.5h → 9.0h total (matches plan).

## Plan reconciliation

<!-- One row per method_sensitive field declared on the intervention milestone(s). -->

**M0.5 method_sensitive fields:**
- n_pairs: plan=30 features × 20 probes = 600 → matches (SAE submethod uses feature-id set directly; no re-bind needed).
- sites: plan=residual_stream @ {L4, L12, L20} → matches (Gemma-Scope 16k SAEs are trained per-layer on residual stream; layer indices align with released checkpoints).
- metric: plan=paired detection AUROC + simple correlation → matches (Paulo-2024-style detection scoring is the standard SAE-feature evaluation metric).
- gpu_hours: plan~0.5 → revised ~0.5 (unchanged — GPT-5 API calls dominate wall-clock; target LLM forward is cheap).

**M1 method_sensitive fields:**
- n_pairs: plan=300 features × 25 texts × 3 methods = 22,500 forward passes → matches (paired same-feature-id design is intrinsic to the SAE-feature comparison protocol).
- sites: plan=residual_stream @ {L4, L12, L20} → matches (Gemma-Scope release provides SAEs at these layers).
- metric: plan={GenAcc, Pearson ρ, detection AUROC} → matches (standard SAE-feature auto-interpretation metrics).
- gpu_hours: plan~5.0 → revised ~5.0 (unchanged — batched Gemma-2-2B forward with SAELens hooks is the dominant compute; GPT-5 wall-clock is parallel).
- K_max_sage_rounds: plan=3 → matches (SAE submethod does not constrain agent loop depth; 3 is the design default).

**M2 method_sensitive fields:**
- n_pairs: plan=150 features × 25 texts × 2 methods = 7,500 forward passes → matches.
- sites: plan=3 depths in Qwen3-4B → matches (transcoder-hp for Qwen3-4B; if unavailable, fall back to Qwen SAE or GPT-OSS-20B + resid-post-aa).
- metric: plan={GenAcc, Pearson ρ} → matches.
- gpu_hours: plan~3.5 → revised ~3.5 (unchanged).
- target_pair: plan=Qwen3-4B + transcoder-hp (default) → matches; fallback to GPT-OSS-20B + resid-post-aa if unavailable.

reconciliation_status: ok

## Rationale

The plan's `mechanism_strategy.directions = [Unit Interpretation]` (Direction 5) with SAE feature ids as the target unit maps unambiguously to Family 5 (Feature Dictionary Learning), submethod SAE. SAGE is exactly the "auto-interpretation" application described in Direction 5 — a model-explains-model loop on top of an SAE dictionary. The other five directions were correctly rejected in the plan (Location: unit is given; Causal Intervention: claim is explanation quality, not causal drive; Tuning & Editing: no capability tuning; Formation Tracing: no training-time claim; Decision Auditing: no downstream decision). No cross-round `families_already_settled` list exists (round 1). SAELens is the reference library — it handles Gemma-Scope loading, hook registration on the target LLM, and Neuronpedia integration (public feature explanations + activation corpus).
