# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Probing / Residual Stream States  +  Causal Attribution / Patching  +  Causal Attribution / Ablation  +  Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: "Verbal-Confidence Cache Hypothesis (C1)"
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md
  - skills/mechanism-skills/causal-attribution/ablation/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

## Candidates

1. **[recommended]** Probing / Residual Stream States  +  Causal Attribution / Patching  +  Causal Attribution / Ablation  +  Representation and Parameter Analysis / Steering Vectors — full Location → Causal Intervention composition matching the plan's four mechanism milestones (M2 probe screen → M3 patching → M4 attention-block → M5 steering). Each family/submethod is the canonical catalog match for one plan milestone; no downstream substitution.
   - paths:
     - skills/mechanism-skills/probing/residual-stream-states/SKILL.md  → M2 per-position × per-layer ridge probe
     - skills/mechanism-skills/causal-attribution/patching/SKILL.md    → M3 residual-stream patching (clean/corrupted pairs)
     - skills/mechanism-skills/causal-attribution/ablation/SKILL.md    → M4 attention-block cache → conf-gen (directional ablation on attention weights)
     - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md → M5 signed dose-response (diff-of-means / LDA + α sweep)

2. Feature Dictionary Learning / SAE (rejected) — Would attempt to decompose the cache position into monosemantic features. Cost: pre-trained SAE for gemma-3-27b-pt may not exist at the needed layer band; training one is prohibitive under 10 GPU-h. Also: `FINAL_PROPOSAL.md` §4 explicitly rejects Unit Interpretation (SAE/dictionary) — "confidence" is already the interpretable label.
   - path: skills/mechanism-skills/feature-dictionary-learning/SKILL.md

3. Causal Attribution / Attribution Patching (rejected as primary) — First-order approximation via a single backward pass; would trade fidelity for speed. Reserved as a fallback screen if exact patching (M3) is too expensive at the top-K sites, but the plan's `n_pairs=200` × 3 seeds × 3 sites is tractable exactly; no need for the approximation.
   - path: skills/mechanism-skills/causal-attribution/attribution-patching/SKILL.md

## Composition plan

Screen → decode → verify → recover, matched to plan milestones:

1. **M1 — data prep + activation cache** (2.0h). Not a mechanism method; feeds the cache for every subsequent milestone. Extracts residual-stream activations at the 12-layer grid × post-answer window positions (E0..E4) + confidence-gen position. Provides `answer.normalized_aliases` labels (from TriviaQA) + parsed verbal-conf integers + answer-token log-probs.
2. **M2 — Probing / Residual Stream States** (1.0h) — SCREEN & LOCATION. Ridge linear probe at every `(position ∈ {E0..E4}, layer ∈ {5,10,15,20,25,30,35,40,45,50,55,60})` cell against parsed verbal-conf (regression, R²). Three baselines in-script: `log_prob_only` / `conf_gen_position` / `shuffled`. Emits `top_k_sites.json` = top-K `(position, layer)` cells by R² gap over `log_prob_only`. This is the Location output ranking cache candidates.
3. **M3 — Causal Attribution / Patching** (1.5h) — SUFFICIENCY. Residual-stream patching at `top_k_sites_from_M2` (top-3), clean/corrupted pair pool (clean = high-verbal-conf item, corrupted = low-verbal-conf item) matched on template length; `n_pairs=200` × 3 seeds. Measures `verb_conf_shift`, `answer_acc_preserved`, `answer_logprob_shift` — the last two are the General Rule for mechanism/Interpretability's *general ability* metric (M6(d)).
4. **M4 — Causal Attribution / Ablation** (1.5h) — RETRIEVAL PATH. Directional attention-block (mask attention weights from confidence-gen query position to cache-position keys) at `top_layer_from_M2` and the `top_layer_from_M2..final_layer` band. Metrics: KL to template-conditional prior + Δ mean verbal-conf + answer accuracy preservation.
5. **M5 — Representation and Parameter Analysis / Steering Vectors** (1.0h) — DIRECTIONAL SUFFICIENCY. Extract a **confidence direction** at `top_1_site_from_M2` via diff-of-means AND LDA between high-verbal-conf and low-verbal-conf items on a training split; α sweep `[-4,-2,-1,0,1,2,4]` on held-out. Following `experiment-tips/steering-coefficient-tuning`, α is expressed in **σ_proj units** (α_effective = β · σ_l where σ_l = std(h_l ᵀ u_l)); this makes the α grid comparable across the two direction-extraction methods. `verb_conf_mean` (target) + `monotone_r_squared` (target) + `answer_acc_preserved` (general ability).
6. **M6 — Specificity + null controls** (2.0h). Four sub-experiments bundled: (a) matched non-cache-position controls repeating M3/M4/M5 at a control site (question body / template prefix), (b) recall-strength null within log-prob-percentile bins, (c) log-prob-restatement null (freeze answer token through EOA and patch), (d) answer-accuracy preservation across M3/M4/M5. Uses the same submethods as M3/M4/M5.

Reconciled cost per submethod (matches the plan's 9.0h sum):

| Milestone | Family / Submethod | Plan GPU-h | Reconciled GPU-h |
|---|---|---|---|
| M1 | (data prep) | 2.0 | 2.0 |
| M2 | Probing / Residual Stream States | 1.0 | 1.0 |
| M3 | Causal Attribution / Patching | 1.5 | 1.5 |
| M4 | Causal Attribution / Ablation | 1.5 | 1.5 |
| M5 | Representation and Parameter Analysis / Steering Vectors | 1.0 | 1.0 |
| M6 | (mix of above at control sites) | 2.0 | 2.0 |
| **Total** | | **9.0** | **9.0** |

## Plan reconciliation

<!-- One row per method_sensitive field declared on the intervention milestones. -->

**M3 (Causal Attribution / Patching):**
- n_pairs: plan=200 → matches — patching is exact per-pair (one forward pass per patch), so 200 pairs × 3 sites × 3 seeds = 1800 forward passes is tractable in 1.5h on a single 80GB A800.
- sites: plan=`${top_k_sites_from_M2}` (top-3 from M2) → matches — submethod ranks candidates by causal effect at those cells, exactly what M2 emits.
- metric: plan=`verb_conf_shift, answer_acc_preserved, answer_logprob_shift` → matches — canonical patching outputs (target metric + general-ability metric per General Rule for mechanism/Interpretability).
- gpu_hours: plan~1.5 → matches — no re-bind.

**M4 (Causal Attribution / Ablation):**
- sites: plan=`block_from=${top_k_sites_from_M2}, block_to=[confidence_gen_position], block_at_layer=[top_layer, top_layer..last_layer]` → matches — directional path-blocking on attention weights uses precisely this from/to/layer specification.
- metric: plan=`verb_conf_dist_kl_to_prior, verb_conf_mean_shift, answer_acc_preserved` → matches — KL to prior + Δ-mean + general-ability metric.
- gpu_hours: plan~1.5 → matches — no re-bind.

**M5 (Representation and Parameter Analysis / Steering Vectors):**
- sites: plan=`${top_1_site_from_M2}` → matches — steering targets the single top cache site.
- α_grid: plan=`[-4, -2, -1, 0, 1, 2, 4]` → re-bound to β · σ_l units — following `experiment-tips/steering-coefficient-tuning` we scale the plan's α values by the per-layer projection std σ_l = std(h_l ᵀ u_l). β = {-4,-2,-1,0,1,2,4} maps to α_effective = β · σ_l. This does not change the number of grid points or the scientific intent (signed dose-response with symmetric range) — it just makes the sweep comparable across the two direction-extraction methods (diff-of-means vs LDA) and across the recall-strength bins in M6(b). Recorded here as planned-vs-actual; plan file itself is not edited.
- metric: plan=`verb_conf_mean, monotone_r_squared, answer_acc_preserved` → matches — plus a fluency-collapse diagnostic (Δ answer accuracy > ~10% at α=±4 flags collapse-driven effect).
- gpu_hours: plan~1.0 → matches — no re-bind.

**M6 (mixed submethods at control sites):**
- n_pairs / sites / metric: match — M6 re-uses M3/M4/M5 hooks at a control site; no method-level re-binding.
- gpu_hours: plan~2.0 → matches — no re-bind.

reconciliation_status: ok

## Rationale

The plan's `mechanism_strategy: Location → Causal Intervention` is the shortest strategy that lands the claim. Location is a Probing family screen (M2); Causal Intervention is a **three-submethod** composition — patching for sufficiency (M3), attention-ablation for retrieval-path bottleneck (M4), steering vectors for signed dose-response (M5). Every rejected direction is explicitly named in `FINAL_PROPOSAL.md` §4 (Tuning & Editing — diagnostic not applied; Formation Tracing — inference-time not training-time; Unit Interpretation — "confidence" is already the interpretable label; Decision Auditing — confidence IS the model output; head-level circuit discovery — reserved as follow-up, position × layer × attention-block is sufficient to land the claim).

**Cross-round avoid-set (Rule 1):** `families_already_settled` is empty — round 1, single-claim `BEHAVIOR_SOURCE=given` reproduction — nothing to exclude.

**Aligned with tagging:** yes — the plan's Location tag maps to Probing family, the Causal Intervention tag maps to Causal Attribution + Representation and Parameter Analysis families.

**Compose-with-tips:** matched `experiment-tips/steering-block-selection` (M3/M4/M5 site set locked FIRST by M2's activation-based screen — probe R² across the 12-layer grid) and `experiment-tips/steering-coefficient-tuning` (M5's α sweep expressed in σ_proj units, with answer accuracy as the parallel general-ability metric).
