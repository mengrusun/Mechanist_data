# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Probing / Residual Stream States + Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Candidates

1. **[recommended]** Probing / Residual Stream States  +  Representation and Parameter Analysis / Steering Vectors — Location via layer-swept L2-logistic-regression probes on `outputs.hidden_states[L]` at last-input-token yields `v_c^L, v_v^L` with per-layer AUROC / Spearman ρ / ECE + retrain-on-bootstrap CIs (C1, C2, and — via cosine on the probe weight vectors — C3a). The same two direction vectors are then used as additive steering interventions (α·σ_L\*·v with matched-magnitude random control) at layer L\* via HF forward hook to test C3b. Screen (probing) → verify (steering) is the canonical composition for a matched-pair geometric-and-causal claim about linear representation directions.
   - path (Location): skills/mechanism-skills/probing/residual-stream-states/SKILL.md
   - path (Causal Intervention): skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

2. Probing / Residual Stream States  +  Representation and Parameter Analysis / Representation Engineering — same Location step, but Causal Intervention via RepE's RepControl (concept-direction addition during generation with LAT-selected read/write layers). Provides a lightly different intervention protocol; requires the RepE Reading step to extract a direction from contrastive prompts, which for gold correctness would require constructing correct-vs-incorrect contrastive pairs — probe-based v_c is more directly matched to the plan's paired-sample framing.
   - path (Location): skills/mechanism-skills/probing/residual-stream-states/SKILL.md
   - path (Causal Intervention): skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

3. Probing / Residual Stream States  +  Causal Attribution / Patching — Location via probes; Causal Intervention via exact activation patching (swap the residual state at layer L* with a counterfactual state). Would establish causal necessity of the extracted direction more strongly than additive steering, but the plan's C3b test is specifically framed as **cross-direction additive steering with matched-magnitude random-direction control on a probe readout**. Patching is not a direct match for the ratio/absolute-effect criterion the plan already declares and would require redefining the C3b metric.
   - path (Location): skills/mechanism-skills/probing/residual-stream-states/SKILL.md
   - path (Causal Intervention): skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Composition plan

1. **Screen — Probing / Residual Stream States**: fit L2-logistic-regression probes (`probe_c_binary`, `probe_v_binary`) and L2-linear-regression / ordinal probes (`probe_v_primary`) on the residual stream `outputs.hidden_states[L]` at the last-input-token across all 32 layers (~2.5 h for hidden-state + label collection). Select `L*` by mean of normalized AUROC across the two binary probes (matches plan §6.8). Extract `v_c^L*` and `v_v^L*` as the L2-normalized weight vectors of the two binary probes.
2. **Decode — Cosine similarity + neighborhood robustness**: analysis on probe weights only (~0.2 h). Compute `|cos(v_c^L, v_v^L)|` per layer + 1000-resample retrain-on-bootstrap 95% CI at L*, plus L*±2 neighborhood robustness. Directly tests C3a. Belongs in the composition plan as post-processing on Location outputs (per the routing rule: pure linear algebra on collected direction vectors is not a mechanism family in its own right).
3. **Verify — Representation and Parameter Analysis / Steering Vectors**: HF forward hook on residual-stream output of block L*, add `α·σ_L*·v` at last-input-token, α ∈ {−1σ, 0, +1σ}, direction v ∈ {v_c^L*, v_v^L*, random_unit^L* matched norm}, 500 held-out test samples → 4500 forward passes (~1.0 h). Measure Δ probe readout (primary: internal) and Δ emitted output (secondary: verbalized confidence number under v_v-steering, correctness under v_c-steering). Perplexity safety cap ≥ 3× baseline → halve α (Tip 2 fluency-metric requirement).
4. **Recover — (out of scope)**: no circuit-discovery step — the plan's C3 is a direction-level geometric/causal claim, not an edge-level circuit claim. Explicitly cut per FINAL_PROPOSAL.md §5 non-contributions.
5. **Robustness ablations**: nulls (random-direction probes, shuffled-label probes) + paraphrase (P1 Tian, P2 Likert) on 500 dev — analytical / cached hidden state re-use, ~0.5 h. Feeds C1/C2/C3a robustness.
6. **Single-pass unified-prompt variant**: 10k forward passes with the unified prompt, re-fit probes on H_single^L, re-compute cos — ~1.0 h. Directly tests the two-context objection on C3a.

**Post-routing total GPU-hours estimate**: ~5.2 h main experiment (unchanged from plan — the committed submethods use exactly the hook, position, and probe conventions the plan is already sized for).

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- n_pairs: plan=10,000 probe-training / 2,000 dev / 2,000 test; 500-sample steering slice → matches — the plan's 10k / 2k / 2k / 500 breakdown is standard for Probing (thousands-scale) + Steering (~500 held-out for stable Δ CIs). No re-bind.
- sites: plan=`outputs.hidden_states[L]` (post-block residual) at last-input-token, layer L* selected by normalized-AUROC max → matches — this is Probing / Residual Stream States' canonical site convention, and Steering Vectors uses the same site for its additive hook (last-input-token residual at block L*). No re-bind.
- metric: plan=probe AUROC / Spearman ρ / ECE (C1, C2); `|cos(v_c, v_v)|` with bootstrap CI (C3a); Δ probe readout + Δ emitted output with matched-magnitude random-direction control on ratio-or-absolute criterion (C3b); McNemar on binned accuracy cells (C3c) → matches — every metric is a direct output of Probing (AUROC/ρ/ECE), linear algebra on probe weights (cos), or Steering Vectors (Δ under intervention). No re-bind.
- gpu_hours: plan~5.2 h → revised ~5.2 h — the committed submethods (linear-probe fits + HF-hook additive steering) use no more compute than the generic pre-routing estimate accounted for. Hidden-state collection (~2 h) dominates; probe fits are CPU; steering is 4500 short generations. No re-bind.
reconciliation_status: ok

## Rationale

Recommended candidate #1 is the *canonical* Location → Causal Intervention pair for a residual-stream linear-direction claim on a language model:

- The plan's `mechanism_strategy: [Location, Causal Intervention]` directly maps to Probing (Location) + Steering Vectors (Causal Intervention). Any other pairing would either mis-match the claim structure (Patching would require redefining C3b's Δ-under-additive-steering metric) or require re-extracting the directions from a different signal (Representation Engineering uses contrastive-prompt LAT to extract directions, whereas the plan's `v_c` and `v_v` are probe-derived — a direct match to Probing's decoder weights).
- Both submethods' canonical sites (residual stream at last-input-token, `outputs.hidden_states[L]`) match the plan §6.1–§6.2 hook definition **exactly** — no reconciliation needed on sites.
- The submethods' method_sensitive parameters (n_pairs, sites, metric, gpu_hours) all match the plan's declared values — reconciliation is `ok` on every field.
- `families_already_settled`: none listed in the plan metadata → no cross-round avoid-set exclusion applies.
- Auto-selection under `AUTO_PROCEED=true` fired immediately on candidate #1.
