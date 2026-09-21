# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors — The claim is literally that each reasoning behaviour occupies a linear direction in the residual stream, extractable from contrastive pairs and dose-response steerable via a single scalar α. This is the textbook CAA / steering-vector setup: mean-of-differences on paired activations, additive residual-stream hook at inference, α·σ_proj coefficient. Every one of C1–C4 maps onto this family's primitive; the direction serves as both the read-out (probe direction for C1) and the write-in (steering knob for C3/C4). No frontier extractor is required by the claim, and the plan's `method_sensitive: [n_pairs, sites, metric, gpu_hours]` licenses this binding.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
2. Probing / Residual Stream States — Complementary Location submethod for M1: trains a linear probe on cached residual-stream activations per layer and picks L*(b) via held-out ROC-AUC (exactly what M1 predicate 1 asks for). Used as the *screen* that decides which layer to steer at, not as the intervention.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md
3. Causal Attribution / Patching — A stronger causal test for C3 (patch-in the target-behaviour activation at L*(b) rather than add a linear direction). Rejected as primary because it does not test the *linear direction* claim (C1/C2) — it tests component causality irrespective of geometry — and it would consume 2–3× the M3 budget without touching the four sub-claims. Retained as a fallback if additive steering fails to produce a sign under M3.
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Composition plan

- **Screen (Location, M1)** — Probing / Residual Stream States: cache residual-stream activations at the last-token of every contrastive excerpt for all layers; per behaviour, train a linear probe per layer and pick L*(b) = argmax(held-out ROC-AUC); also report first-PC alignment of the diff-activation set at L*(b).
- **Decode (Location, M1 same milestone, plus M2)** — Steering Vectors: for each behaviour compute mean-difference direction v_b(L*) from the extract pool. In M2, sweep n_pairs ∈ {10,25,50,100,200} × seed ∈ {0,1,2} and measure split-half cosine + steering-effect ratio on a 50-task subset.
- **Verify (Causal Intervention, M3)** — Steering Vectors additive hook on the residual stream at L*(b), coefficient α·σ_L* over α ∈ {−3,…,+3}, generating 500-task chains under each α. LLM-judge (DMX gpt-5.4) scores per-chain behaviour presence + gibberish flag. Sign check + Spearman monotonicity + off-target specificity.
- **Recover (Tuning & Editing, M4)** — Steering Vectors at α ∈ {−2,−1,+1,+2}·σ_L* vs. NL-instruction prompts + Thinking Intervention token-insertion baseline, all scored on the same 500-task benchmark. Pareto plot of (behaviour-rate, accuracy).

Cost notes: M1 activation cache is a single forward pass over ~200 chain excerpts × 32 layers → cached to disk, reused by M2/M3/M4. M3 dominates the budget (18k generations at max_new_tokens 512).

## Plan reconciliation

<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- n_pairs: plan=up to 200 pairs extract + 40 held-out per behaviour → matches — mean-of-differences on ~160 pairs per behaviour is well within the CAA operating range (the CAA paper used ~50–200; our M2 sweep confirms sample-efficiency).
- sites: plan="residual-stream at L*(b)" (method_sensitive re-bindable) → re-bound to `layers = [model.model.layers[i] for i in range(num_layers)]` for M1 activation capture; steering hook at `model.model.layers[L*(b)]` post-block (residual stream) for M3/M4 — this is the canonical CAA hook site (Panickssery et al. 2024) and the Steering Vectors submethod default.
- metric: plan="held-out ROC-AUC ≥ 0.75; first-PC alignment |cos| ≥ 0.7; Spearman ρ ≥ 0.7; off-target |Δrate| ≤ 50%" → matches — these are exactly the metrics the linear-probe + mean-diff pipeline supports out of the box.
- gpu_hours: plan~5.5 h → revised ~5.0 h — CAA is the cheapest submethod in the family (no probe training on the extract pool beyond a scikit-learn LogisticRegression; the M2 60-run sweep re-uses the M1 cache and needs only inference on a 50-task subset; M3/M4 dominate). The 0.5 h savings are absorbed as headroom.
reconciliation_status: ok

## Rationale

**Why #1 is recommended.** The four claims C1–C4 are, verbatim, about (a) linearity of a per-behaviour direction, (b) extractability from a small contrastive pool, (c) dose-response causal control via a scalar coefficient on that direction, and (d) usability of that scalar knob compared to prompt engineering. The Steering-Vectors submethod is the exact primitive the paper is testing — mean-difference direction + additive residual-stream hook — and rejecting it in favor of a heavier extractor (SAE feature, DAS interchange, patching-based direction) would be over-scoping (see FINAL_PROPOSAL.md "Explicitly Rejected Complexity"). The Probing / Residual Stream States submethod is folded into M1 as the *layer-selection screen* (it operationalises the "held-out ROC-AUC" predicate) — not routed as an independent candidate for the mechanism claim, because probing decodability alone does not test causality, which C3 requires.

**Cross-round exclusions.** No `families_already_settled` list is carried by EXPERIMENT_PLAN.md (round 1), so no families are excluded.
