# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Causal Attribution / Attribution Patching + Patching
chosen_idea_title: A Sparse, Modular Circuit Implements Propositional-Logic Reasoning in LLMs
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/causal-attribution/attribution-patching/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md
  - skills/mechanism-skills/causal-attribution/ablation/SKILL.md
  - skills/mechanism-skills/circuit-discovery/SKILL.md

## Candidates

1. **[recommended]** Causal Attribution / Attribution Patching (screen) + Patching (verify) — anchored to `mechanism_strategy: [Location, Causal Intervention]`. Attribution patching (Syed et al. 2024, arXiv 2310.10348) supplies the cheap Location screen: a single backward pass per (clean, corrupt) pair ranks every `L·H + L` component, producing the shortlist for C1. Path patching / activation patching then verifies C2 (role dissociation), C3 (necessity + sufficiency) with resample ablation (per Best-Practices arXiv 2309.16042). Fits every `method_sensitive` field the plan declared (`n_pairs=500` at anchor cell, `metric ∈ {logit_diff, prob_diff, KL}`, sites at head + MLP level, ~8.7 GPU-h). This is the canonical mechanistic pipeline for this claim family (IOI / ACDC / Attribution-Patching lineage).
   - path: skills/mechanism-skills/causal-attribution/attribution-patching/SKILL.md
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md
2. Causal Attribution / Ablation (resample-ablation-only variant) — same family, but resample-ablate the shortlisted components instead of patching from a clean run. Cheaper (no clean/corrupt matched pairs needed) but weaker: cannot cleanly separate necessity vs sufficiency, and cannot dissociate roles at the head level without a per-role matched-corruption pool. Retained as fallback if OOM on the patching path.
   - path: skills/mechanism-skills/causal-attribution/ablation/SKILL.md
3. Circuit Discovery / EAP-IG (edge-level attribution) — moves from per-component scoring to per-*edge* scoring, recovering communication paths between heads and MLPs. Attractive for a paper-quality circuit diagram, but exceeds the 10-GPU-h budget once the shortlist reaches ~15 components (edges = O(K^2) with per-edge integrated-gradients). Deferred as follow-up.
   - path: skills/mechanism-skills/circuit-discovery/SKILL.md

## Composition plan

**Screen → Decode → Verify → Recover** on Mistral-7B at the anchor cell (k=3, chain=2, natural):

1. **Screen (M1)** — Attribution Patching over all `32 layers × 32 heads + 32 MLPs = 1056 components`. Single backward pass per (clean, corrupt) pair on 500 pairs → gradient-weighted ranking of every component. Shortlist rule: cumulative-effect ≥ 0.9. Completeness / minimality via resample ablation. (~1.2 GPU-h.)
2. **Verify — necessity (M2)** — Path patching (Wang et al. 2022 style) on the shortlist. For each corrupted prompt, replace the shortlisted components' activations with the corresponding clean-prompt activations. Report Recovery on all three metrics + dose-response over 5 shortlist sizes + matched-size random control. (~1.5 GPU-h.)
3. **Verify — sufficiency (M3)** — Reinsertion patching. Take a clean prompt, resample-ablate every non-shortlisted component, reinsert clean activations only at the shortlist. Report Sufficient-Recovery over ≥ 5 resample seeds. (~1.0 GPU-h.)
4. **Verify — role dissociation (M4)** — Per-component single-site activation patching under each of three role-corruption pools (fact-swap / rule-swap / answer-swap), assembling role-assignment matrix `S ∈ [0,1]^{|C|×3}`. Null control: shuffled role labels × 100. (~1.8 GPU-h.)
5. **Recover — cross-cell stability (M4.stab)** — Repeat M4 on (k=5, chain=2) and (k=3, chain=3) cells, 200 pairs each. Compute pairwise Jaccard of role-block partitions. (~0.8 GPU-h.)
6. **Recover — cross-family (M5)** — Repeat the compressed pipeline (M1 → M2 → M3 → M4 abbreviated) on Gemma-2-9B at anchor cell, 500 pairs. Family-level comparison (schema recurrence, no head-index matching). (~2.0 GPU-h.)
7. **Recover — cross-scale contingent (M5.contingent)** — Gemma-2-27B at anchor cell, 250 pairs. Sharded 2×A800. Only if remaining budget ≥ 1.5 h AND model available. (~1.3 GPU-h if run.)

Framework: `TransformerLens` is preferred (native head-level hooks, activation-cache API, mature attribution-patching helpers). Fallback: `nnsight` (only if TransformerLens does not support the target model — Mistral-7B *is* supported natively; Gemma-2-9B is supported via `hooked_transformer.from_pretrained('google/gemma-2-9b')`).

## Plan reconciliation

<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->

- **n_pairs**: plan=500 (anchor cell, M1/M2/M3), 300 (M4), 200 (M4.stab), 500 (M5), 250 (M5.contingent) → **matches** — Attribution Patching's single-backward-pass-per-pair cost profile handles 500 pairs on a 7B model in < 1 h at bf16. Patching (M2/M3/M4) at 500 pairs is where the bulk of the budget lands; no need to trim.
- **sites**: plan=`(layer, head_idx)` per attention head and `(layer,)` per MLP → **matches** — TransformerLens' native `resid_pre`, `attn.hook_z`, `mlp.hook_pre` cache slots deliver exactly these granularities. Role dissociation runs at the same site set.
- **metric**: plan=`{logit_diff, prob_diff, KL}` reported side-by-side → **matches** — all three are computable from the same final-logit tensor, no re-run needed. Answer position: the token position at which "True" / "False" appears (fixed by the prompt template — position `-1` after the "Answer:" cue).
- **gpu_hours**: plan~8.7 h (+ 1.3 h contingent) → **matches** — TransformerLens overhead adds ~5% vs a hand-rolled hook layer, still within budget. Attribution patching at 500 pairs on Mistral-7B (32L × 32H bf16, seq_len ≈ 200) is ~1.0-1.2 h wall-clock on a single A800 80GB; patching sweeps at 500 pairs ~1.5 h. No re-bind needed.

reconciliation_status: ok

## Rationale

**Why #1 is recommended.** The plan is a **reproduction** of the sparse-modular-circuit hypothesis with three sub-claims (C1 sparsity / C2 modularity / C3 necessity+sufficiency) — the canonical instrument for each is:

- **C1 (locate a sparse component set)** — Attribution Patching is the state-of-the-art cheap screen. Faster than full ACDC iterative pruning by ~1-2 orders of magnitude, and its scores align with full-patching effect at r > 0.9 for effect ratios > 0.05.
- **C2 (modular decomposition into fact/rule/answer roles)** — Path patching per component × per role-corruption is the direct experimental instrument for assembling a role-assignment matrix. Alternatives (probing, SAE features) would take an indirect route (probe-per-role, feature-labeling) that would require additional labels or auxiliary training — a compute detour outside the 10-GPU-h budget and orthogonal to the anchor claim.
- **C3 (necessity + sufficiency by activation patching)** — this is the definition of C3 as stated in `task.md` verbatim. Anything else (ablation-only, correlational, gradient-only) would fail to test *both* directions.

The family is not settled from any prior round (round 1, no `families_already_settled` entries in the plan). No pin overrides.

**Why #2/#3 are not recommended.**

- **#2 (Ablation-only)**: fails the C3 verbatim test — cannot cleanly separate necessity from sufficiency using ablation alone, since resample-ablation gives you necessity (drop when removed) but not sufficiency (does the isolated set alone drive the output). Kept as OOM fallback for the M4 role-dissociation matrix if the memory footprint of the full clean-activation cache × 300 pairs exceeds available GPU memory (unlikely on 80GB A800 at 7B).
- **#3 (EAP-IG edge-level)**: over-budget and orthogonal to the sub-claim granularity. The three sub-claims are at the *component* level (heads and MLPs), not the *edge* level. EAP-IG would return additional interesting artifacts (the wiring between heads) but at the cost of ~3× more compute per pair, blowing the budget without changing any of the three verdicts.

**General Rule for mechanism/Interpretability compliance**: (1) locate — attribution patching pinpoints the component set (M1). (2) intervene without damaging general ability — the specificity control (matched-size random component set) is the direct off-target check; a random component set drawn from *outside* the shortlist should recover ≤ 0.2, giving a specificity gap ≥ 0.6. Reporting all three metrics (`logit_diff`, `prob_diff`, `KL`) ensures a `KL`-blowup (off-distribution) is visible even if `logit_diff` looks fine.
