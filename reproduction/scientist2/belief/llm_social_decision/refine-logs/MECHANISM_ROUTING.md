# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: Steerable per-variable pure directions in Llama-3.1-8B-Instruct's dictator decision (unified 4-claim mechanistic-evidence pipeline)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Inputs read

- `refine-logs/FINAL_PROPOSAL.md` — 4 mechanistic-evidence claims (C1 linear encoding · C2 purity · C3 bidirectional causal steering · C4 selectivity) on Llama-3.1-8B-Instruct dictator decisions over G/A/I/M.
- `refine-logs/EXPERIMENT_PLAN.md` — `mechanism_strategy: [Location, Causal Intervention]`; M2 extraction + probing; M3 LEACE / Gram-Schmidt decorrelation; M4 signed-α CAA activation-addition; M5 4×4 selectivity matrix; M6 ablations; M7 portability seed.
- `families_already_settled: []` (round 1, no exclusions).
- `/mechanism-skills` catalog (routing entry point) loaded in full; `representation-and-parameter-analysis/SKILL.md`, `representation-and-parameter-analysis/steering-vectors/SKILL.md`, `causal-attribution/SKILL.md`, `probing/SKILL.md` loaded from disk in this turn.

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors — CAA-style paired difference-of-means directions on residual stream + additive intervention at inference time with signed magnitudes, both signs. Matches the plan's stated method thesis verbatim: extract `v̂_V = mean(h(p') − h(p))` on paired minimal-edit prompts (Location), purify via LEACE / Gram-Schmidt against the other three variables (still Location, downstream linear-algebra post-processing), then `h ← h + α · ṽ_V` at α ∈ {−4σ, −2σ, −σ, 0, +σ, +2σ, +4σ} (Causal Intervention). Directly supports C1/C2 (extraction + probes decoded off the direction) and C3/C4 (dose-response + off-diagonal selectivity). Compute cost: ≤ 4 GPU-hours across M2 + M4 + M5.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
2. Representation and Parameter Analysis / Representation Engineering (RepE) — RepReading + RepControl protocol with contrastive-pair concept extraction. Equivalent primitive at the intervention layer (add a scalar-scaled direction into the residual stream) but with heavier scaffolding (PCA over stimulus batches, task-labelled positive/negative concept templates). Overkill for the plan's four pre-named social variables — a difference-of-means extractor is simpler and sufficient. Left as fallback if the pilot shows CAA directions are unstable across paraphrasings.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md
3. Probing / Residual Stream States — auxiliary linear probes to *decode* V from residual activations. Supports C1's linear-encoding evidence and C2's cross-leakage matrix, but does NOT by itself produce a causal intervention handle (probe accuracy is decodability, not causality — see the family SKILL). Included because the plan's M2 already uses probes; probing must compose with steering-vectors for C3/C4 rather than replace it.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Composition plan

**Screen → Decode → Verify → Recover** across the four claims:

1. **Screen** (M2 layer selection) — cheap probe-accuracy sweep per V × layer on the 800-train paired-partner activations picks each variable's best residual-stream layer ℓ_V*. This is a decodability filter (Probing family) that narrows the candidate site for the more expensive causal intervention.
2. **Decode** (C1 + C2 evidence, M2/M3) — Probing / residual-stream-states linear probes on `v̂_V` and on the LEACE / Gram-Schmidt purified `ṽ_V` produce the linear-encoding evidence (C1: probe accuracy ≥ 0.80; projection–transfer β) and the 4×4 cross-leakage matrix (C2: off-diagonal drops to chance + 0.05). LEACE / Gram-Schmidt are downstream linear-algebra post-processing on directions already extracted by Steering Vectors — they belong here, not in the candidate slot.
3. **Verify** (C3 + C4, M4/M5) — Representation and Parameter Analysis / Steering Vectors' additive intervention `h ← h + α · ṽ_V^decorr` on residual stream at S = {ℓ_V*} (default) and S = {ℓ_V* − 1, ℓ_V*, ℓ_V* + 1} (ablation window), α ∈ {−4σ, −2σ, −σ, 0, +σ, +2σ, +4σ} × decorrelator ∈ {GS, LEACE} × site ∈ {single, window3} = 112 M4 runs. C3 dose-response + inversion; C4 4×4 selectivity matrix at α ∈ {−2σ, 0, +2σ}. Coherence gate on free-form generations at |α| ≥ 2σ.
4. **Recover** (M6 baselines & ablations) — random-direction control (B2), raw-vs-pure ablation (B1), mean-centred control (B3), directional ablation `h ← (I − ṽ_V ṽ_Vᵀ)h` (B4) — separates the linear-direction mechanism from generic norm-perturbation confounds.

Cost notes (GPU-hours per milestone, revised where reconciliation applies):
- M2 (extraction + probes) ≈ 1.5 h (5,000 prompts × forward-only × bf16 batched at 32).
- M3 (decorrelation) ≈ 0.3 h (mostly CPU; small extra held-out cache).
- M4 (α-sweep) ≈ 2.5 h (112 conditioned decodes × 200 held-out with cached model).
- M5 (selectivity) ≈ 0.5 h (reuses M4 cache at α ∈ {−2σ, 0, +2σ}, LEACE, single-site).
- M6 (ablations) ≈ 1.5 h (~64 runs, cached model).
- M7 (portability, DeepSeek-8B) ≈ 1.5 h (nice-to-have).
- Total ≤ 7.8 GPU-h ; well under the 10 GPU-h HARD budget.

## Plan reconciliation
<!-- Written once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- n_pairs (M2, M4, M7): plan=800-train + 200-held-out per V (4,000-total paired pairs) → matches. CAA / difference-of-means is stable at n≥800 per variable per Rimsky et al. 2024; the plan already exceeds any published CAA sample floor.
- sites (M2, M4, M7): plan=residual-stream layers picked by probe accuracy × projection-transfer β on train, evaluated on held-out → matches. This is Steering Vectors' canonical site convention (last input token; per-layer probe screen; residual stream). No re-bind needed.
- metric (M2, M3, M4, M5, M7): plan= per-V probe test accuracy, projection–transfer β, mean-transfer shift, 4×4 leakage / selectivity matrix, coherence gate on free-form generations → matches. All five are standard Steering Vectors + LEACE readouts; the family's `SKILL.md` script uses precisely `norm`, `projection`, `dose-response`, `similarity heatmap` primitives.
- gpu_hours (M2, M4, M5, M7): plan~1.5 / 2.5 / 1.5 / 1.5 → revised ~1.5 / 2.5 / 0.5 / 1.5 — M5 estimate cut in half because the committed Steering Vectors implementation shares model + direction in memory with M4 and reuses M4's α ∈ {−2σ, 0, +2σ} decodes verbatim (only the *measurement* differs, not the forward pass). No re-bind on M2/M4/M7; the plan's estimate already assumes a residual-stream-hook CAA implementation.

reconciliation_status: ok

## Rationale

Selected #1 (**Steering Vectors / CAA**) because:
- It is the exact primitive the plan and proposal describe verbatim — paired-difference direction extraction plus additive `α · v̂` intervention. Any other candidate would re-derive it from a heavier scaffold.
- The demo library (`steering_vectors`) ships a Llama-family forward-hook `apply_steering_hook` implementation that matches Llama-3.1-8B-Instruct's `model.model.layers[layer]` structure directly (Llama-2/3 share the HF module tree; port is trivial — `token=os.getenv("HF_TOKEN")` becomes `local_files_only=True` for our local checkpoint).
- Composes cleanly with LEACE / Gram-Schmidt (linear post-processing on the extracted direction) for C2 and with the 4×4 selectivity measurement for C4 without needing a second family. Downstream analysis stays inside the same forward pass.
- Rejects **Causal Attribution / Patching** family: patching replaces one activation with another counterfactual, but our claim is that a *direction* is causally sufficient (add α · v̂, not swap in h(p')). Steering Vectors is the correct primitive for "is a *direction* sufficient?" per the family SKILL selection table.
- Rejects **Circuit Discovery**: the plan explicitly excludes it (not part of the four-part claim; no edge-level story).
- Rejects **Feature Dictionary Learning / SAE**: the concepts are pre-named (G/A/I/M); auto-interp not required. Overkill for a diagnostic claim on four labelled variables.
- Rejects **Vocabulary Projection / logit-lens**: reads token-space content; does not test causal sufficiency of a direction.
- Rejects **Parameter-Space Task Vectors**: parameter-space edits require fine-tuned checkpoints per variable; wildly out-of-budget and off-thesis.

Cross-round avoid-set: empty (round 1). No family excluded.
