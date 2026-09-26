# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors (CAA)
chosen_idea_title: given-validation faithful capture — Cross-Modal Subliminal Safety-Competence Transfer on Qwen3.5-9B Multimodal
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors (CAA) — the plan's M1/M2 shape is exactly CAA: extract a direction from mean-difference of activations on a safety-relevant prompt slice (M1), then intervene additively with a dose-response sweep α ∈ {-2,-1,0,+1,+2} (M2). One direction = one handle for both read-out (M1 correlational — effective rank of activation-difference per layer) and write-in (M2 causal — dose-response on QA_I plus matched-control and MMLU-specificity gates). Layer selection is method-native (mid-to-late residual stream). Composes with `experiment-tips/steering-block-selection` (site) and `experiment-tips/steering-coefficient-tuning` (σ-normalized α). Perfect fit for the C3a low-dim substrate hypothesis and its natural fallback (distributed rewrite negative result).
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

2. Causal Attribution / Activation Patching — swap treated's residual stream slice into Ctrl's forward pass (and Ctrl's into treated's). Higher causal fidelity than steering because it replaces the entire activation rather than adding a fixed direction. Trade-off: (a) does not naturally produce a *rank* metric for M1 (patching is per-site, not per-direction), (b) does not compose cleanly with matched-control direction (M2.b needs a random direction of the same rank at the same layer — patching's counterfactual is an activation, not a direction), (c) more expensive per α point (M2's 20-run grid becomes 20 forward-pass patch runs).
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

3. Probing / Residual Stream States — train a linear probe on treated-vs-Ctrl activations at each layer, use probe weights as the direction. Alternative to mean-difference. Trade-off: adds a train-time step (probe fit + eval), and probe accuracy is decodability not causality — still needs M2 causal to close the loop. Since the plan already frames M1 as difference-of-means (kind-level effective rank), probing is redundant.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

## Composition plan

Screen → Decode → Verify pipeline aligned to M1 → M2 → M3:

- **Screen (M1)**: extract `v_l = mean(h_l^treated) - mean(h_l^ctrl)` at every language-tower layer on the safety-relevant prompt slice (QA_I + text-only chemistry-safety paraphrases). Compute per-layer **effective rank** of the activation-difference matrix (PCA cumulative-variance-explained crossing 90 % OR participation ratio), report the top-K=1,2,4,8,16 directions at the top-3 most-divergent layers. Cost: 1 GPU × 30 min–1 h (one forward pass through the safety-relevant set per model arm; hooked to capture residual-stream at every layer once).
- **Decode (M1 partial)**: report layer indices, layer types (linear-attention vs full-attention, 3:1 pattern in Qwen3.5-9B), and effective rank per layer. Save direction vectors + σ_projs to `mechanism/m1_directions.pt`. Downstream analysis; no additional forward passes.
- **Verify (M2)**: apply the top-K M1 directions as additive hooks on the residual stream at the M1-selected layer(s), sweep α ∈ {-2, -1, 0, +1, +2} in σ_proj units (so α=1 means +1σ of on-direction magnitude — per `experiment-tips/steering-coefficient-tuning`). Four `run_kind` arms:
    - `real_treated`: negative-steering on treated (α ≤ 0 should raise treated toward Ctrl).
    - `real_ctrl_injection`: injection on Ctrl (α ≥ 0 should reproduce the drop).
    - `control_treated`: matched random direction at same layer, same rank, same α — must produce < 1/3 the effect.
    - `mmlu_treated`: real direction on treated but eval on MMLU slice — must drop ≤ 2 pp.
  Cost: 1–2 GPU × 2–4 h. Cross-seed replication (M2.d) adds 4 runs at α=-1, α=-2 on seed-200 and seed-300.
- **Recover (M3 optional)**: decode the top direction via vocabulary projection through `W_U` (logit-lens tokens) — this is the simplest of M3's three recipes and adds nothing to routing cost.

Downstream analysis (PCA / participation ratio / effective rank) is post-processing on collected activations — not a separate mechanism family, per the routing file's guidance.

## Plan reconciliation

<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->

M1 (`method_sensitive: [n_pairs, sites, metric, gpu_hours]`):
- n_pairs: plan=500 image + 300 text-only → **matches** — CAA converges well with 300–800 contrastive pairs (see `experiment-tips/steering-coefficient-tuning` recipe). 800 total meets the family's floor.
- sites: plan="most-divergent layers (top-K = 1,2,4,8,16)" → **re-bound "screen every 2 layers across all 32 language-tower blocks, then focus on the top-3 divergent"** — per `experiment-tips/steering-block-selection`, spaced-interval screening is the right protocol; single-layer picks are brittle. Concrete layers: `{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}`.
- metric: plan="effective rank (PCA cumulative-variance-explained crossing 90 %) OR participation ratio" → **matches** — both are standard CAA screens.
- gpu_hours: plan~1–2 h → **matches (~1 h realized)** — one hooked forward pass per model arm on 800 pairs, no backprop.

M2 (`method_sensitive: [n_pairs, sites, metric, gpu_hours]`):
- n_pairs: plan=200 QA_I items → **re-bound to 500 QA_I items** — since QA_I total N is ~948 (from `eval_pairs_948.json` in the data dir; full parquet is ~133 with paired items) the CAA dose-response needs adequate n for the ±2 pp specificity check on MMLU. Use full QA_I (matches plan's "no subset" rule).
- sites: plan="most-divergent layer(s) from M1" → **re-bound to top-3 M1 layers with a fallback to 3-layer sliding window if single-layer inert** — per `experiment-tips/steering-block-selection`.
- metric: plan="QA_I accuracy per α, per direction, per arm" → **matches** — add per-α fluency check (mean output length + repetition rate) per `experiment-tips/steering-coefficient-tuning`.
- gpu_hours: plan~2–4 h → **matches (~3 h realized)** — 20 runs of full QA_I eval + 4 cross-seed runs, each ~10 min on one GPU with `bs=32` and greedy decoding.

M3 (`method_sensitive: [metric]`): metric: plan="concept_dict / sae_topk / logit_lens" → **re-bound to logit_lens only** — vocabulary projection through the model's own W_U is the cheapest and most compatible with the CAA direction; concept_dict + sae_topk require external artifacts not available for Qwen3.5-9B. **This drops M3 from 3 runs to 1 run** (~30 min).

reconciliation_status: ok

## Rationale

Recommended #1 (Steering Vectors / CAA) is selected because:
- **Method-to-claim fit**: the C3 hypothesis is "low-dim safety-relevant activation subspace inside the language tower shifts between treated and Ctrl". A steering-vector direction *is* that subspace (rank-K). CAA turns C3a (low-dim) and C3b (causal sufficiency) into a single primitive.
- **Compose with tips**: `experiment-tips/steering-block-selection` and `experiment-tips/steering-coefficient-tuning` were both matched in Phase 1.1 — they are designed for exactly this family (loading protocol requirement: "steering vector or CAA" triggers them). Applying them protects against the two silent failures (α too small → no effect; α too large → collapse mimics C3b).
- **Cheap enough at full scale**: no probe training, no SAE training, no backward-pass attribution. One hooked forward pass per arm per (α, direction) point.
- **Natural distributed-rewrite negative-result path**: if M1 effective rank > 32 broadly, the direction is not low-dim and the paper narrative pivots cleanly to "distributed rewrite" — a legitimate CAA null.

Reconciliation exceptions: M1 sites are re-bound from "top-K divergent" to "spaced-interval screen of all 32 layers, then top-3 divergent" per the block-selection tip's spaced-sweep rule (a single hardcoded top layer is brittle across models of different depth). M3 recipe is re-bound from 3 recipes to logit_lens only because concept_dict and sae_topk require external artifacts that are not available for Qwen3.5-9B and are out of scope for this stage — this is a compute reduction from ~3 h to ~30 min, not a science change.
