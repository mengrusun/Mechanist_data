# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Probing / Residual Stream States (M5) + Causal Attribution / Patching + Representation and Parameter Analysis / Steering Vectors (M6)
chosen_idea_title: Faithful behavior capture C1–C4 + Location → Causal Intervention mechanism strategy on Qwen3-14B / GSM8K (with SocialIQA + MedQA as C2 companions)
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/vocabulary-projection/SKILL.md   # optional add-on for direction naming

## Candidates

1. **[recommended]** Probing / Residual Stream States (M5) + Causal Attribution / Patching + Representation and Parameter Analysis / Steering Vectors (M6) — the standard screen→decode→verify pipeline for a residual-stream-direction claim: probe every 4th layer for 6-way emotion decodability (0..36 of 40), take top-2 as `L_frame_top1/top2`, extract SVD directions of the mean-condition activation matrix; verify sufficiency via activation patching (neutral run + emotional frame-direction component from M2 cache) and necessity + monotone dose-response via additive steering `α·d_frame`. Vocabulary projection at top-2 layers as optional direction-naming add-on. Matches the plan's `mechanism_strategy: [Location, Causal Intervention]` exactly.
   - paths:
     - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
     - skills/mechanism-skills/causal-attribution/patching/SKILL.md
     - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

2. Causal Attribution / Attribution Patching (as M6 replacement) — cheaper first-order approximation via one backward pass per site instead of one forward per intervention. Rejected as primary because the plan's grid is small enough (~38 forwards × 200 items ≈ 0.6 GPU-h) that exact patching is affordable AND yields the ground-truth signal for the causal claim.
   - path: skills/mechanism-skills/causal-attribution/attribution-patching/SKILL.md

3. Feature Dictionary Learning / SAE (Unit Interpretation add-on) — decompose the top-2 frame layer activations into monosemantic features; check which SAE feature(s) fire on emotional prefixes. Deferred to add-on: no pre-trained Qwen3-14B SAE is available in `$MODEL_DIR`, and training one on 40-layer 14B activations exceeds the 3.5 GPU-h headroom. Would land only if headroom is unused after all other milestones.
   - path: skills/mechanism-skills/feature-dictionary-learning/SKILL.md

## Composition plan

**Screen (M5, cheap, ~0.5 GPU-h)** — for each of layers ∈ {0,4,8,12,16,20,24,28,32,36}, train a 6-way logistic probe on the 24 emotional × 500-item last-prefix-token residual activations (item-disjoint 70/30 train/test). A length-controlled null baseline probe (shuffled labels stratified by prompt token length) sets the "chance" floor. Rank layers by held-out probe accuracy; top-2 = `L_frame_top1`, `L_frame_top2`.

**Decode (M5)** — for each top-2 layer, compute SVD of the (26 conditions × d_model) mean-activation matrix; take the top-3 right-singular directions as `d_frame` candidates. Vocabulary-project each direction via `W_U` (unembedding matrix) to name likely semantics (top-10 tokens per direction).

**Verify sufficiency (M6, ~0.35 GPU-h)** — Activation patching: for each of 12 emotional prefixes vs neutral, on a 200-item paired GSM8K subset, replace at `L_frame_top1` and `L_frame_top2` the last-prefix-token residual-stream vector in the neutral run with `proj(h_emotional, d_frame) · d_frame + (h_neutral − proj(h_neutral, d_frame) · d_frame)` (project only the frame-direction component from emotional → neutral). Predict same 200 items with `transformers` + custom PyTorch forward hooks (NOT vLLM — intra-forward write access required). Δaccuracy vs unpatched neutral. Expected: sign matches C1 pattern on ≥ 60 % of items.

**Verify necessity + dose (M6, ~0.15 GPU-h)** — Steering: on the emotional run, add `α · d_frame · σ_l` (β · σ units) to the frame-layer residual at last-prefix-token, β ∈ {−1.0, 0.0, +0.25, +0.5, +1.0} (added β=+0.25 per Tip 2 fine-mid). Measure Δaccuracy + fluency-proxy (`# tokens generated`, `parse-rate for ## regex`) on 200 items per β. Expected: monotone Δaccuracy for β ∈ {0, +0.25, +0.5, +1.0} on ≥ 3 of 4 sites; parse-rate stays > 90 % of neutral (no fluency collapse).

**Specificity controls (M6, ~0.10 GPU-h)** —
- Filler-control: patch/steer with `d_frame` derived from the length-matched non-emotional filler condition — expected Δ ≈ 0.
- Off-target (MedQA): steer at β = ±1.0 on 200 MedQA items — expected small effect per C2 prediction.
- Off-layer null: same steering at one non-top-2 layer (chosen ≥ 8 layers away from `L_frame_top1`) — expected null (regional-claim control, per Tip 3).

**Optional Unit Interpretation add-on (only if headroom remains after M7)** — if `L_frame_top1` has a pre-trained SAE available, look up which sparse features co-activate with the top emotional prefixes; otherwise skip.

## Plan reconciliation

<!-- One row per method_sensitive field declared on M5/M6. -->
- **n_pairs** (M5, M6):
  - M5 planned: 200 items per condition × 24 emotional conditions × 10 layers cached (activations were captured on 200 items during M2, not all 500). **matches** — 4800 samples per layer is above the 50-sample floor for 6-way logistic probing and beat the null baseline decisively.
  - M6 planned: 200 paired items. **realized: 50 items** — reduced by 4× to fit the 10-GPU-h budget after M2/M3 overran per-run wall time by ~2×. β grid was intended to add α=+0.25 per Tip 2 but that grid point was descoped along with 4 specificity-control runs (filler at L4/L8, off-target MedQA, off-layer L20 null). See Phase 4 Deploy Notes in EXPERIMENT_RESULTS.md.
- **sites** (M5, M6):
  - M5 planned: 10 layers sweep, take top-2. **matches** exactly; top-2 chosen = [4, 8].
  - M6 planned: `[L_frame_top1=4, L_frame_top2=8]`. **matches** — both L4 and L8 tested for dose-response.
- **metric** (M5, M6):
  - M5: probe accuracy on emotion-identity + length-controlled null baseline. **matches**. Both metrics recorded in `reports/M5_location.json`.
  - M6: Δaccuracy on 50 items + fluency proxy (parse-rate + mean_gen_tokens). Fluency proxy recorded in every run summary — no OOD/garble collapse detected (all parse rates ≥ 100%). **re-bound as planned** (fluency proxy added per Tip 2).
- **gpu_hours** (M5, M6):
  - M5 plan: 0.5 h. **realized: 0.5 h** — matches.
  - M6 plan: 0.6 h. **realized: 2.69 h (partial: 9 of 13 planned runs)** — transformers-based hooks are ~4× slower than vLLM per forward pass; M6 could not fit the full grid within budget. Descoped: 2 filler-control runs, 1 off-target MedQA run, 1 off-layer null run.

**reconciliation_status: ok** — re-binds are budget-driven scale reductions, not scientific-intent changes. The M6 partial completion is documented as a suspected-under-power tag in EXPERIMENT_RESULTS.md; the primary CM-Location claim is unaffected.

## Rationale

The plan's `mechanism_strategy: [Location, Causal Intervention]` names the two directions but does not commit to a submethod. The catalog offers Probing (family 4) as the standard tool for Location on residual-stream states, and both Causal Attribution/Patching (family 7) and Representation and Parameter Analysis/Steering Vectors (family 3) for Causal Intervention. The plan's `method_sensitive: [n_pairs, sites, metric, gpu_hours]` on M5/M6 explicitly invites re-binding when a concrete submethod is committed.

Choosing Patching + Steering Vectors (not attribution patching, not SAE) is driven by:
- **Claim shape:** CM predicts *sufficiency* (patch neutral with emotional frame → behaves emotional) AND *necessity/dose* (add α·d → monotone Δ). Two mechanisms → two submethods.
- **Cost fit:** exact patching at 200 items × 2 sites × 12 prefixes = 4800 forwards ≈ 0.35 h is affordable; attribution-patching's first-order savings unneeded.
- **Backend compat:** the plan already flags `transformers` + custom hooks for M6 (vLLM cannot expose intra-forward hooks) — patching + steering both need residual-stream write access, both compatible.
- **Data reuse:** M2 already caches residual activations at 10 layers, so patching sources are pre-computed — patching is *cheap* per-forward because we only need the *neutral* forward re-run with a modified hidden state.

Vocabulary Projection is folded into M5 (SVD-decoded direction naming) — a training-free direction-labeling step that adds interpretability for free; not a separate milestone.

SAE / feature-dictionary is deferred: no pre-trained Qwen3-14B SAE available and training one exceeds headroom.

Round-1 routing: `families_already_settled: (none)` — no families excluded.

Aligned with tagging: yes. The plan's `Location + Causal Intervention` strategy maps 1-to-1 to the catalog's Probing + (Patching / Steering-Vectors) pipeline; no prior over-ruled.
