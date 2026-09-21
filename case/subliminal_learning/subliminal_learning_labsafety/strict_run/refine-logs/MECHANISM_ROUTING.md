# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: Cross-modal covert transfer of unsafe behavior via a text-only teacher-generated channel
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/probing/SKILL.md
  - skills/mechanism-skills/causal-attribution/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors — CAA (Contrastive Activation Addition) natively provides both Location (v_l = mean(h_treated) − mean(h_CtrlB) per layer, our `d_diff`) and Causal Intervention (additive α · v_l hook + project-out variant, matched-random-direction specificity control, 7-α dose-response) in one primitive. Exactly matches plan's Location → Causal Intervention arc with the ontology (low-rank residual-stream direction in the language tower) already asserted in C2.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
2. Probing — a diagnostic-only companion for Location: train a linear probe (treated=1 / Ctrl-B=0) per layer, use AUC to cross-check `d_diff` layer ranking and cos(d_diff, probe_normal) as an alignment check. Cannot itself do the causal intervention leg, so not sufficient standalone — folded into the composition plan as a Location-diagnostic under the primary CAA family.
   - path: skills/mechanism-skills/probing/SKILL.md
3. Causal Attribution / Ablation — could handle the causal leg (project-out on treated is essentially a directional ablation), but does NOT natively handle Location (no direction extraction) and does NOT natively handle dose-response steering (α-sweep on base). Would force a two-family solution where CAA already handles both.
   - path: skills/mechanism-skills/causal-attribution/SKILL.md

## Composition plan

**Screen → Decode → Verify → Specificity** — all under one primitive (contrastive activation direction):

1. **Screen + Decode (M1.L-Core)** — cache pinned-site (last-text-token, post-attn+MLP-sum) residuals bf16 per (item ∈ flipped-wrong ∪ matched-agree, layer ∈ [0..32], arm ∈ {treated, Ctrl-B}, seed ∈ {42, 123, 2026}); build per-layer per-seed `d_diff_{l,s} = mean(h_treated) − mean(h_CtrlB)` with L2 normalization; Borda cross-seed rank aggregation on layer-metric `‖d_diff‖ · accuracy-conditioning`; select top-3 layers; run linear probe (treated=1 / Ctrl-B=0, 5-fold CV) at each layer as a diagnostic-only companion (Probing family cameo, per candidate #2); stability gate: top-6 layers appear in ≥ 2/3 seeds. If gate fails → L-Secondary LoRA-attribution fallback (Causal Attribution / Ablation on LoRA A rows), hard-stoppable to appendix if it would push total compute > 60 GPU-hrs.

2. **Verify — Sign (M2.2a)** — attach forward hook on the top-3 language-tower layers of the treated student that project OUT the `d_diff` direction (i.e., `h ← h − (h·û) û`) on every forward pass, run QA_I eval, compute recovery fraction r_s = (Acc_ablated − Acc_treated) / (Acc_Ctrl-A − Acc_treated) per seed. Pass iff r_s ≥ 0.30 in ≥ 2/3 seeds (STRONG POSITIVE threshold).

3. **Verify — Dose-response (M2.2b)** — attach forward hook on the base student's top-3 layers that ADDs α · v_layer for α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2} in units of σ_proj (per-layer std of h·û on the base cache; jointly normalized across top-3 layers), run QA_I eval per (α, seed) grid (7 × 3 = 21 runs); compute per-seed Spearman ρ over the 7 accuracy points; pass iff median ρ ≤ −0.5 AND ≥ 2/3 seeds have ρ < 0.

4. **Verify — Specificity (M2.2c)** — (a) matched-random-direction control: replace v_layer with v_random of equal L2 norm per layer, sweep same α grid; pass iff mean |ΔAcc| across α ≤ 1 pp; (b) off-target competence: run project-out on treated on `eval_pairs_948.json` items with safety-tag downsample if > 5 % safety-adjacent, else DELETE this milestone (task.md rule — no invented benchmarks); pass iff |Δ off-target Acc| ≤ 1 pp mean.

5. **Recover — Aggregate (M2.2d)** — pre-registered 3-level verdict: STRONG POSITIVE ⇔ 2a+2b+2c all pass; PARTIAL POSITIVE ⇔ 2a passes but 2b or 2c misses; BOUNDED NULL ⇔ L-Core stability gate fails AND (if fallback) L-Secondary fails, OR 2a recovery < 30 % in ≥ 2/3 seeds.

**Cost notes** — M1 cache ~12 GPU-hrs (6 runs × 2h) + M1.L-Core offline ~0.5h + M2.2a 3h + M2.2b ~6.3h + M2.2c ~3.5h ≈ **~25 GPU-hrs** for the mechanism arc (matches plan's ~24 hr estimate). L-Secondary fallback adds up to ~12 GPU-hrs (hard-stopped at 60 GPU-hr total).

## Plan reconciliation
<!-- Written per Phase 1.5 Step 7. One row per method_sensitive field declared on M1 / M2. -->
- n_pairs: plan="≤ 4000 items/seed × 2 arms × 3 seeds" → matches (CAA needs contrastive pairs; the flipped-wrong ∪ matched-agree slice per seed, capped at 4000, is exactly the plan's specification — CAA has no minimum-n requirement beyond `n ≥ 30–50` per class for stable mean-diff, which 4000 easily satisfies)
- sites: plan="top-3 language-tower layers by Borda cross-seed rank; residual after post-attn+MLP sum, last-text-token" → matches (CAA operates on residual-stream states; last-text-token pooling is the CAA default per steering-vectors/SKILL.md line 85; top-3 mid-to-late layers matches the block-selection tip's heuristic)
- metric: plan="Acc(QA_I), recovery fraction, Spearman ρ, |ΔAcc| specificity, cos(d_diff, probe_normal) diagnostic" → matches (all are standard CAA readouts; the tip 4 fluency-collapse companion metric is added as `OTHER` rate + eval_pairs_948 off-target accuracy — already in M2.2c)
- gpu_hours: plan~24 GPU-hrs (M1 + M2 without L-Secondary) → matches (routed submethod cost estimate ~25 GPU-hrs matches within 5 %; no re-bind needed)
reconciliation_status: ok

## Rationale

**Why #1 (Steering Vectors / CAA)** — The plan's C2 asserts a specific mechanism-ontology hypothesis ("low-rank residual-stream direction in the language tower"). CAA is the single method family whose primitive natively delivers all four legs (Location via mean-diff, sign via project-out, dose via α-sweep, specificity via random-direction match) under one direction extraction protocol — no cross-family adaptation needed. The 2025 EM literature (arXiv 2506.11618) uses exactly this primitive on the same ontology, so the CAA framing is also the current-literature-aligned way to make the mechanism claim publishable. Candidates #2 and #3 each cover only part of the arc; using them together would triple the surface area for a strictly worse result. The **General Rule for mechanism/Interpretability** (locate then intervene, always report both target and general-ability metric) is honored — Location fires before Intervention, and M2.2c's `off_target eval_pairs_948.json` plus the `OTHER` verdict rate serve as the general-ability companion metrics per the block-selection + coefficient-tuning tips.

**Cross-round avoid-set** — `families_already_settled: []` for this behavior+direction (this is round 1 for the multi_modal_strict subproject); no families excluded.
