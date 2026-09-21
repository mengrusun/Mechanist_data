# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: Unified verification suite for Claims 1-5 on Llama-3-8B-Instruct + AdvBench
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors — Difference-in-means direction extraction from contrastive activation pairs (harmful vs benign; refused vs complied) is the canonical CAA / Arditi 2024 lineage. Reads out via projection (M1, M2, M4, M5) and writes in via additive steering hook at (best-layer, best-position) (M3). One primitive covers the whole plan: read for correlational claims, add for causal claims. Handles the two-direction case cleanly (h and r as independent handles) and supports both random-direction and swap-direction specificity controls with no extra infrastructure.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md

2. Probing / Residual Stream States — Layer-wise linear-probe AUROC on residual-stream state is the standardized cross-layer scoring protocol M-prep uses as its diagnostic, M1 uses as its verdict, and M5 uses head-to-head against Llama Guard 3 8B. Supports the residual-stream location the plan targets (`residual_pre_ln` at layer × position); logistic regression is the "restricted hypothesis class" the plan specifies.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

3. Representation and Parameter Analysis / Representation Engineering (RepE) — The RepReading+RepControl view is the same operator as CAA but explicitly frames read-out (project onto direction to detect an attribute) and write-in (add direction to steer) as one framework — a clean alternative if the CAA framing had any gap. Kept as a fallback; not the primary because for the two-direction dissociation task the direct steering-vector recipe is more literal.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

## Composition plan

Screen → Decode → Verify → Recover, mapped to the six milestones:

- **Screen (M-prep)** — one forward-pass sweep over 32 layers × 6-position ladder on the training split of the AdvBench-harmful × Alpaca-benign contrast set. Cache residual-stream activations. Compute diff-mean directions h and r per (layer, position). Score each candidate by held-out logistic-regression probe AUROC. Cost ~1 GPU-h. This *is* the mechanism-family primary compute; downstream milestones are post-processing on the cache.
- **Decode / correlational (M1, M2)** — read out from cached activations only. M1: AUROC (h, r vs random-direction control + shuffled-refusal control), cosine(h, r) vs split-half reference. M2: position ladder × 2 attributes AUROC, crossover-Δs, bootstrap peak distinguishability. Both are pure NumPy/sklearn on cached tensors. Cost ~0.4 GPU-h combined.
- **Verify / causal (M3)** — the causal-intervention milestone. Additive steering hook `h_{l,p} ← h_{l,p} + α · d̂` where `d̂ ∈ {ĥ, r̂, random_norm-matched, swap}`. 4 directions × 7 α values × 200 pairs × short generation. Companion column `α_σ = α_‖d‖ × ‖d‖ / σ_l` written into results for cross-layer comparability (per experiment-tips/steering-coefficient-tuning). Fluency proxy (mean per-token log-prob + repetition rate on completion) logged alongside refusal (per experiment-tips). Cost ~4 GPU-h.
- **Recover — practical monitor (M4, M5)** — M4: paired-attack projection deltas on r and h, per attack family; detection AUROC of h-probe on successful-attack subset. M5: probe (linear + shallow MLP) trained on ⟨activation, h⟩ vs Llama Guard 3 8B head-to-head; AUROC + F1@FPR=5% + per-query FLOPs/wall-clock. Cost ~4 GPU-h combined.

Total: ~9.4 GPU-h — matches the plan.

Downstream analysis (bootstrap CIs, cosine similarity, F1@matched-FPR) is post-processing, not a mechanism family.

## Plan reconciliation

<!-- Reconciles committed submethod's needs vs the fields declared method_sensitive on the intervention milestones. -->
- **n_pairs**: plan=~520 harmful + ~520 matched benign (60/20/20 split → ~312 direction-extraction + ~104 held-out AUROC + ~104 Claim-5 held-out) → matches. Steering Vectors' diff-mean recipe is n-linear in a well-behaved regime by ~200-500 pairs; 312 direction-extraction pairs is comfortable.
- **sites**: plan=(best-layer, t_final-instr) for h; (best-layer, t_post-instr) for r; layer selected by M-prep held-out AUROC → matches. This is exactly what the Steering Vectors demo does (single-layer residual-stream hook at a fixed position), with the added position-crossover story (M2). Position ladder {t_final-instr-2..t_post-instr+2} for M2 is a natural extension of the demo's single-position pooling.
- **metric**: plan=(a) held-out AUROC for correlational claims; (b) Δ internal harmfulness readout + Δ refusal rate for causal claim; (c) per-query AUROC + FLOPs for M5 → matches. AUROC + Δ-projection are the Steering-Vectors-family standard metrics.
- **gpu_hours**: plan~9.4 h → **revised ~9.4 h — no change**. The committed submethod's compute profile is a single forward-pass sweep (M-prep) plus short-generation intervention passes (M3) plus a Llama Guard 3 8B forward pass on ~1000 examples (M5). This matches the plan's estimate. No cross-layer resampling penalty (activations are cached in M-prep), no dictionary training (SAE/transcoder), no ACDC-style edge-search cost.

reconciliation_status: ok

## Rationale

**Why #1 (Steering Vectors) is recommended.** The plan's mechanism_strategy (`Location + Causal Intervention`) is the exact composition Steering Vectors owns:
1. Location — diff-mean on contrastive pairs *is* the direction extraction; extracted directions are handles for both readout (project onto direction to score AUROC) and writeout (add direction to steer). No dictionary, no fine-tune, no per-attention-head sweep — the linear-direction assumption in the claim wording ("*approximately* linear") is exactly what this submethod tests.
2. Causal Intervention — the family's additive-hook primitive is what M3's dose-response + specificity controls require. Random-direction and swap-direction controls are already the family's standard specificity tests (see the demo's `analyze_vectors.py` cosine matrix + PCA).

Aligned with prior: yes — `mechanism_strategy.directions: [Location, Causal Intervention]` in EXPERIMENT_PLAN.md picks exactly this family. `families_already_settled: []` (round 1, no cross-round exclusions).

**Why #2 (Probing / Residual Stream States) is kept as a supporting candidate.** The layerwise probe-AUROC diagnostic in M-prep + M1 + M5 *is* Probing-family work, but downstream causal use lives in Steering-Vectors. In practice the two are complementary and both are used; the primary submethod slot goes to Steering Vectors because the causal claim (Claim 3) is the plan's headline result.

**Why #3 (RepE) is kept as a fallback.** If M3's additive steering shows unexpected off-target coupling (h steering also moves refusal readout or vice versa), the RepE framing's LoRRA (Low-Rank Representation Adaptation) or LoReFT-style richer readouts are the next-family fallback within the same mechanism family. Not the primary because the plan's "approximately linear" wording does not require it.
