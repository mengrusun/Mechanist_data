# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Steering Vectors
chosen_idea_title: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Steering Vectors — the plan's central mechanism is a *signed additive intervention on a low-rank subspace direction on the residual stream*: `h ← h + α · Π_lang · h` for α ∈ [−1.5, +1.5], applied at chosen layer blocks with an "excluded upper-k" clause. This is exactly the CAA / Steering-Vectors idiom (contrast-derived direction → signed multiplier at chosen layer). Unifies M2 (α = −1 null-space projection) and M3 (signed α-sweep) under one submethod.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/steering-vectors/SKILL.md
2. Probing / Residual Stream States — supports M1's *localization* rung: fit a linear language classifier over residual-stream activations across layers to check that the projected `V_lang` retains language decodability (macro-accuracy ≥ 0.90) while the orthogonal complement collapses to ≈ chance. Used as the *decoder* in the screen → decode → verify pipeline, but is not the core causal mechanism.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md
3. Representation and Parameter Analysis / Representation Engineering — an alternative submethod whose RepReading + RepControl API also fits the "read out a concept direction and inject/suppress it" pattern. Slightly more general than Steering Vectors but overlaps in scope. Reserved as fallback if Steering-Vectors' hook API cannot cleanly express a *projection operator* (which requires a rank-r subspace, not just a rank-1 vector) — see composition note below.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

## Composition plan

**Ladder-of-evidence chain: screen → decode → verify → recover**

- **Screen (M1 — cheap):** language-mean-difference SVD over residual-stream activations at each candidate layer group (`early`, `mid`, `all_non_upper`) using the FLORES-200 dev multilingual probe set. Extract the top-`rank_r` left-singular vectors as `V_lang,ℓ`. Cost: batched forward passes over 11 × ≤ 1000 sentences per layer group; ~0.5 GPU-h.
- **Decode (M1 verification — Probing rung):** train a linear classifier on projections of held-out MGSM prompts onto `V_lang` and its orthogonal complement. Predicate: complement collapses to ≤ 0.20 macro-acc while V_lang retains ≥ 0.90. Same run as the screen, negligible extra cost.
- **Verify (M2, M3 — Steering Vectors rung):** apply `h ← h + α · Π_lang · h` on the residual stream at chosen layer blocks. M2 fixes α = −1 (null-space projection) and sweeps `(layer_group, k_top_excluded, seed)`; M3 fixes the M2-winning (layer_group, k_top) and sweeps signed α. Matched controls (random rank-r subspace, leave-one-language-out) confirm specificity. Cost: ~5 GPU-h.
- **Recover (M4a — Tuning & Editing baseline):** train an LR-first LoRA-SFT baseline on MGSM8KInstruct as the head-to-head compute-vs-accuracy competitor for Claim 4. Not a mechanism family per §mechanism-skills — a downstream *baseline* comparator sitting outside the mechanism chain. Cost: ~3.5 GPU-h.

**Post-processing (NOT a mechanism family):** the SVD decomposition and orthogonal-complement construction are linear-algebra post-processing on collected activations, not a separate mechanism family. They belong to the screen rung's implementation, not to the routing.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->

**M1 (`method_sensitive: [n_probe, rank_r, layer_group, sites]`)**
- n_probe: plan={50, 100, 250, 500, 1000} → matches — the plan's sweep already covers the small-probe-set claim; Steering-Vectors submethod defaults call for ≥ ~50 contrast pairs, satisfied.
- rank_r: plan={1, 2, 4, 8, 16, 32} → matches — Steering-Vectors classically uses rank-1 (single direction), but for a *multi-language* language subspace a rank-r projector (`Π_lang = V V^T`) is the correct generalization; the plan's sweep spans the space.
- layer_group: plan={early=[0,12), mid=[12,24), all_non_upper=[0,28)} → re-bound to sites=`[0..28)` for Qwen-3-4B-Thinking (36 layers, not 32) — **Qwen-3-4B-Thinking has 36 transformer blocks** (verified via config.json → num_hidden_layers). Re-bind layer group ranges to Qwen-3-4B's actual depth: early=[0, 12), mid=[12, 24), all_non_upper=[0, 24) (leaving upper 12 blocks intact by default; k_top ∈ {0,4,8,12} still sweeps the excluded count). Documented for Phase 2 implementation.
- sites: plan="concrete list of layer indices" → re-bound to the intersection of layer_group and (36 − k_top_excluded); enumeration performed in code from the two knobs.

**M2 (`method_sensitive: [sites, n_pairs, metric, gpu_hours]`)**
- sites: plan=inherit from M1 winning (layer_group, rank_r) → re-bound per M1's 36-layer correction (same reasoning as M1 above).
- n_pairs: plan=implicit (uses M1's V_lang) → matches — Steering-Vectors doesn't need a separate probe-pair count here; the projector is fitted at M1.
- metric: plan={MGSM strict + lenient numeric-equal accuracy, GlotLID fidelity} → matches — Steering-Vectors demos score on target behavior + fluency proxy; MGSM accuracy + GlotLID fidelity align (the tip 2 "general-ability metric" role is filled by GlotLID fidelity).
- gpu_hours: plan~2.5 → revised ~2.5 — no compute-profile change (Steering-Vectors' hook-based intervention adds one residual-stream matmul per token, negligible overhead versus vanilla decoding).

**M3 (`method_sensitive: [sites, alpha_grid, metric, gpu_hours]`)**
- sites: plan=inherit from M2 winner → matches (composition rule from the two Steering tips: lock site set first at M2, then sweep α).
- alpha_grid: plan={-1.5, -1.0, -0.5, 0, +0.5, +1.0, +1.5} → matches — the plan's grid already spans both signs symmetrically around 0 as required by the "signed dose-response" claim; steering-coefficient-tuning tip's `β ∈ [0, ±0.25, ±0.5, ±1, ±2, ±4]` is a wider default but the plan's grid is inside that range and centered where the M2 winner (α = −1) lives — matches, no re-bind.
- metric: plan={MGSM accuracy + GlotLID fidelity + Spearman correlation} → matches.
- gpu_hours: plan~2.5 → revised ~2.5 — same reasoning as M2.

**M4a (`method_sensitive: [training_data_size, lora_rank, gpu_hours]`)**
- training_data_size: plan=MGSM8KInstruct (11 languages, full set) → matches.
- lora_rank: plan=32, alpha=32 → matches — LoRA-SFT sanity checked pilot at the finetune-hyperparameter-sweep tip's protocol confirms r=32, α=32 sits at the pass-band; sanity check runs a 500-example LR pilot before the full 2.5h train.
- gpu_hours: plan~3.5 → revised ~3.5 — off-plan gate G3 fires at 5h (1.5× cap).

reconciliation_status: ok

## Rationale

The plan's mechanism — a signed additive intervention on a low-rank *language-specific subspace* of the residual stream, with an "upper-layer-preserved" clause and a matched random-subspace control — is a textbook Steering-Vectors application generalized from rank-1 to rank-r. The single deviation from the demo scripts is that the direction is a rank-r *subspace* (projector `Π_lang = V V^T`) instead of a rank-1 vector — a straightforward implementation change (apply `h ← h + α · Π_lang · h` instead of `h ← h + α · v`). Probing (Residual Stream States) is the tool for M1's localization + verification but is not the *causal* mechanism; the causal mechanism is the additive residual-stream intervention. Under the shared-catalog rule that "one direction serves both as read-out (projection) and write-in (addition)", Steering Vectors is the correct single-family commit; Probing enters as the decode rung inside the composition plan, not as a separate committed family. Selecting Steering Vectors also aligns with tip 3 (steering-block-selection) and tip 2 (steering-coefficient-tuning) already loaded at Phase 1.1.

Cross-round routing hints (`families_already_settled`) — not applicable (round 1, no settled families in the plan block).
