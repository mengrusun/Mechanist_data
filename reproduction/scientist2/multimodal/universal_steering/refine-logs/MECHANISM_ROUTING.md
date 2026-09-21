# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / activation-steering (RFM concept-vector, supervised)
chosen_idea_title: RFM Concept-Vector Steering & Monitoring
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/activation-steering/SKILL.md
  - skills/mechanism-skills/probing/linear-probing/SKILL.md
  - skills/mechanism-skills/feature-dictionary-learning/sae-features/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / activation-steering — Additive residual-stream intervention `h_l ← h_l + α · v_c` with a *supervised* per-block concept direction. Exactly matches the plan's intervene-step (`h_l ← h_l + α · v_c` at chosen block) and covers C1–C4 uniformly. RFM (kernel machine + AGOP eigenvector) is the direction-extraction subroutine; the family here is the intervention paradigm.
   - path: skills/mechanism-skills/representation-and-parameter-analysis/activation-steering/SKILL.md
2. Probing / linear-probing — Per-block linear classifier on paired residual activations. Used as (a) the Location screen (block picker) inside every claim's Milestone-1, and (b) the primary classifier baseline for C5 alongside the RFM AUROC. Not the top-level intervention family — it is composed with (1) rather than replacing it.
   - path: skills/mechanism-skills/probing/linear-probing/SKILL.md
3. Feature Dictionary Learning / sparse-autoencoder — Unsupervised concept surfacing. Explicitly *excluded* by the proposal (task.md motivation names SAEs as an approach that "cannot reliably surface specific concepts of interest"). Listed here only to document that the exclusion is intentional, not accidental.
   - path: skills/mechanism-skills/feature-dictionary-learning/sae-features/SKILL.md

## Composition plan

For each concept the pipeline is:

1. **Screen (Location)** — cache residual-stream last-token activations for all 32 blocks on 300 paired train + 100 paired val sequences; fit a linear probe per block on train; score val accuracy; pick argmax block. (linear-probing subroutine — Candidate #2, downstream.)
2. **Extract (RFM step)** — at the chosen block, alternate T=3–5 iterations of: (i) kernel-ridge fit on paired activations with labels ±1, (ii) compute AGOP = E[∇f(x) ⊗ ∇f(x)] on train activations, (iii) reweight kernel by AGOP. Take top eigenvector of the final AGOP as unit-norm concept direction `v_c`. Log cosine sim to CAA mean-difference direction for audit. (Post-processing over Candidate #2's activations — belongs here, not as a top-level family.)
3. **Intervene (Candidate #1)** — additive residual-stream steering `h_l ← h_l + α · v_c` at the chosen block `l`, applied at every generated-token position after the prompt. α ∈ {-3, -2, -1, 0, +1, +2, +3} on dev split (7-point sweep), pick α* by max effect subject to length-degradation guardrail (|steered| ≤ 2× |baseline|).
4. **Evaluate** — task-specific:
   - C1/C3/C4: GPT-4o rubric-judge (`gpt-4o-2024-11-20`, 5-point).
   - C2: HackerRank test-case pass rate via compiled C++ (`g++ -O2 -std=c++17`).
   - C5: **classifier** — use `⟨h_l, v_c⟩` (or a matched linear probe on the same block) as the score; compute AUROC vs GPT-4o judge baseline (and ToxicChat-T5-Large for ToxicChat).

**Controls (per experiment-tips General Rule for mechanism):** matched-random-direction control at same `‖α·v‖` on C1/C2/C4; dev-locked α on held-out; sample size floors ≥50 (C1/C3), ≥20 (C4), ≥1000 (C5); block AUROC reported across all 32 blocks on C5 to guard against cherry-picking.

**Cost notes** (post-reconciliation, forward-looking):
- Screen (all 32 blocks × ~5 concepts × 400 seqs, last-token cache): ~2.0 h on 4 GPUs (parallelized concept-wise).
- Extract (RFM T≤5 iterations × 5 concepts): <0.5 h (RFM is CPU-bound on cached activations).
- C1/C3/C4 steer+judge (~700 held-out generations + GPT-4o judged): ~1.5 h wall.
- C2 steer+compile: ~1.5 h.
- C5 activation cache (2000 HalUeval + 1000 ToxicChat forward passes × 32 blocks last-token): ~1.5 h queue-dispatched.
- Buffer: ~2 h.

## Plan reconciliation
<!-- Written by Step 7 once a family is committed. One row per method_sensitive field declared on the intervention milestone(s). -->
- n_pairs: plan=300 train / 100 val → matches — activation-steering with an RFM extractor is well-served by n≈300 supervised pairs (RFM AGOP top-eigenvector is stable at this scale for linearly-separable concepts).
- sites: plan=residual-stream, per-block screen across all 32 blocks → matches — additive activation-steering canonically operates on residual-stream output of a transformer block; per-block Location screen is the standard block-selection subroutine.
- alpha_grid: plan={-3,-2,-1,0,+1,+2,+3} → matches — 7-point signed dose sweep is the standard steering-coefficient-tuning grid (per `experiment-tips/steering-coefficient-tuning`).
- token_scope: plan=every generated-token position after the prompt → matches — this is the standard additive-steering token-scope for causal steering (prompt tokens unaffected, generation tokens all receive `+α·v_c`).
- metric: plan=GPT-4o rubric shift (C1/C3/C4), test-case pass rate (C2), AUROC (C5) → matches — all three metrics are compatible with additive activation-steering (rubric/pass-rate over generations; AUROC over `⟨h_l, v_c⟩` scores).
- gpu_hours: plan~10 h → revised ~9.5 h — RFM extraction is largely CPU-bound (kernel + AGOP on ~400×d matrices), saving ~0.5 h vs a heavier submethod.
reconciliation_status: ok

## Rationale

The intervention paradigm is fixed by the proposal's four-step recipe (screen → extract → intervene → evaluate) and by the intervene step's formula `h_l ← h_l + α · v_c` — this is textbook additive activation-steering on the residual stream. RFM is a *direction-extraction subroutine* that produces `v_c` from paired data; it sits under this family, not as a competing family. Linear-probing is used twice — as the block-selector inside every claim and as the primary classifier baseline for C5 — so it must be co-loaded. SAEs are explicitly forbidden by the proposal's motivation ("cannot reliably surface specific concepts of interest") and by the plan's "Push AWAY from" list; documented as candidate #3 only to make the exclusion auditable.

No cross-round exclusions apply (this is round 1 for this behavior; `families_already_settled` is absent).

Aligned with proposal tagging: yes (Tuning & Editing → Location strategy, activation-steering family).
