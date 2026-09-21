# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: per-claim-map
chosen_idea_title: Claim 1 — Scale-Dependent Emergence (natural entry point; the plan delivers all 4 claims as a coordinated M1→M2→M3&M4 chain)
effective_domain: mechanistic-interpretability
routing_mode: mode-b-direct-per-claim
candidate_paths:
  - n/a — user-pinned per-claim map (MECHANISM=given, BEHAVIOR_SOURCE=given); no routing candidates were generated
per_claim_map:
  C1: not-applicable                          # behavioural-only (M1); no mechanism intervention
  C2: fisher-information-matrix-zero-ablation # M2.1 - M2.4
  C3: checkpoint-analysis-with-zero-ablation  # M3 (reuses M2's H* on pythia-1b step-143000, applies across 154 → 24-available checkpoints; see note below)
  C4: probe-and-amplify-controller            # M4.1 - M4.3

## Committed per-claim mechanisms

| Claim | Milestone(s) | Mechanism family | Note |
|---|---|---|---|
| C1 | M1 | not-applicable — behavioural-only | 3×3 accuracy matrix + Wilson CI on `world_knowledge` / `personal_belief` / `attributed_belief` across `pythia-{410m, 1b, 2.8b}`. No internal-object intervention. |
| C2 | M2.1, M2.2, M2.3, M2.4 | Fisher-Information-Matrix + Zero-Ablation | Empirical Fisher on `person∈{james,mary}` subsets (n=454) and full `world_knowledge` (n=227), aggregated per-attention-head over `{W_Q^h, W_K^h, W_V^h, W_O^h}` (identified via GPTNeoXAttention's fused `query_key_value` per-head slicing). AND-NOT mask (top 0.1% target AND NOT top 1% knowledge). Deterministic greedy-add + greedy-remove smallest-head-set search under 4 verbatim criteria; 20 random-head + 20 random-mask controls. |
| C3 | M3 | Checkpoint-Analysis + Zero-Ablation | Reuses `H*_personal` / `H*_attributed` from M2 on pythia-1b step-143000. Applies unchanged (same head identities) at each intermediate checkpoint. 9 measurements per checkpoint (3 behavioural + 3 H*_personal-ablated + 3 H*_attributed-ablated). Behavioural + causal emergence detection with 2-of-next-3 persistence check. |
| C4 | M4.1, M4.2, M4.3 | Probe-and-Amplify Controller | 3-class MLP frame classifier (hidden 256, dropout 0.1) on `[L_ctrl-3, L_ctrl-2, L_ctrl-1]` hidden states. Then head-restricted amplification: multiply H*_target head outputs by α_target before residual add. α grid `{1.0, 1.5, 2.0, 3.0, 4.0, 6.0}²` (36 combos) tuned on belief_core val, evaluated OOD on belief_holdout vs prompt-hint baseline. |

## Composition plan (per-claim, machine order)

M1 (behavioural evaluation, 9 runs) →
  M2.1 (Fisher signals, 9 runs) →
    M2.2 (AND-NOT masks, 6 runs) →
      M2.3 (greedy H* search, up to 6 runs — filtered by M1 above-chance gate) →
        M2.4 (40 controls per successful H*) →
          M3 (formation window on pythia-1b intermediate checkpoints — parallel with M4) →
          M4.1 (frame classifier) → M4.2 (α grid, 36 configs) → M4.3 (OOD eval + prompt-hint baseline)

## Plan reconciliation

`resource_fidelity: strict` is active — the reproduction combination (`BEHAVIOR_SOURCE=given` AND `MECHANISM=given`). Under strict fidelity, **method_sensitive fields are absent from the plan** (values are pinned exact). Therefore there is nothing to re-bind:

- n_pairs / used_n: plan pins n=227/454/681 exact per dataset/signal → matches
- sites: plan pins the H* search space (all attention heads) → matches
- metric: plan pins log-prob comparison of gold vs distractor → matches
- gpu_hours: plan estimates ~180h typical / ~350h worst — no re-bind (methods identical to plan)

**Environmental issue (not a re-bind, surfaced here for audit)**: only **24 of 154** enumerated pythia-1b intermediate checkpoints are stored on disk under `/mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/`. The 24 available cover the full step range log-spaced (step0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 13000, 23000, 33000, 43000, 63000, 83000, 103000, 123000, 143000). This is an environmental constraint (missing files), not a cost-saving down-scale, so the strict harness's HALT-rather-than-shrink rule is inapplicable — we cannot fabricate missing checkpoints. Plan proceeds on 24 available checkpoints; the M3 script auto-detects the available set. Reduced resolution is flagged prominently in EXPERIMENT_RESULTS.md's C3 block so downstream verification is aware.

reconciliation_status: n/a   <!-- no method_sensitive fields declared under strict fidelity -->

## Rationale

`MECHANISM=given` — user pinned each claim's mechanism family in `task.md` and the claim stage stamped it in the plan/proposal metadata. No routing candidates were generated (per Mode B direct entry rules); the family is committed directly. The pinned methods are already the exact reference recipe from Chen et al. 2025 (Sensitivity Meets Sparsity) plus the developmental extension via checkpoint sweep and the head-restricted amplification controller — all documented verbatim in `task.md` and `FINAL_PROPOSAL.md`.
