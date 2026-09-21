# Mechanism Audit — C3 (Applied Control) — Phase 2 Baseline Integrity

**Committed family**: Causal Attribution / Ablation (Arm A uses the ablation-family enhancement operator)
**Supporting families**: Steering Vectors (Arm C — RepE/CAA-style), Probing/Residual-Stream (Stage A)

---

## Slot A: Steering Coefficient Sweep for Arm A

Arm A val sweep: alpha_A in {0.5, 1.0, 2.0} x (k_h, k_n) 3-cell neighborhood = 9 configs per emotion.
Best config selected per emotion by val target-prefix logprob gain. This IS a 3-point alpha sweep.
**Slot A FULLY IMPLEMENTED.** PASS.

## Arm A Enhancement Operator (Primary)

The `_install_multi_component_hooks` hook adds:
- Per layer: alpha * sum(d_{e,L} for each head in C_e[e] at layer L) to residual at last-event-token
  (during logprob measurement) or at T=1 decode steps (during generation).
- Per neuron: alpha * sign * std to gate_proj output.

During GENERATION (batched_generate), the hook fires at T=1 (KV-cache decode steps), adding the
circuit activation at every new token. This is the intended "activate C_e at test" semantics.

**Key concern**: with k_h=24 heads all injecting alpha * d_{e,L} into the same layer(s), and
k_n=2000 neurons adding their deltas, the cumulative perturbation at alpha=2.0 may be extremely
large. The plan does not specify a cumulative magnitude bound. The single-layer residual delta
= alpha * sum of 24 head-direction vectors (each of dimension 3072), which at alpha=2.0 and
with 24 vectors each of magnitude ~O(10-100), can produce a total residual perturbation of order
O(1000). This is consistent with the observed "decoherent generation" pattern.

**Verdict**: The operator is correct as implemented. The operator design (cumulative alpha scaling
with 24 heads + 2000 neurons) is the root cause of Arm A's failure. This is a mechanism insight,
not an integrity bug.

## Arm C Steering Vector

Arm C: `_install_arm_c_hook` adds alpha_C * d_e[l_C] to the layer l_C residual at T=1 (every
continuation token). Val sweep: L in top-3 layers, alpha_C in {0.5, 1.0, 2.0} = 9 configs.
Direction d_e is frozen on train fold (pre-eval-split). Implementation matches CAA convention.

**Accuracy = 0.357 macro** — substantially above chance (0.167) and well above Arm A. This confirms
that single-direction steering with a correctly specified operator works on this dataset; the
problem is NOT that emotion steering is impossible, but that Arm A's circuit-injection fails.

## Pre-Eval-Split Freeze

- Arm A: C_e uses kstar fit on val logprob gain; val configs selected on val fold. Eval fold
  only touched for greedy generation and judging. PASS.
- Arm C: direction d_e fit on train fold, frozen. L and alpha_C selected on val fold. PASS.
- Arm B: templates frozen by the plan; position selected on val fold. PASS.

## Overall Mechanism Integrity

**PASS**. The three-arm comparison is methodologically sound. The Causal Attribution / Ablation
family's enhancement operator is correctly implemented at the code level; the failure is a
design-level issue (cumulative perturbation magnitude, val-eval metric mismatch). The cross-arm
comparison fairly attributes the advantage to Arm B (prompting) and Arm C (steering) over
Arm A (circuit injection).
