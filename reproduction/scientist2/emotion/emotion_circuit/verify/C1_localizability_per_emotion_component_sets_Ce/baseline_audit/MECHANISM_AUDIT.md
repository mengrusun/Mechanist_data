# Mechanism Audit — C1 (Localizability) — Phase 2 Baseline Integrity

**Committed family**: Causal Attribution / Ablation (single-component enhancement prefix-logprob as Stage-B causal ranker)
**Claim**: Localizability — Stage A screen + Stage B causal ranker yield sparse, stable C_e.

---

## Slot A: Steering Coefficient Sweep (maps to alpha sweep in Stage-B)

The plan's alpha sweep for M1's Stage B is fixed at alpha=1.0 (alpha_2 only) for component scoring.
This is intentional: Stage B scores components for SELECTION, not for measuring dose-response.
Slot A coverage is therefore partial by design — single alpha for Stage B, but M2 runs the full
alpha in {0.5, 1.0, 2.0} sweep for the causal claims. Low Slot A coverage here is expected.
**Verdict: EXPECTED-PARTIAL — no integrity concern.**

## Operator Soundness

- **Stage B single-component enhancement**: `_install_multi_component_hooks` for a single (l,h,d) or
  (l,n,sign,std) component. For heads: adds alpha * d_{e,L} to the residual stream at the last-event-token
  position (via a layer-level forward hook). For neurons: adds alpha * sign * std to the gate_proj output
  (pre-activation of the SiLU neuron).
  
- **Head intervention approximation**: The per-head enhancement uses the FULL residual direction d_{e,L}
  (shape H=3072) rather than the head-specific projected direction W_O_h @ head_h_mean_diff (shape H=3072,
  but filtered to head h's output). The code comment acknowledges this is a "defensible approximation"
  for Stage B. This means all heads in the same layer receive the SAME direction delta, making the
  per-head stage-B scores identical across heads in the same layer (they differ only in the neuron-level
  hook separation). This is a WARN: the head-level SELECTION in Stage B is effectively layer-level,
  not head-level. However, the Jaccard stability results are based on the Stage-A shortlist (probe AUC),
  which IS head-specific. The Stage-B reranking of heads within the shortlist may be degenerate.

- **Neuron intervention**: Correctly uses sign(d_{e,L} . w_n^out) * std(pre-activation | pos-e on train),
  which is the spec'd formula. PASS.

- **Injection position**: `pos_start = max(0, event_len - 1)` — injects at the last event token AND
  all prefix positions. This is consistent with measuring "does this component raise P(prefix_e | event)?"
  PASS.

## Null/Permutation Construction

- 200-draw permutation null: `permutation_null_jaccard(pool_size, [k]*3, n_draws=200)` samples random
  sets of size k from a pool of pool_size. Pool for heads = TOP_N_LAYERS * n_heads = 3*24 = 72;
  pool for neurons = TOP_N_LAYERS * n_neurons = 3*8192 = 24576. This correctly represents the
  "Stage-A shortlisted universe", so the null is size-matched and pool-matched. PASS.

## Family Freeze

Direction extraction uses emotion-contextualized train contexts only ("event + I feel {e}." on train fold).
kstar selection uses val fold. Jaccard uses train fold subsamples. Eval fold untouched. Pre-eval-split
freeze is maintained. PASS.

## Overall Mechanism Integrity

**PASS (with WARN: head stage-B enhancement is layer-level, not head-level)**

The Stage-A probe AUC IS head-specific and is what drives the primary shortlist. Stage-B reranking
within the shortlist may conflate heads at the same layer. The Jaccard stability result (which is
independent of Stage-B for the Jaccard fold computation since the fold sets use Stage-A scores)
is sound. No integrity block.
