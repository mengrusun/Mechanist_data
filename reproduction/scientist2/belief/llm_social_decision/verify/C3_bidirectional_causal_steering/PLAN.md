# Verify Plan — Claim C3

## Claim C3: Bidirectional causal steering
Injecting a pure direction at inference time causally and substantially shifts the target variable's effect on the dictator-game transfer amount (τ), in both amplifying and attenuating/inverting senses.

**Main-experiment verdict**: supported (at supplementary L=16 with raw v̂_V; inconclusive at ell_V* due to under-power)
**Main-experiment setup**: Llama-3.1-8B-Instruct + DG-1000 (200 held-out) + CAA activation-add at ell_V* (LEACE, single-site) and supplementary L=16 (raw, single-site)
**Main-experiment metric**: V=M sign-inverts at α=+2σ (L=16); V=A amplifies 5.5×; V=I/G bidirectional dose-response

### Main experiment (from /auto-experiment)
- Method: CAA activation-addition with signed-α sweep on residual-stream directions
- Dataset: DG-1000 paired prompts (200 baseline held-out)
- Model: Llama-3.1-8B-Instruct

### Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | Meta-Llama-3-8B-Instruct | Llama-3.1-8B-Instruct | Same architecture family (Llama), RLHF-tuned, 32-layer transformer — tests whether the mid-layer L=16 sign-inversion finding for C3 replicates with a different checkpoint in the same base family. Meta-Llama-3-8B uses the same tokenizer family, so the DG-1000 prompts require minimal adaptation. Evaluate at BOTH the max-probe-cv_acc-picked layer (likely similar to L=2–6 given same architecture) AND at L=16 (proportional mid-layer, same architecture depth). | NOTICE (verify invocation) + MECHANISM_ROUTING.md |

### Evaluation at Two Layers
Per the orchestrator instruction, the model-swap variant evaluates bidirectional dose-response at:
1. **ell_V*-equiv**: the layer picked by max probe cv_acc on Meta-Llama-3-8B-Instruct's activations on the DG-1000 train set (likely L=2–6 again, since the token-embedding linearity property is architectural)
2. **L=16**: mid-layer for a 32-layer model (same as the main experiment's supp run); α ∈ {−2,−1,0,+1,+2}σ with raw v̂_V

### Success Criterion (per variant)
C3 is consistent with main experiment if:
- At L=16: for at least 2/4 variables V ∈ {G,A,I,M}, the dose-response at α≠0 shows the same direction as the main experiment's L=16 result (amplification or inversion in the same sign as Llama-3.1-8B-Instruct)
- AND the parse failure rate remains ≤ 20% (coherence maintained)

Inconsistency (= FAIL) if: the model produces τ=const for all prompts (DeepSeek-style anchor) or no measurable v_effect at any layer.
