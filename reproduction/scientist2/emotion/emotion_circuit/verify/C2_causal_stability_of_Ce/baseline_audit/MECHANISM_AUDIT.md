# Mechanism Audit — C2 (Causal + Stability) — Phase 2 Baseline Integrity

**Committed family**: Causal Attribution / Ablation
**Claim**: C_e causally carries emotion; ablation weakens target, enhancement strengthens it.

---

## Slot A: Steering Coefficient Sweep (alpha in {0.5, 1.0, 2.0})

The plan specifies alpha in {alpha1=0.5, alpha2=1.0, alpha3=2.0} — a 3-point sweep.
All 3 alpha values are implemented and measured in M2. Enhancement results per alpha:

| Emotion | alpha=0.5 Δ | alpha=1.0 Δ | alpha=2.0 Δ | Spearman |
|---------|-------------|-------------|-------------|---------|
| joy     | +0.076      | +1.567      | +3.188      | +1.00   |
| sadness | +0.180      | +1.937      | +4.515      | +1.00   |
| anger   | +1.720      | +1.641      | -0.311      | -1.00   |
| fear    | +1.194      | +5.828      | +5.463      | +0.50   |
| surprise| +2.932      | +2.018      | +1.944      | -1.00   |
| disgust | +0.341      | +1.338      | +0.146      | -0.50   |

Slot A is FULLY IMPLEMENTED. PASS.
Non-monotonic dose-response (anger, surprise, disgust, fear) is a real finding, not a missing sweep.

## Ablation Operator Assessment

**Operator as specified**: mean-substitute C_e components with per-stem mean over OTHER 5 emotion
variants of the SAME stem, applied at the last-event-token position.

**Operator as implemented**: global mean over ALL eval stems × OTHER 5 emotions; applied as a
constant hook at the last-event-token position (or fallback to last position of tensor).

The `_install_mean_substitute_ablation_hooks` function receives pre-computed `mean_head` and
`mean_neuron` dicts that are GLOBAL averages (not per-stem). During the per-stem forward pass,
the hook substitutes the SAME global mean for every stem — this is a constant offset, not a
stem-adaptive neutralization. This implementation is INCORRECT relative to the plan spec.

**Why the wrong-sign delta is consistent with this bug**: If C_e components at a given stem tend
to have emotion-specific activations BELOW the global cross-emotion mean (which can occur when
the positive direction is learned as a departure from the global mean in a non-uniform way), then
replacing with the global mean RAISES the activation, producing positive Δ. Conversely, if the
global mean is above the emotion-specific value for the target emotion, the replacement pushes the
model toward a "generic" state that has higher log-prob for the target prefix (because the global
mean contains signal from all 6 emotions including the target emotion's activation in 1/6 of the
data). The wrong-sign delta is a plausible consequence of this implementation.

## Enhancement Operator Assessment

Enhancement operator: `_install_multi_component_hooks` applies per-head and per-neuron additive
deltas. The implementation correctly accumulates alpha * d_{e,L} per layer for heads, and
alpha * sign * std per neuron pre-activation. The `sign` is computed from d_{e,L} · w_n^out
(the output weight vector), and `std` is from eval neuron pre-activations. This matches the spec.

One concern: `std_val` for neurons in m2_causal.py is computed from `eval_neuron_by_emo[e][:, l, n].std()`
(eval fold), not from `pos_pre` on the train fold as the spec states ("std(activation_n | pos-e on train)").
This is a minor deviation (eval vs. train std) that may inflate or deflate the std estimate but is
unlikely to explain the non-monotonic Spearman for anger/surprise.

## Intervention Scope

The mean-substitute hook applies to the last-event-token position (primary) via
`_install_mean_substitute_ablation_hooks`. For heads: pre-W_O hook modifying the head activation
at position >= event_len. For neurons: gate_proj forward hook at position >= event_len.
The target-prefix logprob is then measured. This is correct: changing the last-event-token
representation affects the model's next-token predictions and hence P(prefix_e | event).

## Overall Mechanism Integrity

**WARN** — operator implementation deviation (global vs. per-stem mean in ablation) is the
primary integrity concern. The enhancement operator is correct in sign/direction but uses
eval-fold std vs. train-fold std for neurons (minor). The Slot A sweep is fully implemented.

C2 baseline integrity: **WARN, not FAIL** — the experiment ran cleanly, the operator deviation
is documented, and the not-supported verdict is the outcome of the implemented (but mis-spec'd)
operator. The deviation plausibly explains the wrong-sign finding and should be flagged for
iteration (method-swap with corrected per-stem operator).
