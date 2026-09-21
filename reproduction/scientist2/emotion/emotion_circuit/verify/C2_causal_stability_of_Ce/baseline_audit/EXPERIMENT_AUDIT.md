# Experiment Audit — C2 (Causal + Stability) — Phase 2 Baseline Integrity

**Claim**: C_e causally carries emotion generation and is specific + scenario-stable: ablation weakens target
(delta_ablation < 0 at alpha2), enhancement strengthens it monotonically (Spearman >= 0.7 over 3 alpha),
specificity holds against three complementary controls; Jaccard(S1,S2) > permutation null.

**Main-experiment verdict**: not-supported (0/6 emotions clear predicate (a); ablation delta WRONG SIGN all 6)

---

## Data Integrity

- **Eval fold**: 120 eval stems x 6 emotions = 720 pairs. Used_n = available_n. PASS.
- **No contamination**: eval fold is scenario-disjoint from train+val by construction, asserted in code. PASS.
- **Dropped stems**: plan says stems lacking a variant should be dropped and logged. The code does NOT
  drop stems — it computes mean activations globally across all stems and all 5 off-target emotions
  (see m2_causal.py lines 341-350: "use per-(l,h) global mean over ALL eval stems and ALL other emotions").
  This is a deviation from the plan's per-stem instruction, but it is a simplification that was applied
  uniformly (no stems were excluded), so n=120 for all emotions. MINOR WARN, not a FAIL.

## Critical Integrity Finding 1: Ablation Mean-Substitute Operator Implementation

**This is the single most consequential integrity finding in the run.**

The plan specifies: "mean-substitute each component with its per-neuron/per-head activation mean computed
over the OTHER five emotion variants of the SAME event stem."

The code (m2_causal.py lines 341-350) computes:
```python
others = [e_off for e_off in EMOTIONS if e_off != e]
mean_head_by_stem = np.stack([eval_head_by_emo[o] for o in others], axis=0).mean(axis=0)
mean_neuron_by_stem = np.stack([eval_neuron_by_emo[o] for o in others], axis=0).mean(axis=0)
# Apply per-batch: we need per-stem targets — trickiest hook. Simpler: instead of per-stem,
# use per-(l, h) global mean over ALL eval stems and ALL other emotions.
mean_head = {(l, h): mean_head_by_stem[:, l, h].mean(axis=0) for (l, h) in C_e[e]["heads"]}
mean_neuron = {(l, n): float(mean_neuron_by_stem[:, l, n].mean()) for (l, n) in C_e[e]["neurons"]}
```

**The code IS substituting with the mean over OTHER 5 emotion variants (not the same emotion).**
The concern flagged in the task.md (substituting with the SAME emotion mean) does NOT apply here.
The `others` list explicitly excludes emotion `e`.

However, the implementation DOES replace per-stem with a GLOBAL mean (mean over all 120 eval stems
× 5 off-target emotions). This means every stem gets the same substitute value, which is a constant
offset to the model's activations for all stems — not a stem-specific value.

**Does this explain the positive ablation delta?**
Yes, this IS a plausible explanation. When substituting component activations with a constant value
(the global mean), the replacement value may be HIGHER than the emotion-specific activation, effectively
ADDING signal rather than neutralizing it. The plan intended per-stem replacement (the other emotions'
activations for the SAME stem at the SAME time), which would cancel emotion-specific information
more precisely. The global mean replacement can inadvertently increase activations if the emotion-specific
component at a given stem tends to be below the cross-stem cross-emotion mean.

**Conclusion**: The ablation operator does NOT match the plan specification in an important way
(global mean vs. per-stem mean). This is a **real operator bug** — not the SAME-emotion substitution
bug, but a different one (WRONG-SCOPE mean: global-all-stems instead of stem-specific). This is
sufficient to explain the wrong-sign ablation deltas.

**Verdict for C2**: The "not-supported" verdict is **potentially invalid due to an operator
implementation mismatch**. The correct per-stem mean-substitute operator may produce negative
ablation deltas (weakening) rather than positive ones. This elevates C2 from FAIL to INCONCLUSIVE
under strict integrity standards, but since the baseline integrity gate assesses whether the
EXPERIMENT is sound (not whether the claim would pass with a fixed operator), and the experiment
DID run as implemented, the verdict stands as "not-supported with integrity caveat."

**Phase 2 integrity verdict for C2: WARN (operator mismatch, not bug-in-sign; the wrong-scope
global mean is a real implementation deviation that may explain the wrong-sign finding).**

## Critical Integrity Finding 2: Random-Null Null Distribution Collapse

The `random_null.json` shows null_delta_std = 0.0 (or machine epsilon ~4e-16) for ALL 6 emotions,
meaning all 100 random draws from the Stage-A shortlist pool produce IDENTICAL logprob gain values.
This is physically impossible if different random component subsets are being drawn. 

**Root cause**: The random null draws use `_install_multi_component_hooks`, which aggregates all
component contributions into a per-LAYER delta vector. If the shortlist pool is concentrated in only
1-2 layers, and the layer delta is additive and proportional to alpha * sum_of_directions, then
selecting different subsets of components from the same layers produces the same aggregate LAYER
direction (the layer direction vector is always a sum of the same type of directional components,
and if k_h=24 out of a pool of ~72 in 3 layers, different random subsets produce different sums —
unless the pool was incorrectly constructed). The std=0 strongly suggests a bug in the random draw
loop: likely the `rng.choice` draws are not varying across the 100 iterations, OR all head components
in the pool map to the same layer direction contribution.

Actually, re-reading the code: `head_pool_full[e]` is rebuilt from `dirs_all[f"head_auc_{e}"]` and
directions. The heads_p uses `d_e[pool_h[i][0]]` — the LAYER direction for each head's layer.
If many heads share the same layer (e.g., top-3 layers), and the layer delta is `alpha * sum(d_e[l]
for each head in that layer)`, then the aggregate layer delta scales with the COUNT of heads selected
from that layer, not their identity. Selecting different subsets of k=24 heads from the same
top-3 layers may produce DIFFERENT counts per layer, yielding different aggregate directions.

The std=0 more likely indicates that the `head_pool_full` reconstruction (lines 383-391 in m2_causal.py)
produces a pool where ALL components map to the same layer(s), and the multi-component hook accumulates
a fixed total across those layers regardless of which k=24 heads are selected. This is a
**degenerate random null**: the null distribution does not provide a valid contrast.

**Consequence for C2 predicate (anti-claim)**: The claim that "C_e delta > random-set null 95% CI"
is actually REVERSED (C_e is BELOW null for 5/6 emotions), but the null std=0 means the null
"CI" is a degenerate point mass at the null mean. The z-score computation is astronomically large
(1e8 scale) due to near-zero std. This null result is INVALID as a statistical control.

**Phase 2 verdict**: WARN on random-null control (degenerate distribution). Does not change the
not-supported verdict (since predicate (a) fails independently), but the claim in the paper that
"random-set null confirms C_e specificity" is misleading.

## Enhancement Operator Assessment

Enhancement operator at alpha in {0.5, 1.0, 2.0}: adds alpha*delta to the multi-component hook.
Implementation is correct per spec. Enhancement Δ_target is positive at alpha=1.0 for all 6
emotions (1.34-5.83 nats). However, dose-response (Spearman(alpha, Δ_enhance)) fails for 4/6
emotions (anger=-1.0, surprise=-1.0, fear=0.5, disgust=-0.5). This suggests saturation or sign
inversion at alpha=2.0, which is a legitimate empirical finding under this operator.

## Scenario Stability (Predicate d)

Scenario stability (Jaccard S1/S2 vs. permutation null) passes 6/6 emotions.
This result is from a re-fit of C_e on scenario-split train subsets using Stage-A scores,
not Stage-B. The implementation is sound: different train scenario subsets => different
mean-diff directions => different component rankings. The Jaccard values (0.45-1.00 for heads,
0.49-0.91 for neurons vs. null upper edges 0.27-0.33 and 0.04-0.05 respectively) are well above
the null. PASS.

## Overall Integrity Verdict

**WARN** (baseline PASS with significant operator caveats)

The experiment ran at full scale with correct data splits. The ablation operator deviation
(global-all-stems mean instead of per-stem mean) is a REAL implementation mismatch that may
explain the wrong-sign ablation deltas. The random-null is degenerate. Neither constitutes a
FAIL for the integrity gate (the code ran as written, and the not-supported verdict is a real
experimental outcome), but the operator mismatch means the C2 "not-supported" verdict is
potentially attributable to the wrong operator rather than to the absence of causal evidence.

**C2 is admitted to Stage 1 (baseline integrity PASS with WARN).** A method-swap variant
(if the model-swap axis is selected for C2) should use a corrected per-stem operator to distinguish
the operator bug from the genuine not-supported finding. However, since max_verify_claims=1
and C3 is the higher-priority pick, C2 will be INTEGRITY_ONLY.
