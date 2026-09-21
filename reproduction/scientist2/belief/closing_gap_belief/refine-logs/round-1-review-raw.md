## Scores

1. **Problem Fidelity**: **8/10**  
2. **Method Specificity**: **8/10**  
3. **Contribution Quality**: **7/10**  
4. **Frontier Leverage**: **8/10**  
5. **Feasibility**: **6/10**  
6. **Validation Focus**: **6/10**  
7. **Venue Readiness**: **6/10**

**OVERALL SCORE**: **7.1/10**

---

## Dimension-by-dimension assessment

### 1. Problem Fidelity — 8/10
This is largely faithful to the anchored claims. The proposal stays centered on exactly the required structure: C1 linear accessibility of correctness, C2 linear accessibility of verbalized confidence, and C3 geometric + causal separability. It also respects the single-model/single-dataset/matched-pair framing and avoids banned directions like SAEs or fine-tuning.

The main fidelity risk is not conceptual drift but **measurement mismatch inside C3**:
- C3 speaks about the geometric relation between the internal calibration direction and the internal verbalization direction.
- But your current implementation compares **probe weights from different forward contexts and potentially different layers**: \(v_c\) from pass 1 on answer-generation states, \(v_v\) from pass 2 on confidence-prompt states, then reports cosine at “matched best layers.”
- If the two directions live in different residual-state distributions induced by different prompts/contexts, the cosine is less cleanly interpretable as “same-space geometry” than the writeup implies. It may still be computable, but a strict reviewer will ask whether this is true subspace geometry or a comparison of context-conditioned readouts.

So: good claim fidelity overall, but C3’s operationalization needs tightening to ensure it tests the claim as stated, not a nearby one.

---

### 2. Method Specificity — 8/10
This is fairly concrete and implementable. The interfaces are clear:
- hidden state extraction points,
- per-layer logistic probes,
- answer scoring,
- confidence elicitation prompt,
- cosine computation,
- matched-norm steering with random null.

An engineer could implement most of this in a day.

The main underspecified pieces are:
- **Exact state location for steering**: residual stream pre-attn? post-MLP? block output? You say “residual-stream hidden states,” but steering requires precise hook placement.
- **How cross-layer steering is performed** if best-C1 and best-C2 layers differ.
- **How isotonic calibration is fit** and on which split.
- **How confidence number parsing and normalization are handled** when outputs are malformed or not on 0–100.
- **How answer correctness is scored** for free-form outputs beyond “alias-list + Wikidata + manual audit on 100.”

These are solvable details, but they matter because this is a characterization paper: ambiguity in interfaces weakens the evidence chain.

---

### 3. Contribution Quality — 7/10
The proposal mostly has one dominant contribution, which is good. The matched-pair characterization is the right center.

However, it is drifting toward a **small bundle of related papers**:
- calibration probe paper,
- verbalized confidence probe paper,
- geometric orthogonality paper,
- steering dissociation paper,
- prompt robustness paper,
- calibration-method comparison paper.

You repeatedly say “one dominant contribution,” but the current validation plan and ablations still read broader than necessary. This does not fully break focus, but it dilutes elegance. The strongest paper here is: **two paired linear readouts, angle, and matched-magnitude cross-steering**. Everything else should support that core, not compete with it.

---

### 4. Frontier Leverage — 8/10
For this claim structure, the chosen primitives are appropriate. Linear probes + linear steering are exactly the field-standard tools for testing linear accessibility and causal separability. Good restraint in rejecting SAEs, tracing, and fine-tuning.

The only caveat is that some pieces are slightly more old-fashioned than needed:
- logistic regression for C2 on median-binarized confidence is acceptable, but the continuous confidence signal is arguably more native to the claim than the binarized target;
- using post-emission hidden state as a “trivial upper bound” is fine diagnostically, but it is not central.

Still, no major methodological primitive is missing.

---

### 5. Feasibility — 6/10
This is the weakest part. The proposal is likely **close** to feasible in 10h, but the budget is optimistic given:
- two separate forward data-collection passes over 10k examples,
- generation of free-form answers and then confidence numbers,
- storage or recomputation of all 32-layer hidden states,
- 36k steering forward passes,
- “verify swaps” with 3h reserved but not concretely defined,
- ablations including paraphrases, token positions, calibration variants, and best-layer robustness.

The compute concern is not the probes; it is the inference/control workload and I/O. On an 8B instruct model, 36k steered passes plus the main collection pass can fit, but only with a very disciplined implementation and likely reduced sequence lengths / cached prompts / no unnecessary recomputation.

Right now the plan assumes too much slack for an actual constrained filesystem/GPU environment.

**Specific weakness**  
The experimental plan includes more generated passes and ablations than are needed for the core claims under the stated hard 10h budget.

**Concrete fix**  
Trim to the minimum claim-sufficient protocol:
1. Collect only the exact hidden states needed for the primary token position.
2. Restrict steering to:
   - best C1 layer,
   - best C2 layer,
   - one shared nearby layer if different,
   - 3 alpha values instead of 6 for the primary run.
3. Move token-position ablation and post-hoc calibration comparison to optional-only if time remains.
4. Run paraphrase robustness on a 500-example dev slice, not full test.
5. Explicitly precompute and cache answer strings and confidence prompts to avoid duplicate generation overhead.

**Priority**: **CRITICAL**

---

### 6. Validation Focus — 6/10
The validation suite is a mixed bag: some controls are exactly right, others are unnecessary, and one core issue is still under-validated.

What is good:
- shuffled-label null,
- random-direction null,
- per-layer trajectories,
- matched-magnitude steering control,
- AUROC/ECE gating before interpreting cosine.

What is not yet tight enough:
1. **C3 geometry is under-identified across different contexts/layers.**  
   If \(v_c\) and \(v_v\) are extracted from different prompt states, low cosine may partly reflect representation shift rather than true disentangled subspaces.
2. **C3b currently measures change in probe readout, not necessarily change in the actual target variable.**  
   For a top-venue reviewer, “cross-steering null” is much stronger if reported both on the probe readout and on the emitted confidence / correctness-related observable when possible.
3. Several ablations are not necessary for defending the anchored claims:
   - post-hoc calibration ablation across isotonic/Platt/temperature,
   - question-only lower bound for C2,
   - post-emission trivial upper bound,
   - possibly token-position ablation as currently proposed.

**Specific weakness**  
The current validation over-spends on side diagnostics while under-resolving the main interpretability threat in C3: whether the compared directions are geometrically comparable and causally separable in a clean matched setting.

**Concrete fix**  
Refocus validation around three claim-critical controls:
1. **Same-hook, same-layer comparability control for C3**: define a canonical residual hook location and report cosine only there; if best layers differ, also report nearest shared-layer cosine and steering.
2. **Cross-steering outcome metrics**: for \(v_v \rightarrow\) confidence prompt, measure actual emitted confidence shift in addition to probe readout; for \(v_c \rightarrow\) answer pass, measure probe_c shift at minimum, and if generation is too expensive, state clearly that C3b is about internal readout separability rather than output behavior.
3. **Delete non-essential ablations**: calibration-method comparison, question-only lower bound, and post-emission upper bound from the must-run list.

**Priority**: **CRITICAL**

---

### 7. Venue Readiness — 6/10
If executed cleanly, this is potentially publishable as a compact characterization result, but it is not yet at top-venue readiness. The main reasons are:
- C3 needs a cleaner operationalization to avoid reviewer skepticism about cross-context geometry.
- The compute/story needs simplification so the paper feels deliberate rather than overextended.
- The evaluation should privilege a small number of high-value controls over a long checklist.

This could become a neat workshop-or-main-track empirical interpretability paper if tightened. In current form, it is promising but not yet persuasive enough.

**Specific weakness**  
The current draft does not yet make C3 look airtight enough for a strong reviewer, which is the novelty-bearing claim.

**Concrete fix**  
Make the paper explicitly a **matched-pair same-representation-space characterization**:
- fix one canonical hook definition,
- report same-layer/same-hook cosine as primary,
- use matched-magnitude steering at that same hook,
- relegate broader diagnostics to appendix.
This would make the novelty claim feel much sharper.

**Priority**: **IMPORTANT**

---

## Simplification Opportunities

1. **Collapse C2 primary target to one continuous probe + one thresholded report**  
   Train a single linear regressor for confidence \(c \in [0,100]\), report Spearman as primary and median-binarized AUROC as a derived secondary metric. This removes some duplication while preserving C2.

2. **Reduce steering grid**  
   Use 3 alpha values instead of 6 for the primary C3b claim, with matched random-direction null. Add more alpha points only if the effect looks nonlinear.

3. **Delete non-essential must-run ablations**  
   Move the following out of “must-run”: post-hoc calibration method comparison, question-only baseline, post-emission upper bound, and token-position ablation. Keep only controls that directly defend C1/C2/C3.

---

## Modernization Opportunities

1. **Use a direct linear regressor for verbalized confidence as the primary C2 model**  
   This is more natural for a scalar verbalized-confidence target than leading with median-binarized logistic regression.

2. **Use batched intervention hooks with cached prompts/states**  
   Not a conceptual change, but a more foundation-model-native implementation choice that materially helps the 10h constraint.

3. **Report layerwise probe uncertainty via bootstrap CIs**  
   Lightweight and modern enough for empirical rigor; better than adding more benchmark breadth.

---

## Drift Warning

**NONE**

The proposal still targets the three fixed claims rather than re-scoping them. The issue is not drift; it is whether C3 is measured cleanly enough.

---

## Verdict

**REVISE**

Promising and mostly well-anchored, but not yet tight enough on C3 operationalization or 10h-budget discipline to be READY. The right revision is **simplification plus cleaner same-space measurement**, not expansion.