# Reviewer Memory

## Iteration 1 — Score: 3/10, Verdict: not ready

- **New suspicions**:
  - The project may be conflating linear decodability with a single causal steering direction; C1's high AUC + uniformly low first-PC alignment is a red flag against a simple 1D mechanism story.
  - Several behaviours may be too rare or too ambiguously operationalized to support contrastive extraction at current dataset scale; zero baseline rates for backtracking / self-correction / generating_validation_examples suggest prompt/task design mismatch.
  - Model-swap fragility due to layerwise norm differences (σ_proj=886 on Qwen-14B vs 10.52 on Llama-8B) is a script parameterization bug, not substantive evidence against the idea.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - Whether any non-uncertainty behaviour is actually steerable at all in this model, or whether uncertainty is the only tractable signal.
  - Whether C4's "finer-than-prompt/TI" effect survives after proper normalization, rather than depending on a model-specific scale accident.
  - Whether random-direction controls at α_op match the observed uncertainty effect (specificity concern).
- **Patterns**:
  - Intervention effects near coherence-collapse boundaries can look like signal but be artifacts of collapse (partial gibberish rated as behaviour).
  - Off-target inflation as α increases is a specificity red flag if not controlled by random-direction baseline.
  - Model-swap fragility should be explained by norm differences vs tokenizer / chat-template / behavior-rate shifts before invoking a mechanism-transfer failure.

## Iteration 2 — Score: 2/10, Verdict: not ready

- **New suspicions**:
  - The steering pipeline may be measuring generic residual-stream perturbation sensitivity, not a target-specific causal axis.
  - Original positive uncertainty effect likely driven by boundary artifacts + norm choice, not a meaningful latent control direction.
  - Prompt/TI comparisons were advantaged by unfair steering parameterization; after normalization the apparent fine-grained edge vanishes.
- **Previous suspicions addressed?**:
  - Decodability vs causal-specificity conflation: YES, and it looks WORSE. Random-direction control refutes learned-direction specificity (z=-6.33).
  - Rare/ambiguously operationalized behaviors (C1/C2): NOT addressed; iteration 1 focused on C3/C4 only.
  - C4 parameterization bug: CONFIRMED as bug (scale-invariant fix eliminates collapse), but fixing it also reveals the fine-grained advantage was parameterization-dependent.
- **Unresolved (carried forward)**:
  - Whether ANY non-uncertainty behaviour is genuinely steerable in a target-specific way under matched-coherence, norm-matched controls.
  - Whether C1/C2's decodability retains independent scientific value once paper abandons causal-mechanism rhetoric.
  - Whether a more principled steering objective could outperform random perturbations (future work, not present support).
- **Patterns**:
  - Specificity failure is now the DOMINANT pattern.
  - Coherence-collapse boundary artifacts continue to explain much of apparent action at large α.
  - Proper normalization reduces flashy effects and exposes parameterization-dependent wins.
  - Project consolidating into a cautionary result: linear probes ≠ unique causal steering direction; naive steering evaluations can mislead.

## Iteration 3 — Score: 5/10, Verdict: almost

- **New suspicions**: none new; consolidation.
- **Previous suspicions addressed?**:
  - Rewritten claims aligned with data: YES.
  - C3_v2 is strongest claim now: negative-finding about decodability vs causal-specificity.
  - C4_v2 is credible dual-finding but must remove cross-model language (Qwen incomplete).
  - C1/C2 remain supporting context (not standalone contributions).
- **Unresolved (carried forward)**:
  - Contribution breadth is limited for a top venue — scope model/setup-specific.
  - No positive methodological replacement offered beyond "be cautious with controls".
  - Cross-model validation (Qwen-14B) incomplete due to budget exhaustion.
- **Patterns**:
  - Specificity failure is the main scientific result.
  - Normalization/parameterization choices critically affect apparent steering wins.
  - Best framing: diagnostic negative result / evaluation methodology paper.
  - Target venue: NeurIPS/ICML mechanistic interpretability WORKSHOP (borderline for main conf).
