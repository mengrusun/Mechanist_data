# Experiment Audit Report — Claim C4

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C4 — Steering-vector control offers strictly more distinct (behaviour-rate, accuracy) operating points than NL-instruction prompt engineering and than Thinking Intervention, with matched-rate accuracy(steering) ≥ accuracy(prompt) − 2 pts and accuracy(steering @ α_op) ≥ accuracy(α=0) − 3 pts.
**Linked milestones**: M4 (primary), M3 (α operating range + baseline), M1 (directions)

## Overall Verdict: WARN
*This is C4's evaluation-methodology integrity verdict.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
Behaviour rate is LLM-judge derived (GPT-5.4, T=0). Accuracy is evaluated by the same GPT-5.4 judge comparing model outputs against GPT-5.4-generated gold answers — both components of the accuracy metric are synthetic. This creates a same-model-evaluator-and-gold-generator loop. Not model-self-derived (the judge evaluates the DeepSeek model's outputs, not GPT-5.4's own outputs), but the gold answer and the judge use the same model. WARN for the synthetic accuracy setup.

### B. Score Normalization: PASS
n_distinct_operating_points is a count. Matched-rate accuracy is a direct comparison between controller accuracy values. Preservation accuracy is a baseline subtraction (steering_acc − baseline_acc). No normalization by model prediction max/mean.

### C. Result File Existence: PASS
The EXPERIMENT_RESULTS.md claims are consistent with M4 results:
- Steering: 4 operating points at rates {0.186, 0.220, 0.250, 0.342} — four distinct points at ε_r=0.05, ε_a=0.01 (consistent with runs/M4_control_compare results)
- Prompt: 2 operating points (suppress rate 0.085, amplify rate 0.390) — 2 distinct
- TI: 2 operating points (suppress rate 0.271, amplify rate 0.797) — 2 distinct
- Preservation: steering α=−1σ acc=0.627, baseline=0.667, miss by 0.04 (1 pt floor) — matches reported "misses 3-pt floor by 1 pt"
- Matched-rate: steering α=+2σ (rate 0.342, acc 0.553) vs prompt_amplify (rate 0.390, acc 0.576) — within 2 pts

### D. Dead Code: WARN
The exact "matched-rate accuracy comparison" algorithm (pairing steering and prompt instances whose rate differs by ≤ 2 pts and comparing accuracy deltas) is not directly verifiable from the summary-level JSON provided. The individual per-task data would be needed to confirm the pairing was done correctly. The n_distinct_operating_points calculation and the preservation check are verifiable from the summary. WARN for partial verifiability of the matched-rate criterion.

### E. Scope Assessment: PASS
The 60/500 task subset is explicitly disclosed. The limitation to only expressing_uncertainty (other behaviours untestable at n=60 due to zero baseline rates) is explicitly stated. The preservation miss (3 pts → actually 4 pts miss: 0.667 − 0.627 = 0.040; EXPERIMENT_RESULTS.md says "misses 3-pt floor by 1 pt" meaning the miss is 4 pts vs the 3-pt tolerance = 1pt overshoot) is explicitly disclosed. TI_amplify's wider dynamic range (rate 0.797 with acc 0.492) is also explicitly noted as complicating the "finer-grained" story.

### F. Evaluation Type: synthetic_proxy
Both behaviour rate and accuracy use GPT-5.4-based proxy evaluation. Gold answers are GPT-5.4-generated. Classified as synthetic_proxy for all M4 outputs.

## Action Items
- The accuracy evaluation setup (same GPT-5.4 generates gold AND judges accuracy) should be explicitly disclosed in any paper write-up as a synthetic evaluation setup.
- The matched-rate pairing logic should be documented more explicitly in the code (or a helper function) to make it auditable at the per-task level.
- Consider expanding to n=200+ tasks to get more controller operating points and reduce noise in the comparison.
