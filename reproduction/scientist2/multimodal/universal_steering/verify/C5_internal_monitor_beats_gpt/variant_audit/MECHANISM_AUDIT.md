# Variant Mechanism Audit — C5 model-swap-deepseek-r1-llama8b

## Scope
Variant: `model-swap-deepseek-r1-llama8b`
Claim: C5 (internal monitoring via RFM concept-vector / linear probe features beats GPT-4o judge)

## Mechanism type
C5 is a **monitoring / classification** task, not an activation-steering intervention task.
The mechanism is: RFM AGOP eigenvector (or linear probe direction) as a linear scorer, evaluated by AUROC.
No alpha sweep, no additive steering intervention, no matched-random-direction control is required.

## Mechanism checks applicable to this variant
- Per-block RFM fit procedure: identical to main experiment (kernel-ridge + AGOP alternation, 3 iters, ridge=1e-2)
- Block selection: val-AUROC argmax (no test leakage)
- AUROC computation: rank-sum formula (same as main experiment)
- No steering intervention → no alpha sweep needed → no matched-random-direction control needed

## Overall verdict
**N/A** — No mechanism intervention in C5 monitoring task. No steering coefficient sweep, no random-direction control required. All applicable procedure steps verified as matching the main experiment.

## Combined verdict (with EXPERIMENT_AUDIT)
max_severity(PASS, N/A) = **PASS**
