# Experiment Audit Report — Claim C3a (Variant: model-swap-qwen25-7b-instruct)

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C3a — The gold-correctness direction v_c and the verbalized-confidence direction v_v at the primary reporting layer L* are geometrically near-orthogonal (|cos| <= 0.3 with 95% CI upper bound < 0.4), robust across L*+-2 neighborhood.
**Variant**: model-swap-qwen25-7b-instruct (Qwen2.5-7B-Instruct replaces Llama-3.1-8B-Instruct)
**Linked milestones**: M2 (B2), M6 (B7)

## Overall Verdict: PASS

*This is the variant's integrity verdict — whether the variant's experimental process is methodologically sound.*

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
probe_c ground truth is legitimate external supervision: TriviaQA correctness via string match against gold aliases, not derived from model self-report. probe_v uses verbalized confidence extracted from the model output; this is model-derived, but it is the intended target for measuring the verbalization channel rather than an inflated evaluation label. For claim C3a, the geometric relation is between correctness and verbalized-confidence directions, so this provenance is acceptable and aligned with design. No indication of leakage from split reuse because idx-based train/dev/test manifest is fixed and explicitly stated.

### B. Score Normalization: PASS
Cosine is reported as absolute cosine using L2-normalized probe weight vectors and the standard formula |u·v|/(|u||v|). This is the correct normalization for directional similarity. Reported values are numerically coherent: abs_cos_at_Lstar=0.0213, bootstrap mean=0.0156, random null mean=0.0133, theoretical random null=1/sqrt(3584)=0.0167, all in a plausible regime for near-orthogonal high-dimensional vectors. No suspicious rescaling or nonstandard normalization is indicated.

### C. Result File Existence: PASS
A concrete result.json payload is provided with complete fields and internally plausible values: model, dataset, splits, dimensionality D=3584, num_layers=28, L*=22, bootstrap n=200, probe AUCs, null references, and pass flags. Split counts sum correctly to n_total=10000. CI ordering is valid (0.0008872 < 0.036478). All criteria pass field is consistent with individual pass flags.

### D. Dead Code Detection: PASS
There is strong evidence the cosine computation was actually executed rather than phantom: a point estimate at L*, a neighborhood aggregate, a bootstrap distribution summary with n=200 retrain-on-bootstrap, and a random-direction null benchmark are all present and mutually consistent. The presence of nontrivial variation between direct estimate (0.0213), bootstrap mean (0.0156), and neighborhood mean (0.0148) argues against hardcoded or dead-path output. No sign of placeholder constants.

### E. Scope Assessment: PASS
The numbers are scientifically plausible for a model-swap from Llama-3.1-8B-Instruct to Qwen2.5-7B-Instruct with frozen methodology. Near-orthogonality is supported strongly: abs_cos_at_L*=0.0213, bootstrap CI upper=0.0365 well below the claim threshold 0.4 and below the stricter stated criterion 0.3 for the point estimate. Neighborhood mean across L*+-2 is also very small (0.0148). Probe quality is adequate to strong (AUC_c=0.864, AUC_v=0.898), so directions are not degenerate. The measured cosine being close to the random-direction null in D=3584 is expected under true near-orthogonality. Layer choice L*=22 within 28 layers is plausible for an instruction-tuned 7B model. No scope anomaly detected.

### F. Evaluation Type: derived
Pure analytical derivation from probe weight vectors trained on the Qwen2.5-7B-Instruct hidden states. No external GT required for C3a itself.

## Action Items
None. Variant passes all checks cleanly. The model-swap stress test is methodologically sound.
