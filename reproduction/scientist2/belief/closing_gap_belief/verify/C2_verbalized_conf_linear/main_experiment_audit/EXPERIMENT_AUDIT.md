# Experiment Audit Report — Claim C2

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C2 — Llama-3.1-8B-Instruct encodes its about-to-be-verbalized confidence in a linearly accessible direction of the residual stream pre-emission, with paraphrase robustness across P0/P1/P2.
**Linked milestones**: M1 (B1), M1.5 (B5), M5 (B6)

## Overall Verdict: WARN

*This is C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
The target for the C2 probe is the model's own verbalized confidence `c` (parsed numeric 0-100 from the model's generation). This is explicitly labeled as a by-design proxy target — C2 claims the model pre-encodes its own verbalization in the residual stream, so using `c` as the probe target is the research question itself, not a GT fraud. The claim is transparent about this. However, strict audit must flag: the probe target derives from model output. The design is sound and disclosed, so this is WARN (not FAIL).

### B. Score Normalization: PASS
No evidence of fraudulent normalization by model prediction statistics. Binarization at the 30th percentile (binarize_threshold=100) is a thresholding choice, not score normalization. AUROC is a standard ranking metric not normalized by model output. Ordinal top-1 accuracy is not normalized.

### C. Result File Existence: WARN
Files exist and core numbers match (`probe_metrics.json`: AUROC(probe_v_bin)=0.948, ordinal_top1=0.994, macro-F1=0.862; `paraphrase.json`: P1 AUC=0.838, P2 AUC=0.870). Two concerns:
1. Bootstrap CI for probe_v_bin [0.913, 0.940] does not contain the point estimate 0.948 — same normal statistical artifact as C1 (bootstrap mean vs full-data estimate). Not fabrication.
2. Paraphrase robustness: planned tolerance Delta AUC <= 0.10; P1 shows Delta=-0.11 (exceeds by 0.01). This is honestly reported in EXPERIMENT_RESULTS.md with the note that AUC remains above the 0.70 accessibility floor. The exceedance is minor and disclosed, but the full C2 robustness criterion is not met.

### D. Dead Code Detection: PASS
All metric functions called and outputs appear in `probe_metrics.json` and `paraphrase.json`. Ordinal probe fitting, binarized AUROC, and paraphrase Delta computations all have corresponding output entries.

### E. Scope Assessment: PASS
Single model, single dataset. No over-broad scope language. The paraphrase robustness sub-claim is appropriately qualified (AUC remains above floor, Delta exceedance disclosed).

### F. Evaluation Type: synthetic_proxy
The probe target (verbalized confidence c) derives from model output. This is by design and explicitly disclosed. Evaluation type is synthetic_proxy.

## Action Items
- Acknowledge in paper that the C2 probe target is the model's own verbalization (make explicit that "pre-emission encoding" is tested by probing against model output as target, not an independent ground truth).
- The paraphrase Delta tolerance exceedance (Delta=-0.11 vs planned <=0.10) should be explicitly noted as a qualification on the C2 robustness claim — already done in EXPERIMENT_RESULTS.md.
