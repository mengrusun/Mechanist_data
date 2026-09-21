# Experiment Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C3 — The same representation-level intervention transfers to multimodal LLMs, blocking image-based jailbreaks (PGD image-hijack attacks against LLaVA-NeXT-Mistral-7B) without materially degrading vision-language task performance.
**Linked milestones**: M6 (partial — M6L, M6T, M6D completed; PGD attack and VLM assembly skipped)

## Overall Verdict: WARN

*C3's experiment methodology for what was run is sound (honest partial reporting, files exist, no inflated claims). However, the decisive C3 evaluations (VLM assembly, PGD image-hijack, VL capability measurement) were skipped due to budget and local-availability constraints. The partial status is honestly documented in c3_verdict.json and EXPERIMENT_RESULTS.md. C3 cannot be judged on its merit — only the mechanism-level sub-evidence (Mistral RR fine-tune failure, same as C1) is available.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN

Provenance for completed M6L/M6T/M6D sub-steps is adequate (real paired prompts, AUC from real labels, diagnostic from genuine base-vs-adapter comparison). However, provenance for the actual C3 claim is incomplete: no PGD image-hijack was run, no VLM assembly occurred, no VL capability measurement was performed. The decisive GT (image-hijack ASR, VL benchmark) does not exist.

### B. Score Normalization: PASS

Reporting correctly normalizes evidence to claim strength: c3_verdict.json marks status="partial", EXPERIMENT_RESULTS.md says "C3: partial (mechanism-level negative; PGD skipped)". No inflation of sub-results into a full success claim. The AUC and delta_cos metrics are computed against real labels/directions, not self-normalized.

### C. Result File Existence: PASS

All cited files for completed substeps exist: artifacts/m6/m1_mistral/summary.json, artifacts/m6/m3_mistral/training_summary.json, artifacts/m6/m4_mistral/activation_drift.json, artifacts/m6/c3_verdict.json. Files for unrun evaluations (asr_pgd_before/after.json, vlm_capability.json) are absent — appropriately so since those runs did not occur, and the partial verdict reflects this.

### D. Dead Code Detection: WARN

The full C3 code pathway (m6_vlm_pgd.py, LLaVA-NeXT assembly, PGD attack loop) was unexecuted. This is not dead code in the traditional sense but unvalidated code for the key claim path. The audit cannot confirm these code paths are correct since they were not run.

### E. Scope Assessment: FAIL

The stated claim requires: VLM assembly, PGD ε=32/255 × 1000 steps image-hijack attack, downstream VL capability measurement. None of these were completed. Only mechanism-level Mistral substeps ran, showing the same RR reroute failure as M3 (delta_cos_harmful=-0.012 vs target ≤-0.30). This is a fundamental scope gap relative to C3's actual requirements — however, it is honestly disclosed as budget-gated with explicit rationale.

### F. Evaluation Type: partial (mechanism-level only — real_gt for completed substeps; missing: VLM attack eval)

## Action Items

1. C3 remains fully untested at the claim level (VLM + PGD attack). The M6 partial artifacts provide only mechanism-level evidence that the Mistral RR fine-tune fails (same as M3/C1).
2. To actually test C3 in a future iteration: obtain LLaVA-NeXT-Mistral-7B weights locally; first fix the RR mechanism (C1b iteration) before rerunning M6T/M6D/PGD.
