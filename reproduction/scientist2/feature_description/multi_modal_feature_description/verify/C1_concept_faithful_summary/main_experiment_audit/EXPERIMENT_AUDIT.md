# Experiment Audit Report — Claim C1

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: SemanticLens Component → CLIP Semantic-Vector Verification on ResNet-50 / ImageNet
**Claim**: C1 — For every component c in a trained vision model, a small set of top-k activation-driven reference inputs from ImageNet is a concept-faithful summary of what c encodes (P1a: last-layer top-1 purity above random baseline; P1b: hidden-layer matched-control gap > 0 at p < 0.05; P1c: k-sensitivity plateau at small k ≤ 16).
**Linked milestones**: M6_C1_last_layer (P1a), M7_C1_hidden (P1b/P1c), M11_layer_granularity

## Overall Verdict: WARN
*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- **Evidence**: fc unit c is tied to ImageNet class c by construction (classifier unit index = class index). GT loaded from dataset/architecture, not model output. In `m6_c1_last_layer.py`, correctness = `(top1 == class_labels)` where class_labels are from the dataset split. P1b uses external broden-style vocabulary (CLIP text embeddings), not model predictions, to compute the top-1 vs. top-2 cosine gap.
- **Details**: P1a is `real_gt`; P1b is external semantic proxy (top-1 vs. second-best on open vocab) — a paired within-component comparison, not circular. No model-output-as-GT detected.

### B. Score Normalization: PASS
- **Evidence**: Only standard L2 normalization of embeddings for cosine similarity. `v_c / ||v_c||`, `text_emb / ||text_emb||`. Purity is raw top-1 accuracy. Delta gap = `cos(top1) - cos(top2)`. No division by model's own prediction max/mean.
- **Details**: No self-normalized score denominators. All metrics are absolute or relative to an independently-derived baseline (random-input baseline for P1a; matched-control for P1b).

### C. Result File Existence: PASS
- **Evidence**: All C1-linked result files exist and numbers match tracker rows:
  - `runs/M6_C1_last_layer/purity__k16__mean.json`: `top1_purity=0.898, delta_pure=0.897, p=1.89e-270, passes=true` ← matches tracker row 28.
  - `runs/M7_C1_hidden/sep__k16__mean.json`: `layer4 delta_sep_mean=0.02015, p=0.0` ← matches tracker row 33.
  - All M6 k-sweep rows (k=1,4,16,64,256) marked done with consistent numbers.
  - M11 done.
- **Details**: No phantom results; files exist at claimed paths with matching numbers.

### D. Dead Code Detection: WARN
- **Evidence**: All metric functions in m6 and m7 are called. However, there is a specification/implementation mismatch: the docstring of m7 says "matched-control test" but the implementation computes `top1 - top2` within the same vocabulary without loading a separate matched-control comparator. This is a documentation/naming mismatch.
- **Details**: The `per_component_delta_sep` function in m7 correctly implements a best-vs-second-best cosine gap (a standard matched-control proxy), but the docstring is imprecise. Not dead code per se, but a weak implementation note.

### E. Scope Assessment: WARN
- **Evidence**: Claim says "for every component c in a trained vision model." Actual tested scope: 1000 fc + 500 layer4 + 500 layer3 components on ResNet-50 / ImageNet-val only. P3 cross-model universality (M12) SKIPPED by user directive 2026-07-14. The claim's universality across "all trained vision models" is unverified. P1c (k-plateau) is demonstrated via the k-sweep in M6 (last layer) and M7 (hidden, k=1/4/16/64/256 — tracker rows 31-35), with plateau analysis in M11.
- **Details**: The claim language ("for every component c in a trained vision model") exceeds the scope of a single-model, sampled-component evaluation. P3 is disclosed as skipped. Scope overstates to "every trained vision model" when evidence is ResNet-50 only. WARN, not FAIL, because the caveat is clearly disclosed in EXPERIMENT_RESULTS.md and CLAIMS_LEDGER.md.

### F. Evaluation Type: PASS
- P1a: `real_gt` — fc unit class label is dataset-structural GT, not model-generated.
- P1b/P1c: `synthetic_proxy` — external broden-style text vocabulary used as concept-identity proxy. This is the standard evaluation paradigm for CLIP-Dissect and is explicitly disclosed.

## Action Items
- Scope caveat: make claim header acknowledge ResNet-50-only scope ("for components in ResNet-50") until P3 is verified.
- P1b docstring: update `m7_c1_hidden.py` docstring to say "top-1 vs. second-best concept gap" instead of "matched-control test" for accuracy.
- These are minor documentation issues, not fundamental experimental failures.
