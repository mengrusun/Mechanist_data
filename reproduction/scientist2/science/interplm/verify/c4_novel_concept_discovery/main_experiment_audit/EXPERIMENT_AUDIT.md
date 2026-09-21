# Experiment Audit Report — Claim C4

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C4 — ≥10% of Swiss-Prot-unaligned SAE features receive coherent non-synonym labels from an external LLM auto-interpreter, above a random-feature control.
**Linked milestones**: M4

## Overall Verdict: WARN
*This is C4's integrity verdict.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
- The novelty gate uses an LLM (gpt-5.4) as a synonym classifier against Swiss-Prot vocabulary. This is an internal LLM judgment, not an external independent dataset-based GT.
- However, this is inherent to the claim type — C4 is about features that are NOT in the vocabulary, so no external annotation exists for them by definition. The synonym-check prompt is a reasonable proxy.
- The auto-interp score (s_auto) also uses LLM scoring on held-out windows — another self-supervised proxy.
- Classification: self_supervised_proxy (two LLM judgment stages).
- Evidence: `scripts/m4_novel_concept_llm.py` — Prompt A (label elicit), Prompt B (predictivity score), Prompt C (synonym check).

### B. Score Normalization: PASS
- No self-normalization. s_auto = (hits ∩ held-out)/|held-out| − (hits ∩ low-activation)/|low-activation| — computed from LLM predictions on window sets, no division by model max.

### C. Result File Existence: PASS
- `runs/m4/control_stats.json` exists: n_labeled_unaligned=100, n_novel_real=0, novel_rate_real=0.0, n_control=50, n_novel_control=0, novel_rate_control=0.0, specificity_ratio=0.0.
- `runs/m4/novel_concepts.parquet` exists.
- EXPERIMENT_RESULTS.md reports: "0 (0.0%) novel-concept-coherent" — matches control_stats.json exactly.
- Tracker row m4: status=done, notes "0/100 novel by strict criterion (synonym-check-null); 3 target-property features for M6."

### D. Dead Code Detection: PASS
- M4 pipeline executed end-to-end: 100 features labeled, 50 control features labeled, synonym-check ran on all, producing the 0/0 result. 3 target-property features extracted for M6 (present in `runs/m4/target_property_features.json`). LLM call cache populated (2,590 entries total).

### E. Scope Assessment: WARN
- Planned: 500 unaligned features × 60 windows/feature (20 top + 20 held-out + 20 low); UniRef windows for top-activating contexts.
- Actual: 100 features × 45 windows/feature; Swiss-Prot windows only (UniRef unavailable).
- The key methodological concern is not the subsample but the criterion discriminativity: both real and control arms yield 0% novel features. EXPERIMENT_RESULTS.md correctly identifies this as "a criterion-design outcome, not evidence that SAE features lack novel concepts" — the synonym-check gate may be too permissive in labeling anything as a synonym.
- Scope is honestly disclosed. The criterion design issue is transparently reported with a proposed fix (embedding-distance-based novelty test).
- Assessment: WARN (honest disclosure of criterion inadequacy, not FAIL).

### F. Evaluation Type: WARN / self_supervised_proxy
- LLM-based synonym check and predictivity score — self-supervised proxy.
- The claim by nature cannot use external GT (unaligned features = features not in any existing vocabulary).

## Action Items
- The synonym-check criterion (LLM prompt C) appears non-discriminative: gpt-5.4 maps most feature labels to some Swiss-Prot vocabulary paraphrase. Fix options: (a) stricter prompt ("must not paraphrase any Swiss-Prot term; if in doubt, answer YES to is_synonym"); (b) embedding-distance threshold (label embedding vs. vocabulary embeddings); (c) human-curated judgment on a sample.
- Re-run with 500 features and UniRef windows (decode input_ids) for adequate power.
