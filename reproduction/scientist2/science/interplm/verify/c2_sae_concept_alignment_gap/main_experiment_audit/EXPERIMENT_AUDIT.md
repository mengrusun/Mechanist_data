# Experiment Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C2 — SAE features cover up to ~143 distinct Swiss-Prot biological concepts vs. raw ESM-2 neurons ~46 concepts / ~15 cleanly recovered, under same alignment protocol.
**Linked milestones**: M2

## Overall Verdict: WARN
*This is C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Ground truth comes from external Swiss-Prot per-residue annotations parsed from the full UniProt uniprot_sprot.dat.gz (466,006 entries, 700MB, downloaded from UniProt FTP). This is independent real annotation data, not derived from model outputs.
- Evidence: `EXPERIMENT_TRACKER.md` notes "full `uniprot_sprot.dat.gz` (700 MB) downloaded from `https://ftp.uniprot.org/pub/databases/uniprot/current_release/...` because the workspace-provided copy was a 50 KB truncated stub."

### B. Score Normalization: PASS
- F1 is computed from standard precision/recall of binary masks (unit-positive mask vs concept-annotation mask). No division by model's own maximum or any prediction statistic.
- Evidence: `scripts/m2_concept_alignment.py` — F1 computed as `2*P*R/(P+R)` from binary residue masks.

### C. Result File Existence: PASS
- All cited numbers match result files.
- `runs/m2/coverage.json`: SAE covered_union=15, neuron covered_union=0 — matches EXPERIMENT_RESULTS.md.
- `runs/m2/sensitivity.json`: τ=0.3 sweep shows SAE=65, neurons=2 — matches EXPERIMENT_RESULTS.md table.
- Top-5 concepts by best F1 reported in results are consistent with `runs/m2/per_concept_best_F1.parquet` (available on disk).
- Tracker row m2: status=done, notes SAE covered=15 (τ=0.5), 65 (τ=0.3); neurons=0/2.

### D. Dead Code Detection: PASS
- M2 alignment script was executed end-to-end and produced all planned output files: `runs/m2/coverage.json`, `runs/m2/sensitivity.json`, `runs/m2/per_concept_best_F1.parquet`, `runs/m2/best_layer.json`, `runs/m2/unaligned_features.json`.

### E. Scope Assessment: WARN
- Planned scope: full Swiss-Prot test split plus 50,000 UniRef sequences for context.
- Actual scope: 1,500 Swiss-Prot test sequences (387,195 residues, ~15% of available set); 400 fine-grained sub-typed concepts (Pfam-family level, more granular than aggregate categories).
- Claim qualifier: "up to ~143" — the "up to" qualifier is appropriate.
- At primary setting (τ=0.5): SAE covered=15 (far from target 143); at τ=0.3: SAE covered=65.
- All subsample sizes and the concept granularity difference are transparently disclosed in EXPERIMENT_RESULTS.md.
- The directional gap (SAE ≫ neurons at every setting) is robustly supported; the absolute count is under-powered due to fine-grained concept definition and smaller test sample.
- Assessment: WARN (honest acknowledged under-power), not FAIL.

### F. Evaluation Type: PASS / real_gt
- Per-residue F1 alignment uses Swiss-Prot annotations as ground truth. External, independently curated, not model-derived.

## Action Items
- Re-run at full 10k test sequences to improve power on rare concepts.
- Consider aggregating Pfam sub-families to match the reference paper's concept granularity if the absolute 143/46/15 targets are needed.
- Ratio criterion is robustly met (SAE ≫ neurons in every setting, including primary: neurons=0 at τ=0.5).
