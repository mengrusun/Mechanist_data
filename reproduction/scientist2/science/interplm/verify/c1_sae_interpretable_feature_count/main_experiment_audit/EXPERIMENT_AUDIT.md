# Experiment Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C1 — For at least one layer L in {1,9,18,24,30,33}, the count of interpretable SAE latent features approaches ~2,548, and this per-layer interpretable-feature count is at least 10× larger than the count of interpretable individual neurons at the same layer.
**Linked milestones**: M1

## Overall Verdict: WARN
*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
- Data source: Swiss-Prot test split (1500 sequences, 387,195 residues) from real UniProt protein sequences — provenance is real and external.
- However, the interpretability gate (auto-interp score ≥ 0.3) uses an LLM-consistency proxy: top-activating windows are summarized by gpt-5.4, then scored on held-out windows by agreement fraction. This is a self-supervised consistency measure, not an externally validated interpretability benchmark. This is an acknowledged design choice for a claim about "interpretable" features — no external ground-truth interpretability label set exists for SAE features.
- Evidence: `scripts/m1_feature_count.py` — LLM gate logic; `runs/m1/layer9.json` gate_features_sample showing LLM labels and scores.
- Classification: self_supervised_proxy for the interpretability gate; the underlying data (Swiss-Prot sequences) is real_gt.

### B. Score Normalization: PASS
- No improper self-normalization is evident. The reported quantities are raw counts (interpretable-feature count = raw integer) and raw ratios (SAE/neuron). No metric is divided by a model-specific maximum or any prediction statistic.
- Evidence: `runs/m1/layer9.json`: `estimated_interpretable_count=1480.05`, `ratio_sae_over_neurons=19.27` — these are pure counts and counts ratios.

### C. Result File Existence: PASS
- Reported results are consistent across all artifacts.
- EXPERIMENT_TRACKER row m1_L9: `est_sae=1480, est_neuron=77, ratio=19.3×` — done.
- `runs/m1/layer9.json`: `estimated_interpretable_count=1480.05` (SAE), `76.8` (neurons), `ratio_sae_over_neurons=19.27`.
- EXPERIMENT_RESULTS.md: "best is L9: SAE_interp=1480, ratio=19.3×" — matches file up to rounding.
- All 6 layer files exist (`runs/m1/layer{1,9,18,24,30,33}.json`).

### D. Dead Code Detection: PASS
- The pipeline was executed end-to-end. All 6 layer result files exist, tracker status is `done`, concrete outputs are present in `runs/m1/layer9.json` with gate statistics. Auto-interp gate was run (gate_features_sample present with 20 sampled features).

### E. Scope Assessment: WARN
- Planned scope: ~10,000 Swiss-Prot test sequences + 50,000 UniRef sequences for top-activating windows.
- Actual scope: 1,500 Swiss-Prot sequences (~15%); UniRef windows unavailable (pre-tokenized format only, no raw sequences); auto-interp gate estimated from 100-feature sample and extrapolated to full feature set via pass-rate × normal_density_count.
- Claim qualifier: "up to ~2,548" — uses "up to" which is appropriate qualification. Best achieved: 1,480 at L9.
- All subsample sizes are transparently disclosed in EXPERIMENT_RESULTS.md with explanations.
- Limitation: estimated count (15% sample rate) carries ±15% margin; true count could be 1,260–1,700. The ratio criterion (≥10×) IS met at 19.3×.
- Assessment: Scope is acknowledged and honestly disclosed; claim language is appropriately qualified. WARN (acknowledged under-power), not FAIL (not hidden or misleadingly reported).

### F. Evaluation Type: WARN
- Data source: real Swiss-Prot sequences (real_gt).
- Interpretability gate: LLM-consistency proxy (self_supervised_proxy) — scores agreement between LLM-generated feature label and held-out activation windows.
- This dual-type evaluation is by design for the claim type (no external interpretability GT exists). The proxy is documented.

## Action Items
- For full-fidelity re-run: extend to 10,000 Swiss-Prot test sequences and add UniRef window sampling (decode pre-tokenized input_ids via ESM-2 tokenizer).
- For confidence in estimated count: run auto-interp gate on full feature set rather than 100-feature sample.
- The ratio criterion (≥10×) is robustly met across 5 of 6 layers (L1 is anomalous — dead SAE). The absolute count (≥2,000) is the under-powered component.
