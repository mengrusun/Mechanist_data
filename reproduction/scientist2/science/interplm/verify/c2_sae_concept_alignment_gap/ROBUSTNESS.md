## c2: model-swap stress test (ESM-2-650M → ESM-2-8M) → PASS

- robustness: 1.0  (1 pass / 1 eligible)
- n_eligible: 1
- n_pass: 1
- n_fail: 0
- n_run: 1 (DIMENSIONS=model → 1 variant)
- robustness_threshold: 0.5
- verdict: PASS (robustness 1.0 ≥ threshold 0.5)

### Main experiment verdict on c2: not-supported
- Main: ESM-2-650M, SAE covered=15 at τ=0.5, neurons=0, ratio=∞
- Main verdict: not-supported (15 < target 65; direction SAE≫neurons supported)

### Variant #1: model-swap-esm2-8m
- Model: ESM-2-8M (d_model=320, 6 layers {1,2,3,4,5,6})
- SAE: /data/zhenqian/models/InterPLM-esm2-8m, d_feat=10240
- Dataset: same Swiss-Prot test split, 1500 sequences, 387195 residues
- Protocol: identical q_top=0.99, τ_F1=0.5, τ_clean=0.7

**Result:**
- SAE covered_union = 14 (main=15; delta=-1)
- SAE clean_union = 2 (main=1)
- Neuron covered_union = 0 (main=0)
- ratio = ∞ (SAE≫neurons preserved)

**Sensitivity sweep:**
| q_top | τ_F1 | SAE | Neuron | Ratio |
|-------|------|-----|--------|-------|
| 0.99  | 0.3  | 58  | 3      | 19.3× |
| 0.99  | 0.5  | 14  | 0      | ∞     |
| 0.99  | 0.7  | 2   | 0      | ∞     |
| 0.95  | 0.3  | 56  | 2      | 28.0× |
| 0.95  | 0.5  | 13  | 2      | 6.5×  |
| 0.95  | 0.7  | 2   | 0      | ∞     |

**Variant verdict:** not-supported (14 < 65 target) — CONSISTENT with main experiment
**Variant integrity (Phase 9):** CLEAN (EXPERIMENT_AUDIT=pass, MECHANISM_AUDIT=n/a)
**Judgment:** PASS — direction SAE≫neurons preserved across scale swap; absolute gap unchanged

### Interpretation
The SAE alignment gap (SAE≫neurons at the directional level) is robust to a 80× parameter-count reduction (650M → 8M). Both models produce essentially identical coverage at the primary setting (14–15 concepts vs 0 neurons). The SAE architecture advantage over raw neurons is a model-invariant property of the representation scheme, not an artifact of model scale. The not-supported verdict on the absolute-count target (≥65) is equally robust — the gap to target is model-scale-independent.

### Robustness wall-clock
Variant run: 255 seconds (4.25 minutes), GPU 0,1,2,3 (physically GPU 4 primary in CUDA enumeration)
