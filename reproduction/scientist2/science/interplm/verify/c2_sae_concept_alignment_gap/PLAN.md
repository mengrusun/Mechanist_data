## C2 Model-Swap Variant Plan — ESM-2-650M → ESM-2-8M

### Claim
C2 (sae_concept_alignment_gap): SAE features cover substantially more Swiss-Prot functional concepts than raw neurons from the same protein language model (SAE covered=15 at τ=0.5, neurons=0; ratio=∞ at 650M scale).

### Main experiment summary
- Model: ESM-2-650M (d_model=1280, 33 layers), SAEs at layers {1,9,18,24,30,33}, d_feat=10240
- Dataset: Swiss-Prot test split, 1500 sequences
- Protocol: per-residue F1 alignment (q_top=0.99, τ_F1=0.5, τ_clean=0.7)
- Result: SAE covered=15 concepts, neurons=0; ratio=∞ at τ=0.5
- Verdict: not-supported (absolute count=15 vs. target≥65, but direction supported)

### Variant #1: model dimension swap

| Field | Value |
|---|---|
| dimension | model |
| swap | ESM-2-650M → ESM-2-8M |
| model | esm2_t6_8M_UR50D (d_model=320, 6 transformer layers) |
| SAE | /data/zhenqian/models/InterPLM-esm2-8m/layer_{1,2,3,4,5,6}/ (d_feat=10240, same expansion factor 32) |
| dataset | same Swiss-Prot test split, same 1500 sequences |
| protocol | identical: q_top=0.99, τ_F1=0.5, τ_clean=0.7, same concept universe |
| output | verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/ |

### Key code changes from main experiment (m2_concept_alignment.py + m1 activations)

1. ESM_DIR: /data/zhenqian/models/ESM-2-8M (esm2_t6_8M_UR50D, d_model=320)
2. SAE_ROOT: /data/zhenqian/models/InterPLM-esm2-8m
3. SAE_LAYERS: [1, 2, 3, 4, 5, 6] (all 6 transformer layers of 8M)
4. The variant is a self-contained two-phase script:
   - Phase A: collect hidden states from ESM-2-8M for the same Swiss-Prot 1500 sequences and run them through the 8M SAEs to produce activation tensors + sparse codes (m1-equivalent for 8M)
   - Phase B: run F1 alignment protocol identical to m2_concept_alignment.py but pointing at Phase A outputs
5. Output files mirror m2/ layout: coverage.json, sensitivity.json, per_concept_best_F1.parquet

### Judgment criterion
Compare the variant result against the main experiment's conclusion on C2:
- Main verdict on C2: not-supported (direction held: SAE≫neurons; but absolute count=15 < target 65)
- Variant is CONSISTENT if: SAE coverage > neuron coverage (direction preserved) regardless of absolute count
- Variant is INCONSISTENT if: SAE coverage ≤ neuron coverage (direction reversed)

Note: the direction claim (SAE≫neurons) is what makes C2 scientifically interesting. If 8M also shows SAE≫neurons (even at low absolute count), the null verdict is robust across scale. If the gap collapses at 8M, the main experiment's null is scale-specific.

### Risks and mitigations
- Risk: 8M activations must be recomputed (not cached). Mitigation: run M1-equivalent for all 6 layers, batch_size=8 (d_model=320 << 1280).
- Risk: 8M SAE layers are 1-6 (not 1,9,18,24,30,33). Mitigation: use all 6, take best per concept.
- Risk: Memory. At d_model=320 vs 1280 (4× smaller), 8M runs use ~4× less GPU memory — well within budget.
- GPU: CUDA_VISIBLE_DEVICES=0,1,2,3
