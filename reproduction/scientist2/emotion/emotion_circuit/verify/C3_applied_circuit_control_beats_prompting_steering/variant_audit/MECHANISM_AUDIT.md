# Mechanism Audit — C3 Variant (Qwen2.5-7B-Instruct) — Phase 9 Variant Integrity

**Variant**: Qwen2.5-7B-Instruct model swap
**Family**: Causal Attribution / Ablation (same three-arm protocol, Arm A uses enhancement operator)

---

## Variant Method Alignment

The Qwen variant re-runs the same three-arm comparison (A=Circuit, B=Prompting, C=Steering) using:
- The same mechanism family (Causal Attribution / Ablation for Arm A's enhancement operator)
- Qwen's own Stage A+B-derived C_e (qwen_C_e.json, qwen_kstar.json)
- Qwen-specific direction vectors (qwen_directions.npz)
- Qwen-tuned hyperparameters (qwen_selected_configs.json)

Within-family constraint: method is NOT swapped (still ablation-family enhancement for Arm A,
CAA-style for Arm C, template prompting for Arm B). Only the model is swapped. PASS.

## Slot A Coverage

The Qwen val sweep covers alpha_C in {0.5, 1.0, 2.0} for Arm C. Arm A sweeps alpha_A in
{0.5, 1.0, 2.0} x (k_h, k_n) neighborhood. Both preserve the 3-point alpha sweep. PASS.

## Architecture Adaptation

Qwen2.5-7B-Instruct has 28 layers x 28 heads x MLP intermediate_size=18944.
The hook implementation in m1_location.py is architecture-agnostic (uses model.model.layers,
self_attn.o_proj, mlp.gate_proj). Confirmed to work on both Llama and Qwen families in M4.
PASS.

## Overall Variant Mechanism Integrity

**PASS**. The Qwen variant correctly applies the same protocol with architecture-adapted parameters.
The within-family constraint (model swap only) is satisfied. The three-arm competition is preserved.
