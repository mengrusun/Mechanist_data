# Mechanism Audit — C4 Variant (model-swap-qwen-14b)

**Claim:** C4 — steering-vector control offers more distinct (rate, accuracy) operating points than prompt/TI  
**Variant:** model-swap-qwen-14b (DeepSeek-R1-Distill-Qwen-14B)  
**Audit type:** Phase 9 variant-level mechanism integrity audit  
**Date:** 2026-07-15  
**Auditor:** auto-verify Phase 9 (audit-by-construction, symmetric to Phase 2)

**Overall verdict: WARN**

---

## Check A — Steering Coefficient Sweep

**Status: WARN** | **Triggered: Yes** (CAA steering intervention)

### Intervention details

- Intervention type: Contrastive Activation Addition (CAA)
- Alpha grid: {-2, -1, +1, +2} × sigma_proj × unit(v), same as main experiment
- sigma_proj (Qwen-14B, expressing_uncertainty): **886.0** (vs 10.52 for Llama-8B = 84x larger)
- Layer: L*=45 (Qwen-14B, 48 total layers — M1 result)
- Direction: extracted from 200 chains on the same auxiliary corpus

### WARN findings

1. **Inherited under-validated operating region**: Variant reuses the same four-point alpha grid {-2,-1,+1,+2} without re-establishing a mid-plateau for Qwen-14B. No random-direction control, no confirmed plateau in M3 (same WARN as Phase 2 mechanism audit).

2. **sigma_proj magnitude effect**: sigma_proj=886 means the actual steering coefficient at alpha=1 is 886 (vs 10.52 for Llama-8B at the same alpha). At alpha=2, the coefficient is 1772. This is dramatically above any coherent steering range, producing near-total output incoherence (coherence_rate in [0.0, 0.40] for all four alpha values). This is a scientifically important finding about cross-backbone sigma_proj scaling, not a methodology failure.

3. **Protocol fidelity**: The variant intentionally replicates the main experiment protocol (same alpha set, same formula) to test robustness — not to find the optimal alpha for Qwen-14B. Re-calibrating alpha for Qwen-14B would change the experiment. This design intent is consistent with the PLAN.md specification and is appropriate for a model-swap robustness test.

### No FAIL findings

The mechanism audit does not FAIL because:
- The evaluation protocol is faithfully executed (M1 + M4 replication on new model)
- The coherence collapse is a genuine scientific result, not an evaluation artefact
- The sigma_proj finding is disclosed and explains the result mechanistically
- Capability metric (accuracy + coherence_rate) is logged for all controllers

---

## Checks B–F — Reserved

Not implemented in current skill version.

---

**Audit conclusion:** WARN (inherited under-validated α_op + sigma_proj magnitude effect; no FAIL). Variant admitted to robustness computation. Combined with EXPERIMENT_AUDIT WARN: Phase 9 integrity_status = WARN.
