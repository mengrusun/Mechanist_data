# Robustness Report — Claim C4

**Claim**: C4 — The same representation-level intervention transfers to LLM agents, materially reducing the rate of harmful tool-use actions executed under attack, while preserving general function-calling capability (BFCL).
**Date**: 2026-07-15
**verdict**: INCONCLUSIVE
**stage2_skip_reason**: —
**robustness**: — (Stage 2 never ran — Phase 2 gate FAIL)
**n_eligible**: 0
**n_run**: 0
**n_pass**: 0
**n_fail**: 0
**integrity_clean**: N/A (no variants)

## Verdict: INCONCLUSIVE

### Phase 2 combined verdict: FAIL

- Experiment audit: WARN
  - BFCL uses real GT (AST function-name match vs BFCL_v3_exec_simple.json) — PASS
  - Harmful tool-use rate uses authored prompts (100 prompts) + LLM judge — two-stage proxy; honestly disclosed
  - Floor effect: B0 harmful_tool_use_rate=0.010 (base RLHF already refuses most harmful tool calls); RR=0.040 (slightly worse, likely noise at this floor)
  - BFCL 50-item substitute documented; negative result (RR worsens harm rate) honestly disclosed
  - Scope WARN: single seed, authored prompts only, no independent attack optimisation

- Mechanism audit: FAIL (Check A — Steering Coefficient Sweep)
  - M7 reuses M3 LoRA adapter without agent-specific re-training or hyperparameter sweep (m7_agent_eval.py:224 loads artifacts/m3/RR_lora)
  - Underlying M3 mechanism has same mandatory FAIL violations: single alpha=10.0 (no sweep), no capability metric at sweep points, no sigma_proj scaling
  - Random-direction control n=1 from M4 d_ctrl (requires ≥30)
  - Reroute not demonstrated: delta_cos_harmful=-0.0197 (from M4), target ≤-0.30
  - C4 negative result (RR worsened harm rate) is consistent with M3 mechanism never activating in the agent scaffold

### inconclusive_reason

main-experiment mechanism rigor broken — see verify/C4_agent_transfer/main_experiment_audit/MECHANISM_AUDIT.md

### Iteration guidance

To convert INCONCLUSIVE to a testable state:
1. Fix M3 RR mechanism (prerequisite — see C1 iteration guidance); C4 cannot be meaningfully tested until M3's mechanism actually activates.
2. Once mechanism is fixed, re-run M7 to check whether the safety transfer holds in the agent scaffold.
3. Consider whether agent-domain prompts may require a separate fine-tune rather than direct reuse of M3 weights.
4. Re-run Phase 2 audits; if combined verdict upgrades, C4 can proceed to Stage 2.

### Stage 2 plan (if C4 were admitted)

C4 was not in the Stage-2 pick pool (all claims rejected by Phase 2 gate). C4 depends mechanistically on C1 (M3 adapter reuse); C4 would only be admitted after C1's mechanism is fixed and demonstrated. No independent swap-variant plan for C4.
