# Mechanism Audit Report — Claim C4

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C4 — The same representation-level intervention transfers to LLM agents, materially reducing the rate of harmful tool-use actions executed under attack, while preserving general function-calling capability (BFCL).
**Linked milestones**: M7 (M3 RR weights reused — no new intervention fine-tune)

## Overall Verdict: FAIL

*Check A triggered because M7 reuses M3's LoRA adapter (same RepE intervention). The mechanism-rigor violations are identical to C1 and C2: no alpha sweep, no sigma_proj scaling, no capability metric at sweep points, n_random=1. The M7 evaluation itself (pure inference + BFCL) is not the intervention, but the intervention (M3) that was never properly swept underlies C4's expected safety transfer. C4's negative result (RR harmful_rate=0.040 vs B0=0.010) is consistent with the mechanism failure.*

## Triggered checks (this run): A (Steering Coefficient Sweep)

Trigger matches:
- `scripts/m3_rr_train.py:161` — M7 loads M3's RR_lora adapter (m7_agent_eval.py:224); the alpha coefficient that produced the adapter feeds C4's safety claim
- `scripts/m7_agent_eval.py:224` — `PeftModel.from_pretrained(model, ARTIFACTS_DIR / "m3" / "RR_lora")`

## Checks

### A. Steering Coefficient Sweep: FAIL

- Triggered: yes — M7 loads M3 adapter; M3's RepE coefficient controls the safety transfer claimed by C4
- Intervention type: RepE (M3 LoRA adapter reused for agent evaluation)
- Sweep grid: [10.0] — single alpha from M3; no C4-specific sweep
- sigma_proj scaling used: no
- Capability metric logged at each sweep point: none (BFCL provides one-point post-hoc capability measurement at the single alpha)
- Plateau range: null
- Locked alpha: 10.0 (position: n/a)
- Random-direction control: run=true, n_random=1 (from M4 d_ctrl) — insufficient
- Sign pattern: n/a
- Output-case spot-check: harmful_tool_use_rate RR=0.040 vs B0=0.010 — RR worsened harm rate (both near-zero, floor effect). BFCL preserved at 1.000 (trivially passable since B0 was already 0.980). Consistent with mechanism failure.
- Evidence: m7_agent_eval.py:224 (loads M3 RR_lora adapter); artifacts/m7/RR_agent.json (harmful_rate=0.040, bfcl=1.000); artifacts/m4/activation_drift.json (delta_cos_harmful=-0.0197, reroute not achieved)
- Verdict reason: FAIL — same mandatory violations as C1-C3. C4 reuses M3 weights without any agent-domain adaptation or agent-specific sweep, and M3's mechanism was never properly exercised.

### B–F. Reserved (not_implemented)

Status: not yet implemented.

## Action Items

1. C4 cannot be meaningfully tested until M3's RR mechanism actually activates (C1b fix first).
2. Once the mechanism is fixed, re-run M7 to check whether the safety transfer holds in the agent scaffold.
3. Consider whether agent-domain prompts may require a separate fine-tune rather than direct reuse of M3 weights.
