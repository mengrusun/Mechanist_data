# Mechanism Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C3 — The same representation-level intervention transfers to multimodal LLMs, blocking image-based jailbreaks (PGD image-hijack attacks against LLaVA-NeXT-Mistral-7B) without materially degrading vision-language task performance.
**Linked milestones**: M6 (M6L: Mistral site location, M6T: Mistral RR fine-tune, M6D: diagnostic)

## Overall Verdict: FAIL

*Check A triggered on M6T (Mistral RR fine-tune — same L_rr objective with same alpha coefficient structure as M3). The Mistral fine-tune used alpha=10.0 (same single hardcoded value), no alpha sweep, same near-zero L_rr EMA (0.000222), and M6D diagnostic confirms the same reroute failure (delta_cos_harmful=-0.012 vs target <=-0.30). The mechanism rigor failures are identical to C1/M3.*

## Triggered checks (this run): A (Steering Coefficient Sweep)

Trigger matches:
- `scripts/m3_rr_train.py:161` — alpha default=10.0 (M6T uses the same script with --model-path pointing to Mistral)
- `artifacts/m6/m3_mistral/training_summary.json` — alpha=10.0, same L_rr objective

## Checks

### A. Steering Coefficient Sweep: FAIL

- Triggered: yes — M6T uses the same m3_rr_train.py script with alpha=10.0 (single value, no sweep)
- Intervention type: RepE (same L_rr cosine-squared rerouting objective applied to Mistral-7B)
- Sweep grid: [10.0] — single hardcoded value; no sweep performed on Mistral
- sigma_proj scaling used: no
- Capability metric logged at each sweep point: none
- Plateau range: null
- Locked alpha: 10.0 (position: n/a)
- Random-direction control: run=true, n_random=1 (M6D d_ctrl — insufficient n_random); delta_cos_harmful_ctrl=-0.0007 (near-zero)
- Sign pattern: n/a
- Output-case spot-check: M6D delta_cos_harmful=-0.012 vs target <=-0.30 — same mechanism failure as M3 on Llama-3. The Mistral reroute did not activate.
- Evidence: artifacts/m6/m3_mistral/training_summary.json (alpha=10.0, final_L_rr_ema=0.000222); artifacts/m6/m4_mistral/activation_drift.json (delta_cos_harmful=-0.0116, criterion_c1_reroute_passed=false)
- Verdict reason: FAIL — same three mandatory violations as C1: no alpha sweep, no capability metric at sweep points, target effect not demonstrated on Mistral either. The reroute failure propagated from Llama-3 to Mistral without any hyperparameter adjustment.

### B–F. Reserved (not_implemented)

Status: not yet implemented.

## Action Items

1. For Mistral: same alpha sweep + capability metric requirement as C1/M3.
2. Note that L_rr EMA on Mistral (0.000222) is actually 7× lower than Llama-3 (0.0016) — the reroute objective is even less engaged on Mistral, possibly due to the different site geometry (sites [10-15] vs [9-14]).
3. Full C3 test (PGD image-hijack) is premature until the base Mistral RR mechanism is fixed.
