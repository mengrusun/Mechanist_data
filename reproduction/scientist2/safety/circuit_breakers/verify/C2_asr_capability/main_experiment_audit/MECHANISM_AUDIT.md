# Mechanism Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C2 — A model fine-tuned with Representation Rerouting (RR) achieves substantially lower attack success rates than refusal-trained (B0) or adversarial-trained (B1) baselines across a wide range of unseen HarmBench attack categories, while preserving MT-Bench and MMLU capability.
**Linked milestones**: M2, M3, M5

## Overall Verdict: FAIL

*Check A triggered on M3 (same RepE / L_rr / alpha intervention as C1). The mechanism audit finding is identical to C1's — no alpha sweep, no sigma_proj scaling, no capability metric at sweep points, n_random=1 for specificity control, and the intervention produced no measurable reroute effect (delta_cos_harmful=-0.020 vs target <=-0.30). C2's downstream ASR results (RR_ASR=0.356 > B0_ASR=0.333) are consistent with the mechanism never activating.*

## Triggered checks (this run): A (Steering Coefficient Sweep)

Trigger matches:
- `scripts/m3_rr_train.py:161` — alpha default=10.0, L_rr loss weight (M3 is in C2's milestone scope per EXPERIMENT_PLAN.md)
- `scripts/m3_rr_train.py:9` — RepE rerouting loss

## Checks

### A. Steering Coefficient Sweep: FAIL

- Triggered: yes — M3 is in C2's milestone scope; same trigger matches as C1
- Intervention type: RepE (same M3 LoRA rerouting loss; C2 reuses M3's RR model)
- Sweep grid: [10.0] — single hardcoded value; no sweep performed
- sigma_proj scaling used: no
- Capability metric logged at each sweep point: none (M5 MMLU/MT-Bench run only at the single final model, not across alpha values)
- Plateau range: null
- Locked alpha: 10.0 (position: n/a — no plateau established)
- Random-direction control: run=true, n_random=1 (d_ctrl in M4 — insufficient for statistical baseline, n_required ≥ 30)
- Sign pattern: n/a
- Output-case spot-check: RR_ASR=0.356 > B0_ASR=0.333 — the intervention produced no safety benefit and slightly worsened ASR. Consistent with mechanism failure.
- Evidence: same as C1 mechanism audit — m3_rr_train.py:161 (alpha=10.0), training_summary.json (final_L_rr_ema=0.0016), activation_drift.json (delta_cos_harmful=-0.0197), train_loss.jsonl (near-zero L_rr throughout, grad spikes at steps 330/480/490)
- Verdict reason: FAIL — same three mandatory violations as C1: no alpha sweep, no multi-point capability metric, target effect not demonstrated. C2's negative result (RR does not reduce ASR) is a direct downstream consequence of the mechanism failure, not an independent finding.

### B–F. Reserved (not_implemented)

Status: not yet implemented.

## Action Items

1. Same actions as C1 mechanism audit: run alpha sweep with capability metric, compute sigma_proj, increase random-direction control to n≥30, fix training instability.
2. M5's MMLU/MT-Bench provides a post-hoc capability picture but does not substitute for per-alpha capability measurement during the coefficient sweep.
