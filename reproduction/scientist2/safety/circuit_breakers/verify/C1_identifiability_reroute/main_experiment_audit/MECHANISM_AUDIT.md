# Mechanism Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C1 — Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable internal representations that can be rerouted to an orthogonal, non-harmful subspace using only paired benign/harmful data, without exposure to any attack prompts.
**Linked milestones**: M1, M3, M4

## Overall Verdict: FAIL

*This is C1's mechanism-rigor verdict. Check A (Steering Coefficient Sweep) was triggered by RepE / alpha / L_rr keyword matches in m3_rr_train.py and m4_diagnostic.py. The reviewer found multiple mandatory FAIL-level violations: no alpha sweep, no sigma_proj scaling, no multi-point capability metric, and the locked alpha produced no measurable reroute effect (delta_cos_harmful=-0.0197 vs target <=-0.30).*

## Triggered checks (this run): A (Steering Coefficient Sweep)

Trigger matches:
- `scripts/m3_rr_train.py:161` — `alpha` argument default 10.0, L_rr loss coefficient
- `scripts/m3_rr_train.py:9` — RepE rerouting loss pattern
- `scripts/m4_diagnostic.py:17-18` — direction d_h used as intervention target

## Checks

### A. Steering Coefficient Sweep: FAIL

- Triggered: yes — via m3_rr_train.py:161 (alpha coefficient), m3_rr_train.py:9 (RepE rerouting), m4_diagnostic.py (d_h direction evaluation)
- Intervention type: RepE (training-time LoRA rerouting loss)
- Sweep grid: [10.0] — single hardcoded value; no sweep was performed
- sigma_proj scaling used: no — sigma_proj was never computed; alpha=10.0 is in raw loss weight units, incomparable across runs
- Capability metric logged at each sweep point: none — M5 MMLU/MT-Bench run only at the single final alpha, not across alpha values; collapse cannot be detected
- Plateau range: null — not identifiable (only one alpha tried)
- Locked alpha: 10.0 (position in plateau: n/a — no plateau established)
- Random-direction control: run=true, n_random=1 (one random direction d_ctrl in M4 — far below the required n_random≥30); n_random=1 does not constitute a statistical baseline
- Sign pattern (if asymmetric): n/a
- Output-case spot-check: cases_available=true; verdict=fail; M4 shows delta_cos_harmful=-0.0197 vs target ≤-0.30 — the reroute effect is within noise range; no multi-alpha behavior/capability spot-check performed
- Evidence: m3_rr_train.py:161 (alpha default=10.0); training_summary.json (alpha=10.0, final_L_rr_ema=0.0016); train_loss.jsonl steps 0,50,100,300-490 (L_rr near-zero throughout, grad_norm spikes 852,159,137 at steps 330,480,490); artifacts/m4/activation_drift.json (delta_cos_harmful=-0.0197, criterion_c1_reroute_passed=false)
- Verdict reason: FAIL on three mandatory criteria simultaneously — (1) single hardcoded alpha, no sweep; (2) no capability metric logged at any alpha grid point; (3) target effect (delta_cos_harmful=-0.020) is within baseline noise relative to the required ≤-0.30, and criterion_c1_reroute_passed=false. Additionally: no sigma_proj scaling; inadequate random-direction control (n=1); training instability (grad_norm spikes to 852 at step 330).

### B–F. Reserved (not_implemented)

Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items

1. **Run an alpha sweep**: try at least [0.1, 0.3, 1.0, 3.0, 10.0, 30.0] and log target metric (delta_cos_harmful) + capability metric (MMLU or MT-Bench) at each point.
2. **Compute sigma_proj**: extract projection std on the held-out set and express alpha in sigma_proj units for comparability.
3. **Increase random-direction control**: run n_random ≥ 30 random-direction controls at the locked alpha to provide a statistical baseline for specificity.
4. **Identify a plateau**: run at low alpha (near zero) to confirm alpha=0 baseline, then increase until capability degrades — lock to the middle of the stable window.
5. **Diagnose training instability**: investigate the grad_norm spikes (852, 159, 128 at steps 330, 480, 470) — these suggest the LoRA subspace is not maintaining stable gradient flow for the RR objective.
