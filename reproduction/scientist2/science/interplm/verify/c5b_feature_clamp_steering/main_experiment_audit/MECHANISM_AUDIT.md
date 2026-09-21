# Mechanism Audit Report — Claim C5b

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C5b — SAE-feature-clamp steering of ESM-2 generation.
**Linked milestones**: M6

## Overall Verdict: WARN
*This is C5b's mechanism-rigor verdict.*

## Triggered checks (this run): A (Steering coefficient sweep)

Trigger matches:
- `scripts/m6_feature_clamp_steering.py:133`: `hs = hs + self.alpha * self.direction.to(hs.dtype).to(hs.device)` — additive scalar × direction intervention on hidden states
- `scripts/m6_feature_clamp_steering.py:376-383`: `for alpha in args.doses` with `eff_alpha = alpha * sigma` per arm

## Checks

### A. Steering Coefficient Sweep: WARN
- **Triggered**: yes — `hs += alpha * direction` with scalar alpha multiplying additive hidden-state contribution
- **Intervention type**: SAE_feature (SAE decoder column d_f added to residual stream at all positions; random-clamp and mean-add arms also use additive hooks)
- **Sweep grid**: alpha ∈ {0.5, 1.0, 2.0, 4.0} × sigma_f (eff_alpha = alpha × sigma_f)
  - Range: 0.5–4.0 × sigma_f = 8-fold range = ~0.9 orders of magnitude
  - WARN criterion triggered: sweep spans < 3 orders of magnitude AND < 5 grid points (only 4)
- **sigma_proj scaling**: YES — eff_alpha = alpha × sigma_f (feature activation std), correctly scaling by per-feature activation magnitude. This IS sigma_proj-equivalent scaling.
- **Capability metric**: pseudo-perplexity logged per completion at every dose; plausibility band-pass rate ≥ 0.67 at all alpha values. Capability is NOT crashing.
- **Plateau identification**: No plateau observed. Results are flat (0.200 across all alpha for feature 3998 SAE-clamp) or declining. No upward dose-response in any feature.
- **Locked alpha**: N/A — no plateau to lock. The null result means no optimal alpha was chosen.
- **Alpha position**: N/A — no plateau
- **Random-direction control**: YES — random_clamp arm included with randomly selected feature direction at same dose ladder. One random feature per target-property feature.
- **Sign pattern**: N/A — symmetric (not asymmetric steering protocol)
- **Output-case spot-check**: No per-completion text cases logged; only aggregate yield/PPL statistics available. Cannot verify that capability metric tracks text quality.
- **Evidence**:
  - `scripts/m6_feature_clamp_steering.py:121-136` (SteeringHook class, additive intervention)
  - `scripts/m6_feature_clamp_steering.py:376-383` (dose loop: eff_alpha = alpha * sigma)
  - `scripts/m6_feature_clamp_steering.py:49` (--doses default=[0.5,1.0,2.0,4.0])
  - `runs/m6/summary.md` (yield/ppl table showing flat or declining response)
- **Details**: The sweep is performed but spans only ~0.9 orders of magnitude (4 points from 0.5 to 4.0 × sigma_f), triggering WARN criterion (a): less than 3 OOM and fewer than 5 grid points. Sigma_f scaling is correctly applied. Pseudo-perplexity capability metric is logged at every sweep point and shows no capability collapse (band-pass ≥ 0.67). A random-clamp control arm is included. No FAIL criteria apply: alpha is sigma-scaled (not raw norm), a sweep is performed (not single hardcoded), and capability has not crashed. The null result (no dose-response, no yield increase over no-steer) is a scientific finding — the alpha range tested may simply be insufficient to detect an effect or the features are not causally sufficient for the properties. A broader sweep (e.g. {0.1, 0.5, 1, 3, 10, 30} × sigma_f) would confirm whether the null is robust across OOM.

### B–F. Reserved (not_implemented)

## Action Items
- Extend dose sweep to ≥ 3 orders of magnitude: try {0.1, 0.3, 1, 3, 10, 30} × sigma_f (6 grid points, 2 OOM) to check whether higher magnitudes produce any effect before capabilities degrade.
- Log at least 5 sampled completion texts per (feature, alpha) for qualitative output-case spot-check.
- Add ≥ 30 random-direction samples for statistical confidence in random-clamp baseline.
