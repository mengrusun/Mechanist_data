# Mechanism Audit Report — Claim C2

**Date**: 2026-08-19
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.6-luna)
**Claim**: C2 — Amplifying the alpha-helix-selective feature set S during Evo2 decoding raises predicted alpha-helix content of translated ORFs vs an identically-decoded unsteered baseline.
**Linked milestones**: M2, M-CTRL, M3

## Overall Verdict: PASS
## Triggered checks (this run): A

## Checks
### A. Steering Coefficient Sweep: PASS
- Triggered: yes — residual-add SAE-feature amplification at blocks.26 (common.py ResidualSteerer)
- Intervention type: SAE feature scaling
- Sweep grid: [0, 0.5, 1, 2, 4, 8, 16] (7 points, incl. 0; 32x span)
- Target criteria met: yes (held-out alpha*=1 n=2205, p<0.05) -> sweep-width question satisfied
- Coefficient unit: raw multiplier on fixed vector v (unit decoder dirs x median-positive activations); stated & reproducible
- Capability metric logged: Evo2 perplexity + ORF-valid rate at every alpha (both flat/non-collapsing)
- Plateau range: [0.5, 4) positive plateau {0.5,1,2}, collapse alpha>=4
- Locked alpha: 1.0 (position: middle of plateau)
- Random-direction control: yes (null_direction n=300, norm-matched) -> learned S (0.264) beats random (0.251) and random_feature (0.227)
- Sign pattern: n/a (uniform push)
- Output-case spot-check: metric_text_consistent (alpha=16 = valid fluent low-helix DNA, not a fluency collapse)
- Verdict reason: all 8 questions satisfied; alpha mid-plateau with behavior + capability evidence.

### B–F. Reserved (not_implemented)

## Action Items
- None; mechanism rigor clean.
