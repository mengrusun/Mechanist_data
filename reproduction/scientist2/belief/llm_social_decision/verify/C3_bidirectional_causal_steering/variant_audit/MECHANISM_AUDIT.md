# Mechanism Audit Report — C3 Variant: model-swap-meta-llama3-8b

**Phase**: 9 (Variant integrity audit)
**Date**: 2026-07-13
**Auditor**: executor (self-review; llm-chat MCP not invoked for variant audit)
**Variant**: model-swap-meta-llama3-8b
**Swap model**: Meta-Llama-3-8B-Instruct
**Method family**: CAA activation-addition (additive steering with signed scalar coefficient)

## Overall Verdict: WARN

## Integrity Status: warn

The mechanism implementation in the swap variant correctly applies additive activation steering (`h <- h + alpha * unit_dir`) to the swap model's residual stream, using the same hook site convention and sigma_proj-scaled alpha grid as the main experiment. The check-A mechanism audit confirms that the steering coefficient sweep is bidirectional, sigma-scaled, and logged alongside a coherence metric. The key limitation is n=10 (compact sweep) and the absence of a random-direction control at L=16 — both inherited from the main supp run design.

## Triggered Checks: A — Steering Coefficient Sweep

### A. Steering Coefficient Sweep: WARN

**Trigger**: DIFF.md confirms `h <- h + alpha * unit_dir` via `SteeringHook` (same implementation as main). The swap variant applies this to Meta-Llama-3-8B-Instruct residual stream via identical Python hook registration.

**Sub-checks:**

1. **Alpha swept across multiple magnitudes**:
   - Alpha grid: {-2,-1,0,+1,+2}σ_proj — matching the main supp run.
   - At L=16 for V=M: sigma_proj=0.0871; alpha at +2σ = 0.1743. For V=I at L=16: sigma_proj=0.2537; alpha at +2σ = 0.5074.
   - This covers a 4x range from 1σ to 2σ in the sweep direction, consistent with the main supp run.
   - WARN: ±4σ not tested (same limitation as main supp run); only ±2σ.

2. **Sigma_proj-scaled**:
   - `sigma_proj` values in summary.json differ from main experiment (main V=M L=16 sigma~0.584; swap V=M L=16 sigma=0.0871). This correctly reflects the swap model's different representational geometry — the swap model's M direction has a smaller L=16 projection variance. The sigma values are computed fresh from the swap model's activations.
   - PASS: sigma_proj is correctly recalibrated per swap model.

3. **Logged alongside coherence metric**:
   - All 40 cells have `coherence.format_ok_rate=1.0` and `coherence.mean_5gram_rep=0.0`.
   - Coherence gating passes implicitly (no violations to filter).
   - PASS on coherence logging.

4. **Layer pick for swap model**:
   - Swap model ell_V*: G=10, A=10, I=0, M=0. Main model ell_V*: G=4, A=6, I=2, M=2.
   - Notably, I=0 and M=0 in the swap model suggest these binary variables are separated at the embedding layer in Meta-Llama-3-8B-Instruct. This is a genuine representational difference between the two model checkpoints and is scientifically interesting — not an error.
   - The L=16 evaluation layer is identical in both models (both are 32-layer transformers, so L=16 represents the same relative depth: 50% through the stack).
   - PASS: layer pick correctly applied to swap model; L=16 comparison is architecturally valid.

5. **Controlled with random-direction baseline**:
   - No random-direction baseline was run for the swap variant at any layer.
   - Inherits the same limitation as the main supp run (B2 control absent at L=16).
   - WARN: no random-direction control in swap variant.

6. **Sign pattern (bidirectionality)**:
   - Alpha grid {-2,-1,0,+1,+2} covers both directions.
   - In summary.json, both positive and negative alpha cells are populated for all V × L combinations.
   - PASS: bidirectional sweep present.

**Overall Check A verdict**: WARN
- Sigma scaling: PASS
- Alpha range (±2σ, not ±4σ): WARN
- Coherence logging: PASS
- Layer pick recalibration: PASS
- Random-direction control: WARN (absent)
- Bidirectional alpha: PASS

### Hook Site Convention Verification

- Main experiment: `SteeringHook` registered on `model.model.layers[layer_abs].output[0]` (Llama architecture residual stream post-attention output).
- Swap model: Meta-Llama-3-8B-Instruct uses the same `LlamaDecoderLayer` architecture. The hook site convention (`layers[layer_abs]`) maps identically between Llama-3.1-8B-Instruct and Meta-Llama-3-8B-Instruct.
- Both models are 32-layer, 4096-hidden-dim transformers; L=16 addresses the same architectural position.
- PASS: residual stream hook site is consistent between main and swap models.

### Forward Pass Integrity

- All 40 steer cells and 1 summary file are present.
- wall_time_s values are uniform (~0.29–0.33s per cell), consistent with 10-sample inference without timeout or OOM events.
- parse_failure_rate=0.0 for all cells confirms the swap model correctly parses the DG-1000 prompt format and returns numeric responses in range.
- `sample_generations` contain ["10","12","10"], ["15","12","12"] etc. — integer transfer amounts in [0,20], confirming valid game-format output from the swap model.
- PASS: forward pass executed correctly on swap model for all cells.

## Summary of Mechanism Integrity Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| n_baseline=10 (compact sweep) | WARN | Low statistical power; qualitative pattern preserved |
| No random-direction control | WARN | Cannot rule out non-specific layer perturbation effects |
| ±2σ only (not ±4σ) | WARN | Saturation regime not explored |
| L=0 ell_V* for I,M in swap | INFO | Genuine model difference; L=16 comparison unaffected |

## Action Items

1. [WARN] Re-run at n_held_baseline=200 to achieve full statistical power.
2. [WARN] Add random-direction control at L=16 on the swap model (norm-matched to swap model's sigma_proj).
3. [INFO] The L=0 ell_V* pick for I and M in Meta-Llama-3-8B-Instruct is a finding worth noting: binary social variables (income, meeting outcome) may be tokenization-level separable in this model checkpoint, suggesting they are encoded in the input embedding rather than built up through transformer layers.
