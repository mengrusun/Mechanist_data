# Integrity Audit

**Overall**: WARN    ← max severity across main experiment + variants
**Main-experiment integrity (Phase 2)**: WARN    ← max severity across main-experiment combined verdicts
**Variant integrity (Phase 9)**: WARN    ← 1 variant (C3 model-swap-meta-llama3-8b); 0 FAIL; 3 WARN findings; eligible for Phase 10

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment: proxy GT not labeled; scope: 48 unique texts, shallow layer pick) | verify/C1_linear_encoding_social_vars/main_experiment_audit/ |
| C2 | WARN | N/A | WARN | continue-with-warn (experiment: GS off-diagonal 0.506 borderline; 1-D probe weaker than full-vector) | verify/C2_purity_via_decorrelation/main_experiment_audit/ |
| C3 | WARN | WARN | WARN | continue-with-warn (experiment: supp run uses raw direction not pure GS/LEACE; mechanism: no L=16 random-direction control for sign-inversion finding) | verify/C3_bidirectional_causal_steering/main_experiment_audit/ |
| C4 | WARN | WARN | WARN | continue-with-warn (experiment: min-diagonal=0 is under-power artifact not phantom; mechanism: inherits C3 sweep issues at ell_V*) | verify/C4_selectivity_matrix/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as `pass`.
> - **Gate decision** — what Phase 2 does with this claim.

**Phase 2 summary**: All 4 claims ADMITTED (combined WARN). No claims are INCONCLUSIVE. Admitted pool = {C1, C2, C3, C4}. Stage 2 cap (MAX_VERIFY_CLAIMS=1) applies to Stage 2 entry; see STAGE2_PICK.json.

## Variant integrity (Phase 9)

**Overall variant integrity verdict**: WARN
**Scope**: C3 only (model-swap-meta-llama3-8b); C1, C2, C4 had Stage 2 skipped (INTEGRITY_ONLY, max_verify_claims_cap).

| Variant | Exp. audit | Mech. audit | Combined | Gate decision | N_eligible impact |
|---------|------------|-------------|----------|---------------|-------------------|
| C3 / model-swap-meta-llama3-8b | WARN | WARN | WARN | admitted-with-warn | INCLUDED in N_eligible (not excluded) |

**Key findings:**

1. **Hook site convention**: PASS — Meta-Llama-3-8B-Instruct and Llama-3.1-8B-Instruct share identical `LlamaDecoderLayer` architecture; L=16 is the same relative depth (50%) in both 32-layer models. Residual-stream additive hook (`h <- h + alpha * unit_dir`) applied correctly.

2. **Coherence gate**: PASS — `format_ok_rate=1.0` and `mean_5gram_rep=0.0` for all 40 steered cells. No coherence violations; no cells dropped.

3. **n_baseline discrepancy**: WARN — `n_baseline=10` in all summary cells vs `n_held_baseline=200` in config.yaml. The variant ran at `coherence_k=10` scale (compact sanity-sweep), not full held-out evaluation. SE(v_effect) ≈ 1.9 at n=10 (vs ≈0.42 at n=200). Most individual shifts are <2 SEs. Verdict: admissible as compact variant, not as full replication. The qualitative pattern (bidirectionality across all 4 V at L=16) is coherent and unlikely under pure noise.

4. **Forward pass integrity**: PASS — wall_time_s ~0.29–0.33s per cell (no OOM/timeout); parse_failure_rate=0.0; sample_generations contain valid integer responses in [0,20].

5. **Sigma_proj recalibration**: PASS — sigma_proj values differ from main experiment (correct: computed fresh from swap model's held-out activations). Swap V=M L=16 sigma=0.087 vs main sigma=0.584 — different representational geometry, correctly handled.

6. **No random-direction control at L=16**: WARN — inherits main experiment's limitation (B2 control absent at mid-layer). Cannot confirm swap model's sign-inversion effect is direction-specific vs. any large-magnitude perturbation.

7. **Swap ell_V* difference**: INFO — swap picks L=10 for G/A, L=0 for I/M (vs main L=4/6/2/2). L=0 for I/M suggests binary variable encoding is tokenization-level in Meta-Llama-3-8B-Instruct. L=16 comparison unaffected.

**Variant integrity summary**: 1 variant audited; 1 admitted (WARN); 0 excluded. N_eligible=1 for Phase 10.

**Findings log**: 3 WARN findings (n_baseline=10, no L=16 random-direction control, alpha range ±2σ only); 0 FAIL findings.

**Details**: `verify/C3_bidirectional_causal_steering/variant_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}`
