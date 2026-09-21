# Integrity Audit

**Overall**: WARN
**Main-experiment integrity (Phase 2)**: WARN
**Variant integrity (Phase 9)**: WARN (1 warn, 0 fail; 1 variant eligible)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment: scope — matched-control A≈D fails claim predicate) | verify/C1_bottleneck_layer_llm/main_experiment_audit/ |
| C2 | WARN | WARN | WARN | continue-with-warn (experiment: scope/downscale; mechanism: lambda not swept) | verify/C2_bottleneck_anchored_dpo/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention (C1's M1+M2 use forward-only diagnostics and activation patching without a steered coefficient).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as pass.
> - **Gate decision** — what Phase 2 does with this claim.

**Phase 2 summary**: Both C1 and C2 admitted to Stage 2 with WARN. No FAIL — main experiment results files exist, numbers verify from raw data, no dead code, no fake GT. Warn sources: (C1) claim predicate requires matched-control specificity A>D which fails (A≈D=0.755); (C2) worst-language tie (not strictly lower), MGSM/sw -5pp capability regression, MultiJail cap at 100/lang, lambda_bottleneck at single untested value.

## Variant integrity (Phase 9)

**Date**: 2026-07-14
**Scope**: C2 only (C1 deferred — stage2_skip_reason: max_verify_claims_cap)

| Claim | Variant | Exp. audit | Mech. audit | Combined | Eligible? | Detail |
|-------|---------|------------|-------------|----------|-----------|--------|
| C2 | model-swap-qwen25-7b | WARN | WARN | WARN | YES | verify/C2_bottleneck_anchored_dpo/variant_audit/ |

**Phase 9 summary**: The model-swap-qwen25-7b variant for C2 passes integrity review with WARN. All result files exist and numbers compute from raw data. Warn sources: (Exp.) SW N_eff=10 (50/60 OTHER responses for all models — unreliable ASR for SW, applied symmetrically); MT-Bench 15 vs 25 prompts (declared budget adjustment); MultiJail 60/lang vs 100/lang (declared). (Mech.) L* relative-depth shifts from 31% (LLaMA, L*=10/32) to 50% (Qwen, L*=14/28); Qwen2.5-7B's higher inherent multilingual safety baseline reduces L*-anchor signal headroom. No FAIL — no fabricated data, no dead code, no score normalization violations. The variant is **eligible for robustness computation** (N_eligible=1).

**Integrity-FAIL variants excluded from robustness**: 0 (no variant failed Phase 9 integrity gate).

**Overall Phase 9 status**: WARN (1 warn, 0 fail out of 1 variant run)
