# Integrity Audit

**Overall**: WARN    ← max severity across main experiment + variants (main-experiment C2 WARN dominates; variants clean)
**Main-experiment integrity (Phase 2)**: WARN    ← max severity across per-claim combined verdicts (C1 PASS, C2 WARN)
**Variant integrity (Phase 9)**: PASS    ← the one picked-claim variant (C1 model-swap) is integrity-clean

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision                  | Detail |
|-------|------------|-------------|----------|--------------------------------|--------|
| C1    | PASS       | PASS        | PASS     | continue                       | verify/C1_helix_gain_doseresponse/main_experiment_audit/ |
| C2    | WARN       | N/A         | WARN     | continue-with-warn (experiment)| verify/C2_validity_preserved_noninferior/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of the A–F experiment-methodology audit on this claim.
> - **Mech. audit** — `overall_verdict` of the mechanism-rigor audit. `N/A` when the claim uses no intervention of its own (C2 inherits C1's CAA setting).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`, `n/a` treated as pass.
> - **Gate decision** — C1 admitted (PASS) → Stage 2 eligible; C2 admitted-with-warn → Stage 2 eligible. Under `MAX_VERIFY_CLAIMS=1` only the top-1-by-importance claim (C1) proceeds to swap variants (see `verify/STAGE2_PICK.json`); C2 → INTEGRITY_ONLY (`stage2_skip_reason: max_verify_claims_cap`).

**C2 WARN driver:** experiment-audit check E — the claim wording "off-target properties not degraded" is optimistic against the documented CAA-specific off-target shifts at the winning setting (protein length −36.6 aa ≈ −37%; GC −0.139). The non-inferiority test itself is methodologically clean and pre-registered. See `verify/C2_validity_preserved_noninferior/main_experiment_audit/EXPERIMENT_AUDIT.md`.

## Variant integrity (Phase 9)

**Variants audited**: 1  (1 pass, 0 warn, 0 fail)

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|-----------------------|------------------------|----------|
| C1    | PASS                  | PASS                   | PASS     |

*(C2 not audited here — deferred to INTEGRITY_ONLY under the MAX_VERIFY_CLAIMS=1 cap; no variants ran.)*

### Findings (per variant)
- C1 / variant `model-swap-evo2-7b-262k`: **[INTEGRITY: PASS]** — reuses the frozen ESMFold→DSSP assay (model-independent GT), same scoring path, anti-circularity preserved (DEV build / TEST-split eval), coefficient re-swept in-model with capability metric co-logged and an α=0 in-model baseline. Counts in both numerator and denominator of robustness. Detail: `verify/C1_helix_gain_doseresponse/variant_audit/`.
