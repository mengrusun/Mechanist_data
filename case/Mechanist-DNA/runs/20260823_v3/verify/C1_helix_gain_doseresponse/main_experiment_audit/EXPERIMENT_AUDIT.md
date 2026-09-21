# Experiment Audit — C1 (helix gain / dose-response) — main experiment

**Scope:** M1–M4 evidence for C1. **Overall verdict: PASS.**

C1: *targeted internal intervention during Evo2-7B decoding raises the encoded protein's mean α-helix fraction significantly above the unintervened baseline, with a monotone dose-response.*

| Check | Verdict | Finding |
|---|---|---|
| A. GT provenance | PASS | Helix "ground truth" = ESMFold→DSSP (independent structure predictor + deterministic geometric SS assignment, codes H/G/I). **Not** derived from Evo2's own output. A predicted-structure proxy (disclosed), model-independent, identical across conditions. |
| B. Score normalization | PASS | `helix_frac = helix_res / total_res`; pLDDT weighting is a fixed policy applied identically. No division by the model's own max/mean. |
| C. Result-file existence | PASS | Cited numbers verified on disk. `results/m3/s28_caa.json`: coef 0.0 → mean helix_all ≈0.414; coef 1.0 → ≈0.482. `results/m4/specificity.json`: caa−baseline Δ=0.0684, Cliff's δ=0.135, Hedges g=0.237, MW p=5.9e-6; matched_control −0.024 (p=0.088); sham −0.005 (p=0.716). Match EXPERIMENT_RESULTS.md. |
| D. Dead code | PASS | ORF/validity/assay/steering-hook/NLL all invoked; `finalize_stats.py` recomputes reported stats from raw per-gen arrays. |
| E. Scope | PASS | Scoped to the pre-registered site-28 CAA family (18 dose points + 750/arm). Site-30 null and SAE-clamp negative honestly reported; C1 does not rest on them. |
| F. Evaluation type | PASS | `synthetic_proxy` — ESMFold-predicted structure + DSSP; inherent to in-silico steering, pre-registered, disclosed. |

**Integrity-focus items (per verify task):**
- **Survivorship-safe endpoint (invalid→0):** confirmed in `score_generations` — no-ORF/invalid generations receive `helix_frac=0` and the primary endpoint means over ALL generations on TEST. This couples C1 to C2 so steering cannot inflate helix by shedding invalid sequences.
- **DEV/TEST anti-circularity:** CAA direction built from the DEV-split contrast set (`contrast_set.json` high/low gids are all DEV); evaluation uses TEST-split primers with seed offset `900000+seed*1000`. No sequence used to build the intervention is used to test it.
- **Specificity controls:** matched-control (orthogonal same-norm) and same-norm sham both show no gain; CAA exceeds both (p=9e-10, p=1e-6). Gain is direction-specific.
- **Length/GC artifact:** checked in-experiment — within a matched [40,90 aa] band CAA still gives +0.089 helix; within-condition corr(length,helix)≈0.06–0.10; helix-favoring aa fraction unchanged/slightly lower. The helix gain is **not** an artifact of the length shortening. This is a methodology strength.

**Combined with mechanism audit → PASS.** C1 admitted to Stage 2.
