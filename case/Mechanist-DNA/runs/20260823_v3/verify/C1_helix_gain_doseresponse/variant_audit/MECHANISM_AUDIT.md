# Variant Mechanism Audit — C1 (model-swap-evo2-7b-262k)

**Intervention:** additive CAA at block 28 on `evo2_7b_262k`, direction rebuilt in-model. **Overall verdict: PASS.**

## A. Steering-coefficient sweep — PASS

| Requirement | Status | Evidence |
|---|---|---|
| Unit stated | ✓ | raw multiplier on in-model diff-of-means; `vnorm=947.21` (vs base 823.91 — analogous scale). |
| Capability metric co-logged | ✓ | validity + NLL at every dose cell (validity 0.997/0.987/0.964/0.894 across coef 0/1/2/4). |
| α=0 baseline included | ✓ | in-model coef-0 baseline (helix 0.426). |
| Range | ✓ (adequate) | [0,1,2,4], 4× nonzero, 4 points. Sufficient for a within-family transfer probe of an already-established sweep; the base experiment established the full [0..8] shape. |

**B–F. Reserved** — `not_implemented`.

**Note.** No in-variant random-direction/sham control — by design. Those established *specificity* in the main experiment; this variant tests whether the *same direction transfers* to a different family checkpoint (transferability), not specificity from scratch. Not a rigor gap.
