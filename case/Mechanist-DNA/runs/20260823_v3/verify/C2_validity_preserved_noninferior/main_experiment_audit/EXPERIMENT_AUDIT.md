# Experiment Audit — C2 (validity preserved / non-inferior) — main experiment

**Scope:** M1 validity harness + M4 non-inferiority / off-target / frontier. **Overall verdict: WARN** (check E scope-wording; non-inferiority test itself is clean).

C2: *at the C1-winning setting, generated DNA stays valid at a rate non-inferior to baseline, with off-target properties not degraded.*

| Check | Verdict | Finding |
|---|---|---|
| A. GT provenance | PASS | Validity = deterministic pre-registered composite (in-frame start `M`, no premature stop, length 30–300) + one disclosed Evo2-NLL≤1.5 naturalness gate. NLL gate is mildly self-referential but a fixed threshold applied identically, not the outcome metric. Off-target metrics (β-sheet, GC, length, aa-composition) are model-independent. |
| B. Score normalization | PASS | validity_rate = fraction valid over the **full** generated set; not normalized to any model max/mean. |
| C. Result-file existence | PASS | validity 0.968 vs 0.991 and NI LB −0.0347 > −0.05 verified in `winning.json` (coef 1.0 ni_pass=true; coef≥2 ni_pass=false). Off-target GC/length shifts consistent with `s28_caa.json`. |
| D. Dead code | PASS | validity + off-target scorers all invoked; `run_m4_stats.py`/`finalize_stats.py` recompute NI + deltas from raw. |
| **E. Scope** | **WARN** | Formal NI test is correctly scoped and passes; frontier honestly reports coef≥2 fails NI. But the claim wording **"off-target properties not degraded"** is optimistic against the documented, large, CAA-specific shifts at the winning setting: **protein length −36.6 aa (~−37%, δ=−0.45)** and **GC −0.139 (δ=−0.81)**. These are transparently reported and pre-registered as "reported, not disqualifying" — a wording/scope tension, not a fabricated result. |
| F. Evaluation type | PASS | Rule-based validity composite + synthetic-proxy off-target structure; disclosed. |

**Assessment.** The non-inferiority machinery is methodologically sound (pre-registered margin, one-sided 95% LB, full-set validity rate, honest frontier). The WARN is purely the claim-wording tension around the documented off-target length/GC shifts — the round's main interpretability caveat, and the exact axis the verify task flags for robustness. C2 is admitted (continue-with-warn) but deferred to **INTEGRITY_ONLY** this pass under `MAX_VERIFY_CLAIMS=1`; the WARN caveat travels with it.
