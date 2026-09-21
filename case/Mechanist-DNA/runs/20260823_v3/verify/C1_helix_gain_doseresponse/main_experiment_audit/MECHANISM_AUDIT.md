# Mechanism Audit — C1 (helix gain / dose-response) — main experiment

**Intervention:** additive contrastive activation addition (CAA) at residual block 28 during Evo2-7B decoding. **Overall verdict: PASS.**

## A. Steering-coefficient sweep — PASS (strong)

| Requirement | Status | Evidence |
|---|---|---|
| Coefficient unit stated | ✓ | Raw multiplier on the diff-of-means vector; `vnorm=823.91` recorded in `s28_caa.json`. |
| Capability/coherence metric co-logged | ✓ | validity_rate + mean Evo2 NLL (naturalness) logged at every sweep point (Pareto). |
| Coefficient locked at a principled point | ✓ | coef 1.0 locked by the pre-registered validity-frontier rule (max helix s.t. validity NI), `winning.json` / `validity_frontier.json`. |
| Random-direction / sham control | ✓✓ | matched-control (orthogonal same-norm) −0.024 n.s.; same-norm sham −0.005 n.s. Both present; CAA exceeds both (p=9e-10, p=1e-6). |
| Wide geometric range incl. α=0 | ✓ | [0, 0.5, 1, 2, 4, 8] = 16× nonzero range, 6 points, α=0 baseline. Run **met** its criteria, so the ≥30× wide-sweep requirement (applies only to shortfalls) does not bind. |

**B–F. Reserved** — `not_implemented` (direction-extraction quality, site/layer choice, n_effective, probe-vs-causal, intervention scope).

**Note.** The M2 localization (probe R²≈0, honestly reported) established only decodability and correctly *feeds* — does not substitute for — the causal steering. C1 rests on the causal dose-response, not the probe. Mechanism rigor is sound.
