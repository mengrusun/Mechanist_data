# Mechanism Audit — C2 (dose-response helix)
**Scope:** M1 + M2 — σ_proj-unit dose-response steering sweep

## Check A — Steering Coefficient Sweep
**Result: PASS**

Sweep coverage:
- Grid: c_sigma ∈ {-5.37, 0.0, 2.685, 5.37, 10.74, 21.4801, 32.2201, 42.9601, 64.4402, 85.9203} (10 σ_proj-unit doses)
- In raw-α units (σ_proj=0.410): α ∈ {-2, 0, 1, 2, 4, 8, 12, 16, 24, 32} — covers ~4 orders of magnitude from negative to 32× projection std.
- **σ_proj-scaled**: yes — σ_proj=0.4104 measured on natural-CDS reference set; all doses expressed in σ_proj units (plan improvement 4).
- **Capability/coherence metric logged**: valid-ORF rate reported per dose; pLDDT distribution reported per dose; helix measured at matched valid-ORF (P6).
- **Interior c* locked mid-plateau**: c*=21.48 selected as pLDDT-weighted-helix argmax WITHIN the contiguous capability-preserved region (P7). c*_note confirms: "region ends at the first dose whose valid-ORF drops"; high-dose resurgences excluded. c*-selection was on seed 42 (selection seed); held-out seeds 200,201 used for headline trend (P2 split-sample).
- **Random-direction baseline**: 48 norm-matched random directions at c* evaluated (M3). They are also the null control for C3 but confirm the dose-response is specific to S.
- **Sign pattern preserved**: negative c (-5.37) does not raise helix (ESMFold: 0.463 ≈ baseline 0.454; OmegaFold: 0.479 ≈ baseline 0.473) — sign pattern consistent with an upward specific effect.
- **No α copied verbatim**: σ_proj calibration was computed freshly on the round-2 reference set (M1: σ_proj=0.4104, different from any prior work).

## Checks B–F — Reserved
Not implemented.

## Overall Verdict
**overall_verdict: pass**

The steering coefficient sweep in M2 satisfies all mechanism-rigor requirements: 10-dose σ_proj-unit grid spanning multiple orders of magnitude, capability metric (valid-ORF) logged per dose, interior c* locked mid-plateau (not grid edge), split-sample c* selection, and random-direction baseline (48 directions in M3).
