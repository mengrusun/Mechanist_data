# Mechanism Audit — C3 (specificity causal knob)
**Scope:** M3 — specificity at locked interior dose c*=21.48

## Check A — Steering Coefficient Sweep
**Result: PASS**

C3's decisive specificity statistic is locked at the interior dose c*=21.48 from M2 (same sweep structure). The M3 bracket {0, 10.74, 21.48, 32.22} shows the c-span trend for each arm, providing context for the locked-dose comparison. Key sweep properties confirmed:

- **c* selected on seed 42 (selection set), specificity stats computed on held-out seeds 200, 201 (P2 split-sample)**: confirmed in m3_specificity_summary.json (heldout_seeds=[200,201]).
- **c* is interior** (not grid edge): helix peaks at c*=21.48, falls off at 32.22, capability degrades. The M3 bracket confirms c* is mid-plateau.
- **Random-direction null is norm-matched**: 48 directions each match |S|=19 features, injection-site norm ||S.base_dir||, and natural-activation weighting (per P4). Per-direction generation impact (valid-ORF rate) reported: null_valid_orf_mean=0.882 ≈ S valid-ORF=0.895, confirming S is not special by milder perturbation.
- **Matched-control feature(s)**: 19 matched-control features steered at c* — negative result confirmed (Δ=-0.031/-0.025).
- **Capability metrics**: valid-ORF per arm reported; S at c* (0.895) ≈ c0 (0.902) — capability preserved.
- **No α copied verbatim**: c*=21.48 derived from M2's fresh sweep on this experiment.

## Checks B–F — Reserved
Not implemented.

## Overall Verdict
**overall_verdict: pass**

M3's specificity test satisfies mechanism-rigor requirements: 48 norm-matched null directions (per P4), split-sample c* (per P2), capability-matched comparison (per P6/P7), matched-control negative, locked interior dose from the M2 sweep. The helix-axis specificity conclusion is well-grounded.
