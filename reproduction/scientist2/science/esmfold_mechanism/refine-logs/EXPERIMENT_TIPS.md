# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **steering-block-selection** — plan scans block-window in 8 bands `[0-3], [4-7], [8-11], [12-15], [16-23], [24-31], [32-39], [40-47]` covering the full 48-block trunk (M1); M3b re-uses M3a-best block within the M1-localized early window. This is the "match-to-claim regional sweep" pattern the tip prescribes: the claim "decision localizes to early blocks" is directly testable because early bands act as the intervention while late bands act as matched null controls.
   - convention to adopt (no rebind needed): plan already conforms — 8 mutually-exclusive bands + matched-window null control. Just enforce the "never copy a raw index" rule: report the *observed* window (`[i*, j*]`) after M1 completes, not a pre-committed one.

2. **steering-coefficient-tuning** — M3b sets `α ∈ {-3, -1, +1, +3}` as raw coefficients on `s[block_k, res, :] += α · v_charge`. Also matches "coefficient copied without justification".
   - convention to adopt (rebind M3b):
     - (a) express α in units of `σ_proj = std(sᵀ v_charge)` computed on the 50-chain calibration split at block_k, so the sweep is physically comparable across runs and across the M3a-selected block; store the effective σ_proj and the multiplier;
     - (b) add α = 0 baseline (identity intervention) as an explicit sweep point;
     - (c) log a **structural-coherence "general-ability" metric alongside the target**: mean pLDDT over the target region, and target-region residue-neighbor RMSD vs no-intervention prediction — so we can distinguish "hairpin flipped because charge steering worked" from "structure collapsed off-distribution". Cross-strand Cα-Cα distance is already in the plan (auxiliary indicator) — keep it.
     - (d) matched-control random direction `v_random` (same L2 norm as `v_charge`) at the same α becomes MUST-RUN (was "optional" in the plan) — this is the tip's explicit collapse-vs-signal control.

## Composition (rebinds routed to Phase 2)

- **M1 rebinds**: none.
- **M3b rebinds** (kept internal to this stage — logged as method-sensitive re-binds in MECHANISM_ROUTING.md when steering site is committed; EXPERIMENT_PLAN.md is not edited):
  - α units: raw → `β · σ_proj` with β ∈ {-3, -1, 0, +1, +3} (5 points, was 4)
  - fluency metric added: mean_plddt_target + neighbor_rmsd_vs_noop
  - random-direction control: upgraded from optional to MUST-RUN

## No-match log

- ImageNet transform: N/A (protein sequences, no vision pipeline).
- Fine-tune sweep: N/A (no fine-tuning — we do frozen ESMFold + linear probe + activation patching).
- MCQ evaluation: N/A (target metric is DSSP boolean on predicted structure, not letter-parsed).
