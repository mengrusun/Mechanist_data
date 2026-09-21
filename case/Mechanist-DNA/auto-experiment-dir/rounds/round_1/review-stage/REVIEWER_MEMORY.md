# Reviewer Memory

## Iteration 1 — Score: 5.5/10, Verdict: not ready

- **New suspicions**:
  - **C3 specificity is the main blocker**, not C2 efficacy. The decisive double-dissociation stat was taken at α=16, a capability-degraded dose (valid-ORF 0.888→0.667). Invalid as a clean specificity operating point.
  - The random-direction null must be *genuine*: ≥30 independent |S|-feature directions, sampled excluding S∪β∪matched, built with the same coefficient convention as S.base_dir then rescaled to match ||S.base_dir|| at the injection site. Per-direction sampling must be large enough that the null is not Monte-Carlo noise. Capability (valid-ORF) of random directions must be reported and compared to S — helix differences could otherwise be selection artifacts.
  - **S vs random-direction null should be the PRIMARY specificity statistic** (empirical one-sided p = (1+#{Δ_rand≥Δ_S})/(N+1)); the single matched_control is only a secondary sanity control ("why should I care you beat one handpicked direction?").
  - **β-sheet arm is a NEGATIVE result** (Δsheet ≈ +0.005), not half a double dissociation. Track whether authors report it honestly or keep overstating "double dissociation" when only one side moved.
  - **C2 is solid but limited to the SS-assignment robustness axis.** Structure-predictor swap (ESMFold→OmegaFold) still untested — keep this caveat live; helix enrichment could partly exploit ESMFold inductive biases.
  - C2 α*=32 is a grid edge (still climbing); ORF validity dips at α=16 then recovers at α=32 — non-monotone degradation needs explicit discussion, not hiding.
  - **C1 likely a real set-level signal but reported with integrity/narrative problems**: "established" overclaims vs plan's "conditional" bar; degenerate seed_jaccard=0.0 not transparently contextualized; eukaryote robustness leg missing. A ⓪ honesty-relabel (established→conditional, disclose the stat, note eukaryote pending) is warranted and sufficient for the INTEGRITY_ONLY record.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all of the above; C3 ② fix pending; C1 ⓪ relabel pending; C2 caveats must reach the paper.
- **Patterns**: Rhetoric runs ahead of the cleanest evidence — decisive stats taken at edge-of-grid / capability-degraded doses (C3 α=16, C2 α*=32); favorable statistics substituted for pre-registered ones (C1 seed_jaccard); "double dissociation" claimed when only one arm moved. Recurring theme: tighten claims to the capability-preserved, properly-nulled regime.

## Iteration 2 — Score: 6.8/10, Verdict: almost

- **New suspicions**:
  - C3 is repaired but only NARROWLY: N=33 null with empirical p=0.029 is the just-barely-significant extreme (0 exceedances → minimum nonzero p); acceptable, not overwhelming. Evidence is one decisive dose, one predictor stack, modest null count.
  - The locked α=8 is much better than α=16 but still sits at the EDGE of the capability-preserved region, not a broad interior plateau optimum — mildly "best surviving point" exposed.
  - Pipeline operational fragility: the mkdssp-PATH bug (which silently zeroed the DSSP readout for all 33 directions until caught) was disclosed honestly, which helps integrity but signals a brittle analysis stack.
- **Previous suspicions addressed?**:
  - C3 α=16 degraded-dose blocker → YES, genuinely fixed (re-locked to capability-preserved α=8; decisive stat moved off the degraded point).
  - Genuine random-direction null → YES (33 independent, norm-matched, excluding S∪β∪matched, capability-matched; S beats all 33, p=0.029, z=3.02). "The comparison I wanted."
  - β-arm honesty → YES (now a documented negative result; no symmetric "double dissociation" claim).
  - C2 caveats → YES, adequate (proxy readout, predictor-swap untested, α*=32 edge, ORF dip).
  - C1 label/seed_jaccard/eukaryote → YES (relabeled conditional/set-level, seed_jaccard disclosed, eukaryote corrected to complete & positive — materially improves the empirical picture).
- **Unresolved (carried forward as caveats, non-blocking)**: C3 narrowly-sufficient (N=33, single dose/predictor axis); C3 mechanism-audit residual WARN (no σ_proj units, α at capability-preserved edge); C1 + C3 swap-robustness deferred by MAX_VERIFY_CLAIMS cap; presentation risk — keep C2 bounded to the proxy/SS-assignment axis.
- **Patterns**: The recurring "rhetoric outruns cleanest evidence, then reeled back after audit" pattern persisted at entry, BUT this iteration the authors did the right thing — corrected the decisive STATISTIC (not just the phrasing), which improves confidence. Trust was taxed by the repeated need for post-hoc honesty corrections, but the substantive fixes were real.
