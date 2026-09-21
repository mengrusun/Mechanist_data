# Reviewer Memory

## Iteration 1 — Score: 8/10, Verdict: almost

- **New suspicions**:
  - C2 scope may be too easily overgeneralized by readers even though the wording says "for pythia models". Fisher signal stability (jackknife ρ) does NOT imply cross-family transferability — an OLMo-1B result where the same Fisher-ranked heads act as suppressors (inverted causal direction) is a real signal about mechanism generality.
  - C3 "formation window" language is vulnerable to overclaiming as precise-timing given only 24/154 checkpoints — the observed windows are interval-censored on a coarse log-spaced grid, not high-resolution onset estimates.
  - C1 "scale-dependent emergence" from only 3 model sizes should not be presented as a scaling-law characterization; the personal-belief non-monotonicity (0.85 → 0.79 → 0.99) undermines any smooth-scale-law reading.
  - C4 needs explicit PPL delta reporting in the final narrative — "WK exactly preserved" alone is insufficient because the claim also promises no destructive perplexity cost. (Verified as already reported at EXPERIMENT_RESULTS.md lines 167, 180.)
  - "Belief heads" terminology may reify a Fisher-selected head set as a semantically-typed object; OLMo-1B result suggests operational localization (this procedure, this family) rather than universal semantic head identity.

- **Previous suspicions addressed?**: n/a (first iteration)

- **Unresolved (carried forward)**:
  - C2 FAIL verify-bookkeeping is out-of-scope for a Pythia-scoped claim but will require an explicit paper-side reframing (⓪) to prevent misinterpretation; verify state itself will remain FAIL because the OLMo-1B variant cannot be un-run and cross-family cannot become an in-scope Pythia neighbor without a download of pythia-160m or pythia-6.9b (BLOCKED by task.md hard constraint: only pythia-{410m,1b,2.8b} on disk).
  - C1/C3/C4 stress-tests deferred by `max_verify_claims_cap` remain unrun.
  - C3 environmental gap (24/154 pythia-1b checkpoints).

- **Patterns**:
  - Fisher signal stability != functional transferability — repeat this in every mechanistic-interpretability iteration.
  - Reproduction-fidelity strictness restricts our variant options: when the claim is family-scoped and the environment lacks in-family neighbors, verify's swap-test may produce a FAIL that is definitionally out-of-scope. The clean remediation is boundary-clarification (⓪), not method redesign (②/③).

## Iteration 2 — Score: 8.5/10, Verdict: almost

- **New suspicions**:
  - **Top-level summary consistency risk**: even with good caveats in detailed sections, some headline location (title/abstract/summary table) may still implicitly flatten "SUPPORTED-within-Pythia" into "SUPPORTED". Minimum fix is a one-line top-level bookkeeping clarification: "Automated verify marks C2 as FAIL because the cross-family model-swap robustness probe failed; this does not contradict the paper claim, which is intentionally restricted to the Pythia family."
  - **Terminology drift risk**: "belief heads" may still appear unqualified in prose/figures/captions. If so, readers may over-infer universality despite the caveat section. Grep-level consistency check needed.

- **Previous suspicions addressed?**:
  - C2 overgeneralization risk — RESOLVED via explicit Pythia scoping, cross-family failure disclosure, and operational terminology in FINAL_PROPOSAL.md §C2 + EXPERIMENT_RESULTS.md M2 + CLAIMS_LEDGER.md C2 row.
  - C1 scaling-law overclaim — RESOLVED (confined to "observed differential behavior across 3 Pythia scales", non-monotonicity explicit).
  - C3 precision overclaim on formation windows — RESOLVED via interval-censoring and 24/154 checkpoint caveat.
  - C4 missing PPL disclosure — RESOLVED (PPL 1.047× on 1b / 1.005× on 2.8b explicit in CLAIMS_LEDGER.md and FINAL_PROPOSAL.md).
  - "Belief heads" reification — PARTIALLY RESOLVED (explicit "operational designator, not universal semantic head identity" in FINAL_PROPOSAL.md §C2 and CLAIMS_LEDGER.md); still needs grep-consistency across other artifacts.

- **Unresolved (carried forward)**:
  - Top-level bookkeeping-vs-scope-support one-liner (asked as final polish this iteration).
  - Terminology consistency grep pass across all artifacts.
  - C2 verify-bookkeeping remains FAIL — reviewer explicitly says "no further experiments warranted under current constraints"; this is a definitional out-of-scope FAIL, not an indictment of the scoped claim.
  - C1/C3/C4 stress-tests deferred by `max_verify_claims_cap` remain unrun.

- **Patterns**:
  - Reviewer's iteration-2 concerns are now editorial (presentation consistency, top-level surface), not scientific — this confirms the science-level correctness is settled.
  - Reproduction context + hard-constraint blocker on in-family Pythia neighbors → the natural terminal state is score ≥8, verdict `almost`, C2 mechanically-FAIL-but-scoped, with a documentation-side reframing.

## Iteration 3 — Score: 8.8/10, Verdict: almost

- **New suspicions**: none substantive. Only minor: scoped-verdict taxonomy is slightly heterogeneous but preferable to false clarity.
- **Previous suspicions addressed?** All iteration-2 asks RESOLVED (top-level bookkeeping-vs-scope one-liner in the right place; terminology-consistency at non-blocking status).
- **Unresolved (carried forward)**: C2 verify-bookkeeping remains FAIL as a definitional out-of-scope-stress-test artifact (no admissible in-family re-entry experiment available without violating environment hard constraints); C1/C3/C4 stress-tests deferred by cap (recorded as Open Items with per-`stage2_skip_reason` upgrade suggestions).
- **Patterns**:
  - Reviewer explicitly declared "this is the natural terminal state" and "further looping is likely to be churn" — the science-level correctness is settled, and the mechanical bookkeeping mismatch (verify FAIL vs scoped-support prose) is an unresolvable process artifact under current constraints.
  - Under hard resource/scope constraints, the proper terminal state can be *scientifically stable, mechanically non-green* — this is one of those cases. The correct resolution is to record the terminal state, not to continue looping.

## Iteration 4 — Score: 8.8/10, Verdict: almost (consecutive_noop_count = 1)

- **New suspicions**: none.
- **Previous suspicions addressed?**: unchanged (all iteration-3 status preserved).
- **Unresolved (carried forward)**: unchanged.
- **Patterns**: reviewer stance stable across iterations 3–4; no on-disk changes; stall-guard counter incremented.

## Iteration 5 — Score: 8.8/10, Verdict: almost (consecutive_noop_count = 2 → stall guard fires → terminate)

- **New suspicions**: none.
- **Previous suspicions addressed?**: unchanged.
- **Unresolved (carried forward)**: unchanged.
- **Patterns**: stall-guard mechanical trigger fires. Loop terminates cleanly. Termination reason: `stalled` (two consecutive no-op iterations with unchanged score/verdict — corresponds to reviewer's iteration-3 explicit "natural terminal state" declaration).

