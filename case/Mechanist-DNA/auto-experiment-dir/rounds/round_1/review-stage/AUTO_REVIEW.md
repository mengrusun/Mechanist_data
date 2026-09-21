# Auto Iteration Review Log — Feature Steering an α-Helix Knob in Evo2-7B

Reviewer: external LLM (gpt-5.4) via llm-chat MCP (source: shell env). Target score 6, positive verdicts {ready, almost}, MAX_ITERATIONS 6, MAX_CLAIM_REENTRIES 2. GPU pin {2,3,4,5}; witnessed {3,4,5} (GPU 2 held by weiyunx).

---

## Iteration 1 (2026-07-18)

### Assessment (Summary)
- Score: 5.5/10
- Verdict: not ready
- Budget after this iteration: iterations 1/6, claim-reentries 0/2
- Key criticisms: C3 specificity not cleanly established — decisive double-dissociation stat taken at α=16, a capability-degraded dose (valid-ORF 0.888→0.667); no genuine random-direction control (only one fixed matched direction); β-arm never rose yet "double dissociation" claimed. C1 "established" overclaims vs the plan's own "conditional" bar + undisclosed degenerate seed_jaccard=0.0. C2 solid but caveats (proxy readout, untested predictor-swap axis, α*=32 grid edge, ORF dip) must surface.

### Reviewer Raw Response
<details><summary>Iteration 1 — full reviewer response (score 5.5/10, not ready)</summary>

Score 5.5/10 (borderline, not top-venue ready). C2 genuinely compelling; C1 plausible but overclaimed; C3 is the main blocker (specificity not cleanly established — decisive stat at capability-degraded α=16). Routing: C3 = type ② (main-experiment mechanism-harness fix — correct; not a claim rewrite). Re-locking to α=8 + a real random-direction null is the right minimum repair. Amendments required: (A) make S-vs-random-direction-null the PRIMARY specificity test (single matched_control → secondary sanity), empirical one-sided p = (1+#{Δ_rand≥Δ_S})/(N+1); (B) define the null carefully — sample |S| features excluding S∪β∪matched, same coefficient convention as S.base_dir then rescale to match ‖S.base_dir‖; (C) per-direction sample floor so the null isn't Monte-Carlo noise; (D) report capability (valid-ORF) of random dirs vs S to rule out a selection artifact — include all directions; (E) report the β-arm honestly as negative, not half a double dissociation. C2 PASS confirmed ("strongest part of the paper") — but the paper must frame it as a causal effect on a predicted structural phenotype under one structure-prediction pipeline, with caveats (proxy readout; predictor-swap untested; α*=32 grid edge; ORF dip at α=16). C1 ⓪ relabel "strongly warranted": soften established→conditional, disclose the degenerate seed_jaccard=0.0, note eukaryote leg. READY: No — C3 (INCONCLUSIVE) unresolved.

</details>

### Verify-Passed Claims (brief audit)
- C2: consistent — claim wording matches m2_dose_response_curve.json (ρ=0.759 p=1.7e-5, helix 0.475→0.834, +0.36) and the pydssp variant (ρ=0.886, +0.327 vs +0.332). Reviewer flagged paper-side caveats to carry (proxy readout; structure-predictor swap axis untested; α*=32 grid edge; non-monotone ORF dip at α=16). Carried as ⓪ (see Actions).

### Actions Taken (per claim, per type)
- **C3 — type ② — main-experiment mechanism-harness fix.** Re-locked the double-dissociation decisive dose from α=16 (capability-crashed) to the mid-plateau α=8 (capability preserved); added a genuine 33-direction random-magnitude-matched control as the PRIMARY specificity statistic; β-arm re-stated honestly as a NEGATIVE result. Re-audited via /mechanism-audit → FAIL→WARN → C3 INCONCLUSIVE→INTEGRITY_ONLY.
  - **New code** (`code/m3_random_control.py`): 33 independent random directions, each 19 SAE features drawn uniformly excluding S∪β∪matched, weighted by natural activation magnitude, rescaled to match ‖S.base_dir‖; identical DNA→ESMFold→DSSP pipeline; two-phase (all Evo2 gen, then all ESMFold) to avoid runtime alternation.
  - **New consolidation** (`code/m3_consolidate_v2.py`): locks decisive dose at α=8, computes S-vs-random-null (empirical p + z), keeps matched-control secondary, reports β-arm as negative → `results/m3_specificity_summary_v2.json`.
  - **Bug caught + fixed mid-run**: the detached launcher lacked `mkdssp` on PATH (base-conda PATH, not the scientist env), silently zeroing DSSP readout for all directions (ESMFold folded fine; `n_folded_gated=0` because DSSP raised FileNotFoundError). Root-caused via a reproducer (esmfold plddt=75.2; only error = `FileNotFoundError: mkdssp`), fixed by `export PATH=<scientist>/bin:$PATH` in `code/launch_random_control.sh`, verified via 1-direction pilot (n_folded=10/12, helix 0.493 ≈ baseline) + `/proc/<pid>/environ` check on live workers, then the full batch re-ran clean.
  - **Plan Before/After** (`refine-logs/EXPERIMENT_PLAN.md` M3): added the random-direction null as PRIMARY control + the locked-α=8 decisive-dose specification + honest β-arm framing (resource_fidelity marker `not-strict` preserved).
  - Re-invoked: `/mechanism-audit — claim C3` on the fixed evidence.
- **C1 — type ⓪ — narrative honesty relabel (no scripts, no runs).** Softened `main_experiment.verdict` "established (set-level)" → "conditional (set-level)" per the plan's own vocabulary (basis: set-vs-single-feature + label honesty, NOT organism restriction); disclosed the degenerate seed_jaccard=0.0 (relaxed-set cross-seed Jaccard ~0.7-0.8 is the substantive stability). Applied to `claims_ledger.json` C1.
  - Mid-iteration correction (team-lead flag): the eukaryote leg is COMPLETE & POSITIVE, not pending — all 6 H. sapiens configs established (350,268 codons, set AUROC 0.858). Corrected the C1 record; cross-organism generality confirmed.
- **C2 — type ⓪ — paper caveats carried (PASS held, no back-edge).** Recorded proxy-readout, untested-predictor-swap, α*=32-grid-edge, ORF-dip caveats in `claims_ledger.json` C2.

### Claim Rewrites (type ③)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- C1 [stage2_skip_reason: max_verify_claims_cap]: upgrade via `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused; only Stages 2–3 run). Main-experiment integrity: warn (warn_source: experiment — label overclaim, now relabeled ⓪).

### Results
- [run-experiment] iteration=1 runs_this_iteration=3 (33 random directions across 3 GPU processes) gpu_hours_this_iteration≈6.14 (fixed batch) + ≈5.4 wasted (mkdssp-PATH-bug run, re-run clean) cumulative_gpu_hours≈11.7
- C3 → /mechanism-audit re-run → mechanism FAIL→WARN → combined integrity WARN → **INCONCLUSIVE → INTEGRITY_ONLY**. Fixed decisive result (α=8): S helix Δ=+0.096 beats all 33 random directions (0/33≥S; empirical p=0.029, z=3.02), capability matched (S valid-ORF 0.888 ≈ random 0.887); matched-control Mann-Whitney p=3.5e-4; β-arm negative (Δsheet +0.005).
- C1 → ⓪ relabel applied (conditional set-level; seed_jaccard disclosed; eukaryote corrected to complete).
- C2 → ⓪ caveats carried; PASS held.

### Status
- continuing to iteration 2 (re-review the fixed work)

---

## Iteration 2 (2026-07-18)

### Assessment (Summary)
- Score: 6.8/10
- Verdict: almost
- Budget after this iteration: iterations 1/6 (this iteration was a pure re-review — no back-edge — so no budget consumed), claim-reentries 0/2
- Key criticisms (residual, non-blocking): C3 repaired but narrowly (N=33 null, p=0.029 is the just-significant extreme; single decisive dose + one predictor stack); α=8 at the capability-preserved edge; pipeline operational fragility (mkdssp-PATH bug, disclosed). C1/C3 swap-robustness deferred by the verify cap; keep C2 bounded to the proxy/SS-assignment axis.

### Reviewer Raw Response
<details><summary>Iteration 2 — full reviewer response (score 6.8/10, almost)</summary>

Score 6.8/10, verdict "Almost ready." C3: mostly YES — a real fix, not a rhetorical sidestep. The re-lock to α=8 (capability-preserved), 33 independent random directions (excluding S∪β∪matched, norm-matched, same pipeline, capability compared) and the primary stat Δ_S=+0.096 vs random mean −0.014/max +0.061, empirical p=0.029 "is the comparison I wanted." Residual: N=33 is acceptable but small-ish (p=0.029 is the zero-exceedance extreme); α=8 still somewhat convenient (edge of capability-preserved region); one readout axis / predictor stack; the mid-run DSSP/PATH bug signals a brittle stack (credit for disclosure). Bottom line: genuine repair of the FAIL-level problem — can claim helix-axis specificity at a capability-preserved dose vs a genuine random null; should NOT oversell as broad mechanistic proof. β-arm: YES, now honest (negative result; no double-dissociation claim) — "genuinely addressed." C2 caveats: adequate; keep bounded to the SS-assignment axis. C1: relabel + eukaryote correction acceptable (eukaryote correction materially improves the empirical picture), with mild trust cost from the earlier overstatement. READY: Almost — by the bookkeeping the blocking claim-state issue is resolved and no fatal flaw remains, but the strongest story is good-but-fragile and depends on disciplined framing. No back-edge actions requested for the INTEGRITY_ONLY claims. Pattern: initial rhetoric tends to outrun the cleanest evidence then gets reeled back after audit — but this iteration the authors corrected the STATISTIC, not just the phrasing, which improves confidence.

</details>

### Verify-Passed Claims (brief audit)
- C2: held PASS (reviewer: "strongest part of the paper"); caveats already carried in iteration 1. No change.

### Actions Taken (per claim, per type)
- none — pure re-review. The three-dimensional STOP rule fired (score 6.8 ≥ 6; verdict "almost" ∈ {ready, almost}; no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS remain — C2 PASS, C1 & C3 INTEGRITY_ONLY). Loop terminates on positive verdict.

### Claim Rewrites (type ③)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- C1 [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true`. Main-experiment integrity: warn (label, relabeled ⓪).
- C3 [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C3 — resume: true`. Main-experiment integrity: warn (warn_source: experiment+mechanism — mechanism-audit residual WARN: α not in σ_proj units, α=8 at capability-preserved edge). Swap-robustness never evaluated for C3.

### Results
- no runs this iteration (pure review).
- Final per-claim state: C2 PASS (held); C1 INTEGRITY_ONLY (⓪ relabeled, eukaryote complete); C3 INTEGRITY_ONLY (repaired from INCONCLUSIVE via ②).

### Status
- completed — positive verdict (score 6.8, almost); termination_reason = positive_verdict.
