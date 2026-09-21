# Reviewer Memory

## Iteration 1 — Score: 4/10, Verdict: almost

- **New suspicions**:
  - **C1 residue framing risk**: paper must not oversell "residue-free" subliminal transfer; strict `banana_residue_count==0` was missed (2/154 = 1.3%). Post-hoc `inconclusive→conditional` override is defensible only because verify variant corroborates.
  - **C3 amplification wrong-sign is either a bug or a real distributed-subspace effect**: reviewer explicitly said "Do NOT spin this as subtle sophistication unless you have ruled out an implementation sign bug." Track this until a code-level audit conclusively rules out a sign flip.
  - **C3 specificity failure**: random_ablate (−4pp) > target ablate (−3pp) is a **major** specificity failure, not a footnote. Combined with the matched_control_ablate being comparable, this is genuine evidence against a clean single-direction causal handle at block 47.
  - **LR sweep unfinished**: `lr=1e-5` and `lr=5e-6` never evaluated (best_LR committed at 5/12 based on partial evidence). Reviewers will correctly note that "best LR" was chosen before the sweep completed.
  - **Budget overrun (~15.5 vs 10.0 GPU-h, +55%)**: signals weak execution discipline. Secondary for science quality; primary for process trust.
  - **Distinction between INTERNAL READY and TOP-VENUE READY**: reviewer explicitly separated these. Internal contract is satisfied (no FAIL/INCONCLUSIVE/ZEV); top-venue mechanism paper is not.
- **Previous suspicions addressed?**: n/a (first iteration).
- **Unresolved (carried forward into iteration 2)**:
  - C2 and C3 are INTEGRITY_ONLY — swap-test skipped per MAX_VERIFY_CLAIMS=1 cap. No back-edge action available in this iteration under contract. Upgrade requires a fresh `/auto-verify <id> -- resume: true` outside this loop.
  - Multi-seed C3 replication is out-of-budget (~1.4 GPU-h vs 0.8 cap).
  - Full LR sweep completion is out-of-budget (~2.4 GPU-h).
  - The whole "mechanism paper is not top-venue ready" concern is a scientific reality, not something more iterations can fix within budget.
- **Patterns**:
  - **Framing risk cluster**: three of the reviewer's flags (C1 residue, C3 amp sign, C3 as "partial") are about how results are *presented* — the actual numbers are honest but the framing/labels overstate what's supported. All three admit ⓪ narrative fixes without new experiments.
  - **Budget-blocked cluster**: the two most-substantive remaining fixes (multi-seed C3, full LR sweep) both exceed the 0.8 GPU-h iteration cap — one by 1.75×, the other by 3×. The iteration loop cannot substantively strengthen the mechanism story from here.

**Iteration-1 code audit finding (⓪, no back-edge)**: Inspected `src/mechanism/mechanism_intervene.py` L127-167 and `src/mechanism/mechanism_location.py` L228-230. Intervention hook is correctly operationalized (amplify=+kσv̂, ablate=−projection). Direction extraction saves the raw top-1 left singular vector with **no sign-orientation step** — the "amp decreases P(banana)" observation is therefore not diagnostic between (a) SVD sign ambiguity (arbitrary sign meaning `+v` is actually `−v_teacher`) and (b) a real distributed-subspace effect. Ablation/random/matched-control results are sign-independent and stand. Documented in CLAIMS_LEDGER Open Items + C3 Caveats.

## Iteration 2 — Score: 5/10, Verdict: almost

- **New suspicions**:
  - **Affordable disambiguation was deferred despite being directly relevant** (execution-choice bias signal): the C3 sign-orientation rerun costs ~0.1 GPU-h — well within the 0.8h iteration cap — yet was deferred. Not fatal, but suggests selective tolerance for ambiguity when the result might remain unfavorable. Track this pattern.
  - **Risk of laundering weak C3 through "distributed subspace" rhetoric**: the ledger now says the result is "consistent with the distributed LoRA-artifact account." Consistency is cheap; the data do NOT positively establish a distributed-subspace mechanism — they mostly refute a clean single-direction account. Watch for wording drift from *negative evidence against simple steering* into *positive evidence for a richer mechanism*.
- **Previous suspicions addressed?**:
  - **C1 residue framing (iteration-1 concern)** → **addressed** — new paper-framing caveat explicitly forbids "residue-free" language and mandates "judge-stochasticity-level residue" + Methods disclosure of override.
  - **C3 amplification wrong-sign as possible bug (iteration-1 concern)** → **partially addressed** — the code audit correctly shows the hook is not sign-flipped and correctly identifies the SVD sign-ambiguity in extraction. The specific "is it a bug?" question is answered (no bug in hook; unresolved orientation in extraction). But the practical amplification-result interpretability remains open because the ~0.1 GPU-h confirmatory rerun was deferred.
  - **C3 specificity failure (random ≥ target)** → **still unresolved, and unaddressable within budget** — this is a scientific reality, not a fixable methodology issue. Reframed honestly (no longer "partial support" but "negative result on single-direction handle") but the underlying evidence is unchanged.
  - **LR sweep unfinished** → **still unresolved** — lr=1e-5, 5e-6 remain unevaluated (2.4h out of budget). Documented as methodological caveat only.
  - **Budget overrun** → **still unresolved** — documented in Open Items; unfixable retroactively.
- **Unresolved (carried forward, if there were an iteration 3)**:
  - Deferred C3 sign-orientation rerun (~0.1 GPU-h) — contract-blocked (C3 is INTEGRITY_ONLY; no back-edge action per verify contract).
  - The mechanism story cannot be strengthened within iteration-loop scope; only a standalone `/auto-verify C3 -- resume: true` or a fresh multi-seed protocol outside the loop would move it.
- **Patterns**:
  - **Narrative repair helped substantially (+1 score point), but evidence did not improve** — this iteration is almost entirely a story-discipline upgrade. Valuable, but not scientific strengthening.
  - **C3 remains the central blocker for ambitious claims** — the code audit removed one possible excuse (implementation bug) but left the causal picture weak; this is the dominant reason the work is not top-tier mechanism-ready.
  - **Internal-ready vs venue-ready distinction persists** — internal contract is satisfied; top-venue standards are stricter.

**Termination signal**: Reviewer explicitly stated "none pending" for FAIL/INCONCLUSIVE/ZEV fixes at both iterations. The deferred C3 sign-orient rerun (the only affordable substantive fix) is contract-blocked because C3 is INTEGRITY_ONLY (no back-edge action permitted per verify_integrity_only contract; only standalone `/auto-verify C3 -- resume: true` outside the loop can move it). MAX_ITERATIONS=2 was set as a reviewer-cycle cap due to the HARD-budget crisis (experiment already 5.5h over); with both cycles complete and no back-edges dispatchable, the loop terminates as `iterations_exhausted` (by reviewer-cycle intent).
