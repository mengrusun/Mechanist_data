# Auto Review — Iteration Loop Audit Log

Project: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
Reviewer LLM: gpt-5.4 (via https://www.dmxapi.cn/v1)
Start: 2026-07-14

Budget context:
- MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6
- GPU budget remaining at start: ~0.79 GPU-h out of 10-h task cap (9.21 already consumed by main experiment + verify).

---

## Iteration 1 (2026-07-14)

### Assessment (Summary)
- Score: 4/10
- Verdict: almost (canonicalized from reviewer's "Almost")
- Budget after this iteration: iterations 0/6 (no back-edge action of type ①/②/③ dispatched — all reviewer recommendations are type ⓪ narrative-only), claim-reentries 0/2
- Key criticisms:
  - Paper's contribution set overclaimed relative to evidence.
  - C2 is a **failed hypothesis** (ordering reversed) — must be presented as such, not "mixed evidence."
  - C4 remains INCONCLUSIVE (M7 descoped) — the only real submission-readiness blocker; reviewer recommends narrative demotion to "untested/deferred" (partial GPU rescue would look like reviewer-bait).
  - CM must be **narratively bifurcated** into representation [strong] and causal [weak/null].
  - C1 needs refined phrasing: emotion-mean level within noise floor; individual-prefix level shows real directional shifts.
  - C3b support is real but **marginal at threshold** — must show exact CIs.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Overall score: **4/10**

Why so low:
- The paper's headline framing is interesting, but the evidence is mixed and several originally proposed claims do **not** survive contact with the results.
- **C2 is directly contradicted** by the main experiments.
- **C4 is untested/inconclusive** due to descoping of the main experiment.
- **CM is only half there**: location/probe evidence is strong, but the causal story is weak at the tested intervention scale and undercontrolled.
- The strongest positive is that the project seems to be converging toward a **more honest and publishable negative result**: emotional framing is mostly weak, unstable, and input-dependent, with some detectable representation-level encoding but little demonstrated causal leverage.

As a top-venue submission **in its current original claim structure**, this is not competitive.
As a **tightened, negative-results paper with mechanistic nuance and aggressive claim pruning**, it could become reasonable.

---

## Claim-by-claim minimum fix path

### C1 — INTEGRITY_ONLY
Current state: now materially improved by post-hoc sign-consistency computation.
- emotion means remain within noise floor
- sign-consistency band holds for 16/24 conditions, fails for 8/24
- out-of-band cells align with the same conditions that cross the noise floor

Assessment: This is **not a clean confirmation** of the original broad reading of C1. It is better framed as:
- at the **emotion-mean level**, effects are small / near noise floor;
- at the **individual-prefix level**, there are real input-dependent shifts, including several consistent wins/losses.

Minimum fix: **⓪ narrative correction only**
- Update text to say C1 is **partially supported**
- Explicitly include the 16/24 in-band vs 8/24 out-of-band result
- State that some individual prefixes induce reliable directional effects despite weak emotion-level averages

### C2 — INTEGRITY_ONLY
Current state: not supported, and there is a real eval-mode confound.
- Observed spread ordering is reversed: GSM8K 18.6 pp, SocialIQA 3.0 pp, MedQA 2.8 pp.
- But GSM8K uses CoT vs MCQ-LL companions — not apples-to-apples.

Assessment: This claim is currently **dead as stated**. The confound does not rescue it; it only prevents a strong comparative interpretation. The actual observed result goes in the opposite direction.

Minimum fix: **⓪ narrative demotion / explicit negative result**
- Rewrite C2 as a **failed hypothesis**
- State clearly that the observed ordering is reversed under current eval setups
- Add that cross-task comparison is further limited by eval-mode mismatch, so the result should not be overgeneralized

Do **not** spend GPU here. A same-mode rerun would be nice scientifically, but it is not needed to reach a submission-ready truthfully scoped paper, and may still not save the claim.

### C3a — INTEGRITY_ONLY
Current state: supported; integrity PASS already.
- GSM8K=fear, SocialIQA=surprise, MedQA=fear
- This does satisfy identity flip across tasks.

Assessment: This is okay, but narrow. It's not a deep result on its own because the flip is only between fear and surprise, and one argmax repeats.

Minimum fix: **⓪ narrative precision**
- Keep the claim
- Phrase conservatively: "the best-performing emotion is task-dependent and not invariant across domains"
- Do not oversell as broad task-specific emotional specialization

### C3b — INTEGRITY_ONLY
Current state: supported, but exactly at threshold.
- fear CI straddles 0
- anger CI straddles 0
- surprise reverses sign
- exactly **3/6**, so threshold is met but marginal

Minimum fix: **⓪ narrative surfacing**
- Include the actual CI bounds in the paper
- Explicitly call this **threshold-level support**
- Mention sadness is also marginally monotone (lower CI at +0.001), which underscores fragility

### C4 — INCONCLUSIVE
Current state: blocker. Main experiment never ran. This is the only true submission-readiness blocker under your rules.

Minimum fix to reach READY: **⓪ narrative demotion**
- Remove C4 from the core contributions/claims
- State explicitly:
  - adaptive policy hypothesis was **not tested**
  - M7 was descoped for budget reasons
  - therefore the paper makes **no empirical claim** about adaptive policies outperforming neutral or best-fixed prompts
  - this is deferred to future standalone verification

Important: do **not** present C4 as "inconclusive but suggestive." There is no result. It must be downgraded to **untested/deferred**. A partial 100-item heuristic demo would not really fix the integrity issue and risks introducing a weak, reviewer-bait appendix result that raises more questions than it answers.

### CM — PASS, but only as a partial mechanism claim
Current state: verify PASS overall, with "Location strong / Causal not-supported at tested scale."

Brief consistency check:
- **Location arm:** strong and internally consistent (probe=1.00 at layers 4-36 vs null 0.28, family verification on Qwen3-4B).
- **Causal arm:** weak / negative (steering ranges 2-4 pp, at or below the 5.24 pp noise floor).

So the data support:
1. emotion identity is encoded in intermediate representations, yes
2. that encoding causally modulates downstream behavior in the predicted direction, not convincingly

Caveat: split CM in wording, even if not in numbering:
- CM-location: supported
- CM-causal: not supported at tested scale

Also note: specificity controls descoped, cross-emotion patching descoped, length-null in the variant was high (0.69).

Minimum action: **⓪ narrative caveat + numeric consistency check**

---

## Is this READY for submission?

## **Almost**

Not "Yes" yet, because under your own criterion:
- READY requires **no FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS blockers**
- **C4 is currently INCONCLUSIVE**

### Fastest path to READY within budget
Use **⓪ narrative demotion of C4**:
- remove it as a tested claim/contribution
- mark as deferred/untested

Once that is done, this becomes **READY-with-caveats** from an integrity perspective.

### But reviewer-honest framing
- emotional framing effects are **mostly weak and unstable**
- one comparative hypothesis **failed**
- task-wise best emotion **changes**
- intensity scaling is **non-monotonic for several emotions**
- representations carry emotion information, but **causal control is not demonstrated**

---

## Recommended minimum-cost fix path

1. Demote C4 to untested/deferred — ⓪
2. Rewrite C2 as a failed hypothesis — ⓪
3. Update C1 with the new sign-consistency results — ⓪
4. Surface C3b CIs in the main text/table — ⓪
5. Split CM narratively into representation vs causal intervention — ⓪

No further GPU spend is necessary to reach a truthful, internally consistent submission state.

---

## Memory update
- Main concern: the paper's contribution set is overclaimed relative to evidence; must be reframed as a negative/qualified result paper.
- C4 is the only readiness blocker; recommend pure narrative demotion rather than weak partial experiment.
- C2 should be treated as a failed hypothesis, not "mixed evidence."
- C1 is now more informative: weak mean effects but genuine per-prefix directional signals; avoid blanket "noise floor only" language.
- C3b support is real but marginal; present exact CIs and threshold-level nature.
- CM should be explicitly bifurcated into strong representation evidence and weak causal evidence; avoid unified causal-mechanism language.
- Watch for reviewer attack on evaluation mismatch (GSM8K generative CoT vs MCQ-LL companions); this must be foregrounded as a limitation, not buried.

</details>

### Verify-Passed Claims (brief audit)
- CM: Reviewer consistency check confirms — Location arm strong (probe=1.00 vs null=0.28); Causal arm weak/null (2-4 pp range at/below 5.24 pp noise floor). Reviewer recommends narrative bifurcation into CM-location [supported] and CM-causal [not-supported at tested scale]. Caveat: variant length-null 0.69 (with 7 conditions) exceeds ≤0.2 designed for 24-condition M5; probe-null gap 31 pp still positive.

### Actions Taken (per claim, per type)

- C1 — type ⓪ — sign-consistency band computed post-hoc via `scripts/c1_sign_consistency.py` on M2 per-item data (CPU-only, no GPU). Result: 16/24 in-band [0.4, 0.6]; 8/24 out-of-band (aligned with noise-floor crossers). Refined C1 reading: partial support at emotion-mean level; individual-prefix directional signals present. No back-edge action.
  - **Analysis script written** (`scripts/c1_sign_consistency.py`) — computes wins/losses/agrees over neutral per condition; writes `reports/C1_sign_consistency.json`.
  - **EXPERIMENT_RESULTS.md M2b section**: added sign-consistency table + revised C1 verdict.
  - **CLAIMS_LEDGER.md C1 section**: added iter-1 line surfacing the new sign-consistency finding.
- C2 — type ⓪ — narrative demotion to failed hypothesis. No back-edge action.
  - **CLAIMS_LEDGER.md C2 section**: iter-1 line — "C2 is a failed hypothesis" with eval-mode confound foregrounded.
- C3a — type ⓪ — narrative precision (conservative rephrasing to "task-dependent best emotion"). No back-edge action.
  - **CLAIMS_LEDGER.md C3a section**: iter-1 line — conservative rephrasing.
- C3b — type ⓪ — surface per-emotion CI bounds from existing `reports/M4_c3_analysis.json` in EXPERIMENT_RESULTS.md; label as threshold-level support. No back-edge action.
  - **EXPERIMENT_RESULTS.md M4 section**: added per-emotion CI table.
  - **CLAIMS_LEDGER.md C3b section**: iter-1 line — threshold-level support with CIs surfaced.
- C4 — type ⓪ — narrative demotion to "untested — deferred, not falsified." Reviewer explicitly recommended AGAINST partial 100-item heuristic demo (would look like reviewer-bait). No back-edge action.
  - **CLAIMS_LEDGER.md C4 section**: iter-1 line — removed from core contributions in paper narrative; INCONCLUSIVE remains in ledger for audit truth.
- CM — type ⓪ — narrative bifurcation into CM-location and CM-causal. No back-edge action.
  - **CLAIMS_LEDGER.md CM section**: iter-1 line — bifurcated narrative.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- C1 [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]:
  * `max_verify_claims_cap` → `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run)
  * WARN: sign-consistency not computed at experiment time — **RESOLVED in iter-1 post-hoc**; the WARN status persists in `INTEGRITY_AUDIT.md` but the underlying finding is now surfaced.
- C2 [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]:
  * `max_verify_claims_cap` → `/auto-verify C2 -- resume: true`
  * WARN: eval-mode confound (CoT vs MCQ-LL) — no cheap fix; paper foregrounds as limitation.
- C3a [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]:
  * `max_verify_claims_cap` → `/auto-verify C3a -- resume: true`
- C3b [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]:
  * `max_verify_claims_cap` → `/auto-verify C3b -- resume: true`
  * WARN: per-emotion CI not surfaced at experiment time — **RESOLVED in iter-1 by surfacing existing M4 analysis JSON**.

### Results
- [run-experiment] iteration=1 runs_this_iteration=0 gpu_hours_this_iteration=0.00 cumulative_gpu_hours=0.00
- C1 → CPU-only post-hoc: `reports/C1_sign_consistency.json` computed (16/24 in-band; 8/24 out-of-band).
- C3b → per-emotion CIs surfaced from `reports/M4_c3_analysis.json` (already computed but not shown in main text).
- All other claims → narrative-only (⓪) — no runs fired.

### Status
- continuing to iteration 2 (score 4/10 < TARGET_SCORE=6; C4 remains INCONCLUSIVE — need reviewer re-scoring with narrative changes in place)

---

## Iteration 2 (2026-07-14)

### Assessment (Summary)
- Score: 6/10 (met TARGET_SCORE=6)
- Verdict: almost (canonicalized from reviewer's "Almost")
- Budget after this iteration: iterations 0/6 (still no back-edge action of type ①/②/③ dispatched — reviewer confirms iter-1 ⓪ fixes materially close the gap; no new action requested), claim-reentries 0/2
- Key criticisms:
  - Front-matter leakage risk (title/abstract/intro must be visibly modest — narrative must match ledger).
  - Novelty/impact ceiling: "we tested a provocative framing and mostly found it does not robustly hold" — publishable only if analysis quality is unusually strong.
  - C4 remains INCONCLUSIVE-in-ledger; reviewer recommends option (C) — grade as a 5-claim paper.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Re-score

**Score: 6/10**
**Verdict: Almost**

This is a meaningful improvement over iteration 1. The main reason is not that the empirical story got stronger, but that the paper-story now more honestly matches the evidence.

What changed materially:
- C2 is no longer being laundered as "mixed evidence"; calling it a failed hypothesis is the right move.
- CM is now split into representation/location (strong) and causal intervention (weak/null). Major overclaim removed.
- C4 is no longer pretending to be a contribution without evidence.
- C1 is better calibrated: the new sign-consistency computation is genuinely useful.
- C3b is now appropriately presented as threshold-level with exact CIs.

Still not a strong top-venue paper on the merits alone. It is now a defensible qualified/negative-result paper.

### C4 handling recommendation

I choose (C): Treat C4 as not applicable to this submission, provided it is fully removed from title/abstract/introduction/contributions/conclusion as a claimed result.

That means I am grading a 5-claim submission (C1, C2, C3a, C3b, CM).

If you insist on the 6-claim framing, then STOP should indeed not fire. But my recommendation is that this is the wrong framing for submission.

### Is this READY for submission?

Almost. Not fully "Yes," but much closer.

Why not fully ready:
1. Empirically modest for top venue.
2. Central phenomenon mostly a negative/qualified result — acceptable only if framing is exceptionally disciplined.
3. Mechanism section only partly strong (encoding strong; causal weak).
4. C3b remains right at threshold.
5. C2 eval-mode confound still limits interpretability.

Why it is no longer a clear "No":
- Main overclaim problems were genuinely fixed.
- Paper now has intellectually honest shape.

### What would still make me nervous as a reviewer
- Novelty/impact ceiling.
- Potential mismatch between title/abstract and actual evidence.
- Mechanism language creep.
- Overinterpretation of C1 sign consistency.
- Top-venue standards: after pruning, is enough left? Maybe, but only if written as careful falsification/qualification study.

### Bottom line

Score: 6/10. Verdict: Almost. Submission readiness: Almost. C4 handling: remove from submission claim set entirely; keep only as deferred future work / audit note.

### Memory update

**Prior suspicions genuinely addressed**:
- Overclaimed contribution set → addressed
- C2 should be failed hypothesis → addressed
- CM bifurcation → addressed
- C1 sign-consistency → addressed
- C3b CI surfacing → addressed
- C4 narrative demotion → addressed at paper level

**Prior suspicions still unresolved**:
- C4 remains inconclusive in strict ledger terms
- C2 eval-mode confound (honestly surfaced now)
- CM causal specificity (acceptable because demoted)

**New suspicions**:
- Front-matter leakage: title/abstract may still oversell.
- Acceptance depends on framing as careful negative-result / falsification study.

</details>

### Verify-Passed Claims (brief audit)
- CM: Reviewer confirms — CM-location (strong: probe=1.00 vs null=0.28) and CM-causal (weak/null: 2-4 pp steering range at/below 5.24 pp noise floor). Iter-1 bifurcation is correct and now visible in ledger. Cleanest contribution of the paper.

### Actions Taken (per claim, per type)
- none — all iter-1 ⓪ narrative fixes remain applied; reviewer confirms they materially close the evidence-to-claim gap. No new back-edge action requested. This iteration's reviewer call was consumed to re-score with the iter-1 changes in place; no on-disk change was necessary.
- C4 handling: reviewer recommends option (C) — remove from submission claim set; grade as 5-claim paper. This is a paper-framing recommendation, not a ledger-state change. C4 remains INCONCLUSIVE in the strict verify-report sense; loop treats it as still blocking STOP.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- C1 [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]:
  * `max_verify_claims_cap` → `/auto-verify C1 -- resume: true` (WARN: sign-consistency — RESOLVED post-hoc in iter-1)
- C2 [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]:
  * `max_verify_claims_cap` → `/auto-verify C2 -- resume: true` (WARN: eval-mode confound — foregrounded in paper)
- C3a [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]:
  * `max_verify_claims_cap` → `/auto-verify C3a -- resume: true`
- C3b [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]:
  * `max_verify_claims_cap` → `/auto-verify C3b -- resume: true` (WARN: per-emotion CIs — RESOLVED via M4 surfacing in iter-1)

### Results
- [run-experiment] iteration=2 runs_this_iteration=0 gpu_hours_this_iteration=0.00 cumulative_gpu_hours=0.00
- Score jumped 4→6 with zero GPU spend, confirming iter-1 ⓪ narrative fixes materially closed the gap.
- Verdict moved from initial-review "almost" (skeptical) to iter-2 "almost" (positive-almost — the paper is now defensible).

### Status
- **Score = TARGET_SCORE = 6, verdict = "almost" (positive)** — dimensions 1 and 2 of STOP satisfied.
- **Dimension 3 (no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS)** — NOT satisfied: C4 remains INCONCLUSIVE. STOP does NOT fire per contract.
- Reviewer explicitly recommends option (C) — treat C4 as removed from submission claim set. But this is a paper-framing decision; the ledger still records C4=INCONCLUSIVE.
- **Both iter-1 and iter-2 were ⓪-only iterations** — `consecutive_noop_count` incremented in iter-1, RESET to 0 in iter-2 because both score (4→6) and verdict-strength changed. iter-2 is a ⓪-only "confirmation" iteration after a real ⓪ change → not a stall.
- **Next iteration decision**: no productive ①/②/③ action is available within the ~0.79 GPU-h budget. A full M7 for C4 requires ≥2.17 GPU-h. Reviewer explicitly rejected the partial-demo path. Continuing to iteration 3 with no new on-disk change would generate a ⓪-only iteration with unchanged score/verdict → `consecutive_noop_count` would hit 2 (the stall guard).
- **Iteration 3 anticipated as the stall-guard trigger**: I will run one more reviewer call to confirm nothing new can be productively addressed; if the score/verdict stay at 6/almost with no ①/②/③ possible, the loop terminates per the stall guard.

---

## Iteration 3 (2026-07-14) — Convergence check

### Assessment (Summary)
- Score: 6/10 (unchanged from iter-2)
- Verdict: almost (unchanged)
- Budget after this iteration: iterations 0/6, claim-reentries 0/2 (still no back-edge action dispatched)
- Key criticism: none new — reviewer explicitly recommends termination.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. **Score / verdict**

Unchanged: **6/10, Almost**.

Nothing material has changed since iter-2, and my assessment does not move. The paper is now a reasonably honest, qualified submission, but in the strict claim ledger it still carries **C4 = INCONCLUSIVE**, which prevents a cleaner "ready" state.

2. **Any productive back-edge action within ~0.79 GPU-h?**

**No productive action remains within budget that would plausibly either**
- **unblock C4**, or
- **raise the score from 6 to 7+.**

### Why not C4
- ⓪ narrative demotion — already done
- ② full M7 execution — requires ~2.17 GPU-h, over budget
So C4 cannot be resolved within the remaining budget.

### What about optional CM strengthening?
The only feasible extra work is the CM causal-specificity controls on a 50-item subset (~0.3–0.5 GPU-h). Honest view: incrementally nice, would not unblock C4, unlikely to move score from 6 to 7+, risks becoming reviewer-bait polish. Score ceiling is dominated by novelty/impact, residual front-matter oversell risk, and unresolved C4.

Would NOT spend budget on CM filler controls now.

### Bottom line
**No productive action remains within budget. Stop.**

3. **READY / NOT READY for final report**

**NOT READY in strict stop-rule terms**, because C4 remains INCONCLUSIVE.

Substantively:
- Paper-level narrative: defensible "almost-ready" qualified submission
- Auto-verify / strict-ledger outcome: not ready, due to unresolved C4

4. Memory update (see REVIEWER_MEMORY.md § Iteration 3).

</details>

### Verify-Passed Claims (brief audit)
- CM: unchanged since iter-2. Reviewer confirms narrative bifurcation stands. Optional specificity controls (0.3-0.5 GPU-h) rejected as reviewer-bait polish that would not change score.

### Actions Taken (per claim, per type)
- none — reviewer explicitly recommends "stop" (no productive back-edge action within budget).
- C4: INCONCLUSIVE, budget-blocked (② requires 2.17 GPU-h, only ~0.79 available). Loop cannot resolve this; C4 surfaces in Open Items of final report.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- C1, C2, C3a, C3b — same as iter-2; all `max_verify_claims_cap` deferrals; upgrade instruction `/auto-verify <id> -- resume: true` for each.

### Results
- [run-experiment] iteration=3 runs_this_iteration=0 gpu_hours_this_iteration=0.00 cumulative_gpu_hours=0.00
- Reviewer's convergence check confirms stable equilibrium: score 6/10, verdict "almost", C4 budget-blocked.

### Status
- **TERMINATE.** Reviewer explicitly recommends stop. Score at TARGET_SCORE=6, verdict "almost" positive — dimensions 1 and 2 of STOP satisfied. Dimension 3 blocked by C4 INCONCLUSIVE (budget-blocked).
- Termination reason: **iterations_exhausted** (interpreted as "no more productive iteration possible within budget" — stall guard-adjacent, since the reviewer's own recommendation is "stop" after a stable no-op iteration).
- iterations_consumed = 0 (all fixes were type ⓪ narrative-only; per skill contract, ⓪ does not consume budget).
- claim_reentries_consumed = 0.
