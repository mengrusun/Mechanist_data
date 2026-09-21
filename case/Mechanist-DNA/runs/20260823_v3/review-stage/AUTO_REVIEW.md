# Auto Review — Steering Evo2-7B toward high α-helical content

Reviewer: external `llm-chat` HTTP fallback (model `gpt-5.4` via `LLM_BASE_URL`) — the MCP `llm-chat` tool
was not exposed in this host context, so the skill's documented curl fallback was used with the
environment-provided credentials (endpoint reachable, HTTP 200). This is a **genuine external review**, not
the `[pending external review]` self-review degradation the NOTICE warned might be necessary.

Config: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, POSITIVE_VERDICT_TERMS={ready, almost},
GPU_ID=auto, RESUME=false. Budget context: ~11.2 GPU-h remain for the iteration stage (whole-round HARD cap 40).

Verify-state buckets (from `verify/VERIFY_REPORT.md`):
- verify_passed: [C1]
- verify_failed: [none]
- verify_inconclusive: [none]
- verify_zero_eligible_variants: [none]
- verify_integrity_only: [C2] (stage2_skip_reason=max_verify_claims_cap; main-experiment integrity=WARN, warn_source=experiment)

All three *blocking* buckets (failed / inconclusive / zero-eligible) are empty at entry, so the
three-dimensional STOP rule can fire on iteration 1 if score ≥ 6 and verdict ∈ {ready, almost}.

---

## Iteration 1 (2026-08-24T09:10Z)

### Assessment (Summary)
- Score: 7/10
- Verdict: almost
- Budget after this iteration: iterations 0/6, claim-reentries 0/2 (only action was type ⓪ narrative-only — does not consume budget)
- Key criticisms:
  - C2 wording "off-target properties not degraded" overclaims vs the documented −37% protein-length / −0.139 GC shift — must be tightened for an honest paper (paper-side / type ⓪, not a back-edge).
  - C1 effect at the validity-preserving winning setting is real but modest (+0.068 abs, Hedges g=0.237, Cliff's δ=0.135); strongest gains occur at coefficients that fail validity NI.
  - Length shortening (100→64 aa) remains an off-target entangled with the mechanism; within-band analysis reduces but does not fully dispel it.
  - Proxy-heavy endpoint (DNA→ORF heuristic→ESMFold→DSSP→pLDDT): frame as computational structural steering, not demonstrated in-vitro helicity.
  - Generalization is within-family checkpoint swap only.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Overall score
**7/10**

Why not higher: the main effect is real, pre-registered, survivorship-safe, and fairly cleanly demonstrated for **one claim**. But the paper is still a bit narrow and somewhat fragile in practical significance: the winning effect is modest, off-targets are substantial, and C2’s narrative framing was initially too rosy. This is closer to a **solid mechanistic/intervention result** than a broadly transformative generative-biology paper.

## 1) Venue score
**7/10 for a top venue**

- **Strengths**
  - Strong pre-registration discipline.
  - Proper ALL-generations endpoint with invalids to zero.
  - Clear held-out TEST split and anti-circularity.
  - Nice specificity controls (orthogonal matched control, sham).
  - Convincing monotone dose-response for C1.
  - External verify-stage replication on related checkpoint materially helps credibility.

- **Weaknesses**
  - Biological endpoint depends on a long proxy chain: DNA → ORF heuristic → ESMFold → DSSP → pLDDT weighting.
  - Effect at the chosen valid setting is **modest**: +0.068 absolute helix fraction, g=0.237, Cliff’s delta=0.135.
  - Strong off-target shifts, especially **length** and **GC**, mean this is not “clean steering.”
  - C2 is statistically supported as stated, but only for validity NI; the broader “not degraded” framing is not tenable.
  - Generalization is only within-family checkpoint swap, not broader model classes or wet-lab-linked validation.

---

## 2) FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS claims
All three buckets are empty.

- **FAIL:** none
- **INCONCLUSIVE:** none
- **ZERO_ELIGIBLE_VARIANTS:** none

So there is **no minimum fix required** in categories (1)/(2)/(3).

---

## 3) PASS claim brief check

### C1 — PASS
**Narrative**
This is the strongest part of the work. The key pre-registered claim is not merely “one setting beats baseline,” but that **site-28 CAA induces a monotone positive dose-response in mean helix fraction on TEST over all generations**, and that is well supported. The result is not obviously explained by trivial artifacts: you checked length, within-band effects, amino-acid composition, and direction-specific controls. The model-swap verify result further reduces concern that this is a one-checkpoint fluke.

**Numeric consistency check**
- Baseline ladder anchor: **coef 0.0 = 0.414**
- Dose ladder: **0.5→0.475, 1.0→0.482, 2.0→0.524, 4.0→0.575, 8.0→0.563**
  - This is visually and numerically monotone up to 4.0 with mild rollover at 8.0, which is fully compatible with the reported Spearman trend test.
- Reported trend test: **rho=0.922, perm p=5e-5, q=1.5e-4**
  - Strong and consistent with the ladder.
- Winning setting comparison: **0.482 vs 0.414 = +0.068**
  - Matches the stated bootstrap CI **[+0.040, +0.098]**
  - Effect sizes **g=0.237**, **Cliff’s delta=0.135** are small-but-real; this is internally consistent with a +0.068 mean shift over n=750/arm.
- Specificity:
  - matched control **-0.024** n.s.
  - sham **-0.005** n.s.
  - CAA-control differences **+0.093** and **+0.073** with strong p-values
  - This is a major plus; it supports the **direction**, not just perturbation magnitude or site perturbation generally.
- Confounds:
  - Length shortens dramatically (**100→64 aa**), but matched-band effect **+0.089** remains.
  - Within-condition corr(length, helix) ~**0.06–0.10** is weak.
  - Helix-favoring AA fraction unchanged/slightly lower (**0.313 vs 0.326**), which argues against trivial sequence-composition explanation.

**Caveats to surface in paper, but no loop action needed**
1. **Practical magnitude at the valid setting is modest.** Do not oversell; the strongest gain is at doses that fail NI.
2. **Length is still a serious off-target.** The within-band analysis helps, but readers will still view shortening as entangled with the mechanism.
3. **The endpoint is proxy-heavy.** Emphasize this as computational structural steering, not demonstrated biophysical helicity in vitro.
4. **Site specificity remains partially unexplained.** Site-28 works, site-30 does not; SAE clamp negative. Interesting, but mechanistically incomplete.

---

## 4) INTEGRITY_ONLY claim

### C2 — INTEGRITY_ONLY
**Judgment**
The **statistical claim actually pre-registered for C2**—validity non-inferiority at the winning setting—is supported and looks methodologically fine.

- Baseline **0.991**
- CAA win **0.968**
- Delta **-0.023**
- One-sided 95% bootstrap LB **-0.035 > -0.05**
- Therefore **non-inferior by the stated margin**

So as a claim, this is acceptable.

**But the wording absolutely should be tightened.**
The phrase **“off-target properties not degraded”** is not defensible given:
- protein length: **-36.6 aa (~-37%)**
- GC: **-0.139**
- NLL: **+0.061** though still under cap

Those are real degradations/changes, even if validity remains non-inferior.

### Type-0 paper-side wording recommendation
Replace any language like:
- “validity preserved and off-target properties not degraded”

with something like:
- **“Validity is non-inferior to baseline at the winning setting, while notable off-target shifts remain, especially reduced protein length and GC content.”**

Or even tighter:
- **“The winning intervention preserves pre-registered validity within the non-inferiority margin, but does not preserve all other sequence properties; it substantially shifts GC and inferred protein length.”**

That wording should be considered mandatory for an honest paper.

### Deferred upgrade
Reasonable but non-blocking:
- **Deferred swap-test upgrade for C2** if resources permit later.

Given your stated rules, I am **not** proposing this as a back-edge action, just noting it would strengthen the paper.

---

## 5) Ready for submission?
**Almost.**

More explicitly:
- **By your operational criterion, this is READY** because there are no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE claims left unresolved.
- **My honest reviewer-style answer:** **Almost ready**, not a clean “yes,” because the paper text must be careful not to overclaim C2 or imply clean steering without tradeoffs.

So:
- **Internal pipeline readiness:** **Yes**
- **Scientific/paper readiness for top venue submission:** **Almost**

If you fix the wording around C2 and present off-targets plainly, then yes, I would call it **submission-ready**. Without that wording discipline, reviewers will ding it for overclaiming.

---

## 6) Memory update
- C1 appears genuinely supported: strong pre-registered monotone dose-response at site 28, winning-setting helix gain +0.068, specificity controls clean, verify-stage model-swap replication helpful.
- Main caveat for C1: effect at validity-preserving setting is statistically solid but modest; strongest gains occur where validity degrades.
- Length shortening is a persistent concern: matched-band analysis reduces but does not fully eliminate reviewer suspicion that steering exploits ORF/length regime shifts.
- Off-targets are substantial and CAA-specific, especially GC drop (-0.139) and protein length drop (~-37%); paper must not describe these as “not degraded.”
- C2’s actual NI claim is fine and should survive review if stated narrowly as validity non-inferiority only.
- Proxy-chain concern remains important for future iterations: ORF heuristic + ESMFold/DSSP endpoint may limit how strongly biological structure claims can be stated.
- Within-family checkpoint transfer is a good robustness point, but broader external validity is still limited.

</details>

### Verify-Passed Claims (brief audit)
- C1 — consistency check PASS. Claim wording matches the supporting numbers verbatim: dose ladder (0.414 / 0.475 / 0.482 / 0.524 / 0.575 / 0.563) is monotone through coef 4 and consistent with ρ=0.922, q=1.5e-4; winning +0.068 matches bootstrap CI [+0.040, +0.098]; specificity controls (matched −0.024 n.s., sham −0.005 n.s.; CAA−control +0.093 p=9e-10) and length-matched +0.089 all reconcile with `refine-logs/EXPERIMENT_RESULTS.md` and `verify/C1_helix_gain_doseresponse/ROBUSTNESS.md` (robustness 1.00, model-swap ρ=0.878 / +0.094). No numeric drift. **Paper-side caveats to carry (no loop action):** modest per-generation effect size (δ=0.135); length/GC off-target coupling is a property of the CAA lever (transfers across checkpoints); proxy-heavy endpoint — frame as computational structural steering.

### Actions Taken (per claim, per type)
- C1 — type ⓪ — no change needed; PASS held. Reviewer consistency check clean; caveats are paper-presentation notes only, no scripts or runs touched.
- C2 — type ⓪ — tightened the overclaimed off-target wording in the authoritative claim statement (paper-side; no scripts, no runs, no experiment). This resolves the substance of the verify Phase-2 WARN on the paper surface. Does NOT consume an iteration.
  - **Claim-statement Before/After** (`CLAIMS_LEDGER.md` C2 `**Statement**`):
    - Before: "... at a rate non-inferior to baseline, with **off-target properties not degraded**."
    - After: "... non-inferior to baseline (pre-registered margin −0.05). Validity is preserved within the non-inferiority margin, but **not all other sequence properties are: the intervention substantially shifts GC content (−0.139) and inferred protein length (−37%), documented CAA-specific off-target effects**." (annotated as an iteration-⓪ supersede)
  - No `/auto-verify` / `/auto-experiment` invoked (INTEGRITY_ONLY is a no-back-edge bucket). Deferred swap-test upgrade recorded in Open Items below.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- C2 [stage2_skip_reason: max_verify_claims_cap]:
  * `max_verify_claims_cap` → `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
  [main-experiment integrity: warn; warn_source: experiment (off-target "not degraded" wording vs documented −37% length / −0.139 GC shift — the non-inferiority test itself is clean)]
  * Note: the paper-side substance of the WARN was addressed this iteration via the C2 type-⓪ wording tightening; the swap-test itself (length-regressed endpoint / alternate pLDDT policy) remains the optional robustness upgrade if budget/venue warrant it (fits inside the ~11.2 GPU-h remaining).

### Results
- [run-experiment] iteration=1 runs_this_iteration=0 gpu_hours_this_iteration=0.0 cumulative_gpu_hours=0.0
- C1 → PASS held (robustness 1.00; no re-run needed).
- C2 → INTEGRITY_ONLY held; paper-side wording superseded via ⓪; swap-test deferred to Open Items.

### Status
- completed — three-dimensional STOP fired (score 7 ≥ 6, verdict `almost` ∈ {ready, almost}, and verify_failed / verify_inconclusive / verify_zero_eligible_variants all empty). Termination reason: positive_verdict.
