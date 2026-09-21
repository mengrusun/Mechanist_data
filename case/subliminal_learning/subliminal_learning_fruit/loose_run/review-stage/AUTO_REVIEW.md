# Auto Review Log — Subliminal Learning in Diffusion Image Models (Qwen-Image)

**Reviewer model**: `gpt-5.4` via `<REDACTED_API_BASE_URL>` (source: shell env)
**Iteration budget**: 2 (back-edge cap) — reduced from default 6 due to HARD-budget crisis
**Claim-reentry sub-budget**: 1 — reduced from default 2
**Target score**: 6 (verdict ∈ {ready, almost}, no FAIL / INCONCLUSIVE / ZEV remaining)
**GPU pin**: `CUDA_VISIBLE_DEVICES=4,5,6,7`
**Iteration compute cap**: 0.8 GPU-h total across all rounds

---

## Iteration 1 (2026-07-20)

### Assessment (Summary)
- **Score**: 4/10
- **Verdict** (canonical): `almost`
- **Budget after this iteration**: iterations 0/2, claim-reentries 0/1 (⓪-only iteration; no back-edge consumed)
- **Consecutive no-op count**: 1 (this iteration was ⓪-only; there is no prior persisted state to compare against — counting this as the first no-op iteration for the stall guard)
- **Key criticisms**:
  1. C1 phenomenon is genuinely strong (64.2pp gap, 8/8 seeds) but strict `residue==0` bar was missed (2/154 = 1.3%); paper framing must not oversell "residue-free"
  2. C2 evidence is diagnostic/correlational, not yet fully verified (INTEGRITY_ONLY — swap deferred by max_verify_claims cap)
  3. C3 target ablation is 10× below the 32pp bar; random ablation exceeds target ablation (specificity fails); amplification has the wrong sign — the causal-handle story does not hold
  4. Budget overrun (~15.5 vs 10.0 GPU-h, +55%) signals execution/integrity blemish
  5. LR sweep was committed early (5/12 evals; lr=1e-5, 5e-6 unevaluated) — methodological caveat
  6. Distinction between INTERNAL-READY (yes) and TOP-VENUE-READY (no) — the mechanism paper story is not competitive at NeurIPS/ICML

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (7953 chars)</summary>

## Overall score
**4/10**

Why so low despite a real-looking C1 effect? Because top-venue bar is not "one strong phenomenon claim exists." It is:
- phenomenon is clean,
- mechanism is credibly localized,
- causal test works,
- evaluation protocol is airtight,
- budget/integrity issues are under control.

This project has **one substantively supported claim (C1)**, **one suggestive integrity-only mechanistic signature (C2)**, and **one failed/negative causal handle story dressed as partial support (C3)**. That is not a strong mechanistic paper yet. It is at best a **promising phenomenon + diagnostic signature + failed causal intervention** package.

---

## 1) Minimum fixes for FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS
**none pending**

Per the ledger, there are **no claims** in FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS.
C2 and C3 are **INTEGRITY_ONLY** and are **not actionable** under the contract.

---

## 2) PASS claim check

### C1 — Subliminal-transfer phenomenon
**Verdict:** passes, but with a nontrivial caveat on the strict residue criterion.

### Numeric consistency check
The stated effect is internally consistent:
- mean_teacher = **0.680**
- max control = **0.038**
- gap = **0.642 = 64.2pp**

That easily clears the claimed **≥ 5pp** bar.

Other support:
- **8/8 seeds positive** vs both controls
- eval size is stated as meeting the **≥160-prompt** standard
- verify says method-swap corroborates with **50.2pp** binary gap
- judge recall on sanity set = **1.000** on **20 images**

So the core phenomenon is not borderline. The effect size is huge.

### Caveat
The claim as written required:
- **banana_residue_count == 0**

But the main experiment had:
- **2/154 = 1.3%** residue

This means the claim did **not** strictly satisfy its own clean-channel criterion. You handled this by marking it formally **conditional**, then treating the miss as judge stochasticity, and verify says the override is defensible because:
- binary variant still shows a large gap (**50.2pp**)
- binary residue is **5/154 = 3.25%**, under a looser **5%** threshold
- integrity variant passed

That is enough for me to accept **the phenomenon is real**, but not enough to pretend the original strict formulation was literally met.

### Bottom line on C1
- **Scientific status:** strong positive result
- **Presentation risk:** must not oversell the residue-free aspect
- **Caveat to flag:** post-hoc override from `inconclusive` to `conditional` is acceptable only because verify corroborated it

---

## 3) INTEGRITY_ONLY Open Items

### Open Item — C2: Location of banana signal in DiT
- top site at **block 47/60**, module **attn.to_out.0**
- **11.30×** top-block ratio, top-1 SVD variance fraction = **0.300**
- consistent with LoRA-artifact, single-steering-vector not established, early-layer divergence-latent falsified
- Interesting, but not yet a fully verified mechanistic result. Diagnostic and correlational in weight space; does not by itself establish causal control.
- **Upgrade suggestion:** `/auto-verify C2 -- resume: true`

### Open Item — C3: Causal intervention on located sites
- baseline **0.744** → ablate **0.713** (−3pp), amplify_x2 **0.675** (−7pp, wrong sign), amplify_x4 **0.656** (−9pp, wrong sign), random_ablate **0.706** (−4pp, > target!), matched_control_ablate **0.719** (−2pp)
- target ablation gives only **−3pp**, far below the required **~32pp**
- amplification has the **wrong sign**
- random ablation is **stronger than target ablation**
- specificity fails
- The only thing that passes is the off-target quality bar. That is not enough.
- The charitable interpretation is: the site is a **signature**, not a usable causal handle, and the mechanism lives in a **distributed low-rank subspace** rather than a single direction. Fine. But then C3 is not support for causal localization; it is a negative result against simplistic intervention.
- **Upgrade suggestion:** `/auto-verify C3 -- resume: true`

---

## 4) Ready for a top venue?
**Almost — but not really as a strong top-venue mechanistic paper.**

If you mean "READY by your internal contract," then **yes**: there are no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS claims remaining, and C2/C3 are non-blocking INTEGRITY_ONLY.
If you mean "ready for acceptance at NeurIPS/ICML/ICLR," then **no**.

Reason: C1 is genuinely compelling; C2 is promising but under-verified; C3 does not support the intended causal story; it mostly undermines a simple localized-direction interpretation. The project also carries credibility dents: ~15.5 GPU-h realized against 10.0 HARD; post-hoc override on C1; LR selection process stopped early leaving 1e-5 and 5e-6 unevaluated; causal experiment is single-seed and weak.

> **Top-venue READY? Almost, for a phenomenon paper with restrained claims; No, for a mechanism paper.**

If submitted now, the paper should be framed around: (1) strong subliminal transfer phenomenon, (2) suggestive late-layer LoRA-subspace signature, (3) failed simple causal intervention implying distributed representation. If framed as "we found and causally manipulated the mechanism," that would be misleading.

---

## 5) Additional brutally honest notes

### Budget overrun
The **~55% overrun** matters because it signals weak execution discipline / underestimation / avoidable reruns. For science quality it is secondary; for trust in the experimental process it is not ideal.

### LR sweep
The unevaluated **1e-5** and **5e-6** means the optimization story is not clean. Given cost constraints, this is a **narrative caveat only**, not something to fix now. But reviewers will correctly note that "best LR" was selected before the sweep completed.

### C3 amplification sign
Do **not** spin this as subtle sophistication unless you have ruled out an implementation sign bug. Right now the safest statement is: either the intervention direction is not causal / not isolated, or the operationalization of "banana_direction" is not aligned with causal increase in banana probability, or there is a bug. Given no affordable rerun and contract constraints, this remains a **paper caveat**, not evidence.

---

## Memory update
- Track the distinction between **internal READY** and **top-venue competitive READY**; they are not the same here.
- Main standing concern: **C1 is real, but C2/C3 do not yet elevate this to a convincing mechanism paper**.
- Keep explicit watch on the **C1 residue criterion**: claim text should not imply literal zero-leak success when main run had **2/154** residue.
- Keep skepticism on **C3 amplification wrong sign** until a code-level audit or independent verify rules out a sign/error bug.
- Track that **random_ablate (−4pp) > target ablate (−3pp)** is a major specificity failure, not a footnote.
- Track **budget overrun (~15.5 vs 10.0 GPU-h)** as an execution/integrity blemish.
- Track **unfinished LR sweep** as a methodological caveat; do not call the LR choice fully justified.
- If future iterations continue, the most valuable scientific question is whether the signal is truly a **distributed late-layer low-rank LoRA artifact** rather than a localized semantic direction.

</details>

### Verify-Passed Claims (brief audit)
- **C1**: numeric consistency confirmed by reviewer — 0.680 − 0.038 = 0.642 gap (64.2pp); 8/8 seeds positive; judge_recall=1.0 on 20 sanity images; method-swap variant corroborates at 50.2pp binary. Caveat: strict `residue==0` bar was missed (2/154 = 1.3%); post-hoc `inconclusive→conditional` override is acceptable *only* because verify variant corroborates. **Paper-side framing caveat added to CLAIMS_LEDGER C1 Caveats** (⓪).

### Actions Taken (per claim, per type)
- **C1** — type ⓪ — added paper-framing guidance caveat to CLAIMS_LEDGER: "do NOT phrase as 'residue-free'; phrase as 'subliminal transfer with judge-stochasticity-level residue (≤3.3% under both MCQ and binary judge templates)'". No scripts touched, no runs fired.
  - **CLAIMS_LEDGER.md C1 Caveats — Before**: (2 items: conditional status, teacher LoRA v1→v2 iter).
  - **CLAIMS_LEDGER.md C1 Caveats — After**: +1 new item — "Paper framing guidance (⓪ iteration-1): do NOT phrase as 'residue-free subliminal transfer'; phrase as '... with judge-stochasticity-level residue (≤ 3.3% under both MCQ and binary judge templates), corroborated by method-swap variant with a 50.2pp binary-judge gap'. The post-hoc `inconclusive→conditional` override is acceptable *only* because the verify variant independently corroborates."
- **C3** — type ⓪ — **iteration-1 code audit finding**: inspected `src/mechanism/mechanism_intervene.py` L127-167 and `mechanism_location.py` L228-230. Intervention hook is correctly operationalized (amplify=+kσv̂, ablate=−projection along v). Direction extraction saves the raw top-1 left singular vector `direction_out = U_m[:, 0]` with **no sign-orientation step** — SVD sign is arbitrary. The "amp decreases P(banana)" observation is therefore not diagnostic between (a) SVD sign ambiguity and (b) a real distributed-subspace effect. Ablation/random/matched-control results are sign-independent and remain valid. Documented in CLAIMS_LEDGER C3 Caveats + Open Items. **No script edits performed** — a proposed one-line sign-orient fix (~0.1 GPU-h to re-run only the two amplify arms) is documented but deferred because it lies within the specificity story that is already established sign-independently.
  - **CLAIMS_LEDGER.md C3 Caveats — Before**: 2 items (single-seed, amplification wrong).
  - **CLAIMS_LEDGER.md C3 Caveats — After**: +2 new items — sign-ambiguity caveat + paper framing guidance (frame as negative result on single-direction causal handle, not partial support).
- **C3** — type ⓪ — paper framing guidance added to CLAIMS_LEDGER Open Items: "Frame as (1) strong subliminal transfer phenomenon, (2) suggestive late-layer LoRA-subspace signature, (3) negative result on single-direction causal handle → distributed LoRA-artifact account (Nief 2606.00831)". Not "we found and causally manipulated the mechanism".
- **Open Items** — type ⓪ — 3 new entries added: C3 SVD sign ambiguity (with proposed cheap follow-up), Paper framing guidance (top-level).

**Total budget consumed this iteration**: 0 iterations (all actions ⓪), 0 claim-reentries, 0 new runs, 0.0 GPU-h.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: pass]:
  * upgrade: `/auto-verify C2 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
- **C3** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: warn; warn_source: experiment+mechanism]:
  * upgrade: `/auto-verify C3 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)

### Results
- [run-experiment] iteration=1 runs_this_iteration=0 gpu_hours_this_iteration=0.0 cumulative_gpu_hours=0.0
- **C1**: no back-edge; PASS retained; +1 paper-framing caveat in ledger
- **C2**: no back-edge (INTEGRITY_ONLY, contract-locked); retained; recorded in Open Items with upgrade suggestion
- **C3**: no back-edge (INTEGRITY_ONLY, contract-locked); retained; +2 new caveats in ledger (sign-ambiguity from code audit; negative-result framing) + 2 new top-level Open Items entries

### Status
- Continuing to iteration 2 to re-review under the sharpened framing and confirm no material score change. If iteration 2's ⓪-only or has unchanged score/verdict, stall guard fires and loop terminates.

---

## Iteration 2 (2026-07-20)

### Assessment (Summary)
- **Score**: 5/10 (up from 4/10)
- **Verdict** (canonical): `almost` (unchanged)
- **Budget after this iteration**: iterations 0/2 (no back-edge dispatched — all actions ⓪), claim-reentries 0/1
- **Consecutive no-op count**: 0 (score changed 4→5, resets the stall counter)
- **Key criticisms**:
  1. C1 residue-framing caveat **addressed** (paper-side language explicitly forbidden overselling "residue-free")
  2. C3 code-audit finding **partially addressed** (hook confirmed correct; SVD sign-ambiguity identified in extraction; ~0.1 GPU-h confirmatory rerun deferred — this is the one substantive concern the reviewer still holds)
  3. C3 specificity failure (random ≥ target, matched ≈ target) remains — this is a scientific reality unfixable within iteration scope
  4. LR sweep unfinished (out of budget)
  5. Budget overrun (~15.5 vs 10.0 GPU-h) documented, unfixable retroactively
  6. New pattern flagged: "affordable disambiguation was deferred despite being directly relevant" — reviewer sees this as an execution-choice bias signal. The C3 sign-orient rerun (~0.1 GPU-h) is affordable but contract-blocked (C3 is INTEGRITY_ONLY per verify contract)

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (7210 chars)</summary>

## Re-score
**5/10**

The score goes up slightly from **4/10** because the authors did something they should have done earlier: they stopped pretending C3 supports a clean causal mechanism and explicitly reframed it as a **negative result on a single-direction handle**. That matters.

But this is still **not a strong top-venue mechanism paper**. The core reason has not changed: the most ambitious mechanistic claim is not supported, and the causal evidence is weak/non-specific by the paper's own numbers:
- baseline **0.744** → ablate **0.713** (**Δ=−3pp**)
- amplify_x4 **0.656** (**−9pp**, wrong sign observationally)
- random_ablate **0.706** (**−4pp**)
- matched_control_ablate **0.719** (**−2pp**)

What improved is **honesty of framing**, not strength of evidence.

---

## 1. Top-venue assessment with sharpened framing
As a **phenomenon paper with limited mechanistic evidence**, this is now more defensible. As a **mechanism paper**, still weak.
- **Phenomenon**: interesting, likely publishable somewhere if written narrowly and honestly.
- **Top-venue mechanism claim**: no.

## 2. For each FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS claim: minimum fix
**none pending**

Per the stated verify state, there are no claims currently in FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS. This should not be misread as "nothing important remains weak." The remaining problems are now mostly **framing / scope / unsupported ambition** issues rather than formal verify-state failures.

## 3. C3 code-audit response
**Rating: partially addressed**
- The audit does show the intervention hook is not sign-flipped. Good.
- The audit identifies the SVD sign ambiguity (`direction_out = U_m[:, 0]` saved raw with no orientation).
- Therefore the observed wrong-sign amplification cannot be interpreted as evidence about the underlying mechanism direction.
- But the practical amplification-result issue **remains uninterpretable** until sign orientation is fixed and the two amplify arms are rerun. That rerun was affordable (**~0.1 GPU-h**) yet deferred. The author is trying to say "the main story is specificity anyway," which is fair for paper framing, but the original wrong-sign concern is **not fully closed**.
- **Not sidestepped**: they did inspect the code and found the relevant ambiguity.
- **Not addressed fully**: they did not perform the cheap confirmatory rerun.

## 4. C1 residue framing response
**Rating: addressed**
The new framing explicitly says: do NOT claim "residue-free subliminal transfer"; frame as transfer with judge-stochasticity-level residue; mention the post-hoc `inconclusive→conditional` override in Methods. That is the right correction.

## 5. Ready for submission under honest phenomenon-paper framing?
**Almost**
Why not "yes":
1. LR sweep still unfinished (lr=1e-5, 5e-6 not evaluated) — sloppy optimization/selection story
2. Budget overrun remains ugly (~15.5 vs 10.0 GPU-h)
3. C3 is still mechanistically negative and non-specific
4. The cheap sign-orientation rerun was not done

**Ready for a mechanism paper**: No
**Ready for submission if framed honestly as phenomenon + suggestive signature + negative causal result**: Almost

## 6. Memory update
### New suspicions
- Affordable disambiguation was deferred despite being directly relevant (execution-choice bias signal)
- Risk of laundering weak C3 through "distributed subspace" rhetoric — data do NOT positively establish distributed-subspace; they refute a clean single-direction account. Watch for wording drift.

### Updated patterns
- Narrative repair helped substantially, but evidence did not improve
- C3 remains the central blocker for ambitious claims
- Internal-ready vs venue-ready distinction still holds

## Final verdict
**Score: 5/10, Verdict: Almost, C3 responsiveness: partially addressed, C1 responsiveness: addressed.**
The authors improved the honesty. They did not improve the core mechanistic evidence.

</details>

### Verify-Passed Claims (brief audit)
- **C1**: reviewer rated the iteration-1 paper-framing caveat as **addressed** — new phrasing correctly avoids overselling "residue-free". No further action.

### Actions Taken (per claim, per type)
none — this iteration is a pure re-review cycle with no new on-disk changes. Reviewer explicitly said "none pending" for FAIL/INCONCLUSIVE/ZEV fixes, and the one substantive fix they flagged (C3 sign-orient rerun ~0.1 GPU-h) targets an INTEGRITY_ONLY claim, which is **contract-blocked** for back-edge actions per the `verify_integrity_only` no-action-with-upgrade-suggestion contract. The reviewer's `partially addressed` rating on C3 is acknowledged but the upgrade path is `/auto-verify C3 -- resume: true` outside this loop.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only, unchanged)
- **C2** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: pass]:
  * upgrade: `/auto-verify C2 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run)
- **C3** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: warn; warn_source: experiment+mechanism]:
  * upgrade: `/auto-verify C3 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run)
  * additional upgrade suggestion (from iteration-1 code audit): add a one-line sign-orient step to `src/mechanism/mechanism_location.py` L228-230 before re-running the two amplify arms in `src/mechanism/mechanism_intervene.py` (~0.1 GPU-h; deferred inside iteration loop because C3 is INTEGRITY_ONLY per contract)

### Results
- [run-experiment] iteration=2 runs_this_iteration=0 gpu_hours_this_iteration=0.0 cumulative_gpu_hours=0.0
- Reviewer score improved 4→5 attributable to iteration-1's ⓪ paper-framing + code-audit refinements
- Reviewer explicitly confirmed: internal contract READY = yes; top-venue mechanism paper READY = no; honest-phenomenon-paper submission = almost
- No FAIL/INCONCLUSIVE/ZEV claims remain (contract STOP condition 3 met throughout)

### Status
completed — terminating the loop.
- **Reason**: `iterations_exhausted` (by task-declared reviewer-cycle cap MAX_ITERATIONS=2, per HARD-budget-crisis reduction; formal back-edge count is 0/2 since all iteration actions were ⓪ or pure-review).
- **Why not continue**: reviewer explicitly said "none pending" for FAIL/INCONCLUSIVE/ZEV fixes at both cycles. The one substantive concern (C3 sign-orient rerun) is contract-blocked (INTEGRITY_ONLY). Further ⓪-only rounds would loop without evidence change — the reviewer's `almost` verdict at score 5 reflects a scientific-evidence ceiling reachable within budget.
- **STOP check**: score 5 < TARGET_SCORE=6 → no positive-verdict STOP; but all three claims are in PASS or INTEGRITY_ONLY (no FAIL/INCONCLUSIVE/ZEV) — internal contract READY criterion is met.
