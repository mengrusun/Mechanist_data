# Reviewer Memory

## Iteration 1 — Score: 3/10, Verdict: not ready

- **New suspicions**:
  - **C1 headline fails preregistered unanimity.** 2/3 seeds pass strongly; seed300 reverses. Under `task.md`'s strict predicate (≥3 pp drop across ≥3 seeds unanimously), C1 is not-established. The `not-supported per strict predicate` verdict is judge-stable, but that is a stable *negative*, not a positive.
  - **LR-cliff confound.** Winning LR=1e-3 is the TOP edge of the {5e-5..1e-3} grid; all lower LRs show ~0 pp effect. Classic sharp-training-regime signature — the effect may be optimization damage / weight-init fragility, not "subliminal transfer".
  - **Missing matched-benign SFT control.** No matched-format text-only SFT on benign / neutral content of same size/steps. Cannot separate treatment-specific effect from generic text-only SFT / synthetic-data adaptation.
  - **Mechanism story materially weak despite headline ablation/patching numbers.**
    - AUROC=0.27 for safety-decisive partition is actively inconsistent with a "safety-relevant axis" interpretation — the direction is treated-vs-Ctrl, not safety.
    - Steering non-monotonic, boundary-hitting (best gc at α=+2, the sweep edge).
    - Random-matched steering matches best real-direction steering (SP-A specificity FAIL).
    - Cross-seed (seed200, seed300) steering ~zero effect.
    - MMLU capability control never run at any α.
  - **Ablation/patching may only show representational importance, not specificity.** The direction being the treated-vs-Ctrl axis (per AUROC=0.27) means removing it "restores" Ctrl-like state — but that is model-identity restoration, not safety-substrate restoration.
  - **Underpower cuts both ways.** M2 n=27 held-out — huge sampling noise. But underpower does not rescue overclaims: gc=0.25 at n=27 is ~7 items, well within noise.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - C3 mechanism-rigor gap (MMLU-missing + no plateau).
  - LR-cliff confound on C1 — no additional LR ablation exists in the current pipeline.
  - Matched-benign-SFT control missing entirely.
  - Cross-seed steering non-replication.
  - Ablation/patching may not be safety-specific — could be identity-restoration.
- **Patterns**: none yet (single iteration); flagging tendency to bury seed300 reversal + hyperparameter-cliff behind the M0.7 "conditional" wording.

## Iteration 2 — Score: 3/10, Verdict: not ready

- **New suspicions**:
  - The improved C3 measurement now suggests the earlier positive-leaning mechanism narrative was not just under-supported but likely **mis-specified**. The intervention seems to track a treated-vs-control identity axis, not a safety-specific one.
  - The flat gc≈0.125 plateau with isolated 0.250 spikes on n=27 looks like **noise-compatible weak modulation**, not controllable mechanism.
  - MMLU-flatness may be over-interpreted — it only rules out gross capability collapse, not subtle degradation or safety-specific meaning.
- **Previous suspicions addressed?**:
  - **C3 MMLU-missing (A.3)**: YES — genuinely addressed at measurement level (500-item slice at every α, max |drop| 1.0 pp).
  - **C3 no plateau / boundary issue (A.4)**: YES — probed honestly; the widened data reveals a noise-floor plateau at gc≈0.125, not a productive one.
  - **LR-cliff confound on C1**: NO — not addressed. No additional LR ablation ran.
  - **Matched-benign-SFT control missing**: NO — still not run.
  - **Cross-seed steering non-replication**: NO — only seed100 was widened. Seed 200/300 still show gc≈0 at all α.
  - **AUROC=0.27 semantic mismatch (M1 direction is not safety axis)**: NO — direction unchanged.
  - **Ablation/patching may be identity-restoration**: NO — no follow-up.
- **Unresolved (carried forward for iter-3+)**:
  - LR-cliff confound on C1 (still).
  - Matched-benign-SFT control (still).
  - Cross-seed steering non-replication (still).
  - AUROC=0.27 semantic mismatch (still).
  - Ablation/patching specificity (still).
- **Patterns**:
  - **Positive pattern**: Iteration DID run the falsification checks it was asked for, and DID report the negative result honestly. Not performative. Good research hygiene.
  - **Negative pattern**: reliance on process-state language ("PASS-of-negative-verdict") to imply maturity even though the science remains scientifically weak. The bookkeeping-vs-science gap is the pattern to watch.
  - **Negative pattern**: continued reliance on seed100-only mechanism follow-up while cross-seed non-replication remains unresolved.

## Iteration 4 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - The paper still has a fundamental identity problem: if C1 is the headline, it's still confounded (LR-cliff, matched-benign missing); if C3 is the headline, it's now a clean negative result about a failed mechanism probe attached to an unstable phenomenon. Neither is a top-venue package as-is.
  - The C3_v2 phrasing "best characterized as a distributed representational shift" is slightly interpretive — should be softened to "more consistent with".
  - The loop's "PASS" bookkeeping still risks implying maturity because one claim was redefined into a negative result and another passes as not-supported per strict predicate — procedurally valid, but not evidence of paper maturity.
- **Previous suspicions addressed?**:
  - Iter-1 A.3/A.4 rigor gaps: YES (already addressed by iter-1).
  - Iter-2 C3 rewrite: **YES — substantive and honest.** Not bookkeeping spin. The rewrite changes the sign of the result, aligns the claim with actual falsification outcomes, stops implying controllable mechanism discovery, narrows surviving positive statements to modest ones.
  - Iter-3 cross-seed widened steering: **YES — good closure**. Not transformative but genuine falsification-value: eliminates the "maybe seed100 was the start of a replicable mechanism" escape hatch by showing gc=0 on both seed200 and seed300 at all α.
  - **LR-cliff confound on C1**: STILL NO.
  - **Matched-benign-SFT control**: STILL NO — this is the highest-value remaining experiment.
  - **AUROC=0.27 semantic mismatch**: STILL NO — but now explicitly stated in C3_v2's predicate (a).
  - **Ablation/patching identity-restoration**: STILL NO — but C3_v2 doesn't claim safety-specificity here; it explicitly notes ablation/patching close the treated-vs-Ctrl gap but with the direction NOT being the specific handle.
- **Unresolved (carried forward for iter-5+)**:
  - LR-cliff confound on C1 (highest priority remaining).
  - Matched-benign-SFT control on C1 (highest priority remaining).
  - Direction semantic identity (ablation/patching = identity-restore vs safety-restore) — hard to disentangle without additional data partitions.
- **Patterns**:
  - **Positive pattern reinforced**: iterations 2 and 3 both delivered substantive scientific value (rewrite + cross-seed closure), not just check-box process. Loop is not spinning on C3.
  - **Negative pattern reinforced**: C1 continues to be spun. The loop keeps closing C3 gaps while C1 gaps sit untouched. Priority allocation is off — C1 is the headline claim.
  - **Convergence forecast**: READY unlikely in 2 remaining iterations. ALMOST plausible only with a substantive C1 control (matched-benign-SFT proxy) + narrative reframing. NOT READY is the modal forecast for termination.

## Iteration 6 — Score: 5/10, Verdict: almost

- **New suspicions**:
  - The LR=1.5e-3 unanimity is a **post-hoc reviewer-requested rescue**, not a clean confirmatory result. It reduces cherry-picking concerns because the reviewer asked for exactly this test, but does not retroactively validate the preregistered LR=1e-3.
  - The transition 7e-4 → 1e-3 → 1.5e-3 (weak → mixed → strong) IS a dose-response, not a knife-edge — this defuses the worst version of the LR-cliff confound but does not fully eliminate LR sensitivity concerns.
  - The narrative risk is very high: the paper could easily read as "we eventually found an LR that works" if written incautiously. Top venues punish that.
- **Previous suspicions addressed?**:
  - **LR-cliff confound on C1**: YES (mostly). Iter-5 shows a monotonic response over 7e-4 → 1e-3 → 1.5e-3, refuting the knife-edge interpretation. The effect is real but strongly LR-sensitive.
  - **Matched-benign-SFT control**: STILL NO. This remains the largest omitted control.
  - **AUROC=0.27 mechanism identity**: unchanged (still not addressed).
  - **Ablation/patching identity-restoration vs safety**: unchanged.
- **What C1 should now say** (reviewer's own suggested text, verbatim):
  > "Under the originally selected LR=1e-3, the strict 3/3-seed unanimity criterion was not met (2/3 seeds passed). In response to reviewer concerns about LR sensitivity, we evaluated nearby LRs and found that LR=1.5e-3 yields strong, unanimous replication across all three seeds. We therefore conclude that the behavioral effect is real but strongly optimization-sensitive, and that the originally selected LR was suboptimal."
- **What C1 should NOT say** (reviewer's own list, verbatim):
  - "The preregistered M0 criterion is satisfied" without caveat
  - "The original claim is fully validated"
  - "This was the planned winning configuration"
  - "The effect robustly holds across learning rates"
- **Unresolved for possible iter-7+**:
  - Matched-benign-SFT (~3h dispatch remaining scope). If run and shows the effect is confined to the specific teacher SFT data, C1 attribution is strengthened; if similar effect on benign-teacher control, C1 attribution collapses.
- **Patterns**:
  - **Positive pattern**: iterations 2, 3, 5 all delivered substantive scientific value — rewrite, cross-seed closure, LR-cliff rescue.
  - **Negative pattern remaining**: matched-benign-SFT is the sole unaddressed reviewer priority (the one that would substantively change the paper).
  - **Overall**: score trajectory 3 → 3 → (deferred) → 4 → (deferred) → 5. Genuine incremental improvement across iterations. Not a stalled loop.
  - **Forecast**: score plausibly reaches 6 (READY) only with matched-benign-SFT proxy + disciplined narrative reframing. Score plausibly stays at 5 (ALMOST) with narrative alone.

## Iteration 8 — Score: 5/10, Verdict: almost (final)

- **New suspicions**: none — score / verdict / evidence unchanged from iter-6. The iter-7 narrative reframing "fixes an honesty/positioning problem rather than a science problem". Better calibrated, not stronger.
- **Previous suspicions addressed?**:
  - Iter-6 narrative-wording concerns: YES — reviewer's own recommended wording adopted verbatim in iter-7.
  - Matched-benign-SFT: STILL NO — this is the sole remaining experiment "with realistic chance to materially improve interpretation / score."
- **Termination judgment (from reviewer)**:
  - "Yes, stopping here is acceptable" — natural stopping point.
  - "Also yes, matched-benign-SFT is the highest-value remaining 3h you could spend" — science-optimal endpoint not reached.
  - Recommendation for the loop: "Stop is acceptable, but not science-optimal. If resources/latency permit, matched-benign-SFT is still worth dispatching; if not, submit with the current explicit caveats."
- **Patterns (final)**:
  - Score trajectory 3 → 3 → 4 → 5 → 5 shows genuine incremental improvement across iterations 1-6, plateau at iter-7/8.
  - Loop delivered substantive scientific value across 3 back-edge actions (iter-1 ②, iter-2 ③, iter-3 ②, iter-5 ②).
  - Terminal state is honest and defensible: submitting the paper with the reviewer's exact caveats is acceptable; the loop is not spinning, it's at its practical ceiling within the on-disk data.
- **What the paper should say (reviewer's final memory update, verbatim carried forward)**:
  - **C1**: qualified positive only. Original LR=1e-3 missed strict unanimity (2/3); nearby LR=1.5e-3 gives 3/3; therefore effect appears real but optimization-sensitive; original LR was suboptimal. This narrative fix does NOT convert C1 into a preregistered-style success or a robust-across-LRs claim.
  - **C2**: integrity-only, acceptable as such.
  - **C3_v2**: acceptable as softened negative-result mechanistic evidence; no single-direction claim.
  - **Missing control**: matched-benign-SFT (largest remaining missing control; would either strengthen interpretation or clarify true scope).
