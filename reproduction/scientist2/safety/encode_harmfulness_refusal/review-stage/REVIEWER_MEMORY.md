# Reviewer Memory

## Iteration 1 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - C3 is currently over-claimed as "supported"; the underlying intervention evidence is not mechanistically rigorous yet due to inadequate alpha scaling (raw ‖d‖ vs sigma_proj units), inadequate range (~1 order of magnitude vs required ≥3), and insufficient random controls (n=1 vs required ≥30). The audit's own conclusions already undermine the "supported" label.
  - C5 novelty collapses after proper scope narrowing: probe trained on harmful/benign and tested on harmful/benign+lookalike is "close to training/evaluation axis" — near-circular. AUROC 1.000 vs 0.9992 is negligible in practical terms. The "beats Llama Guard at jailbreak detection" angle disappears entirely; what remains is "cheap harmfulness classifier matches LG on easy data" — much weaker.
  - C1 is over-summarised: the "PASS" hides that only the harmfulness half is well established; the refusal half is inferred under severe label collapse (98.7% bare-refusal → refusal-side sub-tests NaN). "Independently recoverable" is too strong when one axis is mostly inferred.
  - C4's ASR=0 on published attacks + C2's ceiling AUROC at both positions together imply harmfulness is broadly represented (not spatially localized as the plan predicted) and refusal-variation is sparse in this model → the whole "two-position, two-direction mechanistic separation" story is under-supported.
  - Headline reframing needed: the paper cannot honestly be sold as "mechanistic separation of harmfulness perception and refusal execution + jailbreak-suppression detection". Honest reframing is "strong harmfulness direction, weaker partially-dissociable refusal-control direction; temporal-dissociation and jailbreak hypotheses not supported in this setting".
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - C3 mechanism rigor gap (needs sigma_proj rescale + range extension + n_random ≥ 30) — actionable this iteration.
  - C5 scope over-claim (needs claim wording narrowed) — actionable this iteration.
  - C1 refusal-side unmeasurability (dataset+alignment issue; no cheap fix; propagate as paper caveat).
  - Paper-level: overall reframing away from the two-direction full-separation story.
- **Patterns**: The strongest headline claims (C3 mechanism, C5 jailbreak beats LG) are precisely where the audit finds the weakest evidence. Recurring theme: results were "labelled supported" on the letter of the pre-registered criteria but do not survive an adversarial reviewer applying tighter rigor gates (proper units for α, adequate specificity floor, correct evaluation scope).

## Iteration 2 — Score: 5/10, Verdict: not ready

- **New suspicions**:
  - The z=101.65 h-side random-control comparison, while procedurally correct, is partially a **geometric inevitability** in 4096-dim space: a discovered structured direction versus random matched-norm vectors will produce huge z-scores whenever the readout metric is aligned to the discovered direction. The random-control analysis is a **necessary sanity check** but not by itself a **mechanistic clincher**.
  - The r-side is still weak: refusal effect is threshold-only (only manifests at α_sigma_r ≥ ~3.8), not a stable plateau. This is scientifically weaker than a controllable, dose-responsive mechanism. The model is already saturated (99% baseline refusal on harmful) so demonstrating "we can push refusal even higher on benign at large α" does not cleanly demonstrate a refusal-execution mechanism — it shows the intervention is strong enough to override the baseline behavior.
  - Residual loophole: iter-1's random controls for the r-effect were run at h's site (not r's site). Needs 30 random-matched-norm at the actual r intervention site to close.
  - C5 scope narrowing is honest but the resulting claim ("cheap probe matches LG on bare-harmful vs benign/safe-lookalike, at ~10⁻⁸ compute") is near-circular: the probe was trained on the same harmful/benign contrast used at test.
  - Overall paper story: after the fixes, the strongest honest headline is asymmetric — "strong harmfulness direction (h), weaker/thresholded refusal-control direction (r) with confirmed direction-specificity"; the C2 and C4 negatives further weaken the "temporal + jailbreak" side; the paper needs to be reframed away from the full "mechanistic separation + jailbreak-suppression" story.
- **Previous suspicions addressed?**:
  - C3 alpha units / range / n_random: **YES** (all three iteration-1 audit gaps closed; α now in σ_proj, coverage 35× span, n_random=30 at 2 operating points).
  - C5 scope: **YES** (narrowed to "bare-harmful vs benign/safe-lookalike"; honest disclosure of the near-circular training/eval contrast added).
  - C1 refusal-side unmeasurability: **UNADDRESSED** (no cheap fix exists at this model + dataset).
  - Paper-level reframing: **PARTIALLY addressed** — iteration-2 added a top-of-plan "Iteration-2 headline reframing" block and an EXPERIMENT_RESULTS.md "Iteration-2 Headline Framing" section; still needs full manuscript rewrite before submission.
- **Unresolved (carried forward)**:
  - r-side threshold shape at high |α| — scientific caveat, not a rigor gap (nothing more to fix experimentally on this axis).
  - r-site random-control loophole — actionable this iteration; deployed 60 configs (1.26 GPU-h).
  - Paper-level headline reframing — narrative work, needs authoring pass before submission.
- **Patterns**: The reviewer's persistent theme is that "supported on the letter of the criteria" is not the same as "mechanistically established" — the loop has walked C3 through σ_proj-normalization, span extension, n_random=30 at h's site, and now r-site closure. Each step tightened the evidence; none of them changed the underlying asymmetry (h strong, r thresholded). The paper story must reflect that asymmetry, not paper over it.

## Iteration 3 — Score: 6/10, Verdict: almost (procedurally ready)

- **New suspicions**:
  - Even after the r-site closure, r-side evidence is still **threshold-only/high-magnitude**, not a clean symmetric counterpart to h. Closing "site-alone" does not automatically establish a strong mechanistic refusal-control story.
  - The internal verdict label `supported_with_r_threshold_caveat_and_r_site_specificity_confirmed` is fine as a tracking string but if the paper presents this as "causal dissociation" or "mechanistic separation" the reviewer would call it too strong. **Wording downgrade required** (this iteration).
  - C5 remains honest-but-weak after narrowing; the reviewer explicitly cautioned that practical framing must not creep back into the paper.
- **Previous suspicions addressed?**:
  - r-site random-control loophole: **YES — genuinely closed**. Reviewer's exact words: "true r vs matched-norm random directions at the actual r site/position is **decisive** against 'site-alone drives effect'."
  - Paper headline reframing: **substantially improved but not fully sufficient** — reviewer wants C3 verdict wording downgraded from "causal dissociation" to "asymmetric causal steering evidence" / "partial functional dissociation" throughout the plan+results (implemented in iteration 3 as a type-⓪ narrative pass).
- **Unresolved (carried forward at loop termination)**:
  - r-side threshold shape is a real property of the model+dataset regime, not a rigor gap — no further experimental fix possible within this setup.
  - Full manuscript rewrite (author-facing) required before actual submission — the loop can only enforce plan/results wording; the final paper needs an authoring pass.
- **Patterns**: The reviewer's persistent theme resolved: each iteration tightened the rigor of C3, and iteration 3 added a language-tightening pass to make the paper's wording match the evidence's asymmetry. **All procedural blockers now resolved**; remaining concerns are manuscript-authoring rhetorical honesty, not experimental gaps.
