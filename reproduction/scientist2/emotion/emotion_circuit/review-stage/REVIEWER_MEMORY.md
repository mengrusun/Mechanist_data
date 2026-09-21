# Reviewer Memory

## Iteration 1 — Score: 3/10, Verdict: not ready

- **New suspicions**:
  - **C2 is the highest-priority scientific risk.** Confirmed operator-scope bug in ablation (global mean vs per-stem mean, m2_causal.py:341-350) likely invalidates the negative causal result. Must track whether per-stem fix changes sign / monotonicity / specificity.
  - **C3 likely reflects fundamental operator/eval mismatch, not tuning noise.** Broad additive injection over ~24 heads + ~2000 neurons during 100-token generation appears to induce OOD behavior; Qwen replication supports systematic failure. Even with reduced α, cumulative injection may still fail — cannot pre-judge.
  - **Random-null degeneracy in C2 is another integrity concern.** Even after per-stem bug fix, null/control implementation needs scrutiny (std=0 for all 6 emotions).
  - **C1 strongest asset but not bulletproof.** Flat kstar grid suggests underdetermined hyperparameter choice; head-level Stage-B degeneracy means paper must avoid overclaiming fine-grained head-level identification mechanism.
  - **Claim inflation risk.** Current title "Verifying Locatability, Causality, and Applied Control..." overstates what is supported — even if C2 repair produces causal signal, "Applied Control" language should be dropped or reframed as a negative test unless C3 recovers.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - C2 per-stem operator repair outcome (open — pending iteration 2)
  - C2 random-null degenerate std=0 (open — pending fix)
  - C3 additive-injection OOD failure (open — pending α reduction and/or component reduction)
  - Paper-narrative overreach: title/abstract must be softened downstream of loop
- **Patterns**:
  - Operator implementation drift from EXPERIMENT_PLAN.md spec: two independent operator issues (C2 global-vs-per-stem mean; C3 cumulative injection at generation time). Suggests the plan-vs-implementation alignment step of experiment-stage was incomplete.
  - Val-eval metric misalignment: C3 val uses target-prefix logprob gain, eval uses judge-scored generation accuracy — makes val hyperparameter selection unreliable for the actual test regime.

## Iteration 2 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - **C2 now points to a deeper mechanistic issue, not an implementation issue.** The positive ablation deltas surviving the per-stem fix suggest the identified set is not necessary in the intended sense, and enhancement may mostly exploit generic activation-magnitude effects.
  - **C3 failure looks structural, not just overdosing.** Reduced α and reduced component count improved results but did not change the ranking versus baselines; likely a fundamental mismatch between score-time signal and generation-time control.
  - **Validation-objective mismatch remains a major scientific weakness.** The fact that val prefers the largest reduced α while final judged generation remains poor suggests target-prefix logprob is not a reliable proxy for end-task control quality.
  - **Random sets outperform "causal" sets for half the emotions in C2.** For joy/sadness/anger, random top-K within the same layer pool can produce a LARGER Δ than the causally-ranked C_e. Specificity claims need more careful framing.
  - **Arm C's strong performance on disgust (0.725) — even beats Arm B — is worth understanding.** If an ostensibly weaker control beats prompting on one emotion, either the judge/benchmark has a bias or Arm C is exploiting something specific about disgust prompts.
- **Previous suspicions addressed?**:
  - C2 operator-bug hypothesis: **YES resolved and FALSIFIED**. The per-stem fix produced essentially the same positive Δ, so the bug was not causing the wrong-sign. Reviewer explicitly concedes: "That is exactly the kind of result that rules out the bug as the main explanation."
  - C2 random-null degenerate: **YES resolved**. With enlarged pool, null now has proper variance. New finding uncovered by the fix: half the emotions have C_e Δ below the random-null mean, which is a specificity concern the paper must own.
  - C3 additive-injection OOD failure: **PARTIALLY addressed** (2.6x improvement) but reviewer says "still badly negative; the applied control story is dead in the current setup." Recommends Type ③ over another Type ② round.
  - Claim inflation risk: still open — reviewer strongly says paper must drop "verifying causality" and "applied control" phrases.
- **Unresolved (carried forward)**:
  - Why ablation IMPROVES target-prefix logprob (positive Δ instead of negative). Needs conceptual explanation, not more debugging.
  - Why random sets outperform "causal" sets for joy/sadness/anger (3/6 emotions).
  - Whether targeted-C_{e'} specificity signal (4/6 emotions have majority pass) is enough to hang a paper on.
  - Val-eval metric misalignment — target-prefix logprob is unreliable for judging end-task control quality.
- **Patterns**:
  - **Implementation drift was real, but fixing it did not save the claims.** This is a strong signal that the underlying method has limitations, not that we ran the wrong experiment.
  - **The pipeline systematically converts "predictive localization" into overclaimed mechanistic conclusions.** Framing risk applies to C1 (soft) as well as C2/C3 (hard).
  - **Broad additive interventions appear to act as generic perturbations rather than precise emotion control.** This is now the reviewer's leading hypothesis about the C2/C3 negative results.
  - **The strongest paper now is a negative-result / stress-test paper**, not a positive verification paper.

## Iteration 3 — Score: 5/10, Verdict: almost

- **New suspicions**:
  - **Manuscript-level overclaim residue.** Even though claim statements are now fixed, abstract / intro / related work / conclusion likely still carry "verification" language inconsistent with the data. Common failure mode after late-stage claim narrowing. Every verb ("verifies", "demonstrates", "validates") needs to be swept.
  - **C3_v2 may still be read too positively unless metric scope is explicit.** "Score modulation" can be misread as meaningful steering. Must specify "prefix-level score metric" and sharply separate from open-generation behavior.
  - **The random-set result may imply layer-level enrichment rather than subset-level identification.** Method may be locating the right layers, not uniquely right components. Contribution is weaker but still real — must acknowledge explicitly.
  - **The paper may lack a unifying conceptual takeaway.** The lesson is "discriminative locatability is not sufficient evidence for directional causal necessity or practical steerability" — if not articulated, paper feels like a pile of mixed outcomes.
- **Previous suspicions addressed?**:
  - C3 claim rewrite: **YES accepted with tightening**. Reviewer said "This is the first version of C3 I'd call substantively defensible" but suggested two wording tweaks: (1) "are associated with reliable prefix-level target-emotion score increases under additive activation injection" instead of "support ... score modulation"; (2) "limited/partial" instead of just "partial" specificity.
  - C2 reframing: **YES accepted with addition**. Reviewer wants an explicit statement that positive Δ under ablation means "either components are not necessary in the intended directional sense, OR intervention is removing a competing/suppressive/misaligned signal" — leave interpretation open but frame the anomaly.
  - Title candidate: **acceptable but reviewer prefers "falsification study" framing**. Recommended: "A Falsification Study of Activation-Based Emotion Steering in Llama-3.2-3B: Locatability Survives, Causal Necessity Does Not, and Practical Control Fails" or the less combative "Stress-Testing Activation-Based Emotion Steering in Llama-3.2-3B: Locatability Survives, but Causal Necessity and Practical Control Do Not."
  - Random-null 3/6 concern: **NOT YET addressed prominently enough**. Reviewer says "cannot be a footnote-level caveat"; must be foregrounded.
- **Unresolved (carried forward)**:
  - Need explicit interpretive paragraph for positive-Δ ablation anomaly (points 1 & 2 in reviewer's C2 write-up).
  - Need to foreground random-set 3/6 result in main claims.
  - Need to sweep manuscript verbs (remove "verifies/validates/demonstrates").
  - Need abstract to mention negative findings up front.
- **Patterns**:
  - **Late-stage claim narrowing is scientifically correct but leaves manuscript-level residue** in the "positive paper" scaffold. Standard failure mode.
  - **A rewrite, not an experiment, is now the sole remaining requirement to reach READY.** Reviewer explicitly says "the required item is a rewrite, not an experiment."
  - **Publishable ceiling now identified**: as a stress-test/falsification paper (option 3 from iteration 2's C3 rewrite options).

## Iteration 4 — Score: 5.5/10, Verdict: almost

- **New suspicions**:
  - **One mild suspicion only**: manuscript-level tendency to overcompress the positive result as "localization succeeds," when the actual surviving claim is weaker — "stable, discriminative, coarse localization" rather than "precise component identification." Not a new experiment request; a wording/consistency risk.
- **Previous suspicions addressed?**:
  - All 7 manuscript-writing recommendations from iteration 3 (A: framing, B: interpretive paragraph, C: random-set foregrounding, D: three-way distinction, E: verb sweep, F: caveats in abstract, G: results reorganization): **YES all substantially addressed**. Reviewer confirms: "Yes, my iteration-3 recommendations were actually addressed, not merely papered over."
  - C3_v2 tighter wording adopted: **YES accepted**. "Associated with reliable prefix-level target-emotion score increases under additive activation injection" is appropriately narrow; "limited/partial specificity" is the right downgrade.
  - C2 two-way interpretation of positive-Δ anomaly: **YES accepted**. "The right interpretation, and importantly you do not oversell disambiguation."
  - Falsification title: **YES accepted**. "The title is strong. It is sharper, more honest, and more memorable than the original framing likely was."
- **Unresolved (carried forward)**:
  - "Localization succeeds" wording must consistently mean coarse/stable/discriminative emotion-relevant, not precise/causal/uniquely-privileged. Sweep every paper section.
  - Beyond this loop's scope: stronger null-model / baseline methodology; better causal intervention design; stronger transfer across models; stronger theory of the failure mode; cleaner decomposition of prefix vs generation-time interventions.
- **Patterns**:
  - **The remaining gap is rhetorical, not experimental.** Reviewer explicitly says: "there is no obvious back-edge experiment I think you need inside this loop's scope. The remaining work is rhetorical precision and consistency, not new analysis."
  - Path to READY: one more final scope-control pass ⓪ on "localization" wording. Reviewer commits: "If you do that carefully, I would likely move this to READY in the next iteration."

## Iteration 5 — Score: 6/10, Verdict: READY

- **New suspicions**: none (final cautionary notes for the paper, not iteration-loop blockers):
  - Do not let the abstract drift back into celebratory language.
  - Keep C1/C2/C3_v2 logically decoupled (stable localization does not imply causal, causal weakness does not fix as implementation).
  - Do not oversell fold-stability as mechanistic truth.
  - Keep the random-top-k result visually prominent (do not bury as caveat).
  - Open-generation failure must remain front-and-center.
- **Previous suspicions addressed?**:
  - C1 "coarse localization" scope-control: **YES accepted with the strongest language yet**. Reviewer: "This is the right fix. Your new C1 wording does exactly what I asked for... it distinguishes stable/discriminative component-set discovery from privileged cardinality identification from fine-grained causal ranking. That is the cleanest formulation you've had across the loop."
- **Unresolved (carried forward)**: none within loop scope. Beyond-loop items (stronger null-model methodology, alternative operators, multi-model transfer, mechanistic theory, formalization of scoring-time vs trajectory-time) are for the next paper / next round of research.
- **Patterns**:
  - **The loop terminated correctly**: five iterations, two back-edge actions (② + ③), three ⓪ narrative refinements. Successfully pivoted from "verification" framing to "falsification study" framing while preserving the honest positive findings.
  - **Reviewer verdict verbatim**: "READY for submission as a falsification / stress-test / negative-results paper, not as a triumphant circuit-discovery paper."
