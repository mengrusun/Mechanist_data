# Reviewer Memory

## Iteration 1 — Score: 7/10, Verdict: almost

- **New suspicions**:
  - **C2 overclaim risk**: manuscript language may still imply family-wide Fisher localization. Must be revised to say "at pythia-1b and pythia-2.8b" — not "in Pythia LMs" as a family-general statement. This is the main blocker for READY.
  - **410m interpretation nuance**: at pythia-410m, the pattern is "diffuse or non-unique PB mechanism at small scale, not absence of causal signal" — the K=20 Fisher head set does produce a behaviorally effective drop (tgt_drop = 0.3855 ≥ 0.30) but is not statistically distinct from random K=20 sets. Important distinction that must be presented precisely.
  - **Off-target inversion at 410m**: AB accuracy improves by 0.308 abs when PB Fisher heads are ablated. Could indicate cross-frame interference/inhibition in smaller models; interesting but should be presented cautiously, not overinterpreted (baseline AB is at chance so the signal is fragile).
  - **C4 pre-head classifier shortcut**: L*=1 classifier likely leverages explicit prompt tokens ("believes", "thinks", "In reality"). This does not invalidate the intervention result under task.md's fixed-prompt constraint, but it weakens any claim that the router reads deep latent belief state rather than superficial frame cues. The paper should call this out explicitly.
  - **C3 strongest deferred follow-up**: pythia-2.8b intermediate checkpoints ARE available and would materially strengthen the temporal-emergence story. Recommend as a follow-up round (out-of-scope for the current iteration loop).
  - **C1 descriptive strength**: fine as descriptive evidence; because only three scales are available and PB shows a non-monotonic dip, wording should avoid grand developmental laws beyond what the data support.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all six above until iteration 2 confirms the C2 scope narrowing and paper-side caveats have been recorded correctly.
- **Patterns**:
  - **Scale-dependent circuit concentration**: The Fisher-vs-random gap widens dramatically as model size grows (1.7× → 86× for PB; 6.7× → 35× for AB across pythia-1b → pythia-2.8b). This is the core mechanistic finding and should be foregrounded, with the 410m negative reframed as a data point ON this scaling trend rather than as a robustness failure.
  - **AB emerges sharply, PB matures gradually**: Both behavioral (C1) and formation-window (C3) evidence converges on "AB step-function at 1B, PB non-monotonic across scale + gradual across pretraining." Both claims should be written in mutually-reinforcing language.

## Iteration 2 — Score: 8/10, Verdict: ready

- **New suspicions**: (none new — the residual concerns from iteration 1 are all now downgraded to non-blocking presentation issues)
- **Previous suspicions addressed?**:
  - **C2 overclaim risk**: YES — narrowing to {pythia-1b, pythia-2.8b} in C2_v2 fixes the main correctness problem. The 410m nuance is now correctly distinguished from a robustness failure.
  - **410m interpretation nuance**: YES — the C2_v2 wording explicitly separates "behaviorally effective Fisher ablation" from "lack of statistical uniqueness vs random sets," which is the right level of precision.
  - **Off-target inversion at 410m**: not directly addressed this iteration; carried forward as "cautious exploratory observation only, do not overinterpret."
  - **C4 pre-head classifier shortcut**: recorded in Open Items / paper caveats; carried forward as a wording discipline check.
  - **C3 strongest deferred follow-up**: recorded in Open Items as recommended follow-up round; carried forward.
  - **C1 descriptive strength**: recorded in Open Items / paper caveats; carried forward as a wording discipline check.
- **Unresolved (carried forward to potential future rounds)**:
  - C4 wording must consistently present the pre-head classifier as a **frame router under fixed prompts**, not strong evidence of latent belief-state decoding.
  - 410m AB inversion: keep as a cautious exploratory observation only.
  - C3 follow-up gap: pythia-2.8b intermediate checkpoints remain the most valuable future strengthening axis.
  - C1 scope caution: keep claims descriptive and Pythia-specific; avoid broad developmental-law language.
- **Patterns**:
  - **Scale-dependent concentration of causal circuitry** is now the clearest mechanistic takeaway — smaller models show diffuse/redundant support, larger models show sharp Fisher-vs-random localization. The C2_v2 narrowing FOREGROUNDS this scale-dependence as a positive scientific finding rather than hiding it as a robustness failure.
  - **AB emerges sharply, PB matures more gradually/non-monotonically** remains the best synthesis across C1 and C3.

