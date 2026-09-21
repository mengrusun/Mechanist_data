# Reviewer Memory

## Iteration 1 — Score: 3/10, Verdict: not ready

- **New suspicions**:
  1. Intervention is too strong / poorly normalized — α not in σ_proj units; severe collapse likely a norm-mismatch artifact rather than meaningful nulling.
  2. "Language subspace" appears functionally entangled with reasoning, not a removable nuisance — the core scientific hypothesis (V_lang suppression → better multilingual reasoning) looks false on this model/task.
  3. C2 needs capability-collapse disentanglement — must separate language-identity suppression from generic arithmetic degradation (per-α independent capability metric, e.g., English-only or arithmetic-format sanity).
  4. Random-direction controls are the strongest positive signal; they should become the evidentiary backbone of the revised paper.
  5. C1 rank/orthogonality issue — effective rank capped by n_langs=11; complement still carries 33-49% language accuracy → decomposition is incomplete.
  6. C4 is likely unrecoverable under budget (5-8 GPU-h needed for full M4b+full LoRA+n=250; only 2-3 GPU-h remain).
  7. Paper framing risk — current thesis overclaims; the honest paper is negative-result + specificity.
  8. Variant evidence should support the REVISED (specificity) claim, not the original (monotone-suppression) claim.
  9. Non-collapse-window analysis is the scientifically relevant region — near-zero α, not the extreme collapse points.

- **Previous suspicions addressed?**: n/a (first iteration)

- **Unresolved (carried forward)**: All 9 suspicions above are open.

- **Patterns**: The unifying pattern the reviewer flags is **methodological over-forcing**: α=−1 lands in a capability-collapse regime for both the main experiment and the variant, and each attempt is missing the counterfactual controls (random-direction sweep, independent capability metric, σ_proj-scaled α) that would let a reader distinguish "V_lang suppression damages reasoning specifically" from "any hidden-state perturbation of this magnitude damages reasoning". The reviewer's core recommendation across all claims is: **narrow the operating range, add the specificity controls, and re-frame around what the data actually support (V_lang is language-identity but reasoning is entangled with it, not orthogonal to it).**

## Iteration 2 — Score: 6/10, Verdict: almost

- **New suspicions**: none — the iteration ①-② re-audit brought new evidence (random-subspace α-sweep) that resolved the largest evidentiary gap flagged in iteration 1, and the ③ narrowings honestly re-align the C2 / C4 claims with the observed data. No new suspicions rose to the surface.

- **Previous suspicions addressed?**:
  - #1 (α not in σ_proj units): partially addressed — labeled as WARN future-work in re-audit. Presence of the random control makes σ_proj non-decisive (the random arm establishes specificity in raw units too). Not fully closed, but no longer FAIL-worthy.
  - #2 (V_lang functionally entangled with reasoning): fully addressed — now explicitly the diagnosis in narrowed C2. The paper thesis pivots from "suppress V_lang to improve reasoning" to "V_lang carries language identity, reasoning shares that subspace, so suppression destroys reasoning."
  - #3 (capability-collapse disentanglement): fully addressed — random-subspace α-sweep at same rank/site/seed is on disk; non-collapse-window comparison shows clean specificity (V_lang costs 8-20 pp at |α|≥1; random costs 0 pp).
  - #4 (random-direction controls as evidentiary backbone): fully addressed for C3 variant; main-experiment M3 already had a matched random-subspace α-sweep. Now paired across models.
  - #5 (C1 rank/orthogonality issue): held as an Open Item — recorded in C1's caveats. Would take a separate C1-focused re-run (increased n_probe, LDA vs SVD comparison) to close.
  - #6 (C4 unrecoverable under budget): confirmed — addressed by ③ narrowing rather than a re-run. Section 8 lists the upgrade path if budget resets.
  - #7 (paper framing overclaim): addressed for C2 and C4 via ③ narrowings; C1 held with the "identifiability yes / orthogonal-decomposition no" split; C3 refutation is now formally credited across models.
  - #8 (variant evidence should support revised claim): addressed — C3 variant now formally credits the refutation (PASS by construction).
  - #9 (non-collapse-window analysis): addressed — re-audit explicitly interprets α ∈ [-1.5, +0.25] and labels α ≥ +0.5 as OOD forcing.

- **Unresolved (carried forward)**:
  - #5 partial — C1's orthogonal-decomposition failure and rank cap remain a real caveat; the story is "V_lang identifiable but not the whole story of language representation in the model." The narrowed C2 already carries this diagnosis. Full closure would need a separate C1-improvement study.
  - C1 Stage-2 model-swap (max_verify_claims_cap deferral) — deferred to `/auto-verify C1 -- resume: true` in a follow-up invocation.

- **Patterns**: The unifying pattern this iteration surfaces is that the honest, publishable result of this run is a **negative-result-plus-specificity study** on Qwen-3-4B-Thinking + MGSM. V_lang exists as a direction-specific effect (random control preserves baseline where V_lang does not), but suppressing it does not free reasoning; reasoning is co-embedded with language identity in this subspace. This is a legitimate mechanistic finding, and the narrowed C2 / C4 + PASS on C3's refutation together tell a coherent story without overclaiming.
