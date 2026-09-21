# Reviewer Memory

## Iteration 1 — Score: 6/10, Verdict: almost

- **New suspicions**:
  - The alignment objective at α=1.0 is probably over-regularizing toward THINGS-like similarity structure and damaging generic semantic transfer.
  - C1b may reflect a mismatch between the claim wording ("monotonic hierarchy") and what the teacher naturally induces; likely a paper-claim problem, not an implementation bug.
  - C2b fine-level evaluation is underpowered or malformed; "0 usable pairs" suggests a stratification/construction issue serious enough to avoid strong fine-grained claims.
  - The uncertainty component in C3 is statistically fragile; easy to oversell if not carefully worded.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all four items above.
- **Patterns**:
  - Strong, consistent evidence for human-similarity alignment benefits (C1a/C2a/C2c/C3).
  - Weakest part of the story is utility preservation under full-strength finetuning.
  - Dataset substitution was handled honestly, but it weakens any broad downstream/OOD conclusions and should be foregrounded as a limitation.
  - One targeted training change (lower α, possibly plus LoRA) is the highest-value next experiment under the remaining budget.

## Iteration 2 — Score: 5/10, Verdict: not ready

- **New suspicions**:
  - The main bottleneck is **adaptation scope**, not loss weight: any full-backbone finetuning may erase DINOv2's generic decision structure even when the alignment objective is weak.
  - C4a may be fundamentally misframed for this training recipe: "semantic alignment" and "one-shot classification utility" are not naturally co-monotonic under end-to-end KD finetuning.
  - If LoRA succeeds, the original paper likely confounds "objective works" with "optimization procedure is safe"; if LoRA fails too, then the objective itself may be in tension with generic class separation.
  - C4b's positive evidence may be **task-selective** rather than general OOD robustness: BREEDS gains could reflect taxonomy-compatible reshaping while broader transfer degrades.
- **Previous suspicions addressed?**:
  - α over-regularizes → **rejected as primary cause** (α=0.1 kept alignment and did not recover utility).
  - C1b wording/teacher mismatch → **not addressed** (still a claim-framing issue).
  - C2b fine-level malformed → **not addressed** (still a serious weakness).
  - C3 uncertainty statistically fragile → **not addressed** (still needs careful wording).
- **Unresolved (carried forward)**: C1b, C2b, C3 wording items above.
- **Patterns**:
  - C2a strengthens slightly under α=0.1; the alignment effect is robust to KD-weight scale.
  - C4a remains clearly negative; α tuning is not the fix.
  - C4b remains genuinely mixed: BREEDS positive, other splits negative; conditional success.
  - Iteration-1 α↓ suggestion was useful as a **disambiguation experiment** but the durable fix is **parameter-efficient adaptation** (LoRA).

## Iteration 3 — Score: 6/10, Verdict: not ready → recommending type ③ CLAIM REWRITE on C4a

- **New suspicions**:
  - The downstream effect is better described as **representation reorganization toward coarse/taxonomic structure** than as generic semantic utility preservation.
  - Fine-grained ImageNet one-shot transfer may be intrinsically antagonistic to this alignment objective under the current recipe, even with parameter-efficient adaptation.
  - The apparent OOD gains likely depend on benchmark-label structure matching the induced geometry; BREEDS benefits, ImageNet-slice/fashion-style tasks do not.
- **Previous suspicions addressed?**:
  - Adaptation SCOPE is the bottleneck: **partially yes** (LoRA clearly improves over full tuning; scope not the whole story since C4a still negative).
  - C4a fundamentally misframed for this recipe: **yes**, confirmed across three regimes.
  - C4b positive evidence task-selective: **yes**, LoRA strengthens the same 2/5 without broadening.
- **Unresolved (still carried)**: C1b wording, C2b fine-level construction, C3 uncertainty framing — all narrative-level fixes for paper writing, not iteration.
- **Patterns**:
  - Full-backbone finetuning is too destructive; parameter-efficient tuning is strictly better for preserving transfer.
  - α reduction did little; scope reduction mattered more.
  - Gains and recoveries concentrate on easier/coarser or taxonomy-compatible evaluations; fine-grained ImageNet-class distinctions remain the persistent failure mode.
  - **Further compute is unlikely to rescue the original broad utility-preservation claim; the paper now needs sharper claim discipline more than more training.**

## Iteration 4 — Score: 7/10, Verdict: almost → recommending final type ③ CLAIM REWRITE on C4b

- **New suspicions**: none
- **Previous suspicions addressed?**:
  - C4a broad utility-preservation claim was misframed: **YES, RESOLVED** by C4a_v2 rewrite.
  - Adaptation scope, not α, is the main bottleneck: **YES, REINFORCED**; C4a_v2 properly centers LoRA as best regime.
  - C4a fundamentally misframed for this recipe: **YES, RESOLVED** narratively (trade-off/limitation).
  - C4b task-selective, not general OOD: **NO, still unresolved in wording** — evidence remains selective while claim remains universal.
  - C1b wording mismatch: NO, still a caveat item.
  - C2b fine-level 0-pair construction artifact: NO, still a caveat item.
  - C3 uncertainty framing may be oversold: NO, still a caveat item.
- **Unresolved (carried)**: C4b wording, C1b wording, C2b construction, C3 uncertainty framing (all narrative-level).
- **Patterns**:
  - The work is now mostly bottlenecked by CLAIM DISCIPLINE, not missing compute.
  - The empirical picture is coherent: alignment helps some robustness settings, harms one-shot fine-grained utility, and LoRA attenuates but does not remove the trade-off.
  - Remaining risk is concentrated in broad wording, especially universal quantifiers ("every split", "strictly improves").

## Iteration 5 — Score: 8/10, Verdict: ready → STOP (three-dimensional STOP satisfied; loop terminates)

- **New suspicions**: none
- **Previous suspicions addressed?**:
  - α over-regularizes: YES (rejected in iter 1; nothing revives it).
  - Adaptation SCOPE bottleneck: YES, PARTIALLY RESOLVED (LoRA is the right lever; not universal recovery).
  - C4a overclaim: YES, RESOLVED (C4a_v2 narrows to honest trade-off).
  - C4b universal-quantifier wording: YES, RESOLVED (C4b_v2 narrows to selective BREEDS-taxonomy positive).
  - C1b narrative caveat: NO (non-blocking; manuscript writing item).
  - C2b construction artifact caveat: NO (non-blocking; manuscript writing item).
  - C3 uncertainty framing: NO (non-blocking; manuscript writing item — should avoid stronger certainty language than warranted).
- **Unresolved (carried to paper writing, not iteration)**: C1b, C2b, C3 narrative caveats.
- **Patterns**:
  - Reproduction is strongest when reporting regime-specific, scope-limited effects rather than universal benefits.
  - Most remaining weakness is narrative inflation risk, not empirical invalidity.
  - Current version is acceptable because negative results and benchmark substitutions are surfaced explicitly instead of being papered over.
