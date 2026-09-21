# Captured-Behavior Report — Emotional Framing in Prompts as a Weak, Input-Dependent Signal

**Direction**: (empty — behavior/claims sourced from `task.md`; landscape context from `idea-stage/LANDSCAPE.md`)
**Behavior-source**: given
**Mechanism**: discovery
**Date**: 2026-07-13
**Pipeline**: research-lit → faithful behavior capture (task.md) → mechanism-explore (strategy load, no artifact) → research-refine-pipeline
**Resource-fidelity**: cost-aware (this is `given`+`discovery`, not the reproduction combo `given`+`given`; `resource_fidelity: strict` NOT stamped)

---

## Executive Summary

`task.md` states four faithfully-extracted, individually-verifiable claims about how six-basic-emotion static prefixes affect LLM task accuracy: (C1) prefixes produce only small input-dependent shifts; (C2) effects are larger on socially grounded tasks than on math/factual QA; (C3) no single emotion consistently wins and stronger wording does not proportionally help; (C4) an adaptive per-query policy (EmotionRL) beats fixed prefixes / neutral. Primary evaluation is fixed by `task.md` to **Qwen3-14B on GSM8K** (mathematical reasoning — the "small-effect" side of the C2 contrast); C2 requires a socially-grounded companion dataset. The mechanism direction (chosen from `/mechanism-explore` under `MECHANISM=discovery`) is **Location → Causal Intervention** — locate a low-dimensional residual-stream frame direction / small head set, then verify causally with patched/steered controls; C1–C4 provide the behavioral scaffolding this mechanism localizes.

## Literature Landscape

See `idea-stage/LANDSCAPE.md` (9 retrieved papers; six structural gaps G1–G6; mechanic-db unavailable this session; two WebSearch responses voided by the project policy filter targeting arXiv 2604.02236 — expected). Highlights used for claim capture:

- **EmotionPrompt** (Li et al., arXiv:2307.11760) — the phenomenon we are testing; their headline numbers use per-task max over 11 stimuli, so our capture reports *distributions* per emotion.
- **FormatSpread** (Sclar et al., arXiv:2310.11324, ICLR 2024) — the noise-floor benchmark C1 must clear.
- **Persona is a Double-edged Sword** (Kim et al., arXiv:2408.08631) — persona prefixes hurt on 7/12 Llama-3 reasoning datasets, and per-instance selection beats fixed persona; direct precedent for C1 + C4 pattern.
- **Directional Stimulus Prompting** (Li et al., arXiv:2302.11520) — small-policy + RL over discrete stimuli; direct architectural precedent for C4 (EmotionRL).
- **EmotionDecode** (Li et al., arXiv:2312.11111, ICML 2024) — only prior mechanism story ("dopamine analogy"); does not localize to concrete internal objects. Our `MECHANISM=discovery` direction (Location → Causal Intervention) targets exactly this gap.
- **Affect-as-information** (Schwarz & Clore 1983) — psychological prior predicting the socially-grounded > math ordering of C2.

## Claims to Verify

### Claim C1: Static emotional prefixes cause only small, input-dependent accuracy shifts

**Original (verbatim excerpt from task.md, lines 7–8):**
> Across diverse task domains, static emotional prefixes change LLM accuracy only by small, input-dependent amounts and do not behave as a general-purpose enhancement.

**Extracted statement**: For each fixed static emotional prefix drawn from the 13-way action space {6 basic emotions × 2 intensities} ∪ {neutral}, the per-item accuracy delta vs the neutral baseline on a given (model, task) pair has (i) small mean magnitude, (ii) high per-item variance, and (iii) no systematic direction that generalizes across tasks. Concretely: `|mean_delta_accuracy| ≤ small-effect threshold (Cohen's h ≤ 0.2 or ≤ 3 percentage points on GSM8K exact-match)` **and** the item-level `sign(delta)` is not consistent across items — i.e. the prefix helps some items and hurts others.

**Hypothesis**: H1 — Static emotional prefixes act as an input-conditioned perturbation whose mean effect is within (or comparable to) the format-perturbation noise floor of Sclar et al. (2310.11324), not a general-purpose enhancer.

**Measurable predicate**: On Qwen3-14B evaluated on GSM8K (primary), for each of the 12 emotion×intensity prefixes (both human-written and LLM-generated wording) and neutral, compute (a) macro accuracy per (prefix, wording-source), (b) per-item Δaccuracy vs neutral, (c) mean|Δ| and its bootstrap 95% CI, (d) sign-consistency of Δ across items. **C1 holds** if the mean|Δ| for every prefix is ≤ the format-perturbation noise floor established on the same items with neutral-preserving wording paraphrases, **and** the fraction of items where Δ > 0 stays within [0.4, 0.6] for every prefix (no systematic help/hurt).

**Expected direction**: `mean_delta` ≈ 0 with wide per-item CI; the *distribution* of per-item Δaccuracy, not the point estimate, is what supports the claim.

**Resources (preferred; cost-aware — not binding)**: model: **Qwen3-14B** (hard-pinned by task.md); dataset: **GSM8K** test split (primary; hard-pinned by task.md); prefixes: 6 emotions × 2 intensities × 2 wording sources (human, LLM-generated via the dmxapi `gpt-5.4` service in task.md) + neutral = **26 conditions** plus neutral; used_n: full GSM8K test (1319 items) if the 10-GPU-h budget covers it — resolve concretely in Phase 4.5.

**Status**: pending verification
**Notes**: Split from `task.md` §Claim ¶1. The C1 predicate is stated as *distributional*, not as a null-hypothesis significance test on a single number — this respects task.md's phrasing "small, input-dependent amounts" and avoids collapsing per-item variance into a single mean that could hide the input-dependence.

---

### Claim C2: Effects are larger on socially grounded tasks than on math/factual QA

**Original (verbatim excerpt from task.md, line 9):**
> The effect of emotional framing is most pronounced on socially grounded tasks (interpersonal/social reasoning), where the emotional context interacts meaningfully with task content; on tasks like mathematical reasoning or factual QA the effect is markedly smaller.

**Extracted statement**: The *magnitude* of the emotional-prefix effect — measured as the spread of per-item Δaccuracy across the 12 emotional prefixes vs neutral (equivalently, the peak-to-peak or bootstrap 95% inter-emotion spread of accuracy) — is systematically larger on socially grounded tasks (interpersonal / social inference) than on mathematical reasoning or factual QA.

**Hypothesis**: H2 — Emotional wording carries decision-relevant informational content for socially grounded judgments (per Schwarz & Clore's affect-as-information) but is decision-irrelevant for symbolic math, so the model's task-relevant computation gates the prefix influence in.

**Measurable predicate**: Rank at least one socially-grounded task (from the task.md verify-stage candidates — **SocialIQA** as primary social; also acceptable: BoolQ-interpersonal subset) against GSM8K (math) and a factual-QA benchmark (MedQA or OpenBookQA) using the same 12 emotional prefixes on the same model. **C2 holds** if, on the socially-grounded task, the inter-emotion spread of per-condition macro accuracy is at least *k*× that on GSM8K (target: `spread_social ≥ 2 × spread_math`), with the ordering `spread_social > spread_factual > spread_math` observed in ≥ 2 of 3 comparisons.

**Expected direction**: `spread_social > spread_math` (larger); ordering `social > factual ≈ math` expected but the strict ordering across all three is a secondary prediction.

**Resources (preferred; cost-aware)**: model: **Qwen3-14B** (primary; the task.md verify variants — Llama-3.3-70B-Instruct, DeepSeek-V3.2 — enter at `/auto-verify` stage, not here); datasets: **GSM8K** (math, primary), **SocialIQA** (social, primary companion), one factual-QA benchmark (**MedQA** or **OpenBookQA**); prefixes: same 26 conditions as C1; used_n: cost-aware subset to be sized in Phase 4.5 to satisfy statistical power (target ≥ 300 items per task per condition; resolve against the 10-GPU-h budget).

**Status**: pending verification
**Notes**: task.md hard-pins Qwen3-14B + GSM8K for the primary evaluation but does not name the socially-grounded companion. Companion selection from the verify-stage candidate list (SocialIQA) is preferred as it is already sanctioned by task.md's own resource list and matches the "social inference" definition. Recorded here for Phase 4.5 to resolve.

---

### Claim C3: No single basic emotion consistently wins; stronger wording does not proportionally help

**Original (verbatim excerpt from task.md, line 11):**
> No single basic emotion among {happiness, sadness, fear, anger, disgust, surprise} provides a consistent benefit across all models and tasks; stronger emotional wording does not yield proportionally larger gains.

**Extracted statement**: (C3a — no consistent winner) There is no emotion e ∈ {happiness, sadness, fear, anger, disgust, surprise} whose macro-accuracy on Qwen3-14B is strictly greater than neutral on *every* task family evaluated. (C3b — no monotone intensity) For each emotion e, the accuracy delta at intensity-2 is not systematically larger in the same direction than at intensity-1 (no monotone dose-response of *wording strength*).

**Hypothesis**: H3 — Emotion-frame effects are asymmetric across the six emotions and non-monotone in wording intensity, consistent with the affect-as-information view that different affect types bias distinct cognitive styles (heuristic vs analytic vs vigilant) that interact differently with each task family.

**Measurable predicate**:
- **C3a**: Across the (task, emotion) grid used for C1/C2, there is no single emotion that is the argmax on every task-family (math / social / factual). Equivalently, the "best emotion" identity flips at least once between task families in the fixed grid.
- **C3b**: For each of the 6 emotions, the paired difference `Δ_intensity-2 − Δ_intensity-1` (with per-item pairing) has 95% CI that either straddles 0 or reverses sign for at least half of the emotions on GSM8K, and does not consistently satisfy `Δ_intensity-2 > Δ_intensity-1 > 0` for any emotion across all task families.

**Expected direction**: (C3a) task-argmax identity varies across task families (i.e., the argmax "moves"). (C3b) `Δ_intensity-2 − Δ_intensity-1` non-monotone / non-significant for the majority of emotions.

**Resources (preferred; cost-aware)**: same as C1/C2 grid; no new data required. The intensity comparison uses paired items across the intensity-1 vs intensity-2 conditions of the same emotion with the same wording-source (human or LLM).

**Status**: pending verification
**Notes**: Split from `task.md` §Claim ¶3 into C3a (no consistent winner across tasks) and C3b (no monotone intensity gain). The split is faithful because task.md's sentence bundles two distinct testable predicates ("no consistent winner" ∧ "stronger wording ≠ larger gain"). Both retain the exact wording of `task.md` in the source excerpt above.

---

### Claim C4: Adaptive per-query policy (EmotionRL) beats fixed prefixes and neutral

**Original (verbatim excerpt from task.md, line 13):**
> An adaptive policy that selects the emotional prefix per query (EmotionRL) yields more reliable accuracy gains than any fixed emotional prefix or neutral baseline.

**Extracted statement**: A per-query policy π that selects a prefix from the 13-way discrete action space {6 emotions × 2 intensities} ∪ {neutral} — trained by SFT / RL against downstream task accuracy on Qwen3-14B — yields macro accuracy strictly greater than (a) the neutral baseline and (b) every fixed emotional prefix from the 12-way emotional grid, on held-out items of the primary task (GSM8K). "More reliable" means the policy's advantage is stable across multiple seeds and across held-out splits.

**Hypothesis**: H4 — Because the emotional-prefix effect is input-dependent and heterogeneous across items (per C1) and task-dependent across families (per C2), a policy that conditions the prefix choice on the query outperforms any fixed prefix that pays the item-level cost of one-size-fits-all.

**Measurable predicate**: Train EmotionRL π_θ (small policy: BERT-family / Llama-3.2-1B-class classifier over the 13-way action space; SFT on best-per-item labels then RL against Qwen3-14B GSM8K exact-match reward, following the Directional Stimulus Prompting architecture — arXiv:2302.11520). Evaluate on a held-out GSM8K split. **C4 holds** if `acc(π) > acc(neutral)` and `acc(π) > acc(e*)` for the fixed-emotion argmax e*, both with a lower 95% CI above zero across ≥ 3 seeds. Optionally report the same on SocialIQA held-out (secondary).

**Expected direction**: `acc(π) > max_e acc(e_fixed) ≥ acc(neutral)`.

**Resources (preferred; cost-aware)**: model (frozen): **Qwen3-14B** (task.md-pinned); policy backbone: a small tunable model (to be resolved in Phase 4.5 — candidate: Llama-3.2-1B or a BERT-class encoder; not counted as a "hard-pinned" model since task.md names Qwen3-14B as the *evaluation target*, not the policy backbone); dataset: **GSM8K** train/val/test partition (primary; may repurpose GSM8K's official train split for policy training, holding out the test set for the final measurement); reward: Qwen3-14B GSM8K exact-match; used_n: to be sized in Phase 4.5 against the residual GPU budget after C1/C2/C3 runs.

**Status**: pending verification
**Notes**: The EmotionRL architecture is *stated* in task.md but not fully specified; Phase 4.5 must fix the policy backbone, training objective, and evaluation protocol. Kept faithful — the claim is "adaptive > fixed / neutral", not any specific policy architecture.

---

## Mechanism direction (from Phase 1.75 `/mechanism-explore` load)

Because `MECHANISM=discovery`, no specific mechanism family is committed here — the experiment stage's `/mechanism-skills` routing picks the concrete family. Strategy captured for the plan:

- **Directions to pursue (in execution order)**: (1) **Location** — probe/attribution to find residual-stream directions and attention heads that carry the emotional frame on Qwen3-14B early layers; (2) **Causal Intervention** — activation patching + steering (dose-response on the located direction) with matched-length / matched-frequency neutral controls to check specificity.
- **Optional add-on**: **Unit Interpretation** (SAE / vocab projection) to decode *what* the located direction encodes (valence? arousal? social-relevance gating?) — only if Phase 1 mechanism results yield a clean shortlist.
- **Deliberately not pursued** — **Tuning & Editing** (EmotionRL is input-space, not weight-space); **Formation Tracing** (training-time genesis out of scope + expensive); **Decision Auditing** (no reliability audit requested).
- **Altitude**: the mechanism claim is "some low-rank residual direction / small head set on Qwen3-14B carries the frame and causally drives the per-item Δaccuracy pattern" — the specific layer/head identities are discoveries for the experiment stage, not fixed here.

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering C1–C4 + the mechanism_strategy)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies; no M0 gate since BEHAVIOR_SOURCE=given)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md` (plan-level; downstream `/auto-experiment` updates it in place)

## Next Steps

- [ ] `/mechanism-skills` to route the Location → Causal Intervention strategy to a concrete family + submethod (Workflow 1.25)
- [ ] `/auto-experiment` to implement and run the verification suite covering C1–C4 + mechanism milestones (Workflow 1.5)
- [ ] `/auto-verify` to stress-test the verified claims under task.md's variant candidates (Llama-3.3-70B-Instruct, DeepSeek-V3.2; MedQA / BBH / BoolQ / OpenBookQA / SocialIQA) (Workflow 1.75)
- [ ] `/auto-iteration-loop` to iterate until reviewer-ready (Workflow 2)
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain

## Recommended

**Recommended:** #1 — Faithful behavior capture C1–C4 + Location → Causal Intervention mechanism strategy on Qwen3-14B / GSM8K (primary), with SocialIQA + MedQA/OpenBookQA as C2 companions
