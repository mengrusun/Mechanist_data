# Auto Review — Subliminal Learning on Qwen-Image

- **Reviewer LLM**: gpt-5.4 via https://www.dmxapi.cn/v1 (source: shell env, resolved per Reviewer LLM Configuration priority order)
- **Loop config**: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, POSITIVE_VERDICT_TERMS=[ready, almost], AUTO_PROCEED=true, GPU_ID=4,5,6,7
- **Input buckets** (from `verify/VERIFY_REPORT.md`):
  - verify_passed = [C1]
  - verify_failed = []
  - verify_inconclusive = []
  - verify_zero_eligible_variants = []
  - verify_integrity_only = [C2] (`stage2_skip_reason: max_verify_claims_cap`)
- **Budget-context reminder**: 10-h GPU HARD budget already exceeded (~14.2 GPU-h used across main experiment + verify). Loop strongly prefers type-⓪ narrative-only actions.

---

## Iteration 1 (2026-07-16)

### Assessment (Summary)
- **Score**: 6.5/10
- **Verdict**: almost  (canonical — reviewer wrote "almost")
- **Budget after this iteration**: iterations 0/6, claim-reentries 0/2 (no back-edge action ran — reviewer's minimum fix was type-⓪ narrative-only)
- **Key criticisms**:
  - Under-N after filtering (matched N=53/arm) — small statistical footprint
  - Robustness axis is narrow: only model-axis (rank 16→8), no dataset-axis or method-axis stress test
  - **Same-judge coupling**: gpt-4o used for both filter and evaluation → possible circularity / shared bias
  - Single-concept ("banana"), single-model-family (Qwen-Image), single-recipe (LoRA-denoising-SFT) scope → generality untested
  - Extreme raw teacher P(banana)=0.903 → surviving teacher-arm images may be a highly unusual conditional slice
  - Mechanism unresolved beyond one family (LoRA-SVD-additive-steering) — refutation is scoped, not universal

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Overall score

**Overall: 6.5/10**
- **(a) Behavioral contribution, C1:** **7.5/10**
- **(b) Mechanism contribution, C2:** **4.5/10** as a contribution, **7/10** for honesty/integrity

This is **potentially publishable as a focused empirical finding / workshop-to-main-venue borderline paper**, but **not a slam-dunk top-tier paper** in its current form. The reason is simple: the **headline behavioral result is genuinely novel and fairly clean**, but the paper is **thin on external validity and mechanism**, and the training signal is based on a **very small post-filter dataset (N=53/arm)** with **only one meaningful robustness axis verified**.

The strongest thing here is that the authors did **not** overclaim mechanism and reported a real negative result. That improves credibility.

---

## 1. Contribution scoring

### (a) Positive novel behavioral finding: subliminal transfer in diffusion via denoising SFT
**Score: 7.5/10**

Why this is strong:
- The core result is **interesting and novel**: transfer of a hidden preference-like attribute from teacher to student in an **image diffusion setting**, mirroring a known text-LLM phenomenon.
- The design hits the important precondition: **shared base initialization** between teacher and student.
- The filtering story is unusually clean:
  - overt banana images removed from **both** arms,
  - **residue re-scan = 0**,
  - matched-N used.
- The effect is not tiny:
  - **mean gap = +0.169**,
  - **CI [0.110, 0.246]**,
  - **6/7 seeds** in the expected direction,
  - verify run at rank-8 reproduces **89%** of the magnitude.

Why it is not higher:
- **N=53 per arm after filtering** is very small and obviously induced by an extreme teacher bias rate.
- Robustness is narrow: only **model-axis** variant tested, effectively one lightweight architecture perturbation.
- Evaluation depends on a **single judge family (gpt-4o)** for both filtering and banana detection; that creates a possible **same-judge circularity / shared bias** concern.
- The task is narrow: one concept ("banana"), one model family (Qwen-Image), one teacher adaptation recipe, one student adaptation recipe.

So: **good result, real contribution, but not broad enough yet for an easy top-tier accept**.

---

### (b) Honest negative mechanism finding: LoRA-SVD location + additive residual steering refuted
**Score: 4.5/10 as science contribution**

This is **not** a strong mechanism result in the positive sense, because the intervention failed:
- location evidence is correlational,
- causal test is flat,
- random-direction control is also flat.

That said, I would score the **research integrity** of this section much higher:
- they committed to a mechanism family,
- they tested it,
- they included **negative α**, dose sweep, random-direction control, and a wider 9-block window,
- and they **accepted refutation** rather than storytelling around it.

That is good science. It just does **not** elevate the paper much on mechanism.

---

## 2. PASS claim review: C1

### Narrative assessment
**C1 is supported as stated.** The claim is a clean behavioral claim, not a mechanistic one, and the reported numbers meet the declared thresholds with margin.

The most persuasive aspects:
- **residue = 0** after filtering in both arms,
- **effect remains positive across seeds** with strong majority,
- **verify pass** shows the effect is not a one-off artifact of the exact main setting,
- control arm is matched and identically filtered.

The main caveat that absolutely must be surfaced in the paper:
1. **Under-N after filtering**: only **53 images per arm** survive equalized matching. This is not fatal, but it is a real limitation.
2. **Only model-axis verification**: robustness was checked only via a rank-8 variant. No dataset-axis or method-axis stress tests.
3. **Judge dependence**: the same judge family appears central to both residue filtering and evaluation.
4. **Single-concept / single-model-family scope**: banana in Qwen-Image is a compelling proof-of-concept, not yet a general law.

### Numeric consistency check
Claim wording:
- **mean_gap ≥ 0.10**
- **per-seed majority ≥ 4/7**
- **residue = 0**

Numbers on disk:
- **mean_gap = +0.169** → passes threshold comfortably
- **95% bootstrap CI [0.110, 0.246]** → lower bound above 0.10, which is reassuring
- **per-seed majority = 6/7** → passes
- **Wilcoxon one-sided p = 0.008** → consistent with a real across-seed effect
- **residue = 0 both arms** → passes
- **verify robustness = 1.0**, rank-8 variant:
  - **mean_gap = +0.150**
  - **CI [0.113, 0.175]**
  - **3/3 seeds positive**
  - integrity pass

So yes: **the wording matches the numbers.** No inflation detected from the summary you provided.

### Minimum necessary caveat for paper text
The minimum honest sentence the paper needs is something like:

> "The effect is established in a narrow proof-of-concept setting: a single concept (banana), a single model family (Qwen-Image), and a small matched post-filter dataset (N=53/arm), with robustness checked only along a model-axis variant."

That caveat is necessary. Without it, the paper would oversell.

---

## 3. INTEGRITY_ONLY claim review: C2

**C2 should be treated exactly as you framed it: not blocking, but not verified beyond audit.**

My review judgment:
- The current evidence supports **refutation of this specific mechanism family** in the main experiment.
- The lack of Stage 2 swap-test means I would not treat the mechanism section as "fully robustly verified," but that is **not a red flag**, because:
  - the skip reason is explicit,
  - the integrity audit passed,
  - the result is negative rather than load-bearing positive.

Per your instruction, I will **not** propose a back-edge action here. It is enough to note:

> **If compute ever becomes available, C2 would benefit from `/auto-verify C2 --resume true`**, but this is optional and already recorded in Open Items.

That is the right handling.

---

## 4. Is this READY for submission?

**Verdict: almost**

Not "not ready," because:
- there are **no FAIL / INCONCLUSIVE / ZEV** claims,
- the main positive claim is verified and numerically consistent,
- the mechanism negative is honestly reported,
- integrity signals are decent.

Not "ready," because for a **top-venue standard**, I think the paper still needs at least a **presentation-level tightening** to avoid overclaiming:
- make the scope limitation impossible to miss,
- clearly separate **behavior established** from **mechanism unresolved/refuted for one family**,
- foreground judge-dependence and under-N risk.

Since you asked to avoid type-② and type-③ unless methodologically necessary: I do **not** think additional compute is necessary for a submission decision, and I do **not** think claim rewrite is necessary in the sense of changing the claim status. But I do think the **paper framing** must stay disciplined.

If this were my meta-review style summary:
- **Workshop / spotlight / empirical main-track borderline: yes**
- **Strong main-track accept at NeurIPS/ICML: no, not yet**

---

## Brutally honest bottom line

This is **real, interesting evidence** for subliminal transfer in diffusion models. I would not dismiss it as noise or p-hacking from what you've shown. The team did several things right: equal filtering, residue check, matched-N, seed reporting, and an honest failed mechanism attempt.

But the work is also **narrow and fragile-looking**:
- one concept,
- one model family,
- tiny post-filter dataset,
- one evaluation judge family,
- one robustness axis.

So the right framing is:

> **A credible proof-of-concept behavioral finding with limited generality, plus a negative mechanism result.**

If the paper presents it exactly that way, it is defensible. If it tries to sell "general subliminal learning in diffusion models" broadly, I would push back hard.

---

## Minimum concrete fix

Given the compute overrun and your constraints, the **minimum necessary fix is narrative, not experimental**:

1. **Make the scope limitations explicit in the abstract / conclusion.**
2. **State that C2 is a refutation of one mechanism family, not evidence of no mechanism.**
3. **Disclose the judge-dependence and N=53 post-filter bottleneck prominently.**

No new expensive runs are methodologically required for submission. But **careful framing is required**.

---

## Memory update

### New suspicions you want to track
- Possible **same-judge coupling**: gpt-4o is used for both filtering residue and downstream banana evaluation, which could inflate internal consistency.
- The effect may partly reflect **distributional/style residue** rather than a semantically abstract "banana preference" per se.
- Extreme teacher banana rate (0.903 raw) means the surviving teacher examples may be a **highly unusual conditional slice**, not representative "clean neutral fruits."

### Unresolved concerns to carry forward
- **Post-filter matched N=53/arm** is the major statistical and representational limitation.
- Robustness tested on **model axis only**; no dataset-axis or method-axis validation.
- Generality beyond **banana / Qwen-Image / LoRA-denoising-SFT** remains unknown.
- Mechanism remains unresolved: current additive residual steering family appears **negative**, but this does not localize the actual causal pathway.

### Patterns you want the loop to watch for
- Whether future variants keep the effect under **independent judges** or non-overlapping evaluation/filter pipelines.
- Whether subliminal transfer persists for **other concepts** and less extreme teacher adapters.
- Whether effect size collapses when post-filter N increases via a weaker teacher or broader prompt set.
- Whether alternative mechanism probes show **signed dose-response with specificity**, unlike the flat additive steering results.

</details>

### Verify-Passed Claims (brief audit)
- **C1**: consistency check PASS. Reviewer confirmed claim wording ("mean_gap ≥ 0.10, per-seed majority ≥ 4/7, residue = 0") matches on-disk numbers verbatim (mean_gap=0.169, CI [0.110, 0.246] lower bound above threshold, per-seed majority 6/7, Wilcoxon p=0.008, residues=0 both arms, verify robustness=1.0 with rank-8 variant at 89% of main magnitude). No inflation detected. Four paper-side caveats surfaced (see `### Actions Taken` — ⓪).

### Actions Taken (per claim, per type)

- **C1 — type ⓪ narrative-only** — reviewer flagged four scope/limitation caveats that MUST surface in the final paper text (abstract + conclusion + limitations section). No scripts touched, no runs fired.
  - Caveat 1 (Scope disclosure): the effect is established in a narrow proof-of-concept setting — single concept (banana), single model family (Qwen-Image), single teacher-adaptation recipe (LoRA-anchored on 112 pairs), single student-adaptation recipe (LoRA denoising SFT), small matched post-filter dataset (N=53/arm), robustness checked only on the model-axis (LoRA rank 16→8).
  - Caveat 2 (Judge dependence): gpt-4o is used for BOTH residue filtering AND banana-P evaluation → possible same-judge circularity. Recommended disclosure text: "Both the residue-filtering step and the final P(banana) evaluation use gpt-4o. Independent-judge replication is left to follow-up work."
  - Caveat 3 (Under-N): matched N=53/arm is a real statistical limitation induced by an extreme teacher LoRA (raw teacher P(banana)=0.903). Must be disclosed prominently, not buried.
  - Caveat 4 (Conditional-slice framing): the surviving teacher-arm training images are a highly unusual conditional slice (the ~10% of the teacher's generations that did NOT look like a banana to gpt-4o), not a "clean neutral fruit set." This qualifies what "subliminal signal" means here.

- **C2 — type ⓪ narrative-only** — reviewer flagged a scoping caveat that must surface in the paper's mechanism section.
  - Caveat: C2's refutation is **scoped to the LoRA-SVD-derived additive-steering family** — it does NOT constitute evidence that "no DiT component causally carries the banana signal." Alternative mechanism families (activation patching, MLP-path steering, full-LoRA weight-space swap) remain open. Paper must be phrased "we refute mechanism X for family Y" not "no mechanism exists."

### Claim Rewrites (type ③)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]:
  * Upgrade: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
  * main-experiment integrity: pass; warn_source: null
  * Note: recorded here for potential follow-up; not a blocker for READY (INTEGRITY_ONLY is by-policy no-action).

### Results
- No back-edge action fired → no new `/run-experiment` calls this iteration.
- `[run-experiment] iteration=1 runs_this_iteration=0 gpu_hours_this_iteration=0.0 cumulative_gpu_hours=0.0`
- Per-claim outcomes: C1 → held PASS (numeric consistency confirmed); C2 → held INTEGRITY_ONLY (recorded in Open Items with upgrade command).

### Status
- **completed** — STOP rule three-dimensional check satisfied on iteration 1:
  1. score=6.5 ≥ TARGET_SCORE=6 ✓
  2. canonical verdict=`almost` ∈ POSITIVE_VERDICT_TERMS ✓
  3. verify_failed=[] AND verify_inconclusive=[] AND verify_zero_eligible_variants=[] ✓ (only PASS + INTEGRITY_ONLY buckets are non-empty; INTEGRITY_ONLY does not block STOP by design)

Termination reason: **positive_verdict** (first iteration reached STOP; reviewer explicitly said "No new expensive runs are methodologically required for submission").
