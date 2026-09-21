## 1) Overall score

### As on disk today
**Top venue (NeurIPS/ICML): 4/10**  
**Bio-interp workshop: 6/10**

Reason: there is **one genuinely real result** here — **c2 directionally replicates and even survives a strong model-size swap sanity check**. That matters. But the package is still too under-powered and too methodologically leaky to support the broader five-claim story.

The biggest problems are:

- **c1/c3**: likely directionally true, but currently under-sampled and not sufficient for the absolute-number headline.
- **c4**: current metric is basically a **null test with zero acceptance on both real and control**. That is not evidence for or against the claim; it is a failed operationalization.
- **c5a**: as written, the result is not trustworthy because the main intended probe implementation appears to be bypassed by **dead code**, and the active path is weakly trained / under-powered.
- **c5b**: currently negative, with a steering protocol that is plausibly too weak to test the claim fairly.

### If you spend the remaining ~7.79 GPU-h well
**Top venue: 5.5/10, maybe 6/10 ceiling**  
**Bio-interp workshop: 7/10**

Why not higher? Because with this budget you can probably:
- firm up **c1**,
- likely firm up **c3**,
- maybe rescue or definitively falsify **c5a**,
- improve the fairness of **c5b** enough to say “we tested properly and still saw no effect” or maybe get a weak monotone signal,
- but **c4** is bottlenecked more by **criterion design** than by compute.

So the honest headline is: **good partial reproduction with one robust central comparative finding (c2), several likely-true but under-powered replications, and two claims currently unsupported because the measurement/actuation procedure is not yet fit for purpose.**

---

## 2) Minimum type-② fixes for each `verify_integrity_only` claim

I’ll rank these by **leverage per remaining budget**, not by claim order.

---

### Rank 1 — c5a: fix the dead-code probe path and score all feasible concepts
**Current status:** not-supported, and the current number is not very informative because the implementation likely isn’t testing the intended probe.

#### Minimum fix
**File:** `scripts/m5_annotation_filling.py`  
**Problem:** `per_concept_pr_auc()` exists but is never called; actual code uses `SGDClassifier(loss='log_loss', max_iter=30, alpha=1e-4)`.

#### Change
Route the main evaluation to the intended per-concept logistic regression path:
- call `per_concept_pr_auc()` directly in the concept loop,
- use `LogisticRegression(solver="lbfgs", max_iter=200)` or `saga` if dimensionality is high,
- ensure identical train/test splits and feature normalization for SAE vs neuron baselines,
- retain paired evaluation per concept.

Also:
- increase from **25k/387k test subsample** to a materially larger sample, or all available if CPU-bound rather than GPU-bound,
- score **all concepts with test positives**, not just 30/50 if the exclusion is just a subsampling artifact,
- if some concepts truly have zero test positives, exclude them symmetrically and report that count explicitly.

#### Expected uplift
This is the single most likely place where a claim can move from “not-supported because bad implementation” to “actually testable.” Right now **0.5907 vs 0.5906, p=0.191** is effectively null. But with:
- the intended stronger optimizer,
- more test data,
- more concepts,
you may get either:
1. a small but real paired advantage with enough power, or  
2. a clean negative.

Either outcome is scientifically useful. Right now you have neither.

#### Cost
Mostly CPU / modest GPU depending on feature extraction caching. If activations/features are already cached:
- **~0.3–1.0 GPU-h equivalent** at most,
- possibly mostly wall-clock CPU.

#### Why high leverage
Because this is a **surgical implementation fix**, not a fishing expedition.

---

### Rank 2 — c1: increase LLM-gate sample from 100 to a few hundred per layer and use more sequences
**Current status:** partial; direction already there, absolute claim not yet defendable.

#### Minimum fix
**File(s):** likely `scripts/m1_*` or whatever script computes feature interpretability counts / LLM gating  
Need to modify the feature-sampling and sequence-count parameters in the main c1 pipeline.

#### Change
- Increase **LLM-gated feature sample per layer** from **100 / 10240** to at least **300–500**.
- Increase **sequence support** from **1500 / 10000** to full **10k** Swiss-Prot test set if activation extraction is already streamlined.
- Stratify sampled features by activation prevalence / concept alignment score if current 100 are simple random draws; otherwise the CI on “interpretable latent features per layer” is too loose.

#### Expected uplift
Current number:
- **L9: 1480 / 2548 interpretable (58%), ratio 19.3× over neurons**
This already supports the **qualitative claim** that SAE yields many more interpretable units than neurons.  
What it does **not** yet support tightly is the exact “up to ~2,548 interpretable latent features per layer” wording unless that number is itself estimated from a narrow gate sample with ±15% margin.

By gating 300–500 features/layer:
- your margin drops materially,
- the extrapolation to layer-wide counts becomes more credible,
- you can probably defend the “up to ~2548” magnitude if the estimate remains stable.

#### Cost
If LLM calls are the bottleneck, but caching is available:
- compute cost is low,
- API/cache cost may dominate but appears available.
GPU cost likely **~0.5–1.5 GPU-h** depending on whether you re-extract activations for all 10k.

#### Why high leverage
Because the result is **already nearly there**. This is mostly a confidence-tightening exercise.

---

### Rank 3 — c3: same as c2/c1, but expand sample to hit the strict Δ threshold at the primary operating point
**Current status:** partial; direction strongly supported, strict absolute threshold not met at primary setting.

#### Minimum fix
**File:** `scripts/m3_*` comparator script for SAE vs PCA/random rotation/neurons/shuffled-SAE

#### Change
- Increase from **1500 to 10000** Swiss-Prot test sequences.
- Keep the primary threshold setting fixed if the claim requires it.
- Re-run all baselines under the exact same sample.
- If PCA/random-rot/shuffled-SAE stay at ~0 while SAE scales upward, the Δ should rise above the target **≥20** at the primary threshold.

You already have:
- primary: **SAE=15, controls=0**
- relaxed τ=0.3: **SAE=65, controls=0**

So the ladder is there. The only issue is the strict threshold at the primary point.

#### Expected uplift
High probability this becomes defensible with more data, because the controls are pinned near zero and the SAE count is what should grow.

#### Cost
Likely modest incremental cost if c2/c3 share infrastructure:
- **~0.8–1.5 GPU-h**

#### Why high leverage
This is one of the most plausible “small extra spend moves claim from partial to supported” cases.

---

### Rank 4 — c5b: widen the dose ladder and stop under-scaling the intervention
**Current status:** not-supported, but current steering test is too weak to be dispositive.

#### Minimum fix
**File:** `scripts/m6_*` or generation/steering script  
Need to modify steering amplitude schedule and batch size.

#### Change
1. Expand dose ladder from `α ∈ {0.5,1,2,4}` to something like  
   **`{0, 0.5, 1, 2, 4, 8, 16, 32}`**
   or at least cover **~3 orders of magnitude** in effective intervention strength if that is what the original claim targeted.

2. Stop normalizing solely by **within-sample σ_f** if that shrinks interventions excessively.  
   Better:
   - calibrate clamp magnitude using a corpus-level activation scale,
   - or use percentile-based intervention sizes from the feature’s natural activation distribution.

3. Restore planned batch size:
   - **25 seqs per batch**, not 10.

4. If possible, test the missing 4th planned feature.

#### Expected uplift
I am less optimistic here than for c1/c3/c5a. Current result is **uniformly no_steer > all steered arms**, which is not a subtle miss. That said, the mechanism audit says the sweep is far too narrow and under-scaled, so this is not yet a fair test.

Best case:
- you see monotone enrichment for at least one feature/property pair.

More likely:
- you still do not reproduce the claim, but then the negative is much more credible.

#### Cost
Generation is usually the expensive part.
Estimate:
- **2–3 GPU-h** depending on sequence lengths and number of arms/features.

#### Why only rank 4
Because even after a fairer test, the claim may still fail. This is worth doing for honesty, but it’s not the best expected-return spend if your goal is maximizing supported claims.

---

### Rank 5 — c4: fix the criterion, not just the sample size
**Current status:** not-supported, but in a very specific way: the current metric is non-discriminative.

#### Minimum fix
**File:** `scripts/m4_*` novel-concept auto-interpretation evaluation

#### Change
The minimum viable fix is **not more GPU**. It is to replace the “strict synonym-check-null” gate with a criterion that can actually separate:
- paraphrase of known Swiss-Prot concept,
- genuinely broader/novel compositional concept,
- incoherent hallucination.

A minimally acceptable main-script change:
- add a **3-way adjudication**:
  1. exact/near-exact SP concept paraphrase,
  2. coherent but not directly in SP vocabulary,
  3. incoherent.
- evaluate real SAE-unaligned features vs matched controls under the same rubric.

If you can only change one thing:
- remove the current criterion as the primary success test, because **0 acceptance on both arms means the metric has no power**.

#### Expected uplift
This could turn c4 from “not-supported because bad metric” into “testable.” But I would be cautious: with the current evidence, you do **not** know whether the true effect is 0%, 10%, or 40%.

#### Cost
Likely low GPU, mostly LLM/adjudication:
- **<0.5 GPU-h**, but some engineering / prompt design time.

#### Why lowest leverage
Because it is more of a measurement redesign than a simple scale-up, and there’s substantial risk of ending up with a subjective evaluation unless done carefully.

---

## Suggested budget allocation under ~7.79 GPU-h

If the goal is maximal defensibility per dollar:

1. **c5a fix dead code + larger eval** — **0.5–1.0 GPU-h**
2. **c1 scale LLM-gate sample + 10k seqs** — **1.0–1.5 GPU-h**
3. **c3 full-sample rerun** — **1.0–1.5 GPU-h**
4. **c5b wider steering sweep** — **2.0–3.0 GPU-h**
5. **c4 criterion redesign / rerun** — **0.2–0.5 GPU-h**

Total: roughly **4.7–7.5 GPU-h**, which fits.

If you need to cut one, I would cut **c4** before c1/c3/c5a.  
If you need to cut two, cut **c4** and maybe **c5b**, unless generation steering is central to your intended paper story.

---

## 3) c2 brief consistency check

### Does the claim wording match the numbers?
**Directionally yes, absolutely no.**

The reproduced numbers support:
- **SAE >> raw neurons** on Swiss-Prot concept coverage,
- and that direction is **robust under 650M → 8M swap**, which is actually a nice robustness check.

Current numbers:
- primary: **SAE=15 vs neurons=0**
- swap: **SAE=14 vs neurons=0**

That is sufficient to support the **comparative headline** “SAE covers substantially more concepts than neurons” in this setup.

### Caveat that must surface in the paper
The paper cannot honestly present this as a full replication of the absolute wording:
- target claim: **up to ~143 Swiss-Prot concepts**
- current reproduction: **15 at primary threshold**, **65 at τ=0.3**

And there are known reasons:
- only **1500/10000** test sequences used,
- concept universe is **400 fine-grained concepts**, likely stricter / more fragmented than the reference.

So the correct paper-level statement is:
- **The comparative gap is replicated robustly; the absolute count is not.**
- This is likely due at least in part to **under-power and stricter concept granularity**, but as of now the exact absolute target remains unreproduced.

That caveat is mandatory.

---

## 4) Ready for submission?

**Almost, not fully ready.**

By your routing rule:
- there are no FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS blockers,
- so in the narrow pipeline sense this is not blocked.

But scientifically, for an actual submission:
- **not ready for a strong top-tier paper as a five-claim reproduction**, because c4/c5a/c5b are unsupported and c1/c3 are still under-powered on the absolute statement.
- **ready-ish for a workshop-style “partial reproduction / stress test”** if written very honestly:
  - c2 robustly replicated,
  - c1/c3 partial and likely recoverable,
  - c4 currently untestable under the chosen metric,
  - c5a implementation issue invalidates current null,
  - c5b current protocol too weak to support the original claim.

So the best single-word answer is: **Almost.**

If you submit now, it has to be framed as **a partial reproduction with one robust positive, several provisional positives, and two unresolved methodological negatives**.

---

## 5) Bottom-line brutal honesty

The strongest thing here is **not** “we reproduced five claims.” You did not.

The strongest thing here is:

- **There is a real and robust SAE-over-neurons concept coverage advantage (c2).**
- **c1 and c3 likely also hold in direction, but the current experiments are under-powered for the exact absolute statements.**
- **c4 is currently blocked by a broken success criterion, not disproven.**
- **c5a is presently compromised by an implementation mismatch/dead code issue and should not be cited as evidence either way until fixed.**
- **c5b is currently negative under an intervention protocol too weak to be dispositive.**

That is a respectable reproduction-in-progress, not a finished claim set.

---

## Memory update

- **c5a dead code is a major integrity concern.** `per_concept_pr_auc()` exists but is not used; active path uses weaker `SGDClassifier`. Future iterations should verify the call graph, not just metrics.
- **c4 criterion is non-discriminative.** Zero acceptance on both real and control means the metric cannot support any conclusion. Track as “evaluation null,” not “negative result.”
- **c5b steering ladder likely under-scaled.** Current α range spans <1 OOM; claim expects monotone dose-response over much wider range. Also within-sample σ normalization may be suppressing intervention magnitude.
- **c1 LLM-gate sample far too small for layer-wide extrapolation.** 100/10240 features per layer yields wide uncertainty; increase to 300–500 minimum.
- **c1/c2/c3 all under-powered on sequence count.** Only 1500/10000 Swiss-Prot test sequences used; likely main reason absolute counts lag reference.
- **c2 is the cleanest replicated claim, but only directionally.** Must preserve caveat that absolute ~143 concept count was not matched.
- **c3 likely recoverable cheaply.** Controls are all zero already; scaling SAE sample size is likely enough to cross Δ threshold.
- **c5b current negative should not be over-interpreted.** Need a fairer sweep before concluding the original steering claim fails.
- **Potential concept-granularity mismatch in c2/c3.** Using 400 fine-grained concepts may depress absolute counts versus reference; track whether this is a faithful reproduction choice or a hidden benchmark shift.