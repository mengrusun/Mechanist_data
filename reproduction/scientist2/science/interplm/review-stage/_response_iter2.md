1. **Re-score**

- **Top venue:** **5/10 as-is**, maybe **6/10 ceiling** with a couple targeted fixes.
- **Bio-interp workshop:** **7/10 as-is**, **8/10 ceiling** if c1 or c3 gets one decent power-up rerun and c4 is explicitly reframed as a metric failure rather than a failed science claim.

Why the slight bump from iter 1? Because **c5a moved from “methodologically compromised null” to “credible negative result.”** That is real progress. The paper is still not top-venue ready because too many claims are either partial due to power (c1/c3), structurally broken (c4), or weakly tested (c5b). But it is now more honest and scientifically interpretable.

---

2. **c5a resolution: genuine fix or still under-powered?**

This was a **genuine methodological correction**, not cosmetic.

What was fixed:
- the active probe path now actually matches the intended objective class much better,
- optimization is no longer obviously underfit,
- class imbalance is addressed,
- test size increased,
- result flipped from ambiguous/null-ish to **slight neurons advantage with p=0.516**.

That matters. The original result was contaminated by a bad code path; this one is not.

My judgment:

### What is now supported
- **At the tested setting** — 1000 train seqs, 50k test residues, 30 scorable concepts, layer 9 of ESM-2-650M — **the claim is not supported**.
- More strongly: the corrected result is **not merely “failed to show SAE > neurons”**; it is **directionally slightly against SAE**.

### What is still limited
You should not overclaim this as a universal falsification of the original hypothesis, because:
- only **one layer** tested,
- only **1000 train seqs**,
- only **30/50 concepts** evaluable,
- the dropped 20 concepts may systematically differ,
- PR-AUC on sparse concept annotation can be high-variance across concept base rates.

### Crucial distinction
This is **not** “still obviously under-powered in the same way as before.” The dead-code concern was the main integrity problem, and that has been fixed. The remaining limitations are **scope limitations**, not obvious methodological invalidators.

So my answer is:

- **For the reproduction claim as currently scoped:** c5a is **confirmed not supported**.
- **For a stronger paper-level statement about the broader phenomenon:** one more expansion iteration could be scientifically nice, but it is **not required to safely report a negative**. It would only be needed if you want to say “SAEs do not beat neurons on annotation filling more generally.”

Given budget, I would **not prioritize more c5a** unless the authors are emotionally attached to rescuing it. Right now it is a clean negative and probably more valuable that way.

---

3. **Remaining verify_integrity_only claims: minimum type-② fix + cost + new priority**

## c1
**Current issue:** likely under-powered due to tiny LLM-gated sample and reduced sequence count.

### Minimum fix
- **Increase LLM-gated feature sample from 100 → 300–500 features/layer**
- **Use full 10k SP-test sequences** if not already
- Keep criterion unchanged; just raise precision of the estimate.

### Why this is minimum
The current problem is not that the script is broken; it is that the estimate is too noisy to support/exclude the absolute threshold. This is a pure power fix.

### Expected cost
- **~1.0–1.5 GPU-h**

### Expected payoff
- High. c1 looks plausibly recoverable because the direction is already strong (19.3× ratio), and the current miss is on the absolute rate threshold. This is exactly the kind of claim that often flips from “partial” to “supported” with more samples.

---

## c3
**Current issue:** near-miss on strict Δ threshold; direction holds.

### Minimum fix
- **Rerun with 10k sequences / full evaluation set**
- keep same analysis and thresholds initially
- no criterion redesign yet

### Why this is minimum
This is another likely power issue rather than an implementation issue. You already have a directional effect and a near miss at the primary threshold.

### Expected cost
- **~1.0–1.5 GPU-h**

### Expected payoff
- Moderate-to-high. If c3 crosses the strict Δ threshold, that materially improves the paper’s backbone because it gives you another substantive positive result besides c2.

---

## c4
**Current issue:** the metric is dead-on-arrival as a discriminator. Zero acceptance on both real and control means the test cannot support any conclusion.

### Minimum fix
- **Redesign the acceptance criterion in the main experiment script**
- Specifically: relax the all-or-nothing synonym/null gate into a graded semantic acceptance criterion, or at minimum add a human/LLM adjudication tier that can produce nonzero acceptance on known positives.

### Why this is minimum
More samples are useless if the metric has **0 sensitivity**. This is not under-power; it is assay failure.

### Expected cost
- **~0.2–0.5 GPU-h** compute, but some human/implementation overhead
- If you need prompt tuning / adjudication setup, practical cost is more like “annoying but cheap.”

### Expected payoff
- Mixed. Scientifically important because it repairs a broken claim, but risky because criterion redesign after seeing zero/zero can look opportunistic unless carefully justified. This is the least clean to salvage in a reproduction.

---

## c5b
**Current issue:** intervention protocol too weak to test the claim.

### Minimum fix
- **Expand steering coefficients to a real dose ladder over ≥3 OOM**
  - e.g. α ∈ {0, 0.3, 1, 3, 10, 30, 100} or sigma-scaled equivalent
- **Use more than 3/4 features and more than 10/25 sequences if feasible**
- Preserve the no-steer baseline and report full curve, not just pairwise comparisons.

### Why this is minimum
The current ladder is so compressed that “no effect” is nearly uninformative. You did not really test dose-response.

### Expected cost
- **~2.0–3.0 GPU-h**

### Expected payoff
- Moderate but uncertain. This could still fail even after a proper ladder. Unlike c1/c3, the present result doesn’t look like a near miss; it looks like a weak protocol.

---

## Updated priority ranking with ~7.13 GPU-h left

If the goal is **best submission value per GPU-hour**, I would rank:

1. **c1** — 1.0–1.5 GPU-h  
   Highest chance to upgrade PARTIAL → SUPPORTED.

2. **c3** — 1.0–1.5 GPU-h  
   Another likely under-powered near miss; strong return if it crosses.

3. **c5b** — 2.0–3.0 GPU-h  
   More expensive, less certain, but current protocol is too weak to leave unaddressed if steering is central to the paper.

4. **c4** — 0.2–0.5 GPU-h compute, but lower strategic priority  
   Cheap to patch, but risky from a review optics perspective unless framed as “the original metric was non-discriminative, so we report assay failure and an exploratory alternative.” I would not spend much here unless you can make the redesign extremely principled.

If budget allows **two fixes only**, do **c1 + c3**.  
If budget allows **three**, do **c1 + c3 + c5b**.  
I would **not** spend more on c5a.

---

4. **READY for submission?**

**Almost.**

### Top venue
**No**, not yet.

Reason:
- too many headline claims still sit in awkward categories:
  - c1 partial due to power,
  - c3 partial due to power,
  - c4 unsupported because the metric is broken,
  - c5b unsupported because the protocol is too weak,
  - c5a negative.
- That can still make a good paper, but only if the framing is explicit: this is a **selective/partial reproduction with one robust pass, one robust negative, two plausible under-powered near misses, and two methodological failures of the original assay/protocol.**
- Right now, that story is credible for a workshop, but thin for a top venue unless at least one of c1/c3 gets promoted.

### Workshop
**Yes, basically ready**, if honestly framed.  
But with ~7 GPU-h left, it would be foolish not to try **c1 and c3** first. Those are exactly the kinds of upgrades that can turn the paper from “interesting debugging/reproduction note” into “careful and substantial partial reproduction.”

### Brutally honest bottom line
- If you submit now, the defensible story is **“partial reproduction with important negative corrections.”**
- If you want a stronger story, the best remaining investment is **not broad exploration**; it is **powering up c1 and c3**.
- If those don’t move, then stop. At that point the paper’s contribution is the negative/revisionist story, not a near-complete reproduction.

---

## Memory update

- **Resolved from iter 1**
  - **c5a dead-code / wrong active probe path:** genuinely resolved.
  - c5a prior null is no longer integrity-compromised; we now have a credible negative at tested scope.

- **Still unresolved**
  - **c1 under-power:** still unresolved; no new evidence.
  - **c3 under-power:** still unresolved; no new evidence.
  - **c4 non-discriminative metric:** still unresolved and still serious.
  - **c5b under-scaled steering ladder:** still unresolved.

- **New suspicions / refinements**
  - c5a is now best described as **scope-limited negative**, not just “under-powered null.” Important distinction.
  - The 20/50 dropped concepts in c5a are probably not a sample-size artifact, but they still create a **selection-on-evaluable-concepts** caveat. Fine for scoped reporting, not for broad generalization.
  - The project now bifurcates clearly into:
    1. **well-tested negative**: c5a  
    2. **likely under-powered near misses**: c1, c3  
    3. **methodologically invalid current assays**: c4, c5b

- **Pattern**
  - The paper is improving by replacing ambiguous nulls with more interpretable outcomes.
  - The highest-value remaining work is **not** chasing every claim; it is **separating true negatives from under-powered misses**.
  - c1/c3 are the only remaining claims with a realistic chance of upgrading the paper materially per unit budget.