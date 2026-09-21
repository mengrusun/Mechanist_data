Here’s the blunt reviewer take.

## 1. Re-score after c1 under-power falsification

**Updated score: 6/10**  
(previously 5/10)

Why it goes up:
- You converted c1 from “maybe underpowered” into a **well-characterized partial reproduction**.
- That matters a lot. Negative or partial results are much more credible once the obvious “you just didn’t try hard enough” objection is addressed.
- c5a is already a credible negative after the dead-code fix.
- c2 remains a real pass.

Why it does **not** go higher:
- You still have only **1 clean pass (c2)**.
- c1 is directionally strong but misses a major absolute target in a way that now looks **real, not merely underpowered**.
- c3 is still only partially characterized.
- c4 is effectively a failed assay, not a substantive replication result.
- c5b is weakly tested and currently unsupported.

So this is no longer “probably too broken”; it is now “**mixed but publishable as a scoped partial reproduction**.”

---

## 2. Should the loop continue?

**My recommendation: stop unless you specifically want one last c3 characterization run for polish.**

Your prior stopping heuristic was:  
> if c1/c3 don't move → stop

I think c1 has now “moved enough” in the sense that it is no longer unresolved. It did **not** move toward success, but it moved toward **clarity**. That satisfies the real scientific criterion.

So the question is not “can c1 still be rescued?”  
It is “is there another ambiguity worth spending compute on?”

For me:
- **c1 ambiguity is resolved**
- **c5a ambiguity is resolved**
- **c4 is not worth compute**
- **c5b is very unlikely to become persuasive with only ~3 GPU-h**
- **c3 is the only remaining claim where a cheap extra run could improve the paper’s rigor**

So:

### Practical decision
- **If you want maximum efficiency:** stop now and write.
- **If you want one more iteration for reviewer-proofing:** spend ~1.5h on **c3 only**, then stop regardless of outcome.
- **Do not spend on c5b.**

---

## 3. Is c3 worth another iteration? If yes, what minimum fix?

### Short answer
**Yes, but only as a small “characterization” run, not a rescue attempt.**

Your c1 result is informative here: scaling features/sequences improved confidence and preserved directionality, but did **not** suddenly bridge a large gap to the original absolute claim. That strongly suggests c3 will likely behave similarly:
- direction holds,
- absolute/strict threshold may remain missed or borderline.

That’s still useful. It would let you say c3 is also a **well-characterized partial**, rather than a one-off near-miss under possibly weak sampling.

### Minimum worthwhile fix
I would do:

- **c3 rerun only**
- Increase **sequence count moderately**
- Keep architecture/protocol unchanged
- Do **not** launch a broad sweep
- Predefine success criterion as:
  1. does the ordering/ladder remain stable?
  2. does the primary-threshold gap stay near-miss or improve?
  3. is the τ=0.3 rescue replicated?

This is important:  
**Do not frame the rerun as “trying to hit the original target.”**  
Frame it as testing whether c3, like c1, is a stable partial rather than an underpowered artifact.

### My expected outcome
Probably:
- ladder still holds,
- primary threshold still near-miss,
- relaxed threshold still positive.

That would materially strengthen the paper’s honesty.

### If you do not run c3
That is still defensible. But then c3 should be written more cautiously as:
- “partial directional support under current budget”
rather than
- “well-characterized partial.”

---

## 4. Does the current story hold as an honest workshop paper?

**Yes — with careful wording, this is an honest and defensible workshop paper.**

Your proposed framing is basically right:

- **c2: PASS**
  - strong and clean
  - arguably the strongest result in the reproduction
- **c1: PARTIAL**
  - direction clearly supported
  - ratio criterion met robustly
  - absolute count criterion not met, now credibly so
- **c3: PARTIAL**
  - direction supported
  - strict threshold near-miss
  - passes at relaxed threshold
- **c4: NOT SUPPORTED**
  - importantly, because the assay is non-discriminative
  - this is not just “we failed”; it is “the test as instantiated does not separate signal from control”
- **c5a: NOT SUPPORTED**
  - fair, credible negative after fixing implementation issue
- **c5b: NOT SUPPORTED**
  - protocol too weak to support claim either way; currently negative and under-informative

That is a coherent paper story:
> SAE-based interpretability claims partly reproduce in directional/ranking terms, but several stronger downstream or causal claims do not survive scrutiny under this reproduction budget; some failures reflect assay/protocol weakness rather than direct contradiction, while at least one negative (c5a) appears substantive.

That’s workshop-appropriate.

### What would make it dishonest
- claiming “most claims reproduced” — no
- implying c1 is merely underpowered — no, you now have evidence against that
- overselling c3 as confirmed — not yet
- treating c4 as evidence against the phenomenon instead of against the assay — also no

---

## 5. Is anything critically in need of one more fix?

**No critical fix is required for submission.**

Only optional fix:
- **c3 one more small run** for characterization polish

Not worth it:
- **c5b**. I would not spend the 3 GPU-h. Even if widened, this likely remains too weak to convert into a convincing causal replication. The upside is low.

So my honest view:
- **No claim is currently blocked on one must-do experiment.**
- The paper is already at the point where writing quality and epistemic clarity matter more than extra compute.

---

## 6. Ready / Almost / Not-ready

**READY**, for a **workshop / reproducibility track / negative results venue**.

If you insist on top-tier main-track standard: **Almost**, not ready.  
But for the framing you’re actually using — “reproduction of five SAE-on-ESM-2 interpretability claims” — I think this is **ready**.

Reason:
- you have a meaningful positive result (c2),
- a strong partial with under-power falsification (c1),
- a plausible partial (c3),
- and two instructive unsupported claims, one due to assay failure and one due to credible negative.

That is enough for a solid workshop paper **if written soberly**.

### My recommendation in one line
**Stop and write.**  
Optional: spend one final cheap iteration on c3 only if you want one more “well-characterized partial” before freezing.

---

## Suggested decision tree

### Best efficiency path
- **Terminate loop now**
- Write c3 cautiously
- Do not touch c5b

### Best rigor-per-GPU path
- Run **one c3-only** iteration (~1.5h)
- Then terminate regardless of outcome
- Still do not touch c5b

### Worst path
- Spend remaining budget trying to rescue c5b

---

## Submission-facing framing suggestion

I would explicitly summarize the reproduction as:

- **Strongly reproduced:** c2
- **Directionally reproduced but quantitatively short of original:** c1, c3
- **Not reproduced due to invalid/non-discriminative evaluation setup:** c4
- **Not reproduced under corrected fair test:** c5a
- **Not reproduced; evidence limited by weak intervention sweep:** c5b

That is crisp, honest, and reviewer-resistant.

---

## Memory update

## Memory update
- Iter 2 established **c1 under-power hypothesis is falsified** at ESM-2-650M L9 under current protocol family.
- Expanded c1 run (seqs 1500→3000, LLM-gate features 100→300) preserved pass-rate stability (~15%→14.3%) and yielded **SAE_interp=1430 [1034,1825]**, **ratio=12.9×**.
- Interpretation: c1 is now a **well-characterized partial**: directional/ratio claim robustly supported, original absolute-count target (2548) credibly not met and outside CI.
- Review stance updated: extra compute should **not** be spent trying to “rescue” c1 by simple scaling.
- Remaining optional compute target, if any, is **c3 only** for analogous characterization; **c5b not worth further spend** under remaining budget.
- Overall project status upgraded from “almost / unresolved partials” to “ready for workshop-style scoped partial reproduction,” contingent on restrained claims.