P2 — Improve Experiment. One candidate per call, from a fresh pass. Rewrite ONLY the `Experiments`
field (touching `Short Hypothesis` / `Risk Factors and Limitations` solely where a number must stay
consistent); leave the other four fields byte-for-byte identical to the input.

Apply this instruction to each candidate independently:

---
Experiments.

Rewrite `Experiments`. Touch `Short Hypothesis` and `Risk Factors and Limitations` only where a
number has to stay consistent with it; leave the other four fields exactly as they are.

The experiments test the hypothesis as written; they do not enlarge it. Do not argue here
for anything `Short Hypothesis` does not claim, and do not quietly widen the claim's scope to
make the experiment set look complete.

Every assertion in `Short Hypothesis` needs an experiment that establishes it. An assertion with
no experiment behind it either gains one or does not belong in the hypothesis.

Every experiment states what it tests; the data it uses — an existing set used as is, an existing
set adapted, or one built from scratch, with its source, how many items exist, how many you will
use, and why if fewer; the model it runs on — family and parameter scale, what is frozen and what
is trainable, the key hyperparameters and the seeds; the systems compared; the decisive metric
first; the result that would count as convincing; and what a negative result would mean. Give each
threshold the band where the measurement cannot decide, in that threshold's own units, and say
what follows for the hypothesis when a result lands there — before the data, not after.

Plan evidence; never write a result as though it were already measured. Stay inside what an
academic lab can afford, and state the scale plainly enough that a reviewer can check it.

The reviewer reads this JSON and nothing else, so it has to stand on its own: head each
experiment by the hypothesis it tests — `Experiment 1 (H1 — <name>):` — and use no term or
pointer they would have to look up somewhere else.
---

The seven fields are a contract: exactly these names, exactly this order —
Name, Title, Short Hypothesis, Related Work, Abstract, Experiments, Risk Factors and Limitations.
Plain-prose strings. No bookkeeping fields. Write with indent=4, atomically (tmp + os.replace),
flushing after every finalized candidate.
