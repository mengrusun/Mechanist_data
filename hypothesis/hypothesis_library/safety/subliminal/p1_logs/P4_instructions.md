P4 — Improve (outer reflection, Round 1/1 — the FINAL round). WRITER rewrites every candidate from
its own P3 assessment. One candidate per call, from a fresh pass, one at a time.

Input per candidate: its current seven contract fields PLUS its `revision_notes` (the deduplicated,
actionable edit list from the P3 assessment). Both are in claim_1.json.

Apply this instruction to each candidate independently (this is the P4 improvement prompt):

---
Round 1/1.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{revision_notes for this candidate, one per line}

This is the FINAL round for this proposal. You MUST now finalize: respond with ACTION: FinalizeIdea and the complete idea JSON in ARGUMENTS.
---

Rules:
- STICK TO THE SPIRIT of the original idea. An improvement that replaces the phenomenon is a new
  candidate, not a repair — position k must still hold the idea it started as (same core
  phenomenon; the `Name` should stay the same or an obvious refinement of it).
- You may edit ANY of the seven fields to address the revision_notes (unlike P2, which touched only
  Experiments). Prioritise the notes that raise novelty, impact, and research quality; do not add
  complexity that the notes did not ask for.
- P4 ISSUES NO SEARCHES. Work only from the revision_notes already provided.
- OUTPUT = the seven contract fields ONLY, exactly these names, exactly this order, plain-prose
  strings: Name, Title, Short Hypothesis, Related Work, Abstract, Experiments,
  Risk Factors and Limitations. DROP all bookkeeping (novelty/impact/review/revision_notes/rank) —
  claim_2.json carries seven fields only.
- Keep the Abstract ~250 words; keep experiments feasible for an academic lab.
- Write with indent=4, atomically (tmp + os.replace), flushing after every finalized candidate.
