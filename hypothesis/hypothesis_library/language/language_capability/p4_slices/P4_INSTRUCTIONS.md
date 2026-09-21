# P4 — Improve from assessment (you are the WRITER). FINAL round (Round 1/1).

You refine each candidate using the reviewer's `revision_notes`. Work one candidate at a time, in index order — a queue, never a batch. This is the FINAL and only improvement round.

## The improvement instruction (apply to each candidate)

> Round 1/1.
>
> In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created. Include any other factors that you think are important in evaluating the proposal. Ensure the proposal is clear and concise, and the JSON is in the correct format. Do not make things overly complicated. In the next attempt, try to refine and improve your proposal. Stick to the spirit of the original idea unless there are glaring issues.
>
> If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.
>
> Results from your last action (if any):
>
> {revision_notes}
>
> This is the FINAL round for this proposal. You MUST now finalize: respond with ACTION: FinalizeIdea and the complete idea JSON in ARGUMENTS.

## How to improve
- Address the candidate's `revision_notes` (the deduplicated reviewer edits across novelty/impact/research-quality). Improve any of the seven fields as needed to raise the proposal — sharpen the hypothesis, tighten related-work positioning, strengthen the abstract, make experiments more decisive/feasible, and firm up risks.
- **Stick to the spirit of the original idea.** Do NOT replace the phenomenon under study. Position k must still hold the idea it started as — the same core behavior/claim, refined, not swapped for a new one.
- Keep it clear and concise; do not over-complicate. Do not invent measured results — plan evidence only. Stay within an academic lab's budget.
- **Issue NO web/literature searches** in this phase. Your only new input is `revision_notes`.

## Output contract (STRICT)
- Each output record has **exactly the seven canonical keys, in this order**, and NOTHING else:
  `Name`, `Title`, `Short Hypothesis`, `Related Work`, `Abstract`, `Experiments`, `Risk Factors and Limitations`.
- DROP all bookkeeping (`novelty`, `impact`, `review`, `revision_notes`) — they must not appear in the output. (The next phase re-derives ranking.)
- Keep `Name` stable (same identity). Preserve input order. Each field is a plain-prose JSON string.

## Incremental flush
After finishing EACH candidate, use Write to save the slice output file as a JSON array (indent=4) of all completed records so far, in index order — so it grows by one record per candidate and partial progress is never lost.
