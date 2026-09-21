# P3 — Assess (external reviewer). You ORCHESTRATE; the reviewer SCORES.

You do NOT score the ideas yourself. The scores must come from the external reviewer model reached through the MCP tool `mcp__plugin_mechanist_llm-chat__chat` (gpt-5.4). Your job per candidate: (1) gather web context, (2) build a short search brief, (3) call the reviewer with the exact prompt below, (4) parse its strict-JSON verdict, (5) flush to your slice file. Never substitute your own judgment for the reviewer's numbers.

## First step
Call ToolSearch with `select:mcp__plugin_mechanist_llm-chat__chat` to load the reviewer tool schema (WebSearch is already available to you).

## Per candidate (one at a time, in index order)
1. Read the candidate's seven fields from the input pool (index given by orchestrator).
2. **WebSearch** as needed based on the candidate's content — find the closest overlapping work and the likely delta, who cares about the problem, and where the result would land. 1–3 searches is plenty. Condense into a SHORT `SEARCH BRIEF` (a few sentences): closest prior work / actual delta / who would care.
3. Call `mcp__plugin_mechanist_llm-chat__chat` with:
   - `system`: "You are a strict, fair senior ML reviewer. Judge only from the proposal text and the provided search summary. Return STRICT JSON only, no prose outside the JSON."
   - `prompt`: the REVIEWER PROMPT below, then a line `SEARCH SUMMARY:` and your brief, then a line `PROPOSAL (JSON):` and the candidate's seven-field JSON.
4. Parse the reviewer's reply as JSON. It must contain `novelty`, `impact`, `review`, each `{score:int 1-10, reasoning:str, revision_notes:[str,...]}`. If the reply has stray text, extract the single JSON object. If parsing fails or a score is missing/out-of-range, RE-CALL the reviewer once more asking it to "return only the strict JSON object with integer scores 1-10". If it still fails after 2 tries, record the raw reply under `_parse_error` and keep going (do not fabricate scores).
5. Build `revision_notes`: concatenate novelty.revision_notes + impact.revision_notes + review.revision_notes in that order, then DEDUPLICATE (drop exact-duplicate strings, preserve first-seen order).
6. **Flush**: append to your slice file (overwrite whole file each time) as a JSON array of objects, one per completed candidate, each:
   `{"index": <int>, "Name": "<name>", "novelty": {...}, "impact": {...}, "review": {...}, "revision_notes": [ ... ]}`
   in index order. So the file grows by one object per candidate.

## REVIEWER PROMPT (pass verbatim)
You are reviewing a research idea on three axes — novelty, impact, and research quality. Judge from the proposal text alone.

If a search summary is provided, use it as additional context for the closest prior work, the actual delta, and who would care about the result.

NOVELTY. First extract the 3-5 core technical claims that would need to be novel: the problem solved, the proposed mechanism or explanation, what separates it from the obvious baselines, and whether the novelty lives in the finding, the mechanism, the setting, the dataset, or the problem definition. Then judge:
1. Does it propose a genuinely new theory, task, explanation, mechanism, finding, setting, dataset, or problem definition?
2. If it studies an existing idea in a new setting, is that setting meaningfully different enough that the result itself could be novel?
3. Relative to the proposal's own related work, is the core claim non-trivial rather than an obvious next step?
4. What is the closest prior work family or baseline, and what is the actual delta?

IMPACT. First identify what behavior, phenomenon, or problem is actually under study — the problem, not the method — who would have it or care about the result, what downstream research or practice would change if the result holds, and the single strongest one-line case for why it matters. Then judge:
1. Important problem — a real need, or mostly a niche curiosity?
2. Uptake / citation — would follow-up work build on, use, or cite this?
3. Direction-shifting — could it change how people think about or approach an area?
4. Real-world reach — does it matter for applications, industry, society, or cross-disciplinary use?
5. Phenomenon value — even if the method is simple, does it reveal an important phenomenon?
Apply the "so what?" test explicitly: if the result came out either way, would serious researchers or practitioners change what they do?

RESEARCH QUALITY — technical quality and testability. Act like a strict senior ML reviewer:
1. Logical gaps or unjustified claims.
2. Whether the core hypothesis is precise, falsifiable, and appropriately scoped.
3. Whether the related work is specific enough to position the contribution.
4. Whether the experiments are specific, feasible, properly controlled, and actually capable of testing the stated claim.
5. What key experiments, ablations, controls, or analysis are missing.
6. Narrative weaknesses: where the story is underspecified, overclaimed, or hard to defend.
7. Whether the contribution seems strong enough for a serious ML venue if executed well.
8. What the minimum viable edits are to make the proposal fundable or submission-ready.

Score the three axes independently — a weak axis must not drag the others down — and give each axis its own `revision_notes`: the concrete field-level edits that would raise *that* axis's score.

Return strict JSON:
{
  "novelty": { "score": <integer 1-10>, "reasoning": "<brief justification>", "revision_notes": ["<field-level edit that would raise novelty>", "..."] },
  "impact": { "score": <integer 1-10>, "reasoning": "<brief justification>", "revision_notes": ["<field-level edit that would raise impact>", "..."] },
  "review": { "score": <integer 1-10>, "reasoning": "<brief justification>", "revision_notes": ["<field-level edit that would raise research quality>", "..."] }
}

## Hard rules
- Scores come ONLY from the reviewer tool. Do not invent or adjust them.
- One reviewer call per candidate (plus at most 1 reparse retry). Process one candidate at a time — a queue, not a batch.
- Preserve index order in your slice file. Flush after every candidate.
