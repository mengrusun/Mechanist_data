P3 — Assess. One external-reviewer call per candidate, on the llm-chat MCP reviewer
(`mcp__plugin_mechanist_llm-chat__chat`, model gpt-5.4). NOTHING is eliminated or reordered here —
P3 only annotates. Position k stays candidate k.

For EACH candidate in your index range, do this, one candidate at a time:

STEP 1 — Search brief. Run WebSearch as needed (1–2 focused queries) based on the candidate's
Title / Short Hypothesis / Related Work to gather what helps judge novelty and impact: the closest
overlapping work and the likely delta, who cares about the problem, and where the result would
land. Condense findings into a SHORT plain-text "Search summary" (a few sentences). If a search
returns nothing useful, say so briefly.

STEP 2 — Reviewer call. Call `mcp__plugin_mechanist_llm-chat__chat` once, passing the prompt below
verbatim, then the Search summary, then the full seven-field proposal JSON. Use this exact reviewer
prompt text:

<<<REVIEWER_PROMPT
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
  "novelty": {
    "score": <integer 1-10>,
    "reasoning": "<brief justification>",
    "revision_notes": [
      "<field-level edit that would raise novelty>",
      "..."
    ]
  },
  "impact": {
    "score": <integer 1-10>,
    "reasoning": "<brief justification>",
    "revision_notes": [
      "<field-level edit that would raise impact>",
      "..."
    ]
  },
  "review": {
    "score": <integer 1-10>,
    "reasoning": "<brief justification>",
    "revision_notes": [
      "<field-level edit that would raise research quality>",
      "..."
    ]
  }
}
REVIEWER_PROMPT

STEP 3 — Parse. Extract strict JSON from the reply (strip any ``` fences). It must contain
`novelty`, `impact`, `review`, each with integer `score` (1–10), string `reasoning`, and a list
`revision_notes`. If parsing fails or a field is missing/out of range, call the reviewer ONE more
time appending "Return ONLY the strict JSON object, no prose, no code fences." If it still fails,
record the axis with score null and an empty revision_notes list and note the failure.

STEP 4 — Build `revision_notes` (top-level): concatenate the three axes' `revision_notes` lists in
novelty → impact → review order, then deduplicate exact-duplicate strings while preserving first
occurrence order. This must be actionable, not evaluative.

STEP 5 — Flush. Append the result for this candidate to your slice output file and re-write it
atomically (tmp + os.replace) after every candidate, so an interruption keeps finished work.

Do NOT modify the seven contract fields. Do NOT reorder. You are only producing annotations.
