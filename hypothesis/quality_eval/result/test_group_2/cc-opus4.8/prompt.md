# Prompt: Hypothesis Generation

For each subfolder that contains a `task.md` (e.g. `knowledge/belief/`, `safety/subliminal/`, `science/mechanism/`):

1. Read the research direction in `task.md`.
2. Generate **50 different hypotheses** for papers in the **mechanistic interpretability** field, grounded in that `task.md`. Avoid near-duplicates.
3. From those 50, pick the **10 you think are best** and write them out in full.

**Do NOT run any experiments** — no code execution, no model calls, no results. The `Experiments` field below is a *written plan only*: describe what you would run, never actually run it.

Output into the same folder as JSON:
- `hypothesis_all.json` — the 50 hypotheses.
- `hypothesis.json` — the 10 best.

## Output schema

`hypothesis.json` is a **JSON array of 10 objects**. Each object must contain **exactly** these seven fields, in this order:

```json
{
  "Name": "...",
  "Title": "...",
  "Short Hypothesis": "...",
  "Related Work": "...",
  "Abstract": "...",
  "Experiments": "...",
  "Risk Factors and Limitations": "..."
}
```

The claim JSON should include the following fields:

- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.

Keep each object tightly matched to its own claim — do not reuse a generic `Experiments` plan or a generic `Related Work` section across claims.

Process every subfolder that has a `task.md`.
