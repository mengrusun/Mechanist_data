# Raw Literature Retrieval: SAE-based interpretability of protein language models (ESM-2), Swiss-Prot concept alignment, superposition, auto-interpretation, and downstream utility

**Date**: 2026-07-15
**Query**: sparse autoencoders for protein language models — interpretability of ESM-2 residual-stream features via SAE dictionaries, comparison against raw neurons, alignment with Swiss-Prot biological concepts (binding sites, active sites, motifs, domains, PTM sites), superposition in transformer PLMs, auto-interpretation of latent features using external LLMs over top-activating protein contexts, downstream applications of PLM feature dictionaries: annotation-filling and steered protein sequence generation
**Sources scanned**: WebSearch (returned only forbidden-URL matches — voided by post-search-filter), arXiv API (blocked by arxiv-cutoff 2412 policy), mechanic-db (not invoked — WebSearch already established that the near-topic literature is downstream of the target paper the project blocks), Zotero (not configured), Obsidian (not configured), local `papers/` and `literature/` (empty)
**Query formulations used**:
- sparse autoencoders protein language model ESM-2 interpretability biological features 2024
- sparse autoencoders SAE monosemantic features residual stream superposition dictionary learning Anthropic 2024
- auto-interpretation SAE features LLM labeling top activating contexts feature steering
- Bricken Anthropic "Towards Monosemanticity" sparse autoencoder decomposing language models 2023 transformer-circuits
- Cunningham Ewart Riggs "Sparse Autoencoders Find Highly Interpretable Features in Language Models" 2023
- ESM-2 protein language model Rives Lin Meta 2022 residual stream representations biological features

## Retrieval status

The project's `.claude/forbidden-urls.txt` blocks (a) the specific reference paper this task reproduces and (b) all arXiv IDs at or after cutoff 2412 (i.e., every paper submitted from Dec-2024 onwards, which is exactly the window where nearly every SAE-on-PLM paper lives). The WebSearch post-filter voided six web-search rounds because either the reference paper appeared by title, or 2412+ arXiv IDs appeared in the results. mechanic-db was not invoked because the near-topic literature the cloud DB would return is either the blocked paper or its post-cutoff citing works.

**Consequence for this run.** Because `BEHAVIOR_SOURCE=given`, the literature survey is only informational — the behavior/claims are already fully specified by `task.md`, novelty checks are skipped, and the mechanism strategy in Phase 4.5 needs only enough grounding in general SAE / mechanistic-interpretability principles to be well-shaped. That grounding comes from **pre-cutoff, non-blocked, canonical works** the model retains, listed in `LANDSCAPE.md`. No external retrieval was successful; the retrieved-papers section is empty by construction.

## Retrieved Papers

_(none — all WebSearch responses were voided by policy; mechanic-db not invoked; local channels empty)_
