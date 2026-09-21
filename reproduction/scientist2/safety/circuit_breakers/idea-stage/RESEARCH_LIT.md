# Raw Literature Retrieval: Representation-level safety intervention in instruction-tuned LLMs

**Date**: 2026-07-15
**Query**: representation-level safety defenses for LLMs (Representation Rerouting / circuit breakers), representation engineering, refusal directions, harmful-representation localization, activation steering; transfer to multimodal LLMs and LLM agents; jailbreak attack baselines (GCG, PAIR, TAP, AutoDAN); adversarial training baselines.
**Sources scanned**: web (attempted, aggressively filtered), arxiv API (attempted, network timeout), mechanic-db (skipped: MCP unavailable), Zotero (skipped: not configured), Obsidian (skipped: not configured), local PDFs (skipped: `papers/` and `literature/` both empty).

**Query formulations attempted**:
- "Improving Alignment and Robustness with Circuit Breakers" Zou 2024 — **blocked** (target paper on the project forbidden-urls list — expected, respected).
- Representation engineering RepE Zou 2023 LLM safety honesty activation steering — filtered out (returns contain post-cutoff arxiv IDs).
- Arditi refusal direction single direction LLM 2024 activation ablation — filtered out.
- Jailbreak defense LLM safety RLHF adversarial training 2023 attack success rate benchmark — filtered out.
- GCG universal transferable attack Zou 2023 llama vicuna jailbreak — filtered out.

## Retrieval blocker (honest report)

Two hard constraints on this run's retrieval:

1. **Target paper embargo (project policy).** The project's `.claude/forbidden-urls.txt` lists the target paper (Zou et al., 2024 — "Improving Alignment and Robustness with Circuit Breakers"), the GraySwanAI/circuit-breakers repo, and the released RR / Cygnet checkpoints (Mistral-7B-Instruct-RR, Llama-3-8B-Instruct-RR, Cygnet) as forbidden. **This is a deliberate blind-reproduction setup** — the claim stage must not read the paper or its released weights. Respected in full.
2. **Knowledge-cutoff filter (post-search hook).** The project also blocks any web-search response that contains an arxiv ID ≥ 2406 (June 2024) — voiding the *whole* response, not just the offending line. Any modern query for "refusal direction", "representation engineering safety", "jailbreak defense", etc. inevitably returns some post-cutoff results, so WebSearch cannot deliver a usable landscape file for this project.

Result: no external retrieval delivered structured, citable paper metadata for the raw dump. The synthesized landscape (below in `LANDSCAPE.md`) is therefore built **only from pre-June-2024 domain concepts that predate the target paper** — foundational lines of work in representation engineering, refusal directions, jailbreak attacks, and multimodal / agent safety that are common knowledge in the field and independent of the target paper's specific method.

## Retrieved papers

None — see the retrieval blocker above. `LANDSCAPE.md` synthesis proceeds from general pre-cutoff domain knowledge; downstream ideation is skipped (BEHAVIOR_SOURCE=given), so the landscape is used only as *supporting context* for baselines / datasets / metric definitions, never to alter the claims captured from `task.md`.
