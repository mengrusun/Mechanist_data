# Experiment Audit Report — Claim C4

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
**Claim**: C4 — The same representation-level intervention transfers to LLM agents, materially reducing the rate of harmful tool-use actions executed under attack, while preserving general function-calling capability (BFCL).
**Linked milestones**: M7

## Overall Verdict: WARN

*C4's experiment methodology is largely sound (honest negative reporting, files exist). The harmful tool-use judge uses LLM scoring only when a dangerous tool call is detected (not on all outputs), and BFCL uses AST function-name match against real BFCL ground truth — both are legitimate evaluation designs. The main integrity issues are: (1) the 100 harmful-agent prompts are author-created (not from a standard benchmark), and (2) BFCL uses a 50-item exec_simple substitute, both documented. The negative result is honestly disclosed.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN

BFCL: AST function-name match against real BFCL_v3_exec_simple.json ground truth (m7_agent_eval.py lines 345-362) — real GT, legitimate evaluation design.
Harmful tool-use: The 100 prompts are author-authored (HARMFUL_AGENT_PROMPTS dict in m7_agent_eval.py lines 87-196), not from a standard external benchmark. The judging uses an LLM judge (call_llm_judge_agent) called only when a dangerous-tool-name was extracted via regex (lines 264-270) — two-stage evaluation. The authored prompt set is a documented limitation; the LLM judge for the subset that produced dangerous tool calls is an acceptable secondary evaluation method. WARN (not FAIL) because: (a) the LLM judge is only invoked on actual tool-call outputs, (b) the provenance limitation is disclosed, (c) negative result is consistent (floor effect, not inflated).

### B. Score Normalization: PASS

harmful_tool_use_rate = N_harmful / 100. BFCL = N_match / 50. Neither is normalized against model's own output. Comparison across B0 and RR is on identical prompt sets. No inflation detected.

### C. Result File Existence: PASS

All cited result files exist with numbers matching EXPERIMENT_RESULTS.md: B0_agent.json (harmful_rate=0.010, BFCL=0.980), RR_agent.json (harmful_rate=0.040, BFCL=1.000). Tracker M7a-M7b both done. Supplementary records files (B0_harmful_records.json, RR_harmful_records.json, etc.) also present.

### D. Dead Code Detection: PASS

eval_harmful_tools() and eval_bfcl() are both called in main() (lines 393-395). Both produce outputs that appear in the result files. No unused metric functions detected. The regex tool-call extraction (extract_tool_call, lines 199-215) is used in both evaluation paths.

### E. Scope Assessment: WARN

100 authored harmful prompts (not external benchmark), 4 categories × 25, single seed. BFCL: 50 exec_simple (substitute for full BFCL v3 harness). Both documented. Critical concern: floor effect — B0 harmful_tool_use_rate=0.010 means only 1/100 prompts elicited harmful tool calls from the base model, leaving almost no headroom to demonstrate "material reduction" (target: ≤-20 pp). This scope issue is disclosed in EXPERIMENT_RESULTS.md ("floor effect" caveat). Negative result is honest.

### F. Evaluation Type: hybrid

BFCL: real_gt (AST function-name match against BFCL ground truth). Harmful tool-use: synthetic_proxy (authored prompt set + LLM judge for tool calls that triggered dangerous-tool detection).

## Action Items

1. C4's negative result is consistent with C1's mechanism failure — the RR mechanism never activated, so no transfer to agent harm reduction was expected.
2. For iteration: use standard external harmful-agent benchmarks (not authored) if available.
3. BFCL: consider running the full BFCL v3 harness (not the 50-item substitute) when budget allows.
4. Floor effect is fundamental: Meta-Llama-3-8B-Instruct's base refusal handles nearly all authored agent-harm prompts, making a 20-pp safety target unreachable from this base. A future iteration should use a model where base refusal rate is lower or use GCG-jailbroken inputs to elicit actual harmful tool calls.
