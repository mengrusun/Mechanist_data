# Reproduction Report: `circuit_breakers`

- **Category:** `safety`
- **Experiment directory:** `scientist3/safety/circuit_breakers`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Representation-Level Circuit Breakers for Safe LLMs

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 20 (good: 3, buggy: 17, best_solutions: 1)
- `stage_2_baseline_tuning_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)
- `stage_3_creative_research_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)
- `stage_4_ablation_studies_1_first_attempt` — nodes: 4 (good: 4, buggy: 0, best_solutions: 1)

**Models used in this run (from `bfts_config.yaml`):**
- code: `claude-opus-4-7`
- feedback: `claude-opus-4-7`
- report: `claude-opus-4-7`
- select_node: `claude-opus-4-7`
- summary: `claude-opus-4-7`
- vlm_feedback: `claude-opus-4-7`

**Writeup / review flags:** launch used `--skip_writeup --skip_review` (no PDF; in-run bug review only).

## 2. Verdict: Do Results Support the Claims?

### Overall (automated heuristic)

**Partially or fully supported** — summaries suggest supportive evidence, but human review against the source paper is still required.

### Agent summary excerpt (`Significance` from draft_summary.json)

> This establishes a working seed pipeline for training and evaluating representation-level circuit breakers, including data tokenization with response masks, LoRA-based RR training on late decoder layers, and a coherence-gated ASR judge. It provides an initial numerical baseline against which future methodological refinements can be compared, and validates that the overall harness runs end-to-end.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/4). — *Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable…* |
| 2 | See overall verdict (claim 2/4). — *A model fine-tuned with Representation Rerouting (RR) achieves substantially low…* |
| 3 | See overall verdict (claim 3/4). — *The same representation-level intervention transfers to multimodal LLMs, blockin…* |
| 4 | See overall verdict (claim 4/4). — *The same intervention transfers to LLM agents, materially reducing the rate of h…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/safety/circuit_breakers/idea.json` |
| Human-readable idea | `scientist3/safety/circuit_breakers/idea.md` |
| BFTS config copy | `scientist3/safety/circuit_breakers/bfts_config.yaml` |
| Tree visualization | `scientist3/safety/circuit_breakers/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (99 files) | `scientist3/safety/circuit_breakers/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/safety/circuit_breakers/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/safety/circuit_breakers/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/safety/circuit_breakers/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/safety/circuit_breakers/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable internal representations that can be rerouted to an orthogonal, non-harmful subspace using only paired benign/harmful data, without exposure to any attack prompts.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal end-to-end baseline implementation of Representation Rerouting (RR) circuit-breakers on Llama-3-8B-Instruct, comparing the base model against an RR-trained variant on HarmBench attack success rate (ASR).

### Claim 2

**Original claim (verbatim from task.md):**

> A model fine-tuned with Representation Rerouting (RR) achieves substantially lower attack success rates than refusal-trained or adversarial-trained baselines across a wide range of unseen attacks, while preserving standard capability benchmarks.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal end-to-end baseline implementation of Representation Rerouting (RR) circuit-breakers on Llama-3-8B-Instruct, comparing the base model against an RR-trained variant on HarmBench attack success rate (ASR).

### Claim 3

**Original claim (verbatim from task.md):**

> The same representation-level intervention transfers to multimodal LLMs, blocking image-based jailbreaks (image hijacks) without degrading vision-language task performance.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal end-to-end baseline implementation of Representation Rerouting (RR) circuit-breakers on Llama-3-8B-Instruct, comparing the base model against an RR-trained variant on HarmBench attack success rate (ASR).

### Claim 4

**Original claim (verbatim from task.md):**

> The same intervention transfers to LLM agents, materially reducing the rate of harmful tool-use actions executed under attack.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal end-to-end baseline implementation of Representation Rerouting (RR) circuit-breakers on Llama-3-8B-Instruct, comparing the base model against an RR-trained variant on HarmBench attack success rate (ASR).

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
