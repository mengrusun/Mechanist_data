# Reproduction Report: `encode_harmfulness_refusal`

- **Category:** `safety`
- **Experiment directory:** `scientist3/safety/encode_harmfulness_refusal`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Harmfulness and Refusal Are Encoded Along Separable Linear Directions in LLM Activations

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 15 (good: 3, buggy: 12, best_solutions: 1)
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

> This preliminary implementation establishes strong initial evidence for the core hypothesis that harmfulness and refusal are encoded as distinct but related linear directions in LLM residual streams. Demonstrating causal dissociation via steering (each direction can be manipulated independently) and showing that refusal suppression alone triples-plus attack success rate provides mechanistic insight into jailbreaks and supports the feasibility of a 'Latent Guard' safety probe that monitors internal harmfulness assessment even when refusal fails.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/5). — *Instruction-tuned LLMs represent harmfulness perception and refusal execution as…* |
| 2 | See overall verdict (claim 2/5). — *The two signals live at different token positions: the harmfulness judgment is w…* |
| 3 | See overall verdict (claim 3/5). — *Additive steering along either direction yields dissociated effects — moving alo…* |
| 4 | See overall verdict (claim 4/5). — *A notable class of successful jailbreaks operates by suppressing the refusal sig…* |
| 5 | See overall verdict (claim 5/5). — *A lightweight "Latent Guard" classifier trained on the harmfulness direction mat…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/safety/encode_harmfulness_refusal/idea.json` |
| Human-readable idea | `scientist3/safety/encode_harmfulness_refusal/idea.md` |
| BFTS config copy | `scientist3/safety/encode_harmfulness_refusal/bfts_config.yaml` |
| Tree visualization | `scientist3/safety/encode_harmfulness_refusal/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (111 files) | `scientist3/safety/encode_harmfulness_refusal/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/safety/encode_harmfulness_refusal/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/safety/encode_harmfulness_refusal/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/safety/encode_harmfulness_refusal/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/safety/encode_harmfulness_refusal/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> Instruction-tuned LLMs represent harmfulness perception and refusal execution as two distinct, approximately linear directions in the residual stream, and each can be recovered without disturbing the other.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Initial implementation of a pipeline to identify and analyze a linear 'harmfulness direction' in the residual stream activations of Llama-3-8B-Instruct. The experiment uses AdvBench (harmful) vs Alpaca (benign) prompts, extracts activations across layers (8, 12, 16, 20, 24, 28), computes a difference-of-means direction, and evaluates a 1D logistic regression probe with random-direction and shuffled-labels baselines. Additionally, it explores steering effects along both harmfulness and refusal directions, and tests jailbreak-like refusal suppression.

### Claim 2

**Original claim (verbatim from task.md):**

> The two signals live at different token positions: the harmfulness judgment is written at the final instruction-token position, while the decision to refuse is committed just after the instruction, immediately before the response begins.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Initial implementation of a pipeline to identify and analyze a linear 'harmfulness direction' in the residual stream activations of Llama-3-8B-Instruct. The experiment uses AdvBench (harmful) vs Alpaca (benign) prompts, extracts activations across layers (8, 12, 16, 20, 24, 28), computes a difference-of-means direction, and evaluates a 1D logistic regression probe with random-direction and shuffled-labels baselines. Additionally, it explores steering effects along both harmfulness and refusal directions, and tests jailbreak-like refusal suppression.

### Claim 3

**Original claim (verbatim from task.md):**

> Additive steering along either direction yields dissociated effects — moving along the harmfulness axis flips the model's internal judgment of whether an input is harmful, independent of whether it refuses; moving along the refusal axis flips refusal behaviour without changing that internal judgment.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Initial implementation of a pipeline to identify and analyze a linear 'harmfulness direction' in the residual stream activations of Llama-3-8B-Instruct. The experiment uses AdvBench (harmful) vs Alpaca (benign) prompts, extracts activations across layers (8, 12, 16, 20, 24, 28), computes a difference-of-means direction, and evaluates a 1D logistic regression probe with random-direction and shuffled-labels baselines. Additionally, it explores steering effects along both harmfulness and refusal directions, and tests jailbreak-like refusal suppression.

### Claim 4

**Original claim (verbatim from task.md):**

> A notable class of successful jailbreaks operates by suppressing the refusal signal while the harmfulness signal remains active — the model still internally recognises the input as harmful yet answers — a signature that a hidden-state probe can pick up.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Initial implementation of a pipeline to identify and analyze a linear 'harmfulness direction' in the residual stream activations of Llama-3-8B-Instruct. The experiment uses AdvBench (harmful) vs Alpaca (benign) prompts, extracts activations across layers (8, 12, 16, 20, 24, 28), computes a difference-of-means direction, and evaluates a 1D logistic regression probe with random-direction and shuffled-labels baselines. Additionally, it explores steering effects along both harmfulness and refusal directions, and tests jailbreak-like refusal suppression.

### Claim 5

**Original claim (verbatim from task.md):**

> A lightweight "Latent Guard" classifier trained on the harmfulness direction matches or beats a dedicated safety judge (Llama Guard 3 8B) at flagging jailbreak attempts, at a fraction of the compute.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> Initial implementation of a pipeline to identify and analyze a linear 'harmfulness direction' in the residual stream activations of Llama-3-8B-Instruct. The experiment uses AdvBench (harmful) vs Alpaca (benign) prompts, extracts activations across layers (8, 12, 16, 20, 24, 28), computes a difference-of-means direction, and evaluates a 1D logistic regression probe with random-direction and shuffled-labels baselines. Additionally, it explores steering effects along both harmfulness and refusal directions, and tests jailbreak-like refusal suppression.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
