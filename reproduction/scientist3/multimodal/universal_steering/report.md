# Reproduction Report: `universal_steering`

- **Category:** `multimodal`
- **Experiment directory:** `scientist3/multimodal/universal_steering`
- **Task definition:** `task.md` (see AI-Scientist-v2 reproduction_tasks)

## 1. Global Reproduction Overview

**Theme / hypothesis:** Research Hypothesis: Extracting and Utilizing Concept Representations from AI Model Internals for Steering and Monitoring

**Pipeline status:** completed (stages 1–4 + finalization markers)

**BFTS stages observed:**
- `stage_1_initial_implementation_1_preliminary` — nodes: 3 (good: 3, buggy: 0, best_solutions: 1)
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

> This is the seed implementation establishing a working baseline for concept-vector steering with a simple supervised linear direction (difference-of-means as an RFM proxy). It demonstrates that lightweight linear interventions can substantially shift model behavior for some concept classes (moods, personas) while revealing large disparities across others (experts, topophiles), which is important for understanding which concept types are amenable to activation steering. The layer-wise norm analysis also provides guidance on optimal intervention layers for future work.

### Per-claim verdict

| # | Verdict |
|---|---------|
| 1 | See overall verdict (claim 1/5). — *RFM--a supervised feature learning algorithm can extract per-block linear concep…* |
| 2 | See overall verdict (claim 2/5). — *Steering can improve performance on high-precision tasks: steering from Python t…* |
| 3 | See overall verdict (claim 3/5). — *Concept representations are transferable across human languages: concept vectors…* |
| 4 | See overall verdict (claim 4/5). — *Concept vectors are composable: linear combinations of multiple concept vectors …* |
| 5 | See overall verdict (claim 5/5). — *Internal concept features are more effective for monitoring misaligned content (…* |

## 3. Where to Find Artifacts

| What | Path |
|------|------|
| Idea snapshot | `scientist3/multimodal/universal_steering/idea.json` |
| Human-readable idea | `scientist3/multimodal/universal_steering/idea.md` |
| BFTS config copy | `scientist3/multimodal/universal_steering/bfts_config.yaml` |
| Tree visualization | `scientist3/multimodal/universal_steering/logs/0-run/unified_tree_viz.html` |
| Plots & run outputs (98 files) | `scientist3/multimodal/universal_steering/logs/0-run/experiment_results` |
| Overall draft summary (best starting point for what was run) [✓] | `scientist3/multimodal/universal_steering/logs/0-run/draft_summary.json` |
| Research-stage summary with best-node metrics [✓] | `scientist3/multimodal/universal_steering/logs/0-run/research_summary.json` |
| Baseline tuning summary [✓] | `scientist3/multimodal/universal_steering/logs/0-run/baseline_summary.json` |
| Ablation studies summary [—] | `scientist3/multimodal/universal_steering/logs/0-run/ablation_summary.json` |
| Completion markers | token_tracker.json: ✓, auto_plot_aggregator.py: ✓ |

## 4. Per-Claim Experimental Configuration

### Claim 1

**Original claim (verbatim from task.md):**

> RFM--a supervised feature learning algorithm can extract per-block linear concept vectors from the internal activations of large-scale AI models, and adding these vectors to activations during inference effectively steers model behavior toward or away from the target concept. Example application scenarios: Anti-refusal (jailbreak)，Political stance，Honesty (negative steering).

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal working RFM-style concept-vector steering pipeline was built on Llama-3.1-8B-Instruct. The system extracts linear 'concept vectors' via difference-of-means over hidden states at a chosen layer for positive vs negative prompts across 5 concept classes (fears, experts, moods, topophiles, personas), then applies additive residual-stream steering via a forward hook during generation. Outputs are judged by GPT-4o for concept reflection, producing per-class and overall steering success rates.

### Claim 2

**Original claim (verbatim from task.md):**

> Steering can improve performance on high-precision tasks: steering from Python to C++ when answering coding questions on algorithm problems with different task difficulties, a C++ concept vector can raise llm's test-case pass rate, beating both the default Python output and "Answer in C++" prompting.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal working RFM-style concept-vector steering pipeline was built on Llama-3.1-8B-Instruct. The system extracts linear 'concept vectors' via difference-of-means over hidden states at a chosen layer for positive vs negative prompts across 5 concept classes (fears, experts, moods, topophiles, personas), then applies additive residual-stream steering via a forward hook during generation. Outputs are judged by GPT-4o for concept reflection, producing per-class and overall steering success rates.

### Claim 3

**Original claim (verbatim from task.md):**

> Concept representations are transferable across human languages: concept vectors trained on English text can steer model responses in other languages (e.g., Chinese, French, Spanish).

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal working RFM-style concept-vector steering pipeline was built on Llama-3.1-8B-Instruct. The system extracts linear 'concept vectors' via difference-of-means over hidden states at a chosen layer for positive vs negative prompts across 5 concept classes (fears, experts, moods, topophiles, personas), then applies additive residual-stream steering via a forward hook during generation. Outputs are judged by GPT-4o for concept reflection, producing per-class and overall steering success rates.

### Claim 4

**Original claim (verbatim from task.md):**

> Concept vectors are composable: linear combinations of multiple concept vectors enable simultaneous multi-concept steering.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal working RFM-style concept-vector steering pipeline was built on Llama-3.1-8B-Instruct. The system extracts linear 'concept vectors' via difference-of-means over hidden states at a chosen layer for positive vs negative prompts across 5 concept classes (fears, experts, moods, topophiles, personas), then applies additive residual-stream steering via a forward hook during generation. Outputs are judged by GPT-4o for concept reflection, producing per-class and overall steering success rates.

### Claim 5

**Original claim (verbatim from task.md):**

> Internal concept features are more effective for monitoring misaligned content (hallucinations, toxic content) than LLM judges that evaluate outputs directly — even when the features come from smaller open-source models compared to powerful judges like GPT-4o.

**Source:** `task.md` → section `## Claim`

**Configured resources (from task.md `## Resources`):**

- DATA_DIR: `/data/zhenqian/data`
- MODEL_DIR: `/data/zhenqian/models`
- (see full Resources section in task.md)

**What the agent actually ran (from draft_summary, if available):**

> A minimal working RFM-style concept-vector steering pipeline was built on Llama-3.1-8B-Instruct. The system extracts linear 'concept vectors' via difference-of-means over hidden states at a chosen layer for positive vs negative prompts across 5 concept classes (fears, experts, moods, topophiles, personas), then applies additive residual-stream steering via a forward hook during generation. Outputs are judged by GPT-4o for concept reflection, producing per-class and overall steering success rates.

---
*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*
