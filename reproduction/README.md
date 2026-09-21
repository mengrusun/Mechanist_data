# Reproduction

18 篇机制可解释性论文的多 agent 复现与人工评审工作区。

## Guide

### Human Judge Guide

**任务**：人类专家评估 scientist1 / scientist2 / scientist3 三个 agent 对同一篇论文的复现情况，并将评审结果提交到本仓库。允许使用 AI 辅助（如让 LLM 先按同一份评分指南跑一遍作为参考），但**最终评分与 justification 由人类专家负责**。

#### 1. 阅读评审标准

- 评分指南（人工专家版）：`human_reliablity_judge.md`（英文）/ `human_reliablity_judge_zh.md`（中文），二者内容一致。仓库根目录下的 `reliablity_judge.md` / `reliablity_judge_zh.md` 是 LLM judge 使用的版本，专家评审请以 `human_` 前缀的两份为准。
- 指南定义 9 个可靠性维度，每个维度按 **1 / 3 / 5** 三档锚点评分；介于锚点之间可给 **2 / 4**。
- 维度 9（复现忠实度）拆成 **method / experiment / result** 三个子角度，每个子角度独立打分。
- 不适用的维度（例如无标签、无因果主张）填 `n/a` 并在 justification 中说明理由。

#### 2. 定位相关材料

针对每一个待评审的 `<category>/<exp_dir>/`（例如 `feature_description/multi_modal_feature_description/`），需要横向对照四个位置：

| 位置 | 用途 |
|---|---|
| `paper/<category>/<exp_dir>/` | 原论文 PDF——复现基准 |
| `source_code/<category>/<exp_dir>/` | 原论文开源代码（若有）——辅助佐证 |
| `scientist{1,2,3}/<category>/<exp_dir>/` | 三个 agent 各自的复现产物 |
| `human_judge/<category>/<exp_dir>/` | 你要填写的评审目录 |

`<category>/<exp_dir>` 与原论文标题的对照见根目录 `paper_mapping.md`。

#### 3. 逐 agent 打开产物取证

三个 scientist 的产物组织方式不同，下面给出各自的入口清单；完整字段说明参见 `human_reliablity_judge_zh.md` 中"输入数据"一节。

**scientist1** — 产物结构灵活、无固定目录职能：

- 先浏览 `{case_dir}` 根目录与顶层子目录，识别脚本 / 结果 / 日志 / 最终报告；
- **推荐借助 LLM 辅助取证**：把文件树和可疑目录列表交给 LLM，让它先给出候选证据索引，再由人类核对；
- 关注任意可读汇总（README / *.md / *.log / summary JSON 等）里列的关键数字，逐一回追到落盘 JSON / CSV / log。

**scientist2** — 产物按固定阶段组织：

- `task.md` — 复现目标
- `CLAIMS_LEDGER.md` / `claims_ledger.json` — 每个 claim 的最终状态
- `idea-stage/IDEA_REPORT.md` — claim 阶段的 ideation 与最终提炼
- `refine-logs/EXPERIMENT_PLAN.md` / `FINAL_PROPOSAL.md` / `MECHANISM_ROUTING.md` — 计划的实验设计与机制选取
- `refine-logs/EXPERIMENT_RESULTS.md` / `EXPERIMENT_TRACKER.md` / `main-experiment-verdicts.json` / `PIPELINE_SUMMARY.md` — 实际跑了什么、主实验判定与跨阶段汇总
- `results/<milestone>/**/*.json` / `runs/<run_id>/` — 落盘原始数字（维度 7 溯源必查）
- `verify/VERIFY_REPORT.md` / `verify/INTEGRITY_AUDIT.md` / `verify/STAGE2_PICK.json` / `verify/<claim>/*_audit/*.json` — 稳健性与完整性审计
- `review-stage/AUTO_ITERATION_FINAL_REPORT.md`（+ `AUTO_REVIEW.md` / `REVIEW_STATE.json`）— 迭代日志（判断是否为凑结论而调参 / 移动门柱）

**scientist3** — 产物按"分阶段迭代 + 搜索树"组织：

- `idea.md` / `idea.json` — 输入 scientist3 的任务描述
- `bfts_config.yaml` — 运行配置
- `logs/0-run/stage_{1..4}_.../` — 各阶段（初始实现 → baseline 调优 → 探索性研究 → 消融）的 `journal.json`、`tree_data.json` / `tree_plot.html`、`best_solution_*.py`、`notes/`（对"是否充分尝试超参数"判定有参考价值）
- `logs/0-run/experiment_results/experiment_<hash>/` — 最终实验结果，维度 7 / 8 一手材料
- `logs/0-run/*_summary.json`（`research_summary` / `baseline_summary` / `ablation_summary` / `draft_summary`）— 各阶段汇总
- `report/` — 主结果汇总（若存在，作为 claim 判定的一手材料）
- `token_tracker.json` — token 消耗，可交叉检验运行规模

**关键数字必须回到落盘 JSON 核对**，不要只信汇总报告的转述。

#### 4. 填写评审 JSON

- 从 `template/template.json` 复制一份，重命名为 `<exp_dir>-scientist{i}-<expert_name>.json`（例如 `multi_modal_feature_description-scientist2-Zhenqian_Xu.json`）。
- 填写 `meta.expert_name` 与 `meta.review_date`；`meta.agent` 与 `meta.paper_title` 已由发放方预填。
- 每个维度按顺序写 `justification` → `score` → `evidence_files`：
  - **justification 必须引用具体证据**（文件名 + 具体数字 / 路径 / 模型 id / 原句）；
  - **evidence_files 填相对路径**（相对于该 scientist 的 exp_dir，例如 `runs/M6_C1_last_layer/purity__k16__mean.json`）。
- 维度 9 下每个 claim 分别填 `method` / `experiment` / `result` 三个子角度的 justification 与 score。
- 不做跨维度或跨 claim 的综合聚合分数。

#### 5. 提交到仓库

- 把三份 JSON 放到 `human_judge/<category>/<exp_dir>/` 下（每个 scientist 一份）。

#### 6. 提交时间

（北京时间）7月16日14：00前提交到仓库。

#### 7. 其他事项

## 顶层目录

```
Reproduction/
├── paper/                 # 原论文 PDF
├── source_code/           # 原论文开源代码
├── scientist1/            # Agent 1 的复现产物
├── scientist2/            # Agent 2 的复现产物
├── scientist3/            # Agent 3 的复现产物
├── human_judge/           # 人工专家对 3 个 scientist 的评审结果
├── template/                     # 评审 JSON 模板
├── paper_mapping.md              # exp_dir ↔ 原论文标题对照表
├── human_reliablity_judge.md     # 人工专家评审指南（英文）
├── human_reliablity_judge_zh.md  # 人工专家评审指南（中文）
├── reliablity_judge.md           # LLM judge 评审指南（英文）
└── reliablity_judge_zh.md        # LLM judge 评审指南（中文）
```



## 子文件夹说明

### `paper/`
原论文 PDF。按 `<category>/<exp_dir>/` 两级组织，与其他目录对齐。

### `source_code/`
原论文的开源代码库（若发布），作为评审时的旁证材料。若某论文无开源实现则对应子文件夹可缺省或为空——以 `paper/` 为准。

### `scientist1/`、`scientist2/`、`scientist3/`
三个不同 agent 各自跑出的一整套复现产物（`refine-logs/`、`runs/`、`results/`、`verify/`、`CLAIMS_LEDGER.md` 等）。为了控制体积，各 agent 目录下已删除 hidden state / activation / 模型权重等大型中间产物，保留 `.json` / `.jsonl` / 代码 / 报告 / 图。

### `human_judge/`
人工专家对每个 scientist 的评审结果。目录结构：

```
human_judge/<category>/<exp_dir>/
├── task.md
├── <exp_dir>-scientist1-<expert_name>.json
├── <exp_dir>-scientist2-<expert_name>.json
└── <exp_dir>-scientist3-<expert_name>.json
```

- `task.md`：复现目标（论文行为、待复现 claim、指定的模型/数据/SAE 资源），与 `scientist*/<category>/<exp_dir>/task.md` 一致。
- 三个评审 JSON：同一位专家分别为 scientist1/2/3 打分，文件名末段将 `expert_name` 替换为专家实际姓名（如 `Zhenqian_Xu`）。

### `template/`
`template.json`：评审文件的空白模板。JSON 内 9 个维度的评分标准参见 `human_reliablity_judge.md`。填写规则：
- `meta.expert_name` / `review_date` 由专家填写；
- `meta.agent` / `meta.paper_title` 已按 scientist 编号与所在论文预填。

### `paper_mapping.md`
`<category>/<exp_dir>` 目录名与原论文标题的对照表（18 篇）。评审时可据此快速定位某个 exp_dir 对应哪篇论文，或反向查询某篇论文位于哪个目录。

### `human_reliablity_judge.md` / `human_reliablity_judge_zh.md`
**人工专家评审指南**（英/中）。定义 9 个可靠性维度（数据划分、标签有效性、资源保真、证据充分、统计严谨、因果 claim 有效、结果溯源、跨产物一致、复现忠实度）及 1/3/5 打分标准；含三个 scientist 各自的产物结构说明。

### `reliablity_judge.md` / `reliablity_judge_zh.md`
**LLM judge 使用的评审指南**（英/中）。评分维度与 human 版一致，但保留了流水线特定术语，供程序化审计脚本调用。人工专家评审请以 `human_` 前缀的两份为准。

## 组织约定

- 两级层级 `<category>/<exp_dir>/`（例如 `belief/closing_gap_belief`）在 `paper`、`source_code`、`scientist{1,2,3}`、`human_judge` 之间完全对齐——评审时可直接横向对照同一 `<exp_dir>` 下四个目录的产物。
- 18 篇论文覆盖 9 大类：belief、emotion、feature_description、multi-agent_safety、multilingual、multimodal、reasoning、safety、science。

## 其他说明
scientist1: claude code
scientist2: Mechanist 
scientist3: AI-Scientist-v2
（该说明评估专家/llm judge不可见）
