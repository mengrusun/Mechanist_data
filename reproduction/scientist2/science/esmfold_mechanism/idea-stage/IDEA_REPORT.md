# Idea Report — Captured Behavior

**Direction**: ""（task.md 权威；`behavior_source=given` 从 task.md 直接、忠实捕获；`mechanism=discovery` 由本阶段加载 `/mechanism-explore` 塑造机制假设方向。）
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: task.md（`/data/zhenqian/Reproduction1/mechanica/science/esmfold_mechanism/task.md`，忠实捕获、逐字保留原文）
**Date**: 2026-07-15
**Pipeline**: research-lit → faithful behavior capture (from task.md) → research-refine-pipeline
**Output language**: 中文（结构化机器字段、路径、id、代码、模型/数据集名保留英文）

---

## Executive Summary

本报告忠实捕获 task.md 中给定的三条关于 ESMFold 折叠躯干（48-block folding trunk）的机制性 claim。所有三条 claim 会被 Phase 4.5 一起送入 `/research-refine-pipeline`，并被 refine 成一份**统一的**测试方法与实验路线图。**不做**理念/新颖性/影响力打分；**不设 M0**（because `BEHAVIOR_SOURCE=given`）。因 `MECHANISM=discovery`，本报告已根据 `/mechanism-explore` 决定所采用的机制研究方向组合：**Location → Causal Intervention → Unit Interpretation（linear-probing slice）**，故意不采用 Tuning & Editing / Formation Tracing / Decision Auditing。

---

## Literature Landscape

见 `idea-stage/LANDSCAPE.md`：
- **主线一**：AI 蛋白折叠模型内部机制研究处于起步阶段；ESMFold 48-block 折叠躯干 s/z 二元表示的 block-wise 因果地图是明显空白。
- **主线二**：机制可解释性工具（activation patching、causal mediation、linear probing、SAE steering）已成熟且方法学清晰。
- **主线三**：linear probe 与 causal 之间必须显式跨越（Quantitative Probing, arXiv 2209.03013）——claim 3 必须同时给出 "linear encoding" 与 "causal effect" 两层证据。
- **主线四**：β-hairpin 是 cross-strand 静电相容性敏感的最简 motif，DSSP 是二级结构判定的标准协议。
- **主线五**：PISCES cull list（12,055 条 X-ray、≤25% seq id、40–10000 长度）是业界通用的去冗余高质量集合。

**Structural Gaps 覆盖**：
- G1 = 折叠躯干 block 维度功能地图缺失 → 由 **claim 1** 直接测试。
- G2 = seq2pair 通道作为因果管道的角色未被 causally 验证 → 由 **claim 2** 直接测试。
- G3 = 早期 block 中化学特征（电荷）是否被线性编码并被模型因果使用 → 由 **claim 3** 直接测试。
- G4 = 三层结论未被串成一条 DSSP-based 的统一管线 → 本项目的实验路线图（`refine-logs/EXPERIMENT_PLAN.md`）覆盖。

**Policy note**：目标论文 arXiv 2602.06020（"Mechanisms of AI Protein Folding in ESMFold"）及其 ≥2602 的后续工作被项目 `.claude/forbidden-urls.txt` 列入禁读清单；本 landscape 与本报告**未阅读也未采纳**其任何内容，只使用 2602 之前的公开方法学与生物物理背景文献。

---

## Claims to Verify

### Claim 1: 早期 block 局部化 β-hairpin 折叠决策，且 `s` 是决策活跃位

**Original (verbatim excerpt from task.md, §Claim ①):**
> **Folding decisions for β-hairpin structures are localized in the early blocks of the folding trunk.** Whether the trunk will fold a target region into a β-hairpin is committed to within early blocks, and the sequence representation `s` is the active locus of that decision during this window.

**Extracted statement**: 在 ESMFold 48-block 折叠躯干中，是否把 target region 折成 β-hairpin 的决定，在早期 block window 内已被承诺（committed）；在此 window 内，per-residue 序列表示 `s`（而非 pair 表示 `z`）是该决策的活跃位（active locus）。

**Hypothesis**: H1 — 存在一个早期 block window，在此 window 内对 `s` 施加因果干预可有效改变 β-hairpin 是否形成（DSSP 判定），而在同一 window 之后对 `s` 或在此 window 内对 `z` 施加同强度干预效应显著更弱。

**Measurable predicate**: 在 PISCES cull list 抽取的一批带 native β-hairpin 目标区域的链上，对每个链在 block-window `[i, j]` 上对 `s`（分别的 `z`）做 activation patching / ablation，用 DSSP 对 predicted structure 标注：
- 存在一段 `[i*, j*]`（"早期"），使得在此 window 内对 `s` 的干预导致 target region 的 β-hairpin 形成率显著下降（相对于 unperturbed baseline，effect size 明确、p < 0.05 或对应严格统计量）；
- 相同强度的干预：（a）落在此 window 之后对 `s`——效应显著较弱；（b）落在此 window 内对 `z`——效应显著较弱。
- 定义 "早期" 为躯干前部（block index 明确在报告中给出，例如前 1/4~1/3；具体阈值由实验路线图确定，符合"claim-time 只锁定 kind，不锁定 exact index"的原则）。

**Expected direction**: down（早期对 `s` 干预使 hairpin 形成率下降）。

**Resources (preferred, cost-aware — 非 reproduction combo，不 stamp strict)**:
- Model: ESMFold 完整流水线（ESM-2 backbone + 48-block folding trunk + structure module）——task.md 明确的**唯一**机制分析对象。
- Dataset: PDB 结构文件，按 PISCES cull list `cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055` 过滤后使用（12,055 chains / 11,633 entries）。
- used_n: unspecified — resolve in Phase 4.5（在 10-hour GPU 预算内选取足够多样本以获得统计功效；由 EXPERIMENT_PLAN.md 定 n）。

**Status**: pending verification
**Notes**: 需要 matched control：对同一链上非 β-hairpin 的对照区域施加同强度干预，验证 specificity。

---

### Claim 2: early-block seq2pair 通路是把 β-hairpin 决策从 `s` 写入 `z` 的关键通道

**Original (verbatim excerpt from task.md, §Claim ②):**
> **The early-block seq2pair pathway is the critical channel through which the β-hairpin folding decision is transferred from `s` into `z`.** In the early blocks, the **seq2pair** operation carries the decision to fold a target region into a β-hairpin from the sequence representation `s` into the pairwise representation `z`.

**Extracted statement**: 在早期 block 内，seq2pair 操作（elementwise outer-product 类操作，把 `s` 中的信息写入 `z`）是承担"把是否折成 β-hairpin 的决定从 `s` 传递到 `z`"这一因果职能的关键通道；相比之下 pair2seq（由 `z` 派生的 attention bias 调制 `s`）在此 window 内不承担该角色。

**Hypothesis**: H2 — 在早期 block 内对 seq2pair 通路做因果干预（例如：将其输出替换为 patched / frozen / donor 值），会导致 target region 上 β-hairpin 形成率显著下降，且 pair2seq 通路承受同强度对照干预时效应显著较弱。

**Measurable predicate**: 在同一批链上对每链取 target β-hairpin 区域：
- 在早期 block window 内 patch/ablate/freeze seq2pair 的输出（保持其他通路不动），DSSP 对 predicted structure 标注 β-hairpin 是否形成；
- 对 pair2seq 通路做**匹配的**对照干预（同 block window、同强度、同 residue mask）；
- 结果满足：seq2pair 干预 → hairpin 率显著下降；pair2seq 匹配干预 → 效应显著较小；两条效应差异 statistically 显著。

**Expected direction**: down（seq2pair 干预使 hairpin 率下降；pair2seq 匹配干预无同强度效应）。

**Resources**: 与 claim 1 相同（同一 ESMFold pipeline、同一 PISCES 数据源）。

**Status**: pending verification
**Notes**: 需要严格 pathway-level 干预实现（在 forward 中拦截 seq2pair / pair2seq 的中间张量并替换/冻结），并保证 residue mask、block window、干预强度在两侧对齐。

---

### Claim 3: charge 是早期 block 中线性编码的化学特征，且对 β-hairpin 形成有因果影响

**Original (verbatim excerpt from task.md, §Claim ③):**
> **Charge is a linearly encoded chemical feature in early blocks of ESMFold and causally influences β-hairpin formation.** This charge feature exerts a causal effect on β-hairpin formation, consistent with the physical principle that opposite-charge residues on facing β-strands favor pairing (same-charge configurations correspondingly increase cross-strand distance). The evaluation of β-hairpin formation must be based on DSSP secondary-structure assignment on the predicted structure.

**Extracted statement**: 在早期 block 的 `s`（per-residue 序列表示）上，残基电荷（正 / 中性 / 负，或 signed charge）是可被线性 probe 从表示中解码的化学特征；且沿该线性方向对 `s` 施加因果 steering，会以物理上一致的方式改变 target β-hairpin 区域的 hairpin 形成率——具体而言：把两根 facing β-strand 上一对 cross-strand 残基的 charge 方向"同号化"（same-charge configuration）应减少 hairpin 形成 / 增大 cross-strand 距离；"异号化"（opposite-charge configuration）应增加 hairpin 形成 / 减少 cross-strand 距离。

**Hypothesis**: H3 — 两部分：
- H3a（线性编码）：在早期 block 的 `s` 上，用 held-out linear probe 能显著优于 chance 地从残基表示预测其净电荷（sign 或 signed 值）。
- H3b（因果效应）：在早期 block 沿该 probe 方向对 target 区域上一对 cross-strand 残基做 steering（把电荷方向推向 "same" 或 "opposite"），DSSP 判定的 β-hairpin 形成率会以预测的方向（"same" 下降 / "opposite" 上升；或对应的 cross-strand 距离度量方向）改变，且具备 dose-response（沿方向的强度越大，效应越大）以及 matched-control specificity（对同一链上其他非 target 残基做相同强度 steering，效应显著较弱）。

**Measurable predicate**:
- **H3a**：在 held-out chains 上，probe 对残基电荷（三分类或 signed 回归）的判定准确率 / 相关系数显著优于 chance（sanity threshold 由 report 定，例如 accuracy ≥ 0.75 或 Pearson r ≥ 0.5 等，具体门槛由 Phase 4.5 与 experiment plan 敲定，且要设 permutation / label-shuffle baseline）。
- **H3b**：在同一 target 区域的 cross-strand 一对残基上做 steering，effect on DSSP-β-hairpin rate（或 cross-strand `Cα-Cα` distance）满足 predicted sign，dose-response 单调可分辨（例如至少 3 个 dose 点显示单调趋势），且 matched-control 的 effect size 显著较小（p < 0.05 或对应统计量）。

**Expected direction**:
- H3a: probe > chance（up）。
- H3b: same-charge steering → hairpin 形成率 down / cross-strand distance up；opposite-charge steering → hairpin 形成率 up / distance down。

**Resources**: 与 claim 1、2 相同（ESMFold, PISCES）。

**Status**: pending verification
**Notes**:
- 关键防呆（受 arXiv 2209.03013 启发）：**probe accuracy 高 ≠ 网络因果使用**——H3a 与 H3b 必须同时成立才等于 claim 3 成立；只做 probe 不够。
- 评估必须使用 **DSSP** 对 predicted structure 标注（task.md 硬约束）。cross-strand distance 是可选的辅助度量。
- 电荷特征在残基层面是良定义的（Asp / Glu 负、Lys / Arg / His 正、其余中性），groundtruth 从 primary sequence 直接可得。

---

## Faithfulness Audit（Phase 2 提取规则的说明）

- **无拆分**：三条 claim 在 task.md §Claim 中已被作者独立列出，且各自独立可验证，本报告一比一保留三条。
- **无合并**：三条 claim 的因果指向不同（block-window / pathway / linearly-encoded chemistry），未合并。
- **无更改语义、方向、强度**：所有 Hypothesis / Measurable predicate / Expected direction 严格由 task.md 原文推导。
- **补充结构化字段**：Hypothesis / Measurable predicate / Expected direction / Resources 是**从 task.md 原文严格派生**的结构化重述，不添加 task.md 未主张的新论断。
- **Resources 记录（非 reproduction combo）**：task.md **明确要求**只机制分析 ESMFold（"the sole model whose internals are being mechanistically analysed is **ESMFold**"），也**明确指定** PISCES cull list 作为实验数据源。因 `BEHAVIOR_SOURCE=given` + `MECHANISM=discovery`，本组合**不**是 reproduction combo，故 Resources 记录为 preferred + cost-aware，**不 stamp `resource_fidelity: strict`**。但 task.md 声明了 **10-hour GPU 预算**——按 auto-claim 规范"当声明预算能覆盖 preferred 全尺度时，plan at full scale"——实验路线图会按此原则安排 n 与 block 网格。
- **DSSP 评估硬约束**：task.md 明确要求 β-hairpin 判定必须基于 predicted structure 的 DSSP 标注；本报告在三条 claim 的 Measurable predicate 中均严格贯彻。

---

## Mechanism Strategy（由 `/mechanism-explore` 在 Phase 1.75 决定）

- **directions（execution order）**：Location → Causal Intervention → Unit Interpretation（linear-probing slice）
- **rejected**：
  - **Tuning & Editing** — 三条 claim 是 diagnostic（X 是否引起 B），不是 applied（用 X 提升下游任务）；扩展会脱离忠实复现的范围并挤占 10-hour 预算。
  - **Formation Tracing** — 三条 claim 都不涉及 genesis（何时形成、由哪些训练数据驱动）；该方向最贵且不必要。
  - **Decision Auditing** — 三条 claim 的问题是"机制是否如所述"，不是"模型决策是否可靠 / 是否依赖 spurious 特征"。
- **note**：三个 direction 的组合刚好覆盖三条 claim：Location 支撑 claim 1 的 block window 与 s/z 定位；Causal Intervention 是 claim 1 / 2 / 3 因果部分的公共工具；Unit Interpretation（linear-probing）承担 claim 3 的 "linear encoding" 部分并 name 出 steering 所需的方向。

（此机制策略块会由 Phase 4.5 的 `/research-refine-pipeline` 逐字复写入 `refine-logs/EXPERIMENT_PLAN.md` 与 `refine-logs/FINAL_PROPOSAL.md` 顶部元数据。）

---

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`（统一的三-claim 测试方法）
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`（每条 claim 一个 milestone，claim-tagged）
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

---

## Next Steps

- [ ] `/mechanism-skills` 路由到具体的机制家族 + submethod（Workflow 1.25，将 Location + Causal Intervention 具体化为 activation-patching / probing / steering 等 submethod）
- [ ] `/auto-experiment` 从路由 + 实验计划实现并部署实验（Workflow 1.5）
- [ ] `/auto-verify` 在通过后做 CATH-stratified swap variant 的鲁棒性检验（Workflow 1.75，task.md verify-stage 允许该 dataset）
- [ ] `/auto-iteration-loop` 迭代到 reviewer-ready（Workflow 2）
- [ ] 或直接 `/auto` 走 claim → routing → experiments → verify → review 端到端链
