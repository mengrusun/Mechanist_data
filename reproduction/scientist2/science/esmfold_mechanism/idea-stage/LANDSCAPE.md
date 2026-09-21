# Landscape: ESMFold 折叠躯干机制可解释性——β-发夹决策的早期 block 定位、seq2pair 通路、电荷线性编码与因果

**Date**: 2026-07-15
**Scope**: Interpreted as: 对 ESMFold（含 ESM-2 主干 + 48-block folding trunk + structure module）的**内部机制**做可解释性分析，检验三条已给定的机制 claim（早期 block 定位 & `s` 为决策活跃位；seq2pair 为把决策从 `s` 写入 `z` 的关键通道；charge 作为线性编码的化学特征且对 β-hairpin 有因果影响）。β-hairpin 形成的判定必须基于对 predicted structure 的 DSSP 二级结构标注。
**Policy scope**: 本 landscape 有意避开被禁的目标论文 arXiv 2602.06020 及其 ≥2602 的后续工作；只使用 2602 之前的公开方法学与背景文献。
**Based on**: 15 retrieved papers — see `RESEARCH_LIT.md` for the raw retrieval dump.

---

## 1. Structured Paper Table

| # | Paper (标题缩写) | 年份 | 方法 / 主题 | 关键结果 / 用途 | 与本工作的关系 | 来源 |
|---|---|---|---|---|---|---|
| 1 | How to use and interpret activation patching | 2024 | activation patching 方法论 | clean/corrupted 设计、指标选择、常见误读 | 为 claim 1/2 的因果实验提供方法学基线 | arXiv |
| 2 | Interpreting and Steering PLMs via SAE | 2025 | 稀疏自编码 + steering | 展示可从 PLM 抽取可解释稀疏特征并 steering 引发功能变化 | claim 3 的 linear probe → steering 模板 | arXiv |
| 3 | ProtSAE | 2025 | Semantic-guided SAE | 结构/功能属性的解耦 | 支持 PLM 表示可线性分解，间接支持 claim 3 | arXiv |
| 4 | Endowing PLMs with Structural Knowledge | 2024 | 结构信号注入 + 层间 probing | ESM-2 / ESMFold 内部结构信号的层间画像 | claim 1 的层间 probe 基线 | arXiv |
| 5 | Activation Patching Reveals Language-Agnostic Concepts | 2024 | activation patching 在 concept 上的定位 | clean/corrupted patch 定位 concept 计算 | 为 block-wise 定位 β-hairpin 决策提供模板 | arXiv |
| 6 | Quantitative Probing | 2022 | 因果 probing 方法论 | 高 probe accuracy ≠ 因果使用 | claim 3 必须补足 causal 步骤（不能只做 linear probe） | arXiv |
| 7 | Complex folding pathways in a simple β-hairpin | 2003 | β-发夹折叠动力学 | β-发夹形成对残基相容性敏感 | 支持"cross-strand 静电相容性"为物理先验 | arXiv (q-bio) |
| 8 | Dominant Folding Pathways of a β-Hairpin | 2009 | β-发夹折叠路径 | turn-first vs hydrophobic-collapse-first | 生物物理背景 | arXiv |
| 9 | Wako-Saito-Munoz-Eaton β-hairpin | 2014 | β-发夹热力学模型 | 相图与临界行为 | 理论对照 | arXiv |
| 10 | AlphaFold predicts complex protein knots | 2022 | 结构预测复现罕见拓扑 | 折叠躯干确实承载可复现的折叠决策 | 支持"decision is committed to inside the trunk" | arXiv |
| 11 | Coarse-grained CABS 1-D structural properties | 2015 | DSSP-类二级结构标注 | 二级结构判定协议 | DSSP 评估协议参考 | arXiv |
| 12 | Novel Approach for Protein Structure Prediction | 2012 | 领域历史背景 | 神经网络之前的势能方法 | 领域脉络 | arXiv |
| 13 | Circuit Tracing in Autoregressive PLMs | 2026 | 电路追踪 | **不阅读**（arxiv id ≥ 2602）；仅作存在性证据 | 说明本方向的活跃 | arXiv |
| 14 | Residue-Level Attributions in PLMs | 2026 | 残基归因 | **不阅读**（arxiv id ≥ 2602） | 存在性证据 | arXiv |
| 15 | (以其他重复搜索结果为占位) | — | — | — | — | — |

## 2. Core Landscape Narrative

**主线一：AI 蛋白折叠模型的内部机制研究处于起步阶段。**
ESMFold 的公开架构文档说明其折叠躯干由 48 个迭代 block 组成，每 block 维护 per-residue 的序列表示 `s` 和 per-pair 的表示 `z`；两个方向的信息传递分别由 **seq2pair**（elementwise outer-product 类操作，把 `s` 写入 `z`）与 **pair2seq**（从 `z` 派生的 attention bias 调制序列注意力）承担。这套 s/z 二元表示是 AlphaFold Evoformer 的核心继承，也是"如何把序列信号 iteratively 转化为空间信号"的表征桥梁。尽管在结构预测精度、稳定性、复杂拓扑（含 knot）复现层面已有若干工作证明 ESMFold / AlphaFold 承载了正确的折叠决策，但**在 48-block trunk 内部对某一具体折叠决定（例如 β-发夹是否形成）进行 mechanistic decomposition 的公开研究极少**。本工作的三条 claim 恰好定位在这一空白：早期 block 是决策 window、`s` 是决策活跃位、seq2pair 是通道、charge 是线性编码且因果有效。

**主线二：机制可解释性的通用工具已经成熟。**
在 LLM 领域，activation patching、causal mediation、linear probe、sparse autoencoder（SAE）steering、logit / tuned lens 等一系列方法已有清晰的方法学与最佳实践（Paper 1、Paper 5）；同时，PLM 上的 SAE + steering（Paper 2、Paper 3）也已经被验证：残差流中可分离出与生物学属性对齐的稀疏方向、可通过在编码空间加/减向量对下游预测产生可控效应。这两类方法为本工作提供了完整的方法箱——**问题不再是"有没有工具"，而是"如何把工具组合成一个能检验三条 claim 的最小、干净、可复现的实验管线"**。

**主线三：linear probe 与 causal 之间的鸿沟必须显式跨越（对 claim 3 尤其关键）。**
Paper 6（Quantitative Probing）明确指出：能从表示中线性解码某属性 ≠ 网络在计算下游任务时因果地使用该属性。因此 claim 3 的完整链条**必须同时包含**（a）在早期 block 的 `s` 上训练线性 probe，证明电荷可线性解码；（b）沿 probe 方向对 `s` 施加因果干预（例如在 target 区域两 β-strand 上翻转电荷方向或做 steering），并通过 **DSSP 判定** 该 target 区域在 predicted structure 中是否被折成 β-hairpin 来观测因果效应；这两步缺一不可。

**主线四：β-hairpin 作为 canonical motif 的选择是正当且可操作的。**
β-hairpin 是最基本的 β-sheet 拓扑（两根相邻反平行 β-strand + 中间的 turn），其形成受到 turn 首选残基、hydrophobic 面性、以及 cross-strand 侧链相容性（含**相反电荷相邻优势**、相同电荷排斥）的共同影响（Paper 7–9）。因此把 β-hairpin 作为"最简结构决策"探针，并把 charge 作为最基本的化学特征，既有物理动机、又有可测量的输出（DSSP 判定的 SS 串）。

**主线五：数据与评估协议已成熟且高度约束。**
`task.md` 指定的 PISCES cull list `cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055` 提供了 12,055 条非冗余 X-ray 链（≤25% seq identity、resolution 0.0–2.5 Å、R ≤ 0.3、长度 40–10000）——这是一个业界通行的高质量、去冗余训练/评估集合，且严格避免 sequence identity 引起的信息泄漏。**评估必须使用 DSSP 对 predicted structure 的二级结构标注**是 task 的硬性约束（HARD），本工作将其贯彻到所有 β-hairpin 判定环节。

## 3. Sub-direction-Specific Work

### 3.1 折叠躯干中的层间信息流
- **代表工作**：Paper 4（Endowing PLMs with Structural Knowledge，2024）——层间 probing 协议在 ESM-2/ESMFold 上的可迁移基线。
- **一句话 takeaway**：mid-to-late 层的表示对结构/功能任务是最强的可解码位，但**折叠躯干内部各 block 的角色差异**尚未被系统刻画。
- **留下的 gap**：block 维度（尤其"早期 vs 晚期"的划分与转折点）在折叠躯干内的功能地图缺失。

### 3.2 activation patching 与 causal mediation
- **代表工作**：Paper 1（How to use and interpret activation patching）、Paper 5（Language-Agnostic Concepts via Patching）。
- **takeaway**：clean/corrupted 二分设计 + block/site 网格 patching 是把某一决策定位到网络某个"空间-时间坐标"的成熟工具。
- **gap**：绝大多数工作在 LLM 上做 concept / token-level 因果定位；把这一范式**迁移到 s/z 二元表示 + 48-block trunk 的蛋白折叠模型**，几乎没有公开先例。

### 3.3 PLM 上的稀疏方向、steering、causal 可解释性
- **代表工作**：Paper 2（Steering PLMs via SAE）、Paper 3（ProtSAE）、Paper 6（Quantitative Probing）。
- **takeaway**：PLM 残差流内可分离出与生物属性对齐的稀疏方向，且 steering 可给出因果证据；但必须在方法学层面严格区分 correlational probe 与 causal use。
- **gap**：这些工作聚焦 ESM-2 / ProGen 等的表示层，而**ESMFold 折叠躯干内部**（尤其是 `s → z` 的写入通道 seq2pair）作为线性化学特征的传输载体，缺乏专门研究。

### 3.4 β-hairpin 生物物理背景与 DSSP 评估
- **代表工作**：Paper 7–11。
- **takeaway**：β-hairpin 形成对 cross-strand 残基相容性（含电荷）敏感；DSSP 是标准 SS 判定协议。
- **gap**：把这些生物物理先验**用作机制假设的可 falsify 版本**（即"charge 是模型内部真的用到的因果因子，而不仅是训练数据中的相关模式"）是本工作的目标。

## 4. Structural Gaps（本工作的定位）

- **Gap G1** — **48-block 折叠躯干内部的 block 维度功能地图缺失（尤其早期 vs 晚期的功能差异）**。*Competitive set*: Paper 4 提供层间 probing 基线；Paper 5 提供 activation patching 定位范式。*Why open*: 现有工作或者按"层"分析 ESM-2 主干、或者在 LLM 上做 concept-level 定位，几乎没有在 ESMFold 折叠躯干上做 block-wise 因果定位的公开研究。→ 本工作 claim 1 直接测这个 gap。
- **Gap G2** — **seq2pair 通道作为"从 `s` 写入 `z` 的关键管道"这一角色，在公开可解释性文献中未被 causally 验证**。*Competitive set*: Paper 1（patching 方法学）。*Why open*: `s` 与 `z` 的架构分工是 Evoformer/ESMFold 的公开设计，但"哪条通道是特定折叠决策的实际因果通道"仍是空白。→ 本工作 claim 2 测这个 gap。
- **Gap G3** — **在 ESMFold 折叠躯干中，chemistry-级特征（如电荷）是否被线性编码、且被模型因果使用**——尤其对 β-hairpin 这样对电荷相容性敏感的 motif 未有公开答案。*Competitive set*: Paper 2、Paper 3、Paper 6。*Why open*: PLM 上的 SAE / probing 集中在 ESM-2 主干；对 ESMFold 折叠躯干 `s` / `z` 内部化学特征的线性可解码 + 因果 steering 未被系统研究。→ 本工作 claim 3 测这个 gap。
- **Gap G4** — **在蛋白结构预测模型上，把"机制局部化 + 通道识别 + 特征-因果"三个层级串成一条 unified 实验管线**——文献中或者只做一层（layer probe 或 patching 或 steering）——**同时**做完并使 DSSP-based 结构评估贯穿始终，是本工作的方法学贡献。

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist)_

---

## 附：方法学-任务对应表（供 Phase 4.5 / 实验阶段参照）

| 任务 | 主要工具 | 关键防呆 |
|---|---|---|
| 早期 vs 晚期 block 是否承担 β-hairpin 决策（claim 1） | Block-wise activation patching / ablation on `s`；layer-wise linear probe for "will fold as hairpin" | 需要 matched control（非 β-hairpin 目标区域）；DSSP 判定 |
| 是否 `s` 是决策活跃位（claim 1） | 对 `s` vs `z` 分别 patch/ablate；对比效果 | site 的严格控制；避免只测 `s` |
| seq2pair 是否是把决策从 `s` 写入 `z` 的通道（claim 2） | pathway 消融 / 冻结 / patch seq2pair 输出；对比 pair2seq | 需 pair2seq 作为对照，防止效应错误归因 |
| charge 是否线性编码于 `s`（claim 3 上半） | 在早期 block `s` 上训练 linear probe 预测残基净电荷 | 严格 train/val 分离，PISCES 25% seq id 已足够 |
| charge 是否因果影响 β-hairpin 形成（claim 3 下半） | 沿 probe 方向对 `s` 做 steering（例如把两 β-strand 上一对残基电荷"同号化"或"异号化"）；比较 DSSP 输出 | matched control（对同 seq 做非目标残基的相同强度 steering）；剂量-响应曲线 |
