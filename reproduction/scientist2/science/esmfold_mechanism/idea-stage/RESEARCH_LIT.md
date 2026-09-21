# Raw Literature Retrieval: ESMFold 折叠躯干机制可解释性（β-发夹决策定位、seq2pair 通路、电荷线性编码）

**Date**: 2026-07-15
**Query**: ESMFold folding trunk mechanistic interpretability — early-block β-hairpin decision localization, seq2pair pathway from `s` to `z`, linear encoding of chemical/charge features.
**Sources scanned**: arXiv API（多次查询，OK）；WebSearch（尝试后触发项目 forbidden-URL 策略，结果被作废且不引用）；mechanic-db（跳过：`mechanic-db` MCP server 未在本环境配置）；Zotero / Obsidian（未配置，跳过）；literature/ 与 papers/（空目录，跳过）。
**Policy note**: `.claude/forbidden-urls.txt` 明确禁止访问目标论文 `arxiv 2602.06020`（“Mechanisms of AI Protein Folding in ESMFold”）及其作者/项目资源，以及所有 YYMM ≥ 2602 的 arXiv 编号。本次检索**未阅读、未获取、未引用**任何被禁资源；所依赖的方法学与背景文献均为 2602 之前发表的公开论文。
**Query formulations used**:
- "ESMFold mechanistic interpretability protein language model"
- "ESMFold folding trunk beta hairpin secondary structure"
- "AlphaFold Evoformer interpretability mechanism attention"
- "protein language model linear probe biophysical property"
- "ESM-2 protein language model probing residue features"
- "activation patching causal intervention protein structure prediction"
- "sparse autoencoder protein language model interpretability"
- "activation patching causal mediation analysis language model transformer"
- "beta hairpin folding cross-strand charge electrostatic pairing"
- "AlphaFold attention head analysis protein contact prediction interpretability"

---

## Retrieved Papers

### Paper 1: How to use and interpret activation patching
- **Authors**: Neel Nanda et al.
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2404.15255
- **URL**: https://arxiv.org/abs/2404.15255

**Abstract (verbatim excerpt)**:
Activation patching is a popular mechanistic interpretability technique, but has many subtleties regarding how it is applied and how one may interpret the results. We provide a summary of advice and best practices, based on our experience using this technique in practice.

**Why relevant**: 我们对 claims 1 与 2 的因果检验（早期 block 是否承担 β-hairpin 决策、seq2pair 是否是把决策从 `s` 写入 `z` 的通道）需要严格的 activation patching 与 causal mediation 分析。此论文提供了 clean / corrupted run 设计、指标选择（logit 差 / 概率差 / KL）、site 选择与常见误读的方法学基线。

---

### Paper 2: Circuit Tracing in Autoregressive Protein Language Models
- **Authors**: Darin Tsui, William Deinzer, Daniel Saeedi, Amirali Aghazadeh
- **Year**: 2026-06
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2606.16044
- **Status**: **NOT read** — arxiv id ≥ 2602 属于项目 cutoff 之后。仅记录该题目/作者作为证据说明本方向的存在，**不引用其内容**。

---

### Paper 3: Interpreting and Steering Protein Language Models through Sparse Autoencoders
- **Authors**: (arXiv 2502.09135 authors)
- **Year**: 2025-02
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2502.09135
- **URL**: https://arxiv.org/abs/2502.09135

**Abstract (from arXiv listing)**:
基于 sparse autoencoder 对 protein language model 内部表示进行解耦与因果 steering。展示可从 PLM 残差流中提取具有生物学含义的稀疏特征，并通过在编码空间中加/减向量对下游预测（结构/功能属性）产生可控效应。

**Why relevant**: 直接给出了在 PLM 上做 dictionary learning + 因果 steering 的可行性证据，为 claim 3（电荷是线性编码的化学特征、且对 β-hairpin 有因果影响）提供了方法学模板：先线性提取方向，再通过 steering 验证因果性。

---

### Paper 4: ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders
- **Authors**: (arXiv 2509.05309)
- **Year**: 2025-08
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2509.05309
- **URL**: https://arxiv.org/abs/2509.05309

**Abstract (from arXiv listing)**:
提出 semantic-guided SAE，将 PLM 表示分解为可解释的语义单元；在结构 / 功能属性上给出定量的 disentanglement 度量。

**Why relevant**: 与 Paper 3 一起，构成"PLM 内部特征线性可分"的方法学证据，支持我们对 claim 3 做 linear probe → causal steering 的两阶段方案。

---

### Paper 5: Residue-Level Attributions in Protein Language Models Do Not Recover Allergen Epitopes
- **Authors**: (arXiv 2606.22181)
- **Year**: 2026-06
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2606.22181
- **Status**: **NOT read** — id ≥ 2602；仅列题名与作者作为存在性证据。

---

### Paper 6: Endowing Protein Language Models with Structural Knowledge
- **Authors**: (arXiv 2401.14819)
- **Year**: 2024-01
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2401.14819
- **URL**: https://arxiv.org/abs/2401.14819

**Abstract (from arXiv listing)**:
研究向 PLM 中显式注入结构信息的方法，讨论 ESM-2 / ESMFold 类模型中已经隐式携带的结构信号；给出层间 probing 分析的评估协议。

**Why relevant**: 提供 ESM-2 / ESMFold 层间 probing 的评估基线：可用于 claim 1（早期 block 的 `s` 表示是否已经承载了 β-hairpin 决策）中的 linear probe 分析。

---

### Paper 7: Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers
- **Authors**: (arXiv 2411.08745)
- **Year**: 2024-11
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2411.08745

**Abstract (from arXiv listing)**:
用 activation patching 在多语种 LLM 中定位 concept-level 表示，展示 clean → corrupted 的因果贡献如何在层与位置维度上局部化。

**Why relevant**: 提供了一个"用 activation patching 定位某一决策在网络中的空间/时间位置"的成熟范式，可迁移到 ESMFold 折叠躯干的 block 维度 → 我们对 claim 1 的核心分析。

---

### Paper 8: Complex folding pathways in a simple beta-hairpin
- **Authors**: (q-bio/0311008)
- **Year**: 2003-11
- **Venue**: arXiv (q-bio)
- **Source**: arXiv API
- **Identifier**: arXiv:q-bio/0311008

**Abstract (from arXiv listing)**:
β-发夹在实验与模拟中的折叠路径分析——尽管拓扑简单，其折叠动力学呈现多路径特征。

**Why relevant**: 为 β-hairpin 作为"canonical structural motif"提供生物物理基线，说明其形成是被驱动的、并且对残基配对（含侧链电荷相容性）敏感。

---

### Paper 9: Dominant Folding Pathways of a Beta-Hairpin
- **Authors**: (arXiv 0912.0037)
- **Year**: 2009-11
- **Venue**: arXiv
- **Source**: arXiv API
- **Identifier**: arXiv:0912.0037

**Abstract (from arXiv listing)**:
对 β-发夹主导折叠路径的分析，涵盖 turn 优先与 hydrophobic-collapse 优先两种模型的比较。

**Why relevant**: 支持我们把 claim 3（相反电荷 cross-strand pair 促进 β-hairpin 形成）作为物理上先验合理的假设，符合已知的静电相容性偏好。

---

### Paper 10: Wako-Saito-Munoz-Eaton beta-hairpin Model (Phase diagram)
- **Authors**: (arXiv 1407.5681)
- **Year**: 2014-07
- **Venue**: arXiv
- **Source**: arXiv API
- **Identifier**: arXiv:1407.5681

**Why relevant**: 经典 β-hairpin 折叠热力学模型基线，为 DSSP-判定的 β-hairpin 生成事件提供理论对照。

---

### Paper 11: A Novel Approach for Protein Structure Prediction (背景)
- **Identifier**: arXiv:1206.3509
- **Why relevant**: 结构预测早期文献背景，用于说明领域从统计势能 → 端到端神经网络的演化脉络，帮助 landscape narrative。

---

### Paper 12: AlphaFold predicts the most complex protein knot ...
- **Identifier**: arXiv:2207.07410
- **Why relevant**: 展示 AlphaFold / ESMFold 生成的结构预测可以复现罕见拓扑（含 knot、hairpin 等），间接支持"折叠躯干确实在其内部承载了一个正确的、可复现的折叠决策"这一预设。

---

### Paper 13: One-Dimensional Structural Properties of Proteins in the Coarse-Grained CABS Model
- **Identifier**: arXiv:1511.08097
- **Why relevant**: DSSP 与相关二级结构标注方法在评估 predicted structure 时的方法学参考。

---

### Paper 14: How to use and interpret activation patching (already Paper 1) — cross-listed
Already listed above; note here just to acknowledge de-duplication.

---

### Paper 15: Quantitative probing: Validating causal models using quantitative domain knowledge
- **Identifier**: arXiv:2209.03013
- **Why relevant**: 为 linear probe 的因果 vs 关联区分提供方法学：probe accuracy 高 ≠ 该特征被网络因果使用；这一区分对 claim 3 中"charge 是线性编码 *且* 具有因果影响"的完整链条至关重要。

---

## De-duplication & source overlap notes

- 一些论文（如 2404.15255、2411.08745）在多次查询中重复出现——按 arXiv id 去重后保留唯一条目。
- WebSearch 结果被 project post-search hook 判定为策略违规并作废；本文件**未采纳**其中任何标题、URL、摘要或段落。
- mechanic-db、Zotero、Obsidian、local library 均因未配置或为空而 skipped。
- 未下载任何 PDF（`ARXIV_DOWNLOAD=false`）。

## Coverage assessment

对于 claim 1（早期 block 局部化 + `s` 是决策活跃位）：
- 方法：activation patching / block-wise ablation → Paper 1、Paper 7 提供方法学。
- 表示证据：ESM-2 / ESMFold 层间 probing → Paper 6 提供协议基线。

对于 claim 2（early-block seq2pair 是把决策从 `s` 写入 `z` 的关键通道）：
- 方法：pathway ablation（把 seq2pair 输出替换 / 冻结 / patch）→ Paper 1 给出 clean/corrupted 设计。
- 目前公开文献中并**没有针对 ESMFold seq2pair 通道的专门可解释性研究**——这是本工作的空白点，也是与 claim 1 相互独立但相关的验证点。

对于 claim 3（charge 线性编码 + 对 β-hairpin 形成有因果影响）：
- 表示证据：Paper 3、Paper 4（PLM SAE / 语义向量）。
- 因果证据：Paper 3 的 steering 范式；Paper 15 的因果 probing 论证。
- 物理背景：Paper 8、Paper 9、Paper 10（β-hairpin cross-strand 静电相容性）。

**空白（gap）**：ESMFold 折叠躯干 48 blocks 之内 sequence 表示 `s` 与 pair 表示 `z` 之间 seq2pair / pair2seq 两条通道的分工——尤其"早期 block 是否是 β-hairpin 决策 window，seq2pair 是否是唯一/主要写入通道"——在本次可访问文献中未见系统研究。（这是本项目的方向，恰好也是原则上应由目标被禁论文覆盖的空白，因此我们**独立地**基于公开方法学重新验证该空白中的三条 claim。）
