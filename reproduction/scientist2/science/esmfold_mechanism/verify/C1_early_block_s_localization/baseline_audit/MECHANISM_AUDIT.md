# C1 机制严谨性审计（Phase 2 — Mechanism Audit）

**Claim**: C1 — 折叠决策局部化于早期 block，`s` 是决策活跃位  
**Mechanism family**: Causal Attribution / Patching  
**审计时间**: 2026-07-15

---

## 机制严谨性检查

### 1. 因果解读合法性（Causal Attribution / Patching 家族）

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 干预范式是否为因果必要性（necessity）检验 | PASS | 激活 patching 是 clean/corrupted 范式的标准实现：patching donor s 入早期 block → 目标链"看不到"自己的早期 s 信号 → 如果 hairpin rate 下降，则 s 在那个 window 对 hairpin 决策是必要的。这与 claim 陈述（"s is the active locus"）在逻辑上匹配 |
| donor 选择是否避免 confounding | PASS | Donor 在对应 target residue 上的 native DSSP 不含 hairpin，意味着 donor s 携带"非 hairpin 决策"信号，是合适的 corrupted source |
| Claim 的假设方向是否由数据支持 | PASS | Claim 预测"早期 window s-patching 使 hairpin rate 显著下降（Δ ≥ 0.2pp）"；实测 Δ=-0.862，方向和幅度均一致 |
| 特异性控制是否排除非机制备择解释 | PASS | 三个对照（z-patch, matched-ctrl, late-window）均显示 Δ < 10% of early s-patch，排除了"任何 patching 都会破坏 hairpin"（非选择性 perturbation）和"late-block 也足够"的备择 |

### 2. 方法家族映射正确性

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 家族 = Causal Attribution / Patching | PASS | MECHANISM_ROUTING.md 明确指定，实施与家族 submethod（block-input activation patching）完全匹配 |
| 干预位点（site）与 claim 陈述一致 | PASS | Claim 说"s is the active locus"；实验干预精确作用于 `s[block_k, target_residues, :]`，在 block input pre-hook 层实施，block-specific donor 保证精确性 |
| 效应度量与 claim 量词一致 | PASS | Claim 要求"Δ hairpin rate ≥ 0.2 pp"；实验报告 Wilcoxon-based Δ(hairpin_rate) = -0.862，方向（down）与量词（negative / drop）一致 |

### 3. Causal Attribution 家族特有风险评估

| 风险项 | 状态 | 细节 |
|--------|------|------|
| Block-specific vs. band-averaged donor | PASS | 每个 block k 使用该 block 入口处的 donor s 张量（`collect_donor_activations` 通过 pre-hook 捕获每个 block 的独立 s），而非跨 block 平均；这是精确的 block-specific patching |
| Donor distribution shift | WARN | Donor 链与 target 链的序列不同 → 被 patch 的 s 张量来自不同的序列分布。这是激活 patching 的内生局限（不可避免），但在 N=197 个随机 donor 配对下，系统偏差的风险低。Claim 文本未对 donor 分布做承诺，故只记 WARN |
| No-intervention baseline 确认 | PASS | 每条链均先跑 `baseline` condition（无干预 forward），实测 baseline_rate = 100%，确认 hairpin 在干预前存在 |
| Hook 泄漏检查 | PASS | `patch_s_at_blocks` 使用 context manager + `handle.remove()` 在 `with` 块退出时清除；代码审查未发现泄漏路径 |

### 4. 机制层级声明评估

**C1 claim 的机制层级**：block-window localization + s-representation-level（不是 attention head、不是weight、不是circuit tracing）。

- 该层级由激活 patching 直接可回答，且实验实施在正确层级（block-input s 而非 attention head level）。
- EXPERIMENT_PLAN.md § M1 中 "Location → Causal Intervention" 的双层 ladder of evidence 结构在实验中已完整实现（Step A=correlational probe screen 已体现在 predicate design 中；Step B/C/D/E=causal patching 全部执行）。

---

## 综合评级

**PASS**（1 项 WARN：donor distribution shift，Causal Attribution / Patching 家族内生局限，不影响结论）

**Phase 2 机制审计结论**：C1 **机制严谨性 PASS**。
