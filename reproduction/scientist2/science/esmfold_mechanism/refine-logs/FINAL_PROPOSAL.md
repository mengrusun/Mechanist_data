# Final Proposal — ESMFold 折叠躯干中 β-hairpin 机制的统一因果检验

```yaml
# ---- Machine-readable metadata (do not edit informally) ----
behavior_source: given
mechanism: discovery
resource_fidelity: cost-aware   # NOT strict — this is given + discovery, not the reproduction combo
mechanism_strategy:
  directions: [Location, Causal Intervention, Unit Interpretation]   # execution order
  rejected:
    - "Tuning & Editing — 三条 claim 是 diagnostic（X 是否引起 B），不是 applied（用 X 提升下游任务）；扩展会脱离忠实复现范围并挤占 10-hour 预算"
    - "Formation Tracing — 三条 claim 都不涉及 genesis；该方向最贵且不必要"
    - "Decision Auditing — 三条 claim 的问题是 mechanism 是否如所述，不是 decision 是否可靠/是否 spurious"
  note: "Location 支撑 claim 1 的 block window 与 s/z 定位；Causal Intervention 是 claim 1/2/3 因果部分的公共工具；Unit Interpretation（linear-probing slice）承担 claim 3 的 'linear encoding' 部分并 name 出 steering 所需的方向。"
target_model: ESMFold
dataset: PISCES cull list cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055
evaluation_ground_truth: DSSP secondary-structure assignment on predicted structure  # HARD constraint from task.md
gpu_budget_hours: 10
gpu_ids_allowed: [0, 1, 2, 3]
```

---

## 1. Problem Anchor（不可移动）

**要解释的对象**：ESMFold（ESM-2 主干 + 48-block folding trunk + structure module）内部的折叠决策过程，特别是它把某一 target region 折成 **β-hairpin** 这一最简结构决策所依赖的机制。

**要检验的三条 claim（逐字保留自 task.md，见 `idea-stage/IDEA_REPORT.md`）**：
1. **早期 block 局部化 + `s` 是决策活跃位**——是否折成 β-hairpin 的决定被承诺在早期 block window 内，且序列表示 `s` 在此 window 是决策活跃位。
2. **早期 seq2pair 是 `s → z` 的关键因果通道**——早期 block 中 seq2pair 操作是把该决策从 `s` 写入 `z` 的关键通道。
3. **charge 是早期线性编码化学特征 + 对 β-hairpin 有因果影响**——早期 block 的 `s` 上残基电荷是线性可解码的化学特征，且沿该方向的因果 steering 会以物理上一致的方向改变 β-hairpin 形成（同号 ↓ / 异号 ↑）。

**Problem Anchor 三条约束**（不可为了实验方便而修改）：
- (A) β-hairpin 判定必须使用 **DSSP** 对 predicted structure 的二级结构标注（task.md HARD）。
- (B) 内部机制只在 **ESMFold** 上分析（task.md HARD——不是 AlphaFold / OpenFold / Boltz 等）。
- (C) 数据源是 PISCES cull list `cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055`（HARD 数据源）。

---

## 2. Method Thesis（一句话）

用**三层"ladder of evidence"**——(a) block/site-wise correlational screen（linear probe + attribution）、(b) causal intervention（activation patching / pathway ablation / steering）、(c) matched-control specificity——**串成一条统一的、DSSP-驱动的因果验证管线**，三条 claim 各由该管线的一段独立 milestone 承担；三段共享同一批 target 序列、同一份 DSSP 判定基线、同一套统计功效计算，从而在 10 小时 GPU 预算内联合完成。

## 3. Dominant Contribution

**把 ESMFold 折叠躯干 48-block 内 s/z 二元表示的 block-window × pathway × chemical-feature 三个层级，首次串成一条 DSSP-based 端到端因果验证管线**——从 correlational 定位到 pathway causal 到 feature-level causal steering，逐层带 matched-control specificity，避免了 arXiv 2209.03013 指出的"probe 高准确率 ≠ 因果使用"陷阱。

## 4. Intentionally-Rejected Complexity

- **不做**训练时机制形成的追踪（Formation Tracing / checkpoint scan / influence functions / data attribution）——task.md 未主张。
- **不做**capability tuning / weight editing / task-vector 应用工作——三条 claim 是 diagnostic 不是 applied。
- **不做**Decision Auditing 层面的"模型决策是否可靠 / 是否依赖 shortcut"——scope 外。
- **不做**将 s2z/z2s 拆解成注意力头级 circuit tracing 层次的深挖——task.md 明确到 pathway（seq2pair vs pair2seq）级；再往下拆是 out-of-scope。
- **不引入** SAE 训练。charge 是残基级良定义 chemical label（Asp/Glu 负、Lys/Arg/His 正、其余中性），linear probe 足够；无需 dictionary learning。
- **不做**跨模型迁移（AlphaFold / OpenFold / Boltz）——task.md 明确 ESMFold 是唯一机制分析对象。

## 5. Testing-Method Refinement（对三条 claim 的**测试**做加强，claim 本身不动）

### 5.1 β-hairpin target 区域的定义与选取

- 从 PISCES cull list（12,055 chains）中筛选 **在 native structure DSSP 上包含 ≥1 段 β-hairpin** 的链——β-hairpin 定义为两根反平行 β-strand + 中间 turn（DSSP `E-E` 两段相邻反平行，中间 `T`/`S`/loop 短 turn 桥接），链长在 60–300 之间以便 ESMFold single-GPU forward 稳定。
- 从符合的链中随机抽 **N_main = 200 chains** 作 main experiment 数据；再抽 **N_calib = 50 chains** 作方法学 calibration（不与 main 重叠）。剩余作为 held-out 保留。
- 为每条链找 **1 个 target β-hairpin region**（若一条链有多段，选 DSSP E-长度总和最长的那段），并记录其 cross-strand facing pairs（用 native structure 的最近 Cβ-Cβ 距离配对）。
- **具体化**：所有 target region 的 DSSP baseline 是"ESMFold 在无干预条件下 predict，然后对 predicted structure 跑 DSSP，看 target region 是否被判定为 β-hairpin"——记录 baseline hairpin-formation rate（≥0.7 是选取样本的前置门槛，避免样本本身就不会被 ESMFold 折成 hairpin）。

### 5.2 "早期 block window" 的操作化

- ESMFold 折叠躯干 48 blocks。**在 claim 层面只锁定"存在一个早期 window"**，具体索引由实验发现。
- 探索网格：对每个链在如下 candidate window 上做实验：`[0–3]`, `[4–7]`, `[8–11]`, `[12–15]`, `[16–23]`, `[24–31]`, `[32–39]`, `[40–47]`（8 个 mutually-exclusive block bands）。这既避免一次全 48-block 扫描的过重开销，也保证"早期 vs 晚期"能被明确画出转折点。
- 若某个 band 上的干预效应显著大于其他 band，则该 band 被认定为 "早期决策 window"；否则报告 "no clearly localized early window found"（negative result 也是有效结论）。

### 5.3 干预实施的方法学锁定

- **s-patching**：在指定 block window 的每一个 block 的开头，用一个 "donor" 链（同一批中随机匹配、长度接近、native 不含 β-hairpin 于对应 residue index）的 `s[target_residues, :]` 覆盖当前链的对应位置——按 arXiv:2404.15255 的 clean/corrupted 范式，标准的 clean = 无干预、corrupted = donor-patch。
- **z-patching**：同上，但覆盖 `z[target_pairs, :, :]`（target region 内的所有 pair）。
- **seq2pair 干预**：在指定 block window 内，拦截 seq2pair 层的 output tensor，将其在 target pair 上替换为 donor 值（或做 zero-ablation 做 sanity check）。
- **pair2seq 匹配对照干预**：与 seq2pair 同 block window、同 residue mask、同 donor 强度，只是替换的是 pair2seq 的 output。
- **residue-level charge steering**（claim 3）：先在 M3a 用 linear probe 找到早期 block `s` 上编码 residue charge 的方向 `v_charge`；在 M3b 沿 `s[target_residue, :] += α · v_charge` 施加干预，α ∈ {-3, -1, +1, +3}（4 个 dose 点，含正负号）。
- **matched-control specificity**：每次因果干预都对同一链上一段**非** β-hairpin 且非 target 的相同长度的 residue mask 施加**同强度**同类型的干预，测其对 DSSP-β-hairpin rate 的影响——期望效应显著小。

### 5.4 效应度量与统计功效

- **主指标**：DSSP-β-hairpin formation rate（在每次 predict → DSSP → 判定 target region 是否为 β-hairpin 的 boolean 序列上取平均）。
- **辅助指标（不作为 claim 判据）**：cross-strand `Cα-Cα` 平均距离（Å）——对 claim 3 特别有用（同号增大 / 异号减小距离）。
- **统计检验**：
  - claim 1、2：paired binomial（每条链 clean vs corrupted 的 hairpin 状态），McNemar / Wilcoxon signed-rank；报告 effect size + 95% CI + p 值。
  - claim 3a：held-out probe accuracy vs permutation baseline（1000 label-shuffle 排列）——报告 top-line accuracy、per-class balanced accuracy、AUROC；显著性以 permutation p-value 表征。
  - claim 3b：dose-response 单调性检验（Spearman ρ 沿 α 序列在 hairpin rate 上）+ same vs opposite 差异（paired McNemar）+ target vs matched-control specificity（配对差 t-test 或 Wilcoxon）。
- **样本量**：N_main = 200 chains 对 paired binomial 在 baseline hairpin rate = 0.7、target effect = 0.2 时具有 >0.99 的功效（α=0.05）——足够。

### 5.5 与 verify-stage CATH-stratified variant 的兼容

- main experiment 的 200 条 chain 每条都记录其 **CATH 分类**（若可映射，用 CATH Class / Architecture / Topology level）——即使 main experiment 不做 stratification，也为 verify-stage 提供了直接可复用的 stratified 子集。
- 具体：每条 chain 的 `pdb_id`、`chain_id`、`target_hairpin_region`、`cath_label`、`dssp_baseline_hairpin_rate`、`interventions{}` 全部落盘 `results/main_records.jsonl`——verify-stage 可以直接从这个 jsonl 上按 CATH label 分层重跑 DSSP 判定。

## 6. Frontier-Primitive Necessity Check

本方法**不需要**任何前沿 LLM / VLM / Diffusion / RL 原语。所有工具都是十年内成熟的经典机制可解释性工具：
- Linear probe（十年 +）
- Activation patching（成熟，参见 arXiv:2404.15255）
- Steering vector（数年成熟）
- DSSP（自 1983）

因此不引入前沿组件是**故意**的、有利的：更简单、更可复现、更符合 mechanistic-interpretability 的方法学传统。

## 7. Main Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ESMFold 对某类链 baseline β-hairpin 生成率就低 | 前置门槛 baseline_hairpin_rate ≥ 0.7，链不满足则丢弃 |
| Donor 链选得不好使 patching 效应虚高（confound） | Donor 采用长度相近、native 结构无 hairpin 于对应位置、多个 donor 平均效应；再做 zero-ablation 做 sanity |
| Probe 高准确率但网络不因果使用（Paper 2209.03013 陷阱） | claim 3 必须 H3a AND H3b 都成立才判 PASS；只 H3a 通过 = REFUTED 的中间态 |
| 10 h GPU 预算超支 | 每 milestone 有 gpu_hours 预估、总和 8.5 h（含 buffer）——见 EXPERIMENT_PLAN.md |
| DSSP 判定 boundary（E 与 T 之间的 fuzzy 位）敏感 | 用严格 DSSP 判定规则 + 若 target region 半数以上被判 E-E 反平行则算 hairpin；无 fuzzy fallback |
| 只用一次 run 结果不稳定 | ESMFold forward 是 deterministic given seed；因此单 run 已够。若有非确定性组件，跑 3 seeds 取多数 |

## 8. Must-Prove Claims（三条即 task.md 三条，一一对应到 EXPERIMENT_PLAN.md）

- **C1**（→ M1）：在 200 条链上，存在一个早期 block window `[i*, j*]`，在此 window 对 `s` 做 s-patching 使 target region DSSP-β-hairpin rate 相比 baseline 显著下降（effect size ≥ 0.2 pp, p < 0.05）；同时在此 window 之后对 `s` 的相同 s-patching 效应显著较小，且此 window 内对 `z` 的 z-patching 效应显著较小。
- **C2**（→ M2）：在 M1 找到的早期 window 内，对 seq2pair 做 pathway 干预使 DSSP-β-hairpin rate 显著下降（effect size ≥ 0.2 pp, p < 0.05）；对 pair2seq 做 matched intervention 效应显著较小（差异 p < 0.05）。
- **C3**（→ M3）：
  - **C3a**：M1 找到的早期 window 内某个（或若干个）block 上的 `s` 表示能被 linear probe 显著优于 permutation baseline 地解码残基电荷（three-class balanced accuracy ≥ 0.75，permutation p < 0.001）。
  - **C3b**：沿 M3a 学到的 `v_charge` 方向对 target β-hairpin cross-strand pair 做 steering，DSSP-β-hairpin rate 在 α 轴上呈单调 dose-response（Spearman |ρ| ≥ 0.7），且 same-charge steering → hairpin rate ↓（配对差 p < 0.05），opposite → hairpin rate ↑（配对差 p < 0.05），matched-control 效应显著较小（specificity p < 0.05）。

## 9. Deliverables & Downstream

- 见 `refine-logs/EXPERIMENT_PLAN.md` 与 `refine-logs/EXPERIMENT_TRACKER.md`。
- 下游：`/mechanism-skills` → `/auto-experiment` → `/auto-verify`（跑 CATH-stratified swap variant）→ `/auto-iteration-loop`。
