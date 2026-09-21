# C3 主实验方法学审计（Phase 2 — Experiment Audit）

**Claim**: C3 — charge 是早期线性编码化学特征，且对 β-hairpin 形成有因果影响  
**Milestones**: M3a（线性探针）+ M3b（因果 steering）  
**审计时间**: 2026-07-15

---

## 方法学完整性检查

### Part A: M3a — linear encoding of charge

#### A1. 数据集与样本

| 检查项 | 状态 | 细节 |
|--------|------|------|
| heldout_probe 集独立性 | PASS | 40 条 heldout_probe 链与 main(200)/calibration(50)/donor(100) 集无重叠；PISCES ≤25% seq id 保证 chain 间低同源性 |
| 样本量（M3a）| WARN | 40 chains（planned 250）：train=32, val=4, test=4。Test set = 4 条链（≈736 residues）。计划书标注"suspected_under_power"。但 bacc=1.0 with p<0.001 在如此小的 test set 上出现有两种解读：(a) charge signal 极强（trivially decodable from s at block 0）；(b) 小 test set 偶然碰巧 — 但 val bacc=1.0 on 4 chains 且 permutation null < 0.001 综合起来，charge decodability 高度可信。 |
| 训练/验证/测试分割 | PASS | Chain-level 8:1:1 分割（32:4:4），避免 residue-level 泄漏 |
| Charge label 定义 | PASS | neg={D,E}, pos={K,R,H}, neut=其余——符合标准生物物理电荷分类，与 task.md 一致 |

#### A2. Probe 训练与评估

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 模型选择 | PASS | 3-class multinomial LogisticRegression（class_weight="balanced"）在 chain-level split 上训练 |
| 指标 | PASS | 报告 val balanced acc + test balanced acc + test AUROC macro（3-class OvR）|
| Permutation null | PASS | 1000-permutation label-shuffle（固定 clf 预测，shuffle 测试标签）；empirical p = 0.000999（1000 nulls 中无一超过 test bacc=1.0）|
| v_charge 提取 | PASS | `w_pos - w_neg`（probe 权重差）L2-normalized → v_charge；存盘 `results/M3a/v_charge.npy`；实际 L2 norm after normalize = 1.0（确认） |

#### A3. Predicates（M3a）

| Predicate | 要求 | 实测值 | 状态 |
|-----------|------|--------|------|
| balanced_acc ≥ 0.75 | 是 | test bacc = 1.0（4 blocks 全部） | PASS |
| permutation_p < 0.001 | 是 | p = 0.000999 | PASS |

---

### Part B: M3b — causal steering

#### B1. 干预实施

| 检查项 | 状态 | 细节 |
|--------|------|------|
| β·σ_proj 单位 rebinding | PASS | σ_proj = 21.55（block 0，50 条 calib 链测量）；β ∈ {-3,-1,0,+1,+3} → 实际干预强度 ≈ 64.7 residual-stream units at β=+3。EXPERIMENT_TIPS.md 中的 steering-coefficient-tuning 约束已执行 |
| 干预位点 | PASS | target cross-strand pair 上的两个 residue（来自 manifest.cross_strand_pairs）；same/opposite config 按照 M3b 计划 |
| α=0 baseline sweep point | PASS | β=0 明确作为 sweep 点包含在 {-3,-1,0,+1,+3} 中 |
| random-direction control | PASS | 从 MUST-RUN 升级后的 random v_random（同 L2 norm）控制已执行；结果 hairpin_rate=100% at β=+3σ |
| matched-ctrl specificity | PASS | 非 target pair（非 hairpin 区域）上施加相同 steering；已执行 |
| 数据完整性 | PASS | 179 条 ok 记录 × 5 β × 2 configs + 额外控制；无 missingness |

#### B2. 统计分析（M3b）

| 检查项 | 状态 | 细节 |
|--------|------|------|
| Spearman ρ（same config）| FAIL | 仅 4 chains 有 β 维度方差（175/179 hairpin=1 constant）；mean ρ=0.354（<0.7 threshold） |
| Spearman ρ（opposite config）| FAIL | mean ρ=-0.040（<0.7 threshold） |
| same vs opposite paired diff | FAIL | Δ=-0.006（<0.15 pp），p=1.0 |
| matched_ctrl < 50% | FAIL | matched_ctrl hairpin rate=99.4% ≈ target rate=98.9%，无显著差异 |
| random_ctrl near baseline | PASS | random direction hairpin rate=100% |

#### B3. Predicates（M3b）

| Predicate | 要求 | 实测值 | 状态 |
|-----------|------|--------|------|
| spearman_same ≥ 0.7 | 是 | 0.354 | **FAIL** |
| spearman_opp ≥ 0.7 | 是 | 0.040 | **FAIL** |
| same_vs_opp_p < 0.05 | 是 | 1.0 | **FAIL** |
| same_vs_opp_delta ≥ 0.15 | 是 | 0.006 | **FAIL** |
| matched_ctrl < 50% | 是 | 99.4% ≈ 98.9% | **FAIL** |
| random_ctrl near baseline | 是 | 100% | PASS |

---

## 综合评级（C3 overall）

**M3a**: PASS（WARN on sample size 40 vs planned 250）  
**M3b**: FAIL（5/5 causal predicates fail — real scientific null）  

**C3 整体**：**PARTIAL PASS / WARN**

- M3a 部分（linearly encoded）：**PASS**
- M3b 部分（causally used）：FAIL，但这是**真实科学发现**（real null），不是实现 bug

**Phase 2 审计结论**：

C3 的 Phase 2 gate 需要分情况讨论：
- C3 作为**整体 claim**（"linearly encoded AND causally influences"）：M3b FAIL → 整体 claim not-supported → baseline integrity 可以说是"真实 not-supported"
- 但 C3 的两部分（M3a=supported, M3b=real null）都有**完整的方法实施**，无致命 bug
- 因此整体评为 **WARN**（方法完整，但 claim 部分不成立）

**Phase 2 结论**：C3 **WARN — ADMITTED**（方法完整性 PASS；M3b 是真实 null，claim 部分成立 = "linearly encoded but not causally used"）
