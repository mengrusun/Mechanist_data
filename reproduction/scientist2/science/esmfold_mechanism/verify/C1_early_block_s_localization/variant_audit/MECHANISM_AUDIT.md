# C1 Variant 机制严谨性审计（Phase 9 — Variant Mechanism Audit）

**Claim**: C1  
**审计时间**: 2026-07-15

---

## Variant 1: dataset_cath_strat

### 机制严谨性

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 家族一致性 | PASS | 使用 M1 中相同 Causal Attribution / Patching 的结果；未改变方法家族 |
| 干预位点一致性 | PASS | 干预 site（s, block b_0_3, target_res）与 M1 baseline 完全一致 |
| Null 对照保留 | PASS | 同时重聚合 z_patch、matched_ctrl、late_window 对照，保留 specificity 结构 |
| CATH 分层的机制含义 | WARN | CATH-null 单组无法验证"效应是否在不同结构类中复制"——这是 variant 的主要局限性，由数据（未映射 CATH）引起而非方法设计缺陷 |

### 评级：**PASS（WARN：分层局限）**

---

## Variant 2: method_mean_ablation

### 机制严谨性（代码审查层面）

| 检查项 | 状态 | 细节 |
|--------|------|------|
| Within-family constraint | PASS | Mean-ablation 是 Causal Attribution / Patching 家族内的 submethod swap（不同 ablation baseline），符合 MECHANISM_ROUTING.md 家族约束 |
| 方向有效性 | PASS | Mean-ablation 测试"仅破坏 target chain 自身 s 信号是否足够"——与 donor-patch 互补，共同揭示 M1 效应来自 disruption-of-self 还是 introduction-of-donor |
| DSSP ground truth | PASS（设计） | 脚本使用 predict_and_judge_hairpin()，链路正确 |

### 评级：**EXCLUDED（NOT_RUN）**

---

## 综合评级

- dataset_cath_strat：mechanism PASS（WARN）→ **eligible**  
- method_mean_ablation：NOT_RUN → **excluded**
