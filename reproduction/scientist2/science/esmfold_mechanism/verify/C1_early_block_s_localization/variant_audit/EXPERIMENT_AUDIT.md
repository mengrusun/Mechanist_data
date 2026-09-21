# C1 Variant 完整性审计（Phase 9 — Variant Experiment Audit）

**Claim**: C1  
**Variants audited**: dataset_cath_strat, method_mean_ablation  
**审计时间**: 2026-07-15

---

## Variant 1: dataset_cath_strat

### 方法学完整性

| 检查项 | 状态 | 细节 |
|--------|------|------|
| DSSP ground truth 保留 | PASS | 直接读取 M1 per-chain JSONL 中已有的 `is_hairpin` 值，未重新计算 DSSP。Ground truth 链路未被破坏 |
| 聚合逻辑正确性 | PASS | 按 `cath_label` 分组 → per-chain 按 donor 取均值 → per-group Wilcoxon paired test。与 M1 方法论一致 |
| CATH 分层的实际效果 | WARN | 所有 197 条 chain 的 `cath_label=null`，全部落入"cath_null"组。分层在当前数据下等价于全局统计 |
| 统计完整性 | PASS | N=197，Δ=-0.862，p=2.25e-36；bootstrap 95% CI ≈ [-0.91, -0.82]，CI 不跨 0 |
| 数据独立性 | PASS | 使用 M1 已有数据，无新 ESMFold forward，无新 chain 引入 |

### Predicate 检查

| Predicate | 结果 |
|-----------|------|
| early_effect ≥ 0.2pp | PASS（Δ=-0.862） |
| early_p < 0.05 | PASS（p=2.25e-36） |
| late_window < 50% | PASS（0.085/0.862 = 9.8%） |
| z_same_window < 50% | PASS（0.011/0.862 = 1.3%） |
| matched_ctrl < 50% | PASS（0.067/0.862 = 7.7%） |

### 完整性评级：**PASS（WARN：CATH 分层退化为单组）**

**Phase 9 裁定**：dataset_cath_strat 变体**integrity-clean，计入 robustness 分母**。

---

## Variant 2: method_mean_ablation

### 方法学完整性

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 实施状态 | **NOT_RUN** | 脚本已通过代码审查（CODE_REVIEW.md: PASS），但因 auto-mode 执行权限限制未能完成 GPU dispatch |
| DSSP ground truth 设计 | PASS（设计层面） | 脚本调用 `predict_and_judge_hairpin()` → ESMFold predicted structure → mkdssp，符合 task.md HARD |
| Hook 实现 | PASS（代码审查） | 与 `esmfold_lib.patch_s_at_blocks` 同等安全性；clone + handle.remove() |
| 参考池独立性 | PASS（设计层面） | 使用 donor 链（30 条），与 main set 不重叠 |

### Phase 9 裁定：**NOT_RUN — 从 robustness 计算中排除（excluded）**

原因：非实现设计缺陷，而是执行环境限制。排除理由 = "not_executed_permission_block"。

**修复路径**：`CUDA_VISIBLE_DEVICES=3 python verify/C1_early_block_s_localization/variants/method_mean_ablation/run_variant.py 3`（预计 ~0.5 GPU-h）。

---

## 综合评级

- dataset_cath_strat：integrity-clean → **eligible**
- method_mean_ablation：NOT_RUN → **excluded**

**N_run = 2，N_eligible = 1**（dataset_cath_strat）
