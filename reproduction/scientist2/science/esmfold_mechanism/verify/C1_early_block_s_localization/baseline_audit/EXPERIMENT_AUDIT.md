# C1 主实验方法学审计（Phase 2 — Experiment Audit）

**Claim**: C1 — 折叠决策局部化于早期 block，`s` 是决策活跃位  
**Milestone**: M1 — 早期 block s-patching  
**审计时间**: 2026-07-15  
**审计员**: auto-verify Phase 2

---

## 方法学完整性检查

### 1. 数据集与样本选取

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 数据来源是否与 claim 对齐 | PASS | PISCES cull list `cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055` — task.md HARD 数据源，已照实使用 |
| baseline hairpin-rate 门槛 | PASS | ≥ 0.7 — 实际数据：baseline_rate = 100.0%（所有 197 条 paired chains 均在门槛上） |
| 样本量是否足够 | PASS | N=197 paired chains（目标 200，3 条因 donor 不足/DSSP 失败被剔除）；在 Δ=-0.862 的效应量下统计功效远超 0.99 |
| donor 独立性 | PASS | donor 池（100 条）与 main set（200 条）从 manifest 中独立划分；每条 target chain 随机匹配 3 个 donor，长度差约束 ≤ 20%，donor 在对应 residue native DSSP 不含 hairpin |
| CATH 标注覆盖 | WARN | `cath_label=null` for all chains (CATH 映射未成功写入 manifest)。不影响 main experiment — 仅影响 verify 的 CATH-stratified dataset variant 的分层粒度（所有链会落入同一"CATH-null"组） |

### 2. 干预实施（DSSP-on-predicted-structure 为唯一 ground truth）

| 检查项 | 状态 | 细节 |
|--------|------|------|
| DSSP 判定函数正确性 | PASS | `is_hairpin_region()` in `esmfold_lib.py` — 纯函数，无副作用；规则：target 区域内两段 antiparallel E-E（各 ≥3 残基）+ ≤5 残基 turn 桥接；符合 task.md HARD 定义 |
| DSSP 是否在 ESMFold predicted structure 上跑 | PASS | `predict_and_judge_hairpin()` 流程：`esmfold_forward()` → `output_to_pdb()` → `run_mkdssp()` → `is_hairpin_region()`，全链路无 native structure 混入 |
| 激活 patching 范式正确性 | PASS | `patch_s_at_blocks()` context manager 实现 block-input pre-hook，clone 后替换 `s[0, target_res, :]`，clean/corrupted 范式（arXiv:2404.15255）。Hook 在 context 退出时通过 handle.remove() 清除，无副作用泄漏 |
| 8 个 block band 设计 | PASS | mutually exclusive，完整覆盖 0–47 blocks：{0-3, 4-7, 8-11, 12-15, 16-23, 24-31, 32-39, 40-47} |
| z-patching specificity control（Step C） | PASS | `patch_z_at_blocks()` 对 target cross-strand pairs 的 `z[0, pi, pj, :]` 做 donor 替换；与 s-patching 同 window、同 donor，设计正确 |
| matched-ctrl specificity（Step E） | PASS | `find_matched_control_mask()` 找 non-overlapping 等长 residue mask，避开 N/C 末端；逻辑正确 |
| late-window specificity（Step D） | PASS | bands b_24_31 / b_32_39 / b_40_47 已被 s-patching 全面覆盖，late-window 效应直接从这些 band 的结果读取，无需额外 condition |

### 3. 统计分析

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 效应量报告 | PASS | Δ(hairpin_rate) = mean per-chain difference；Wilcoxon signed-rank 报告配对差异 |
| 对照比较比值 | PASS | z_patch/matched_ctrl/late_window 均以"< 50% of s_patch early"为 predicate，已实现 |
| 多重比较 | WARN | 8 bands × 4 conditions = 32 tests；未做 Bonferroni / FDR 调整。给定早期 band 效应（Δ=-0.862, p=2.25e-36）与对照（Δ<-0.067, p<3.2e-5）之间的量级差异（约 13:1），多重比较不可能反转主结论。风险：WARN（低） |

### 4. 核心 predicate 验证

| Predicate | 要求 | 实测值 | 状态 |
|-----------|------|--------|------|
| early_effect ≥ 0.2 pp | 是 | Δ = -0.862 | PASS |
| early_p < 0.05 | 是 | p = 2.25e-36 | PASS |
| matched_ctrl < 50% of early | 是 | 0.067/0.862 = 7.7% | PASS |
| late_window < 50% of early | 是 | 0.085/0.862 = 9.8% | PASS |
| z_same_window < 50% of early | 是 | 0.011/0.862 = 1.3% | PASS |

---

## 综合评级

**PASS**（1 项 WARN：CATH 标注缺失 + 未做多重比较调整，均不影响 claim 的统计成立性）

**关键发现**：M1 方法实施完整，DSSP 判定链路正确，干预范式符合 clean/corrupted 标准，所有 5 个 predicate 通过。主要风险仅为技术性 WARN，不影响 baseline 完整性门槛。

**Phase 2 审计结论**：C1 **准入（ADMITTED）**，进入 Stage 2 variant 运行。
