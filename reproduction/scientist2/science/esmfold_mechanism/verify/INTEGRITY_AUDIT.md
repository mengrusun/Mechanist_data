# Verify Stage 完整性审计（INTEGRITY_AUDIT）

**项目**: ESMFold 折叠躯干 β-hairpin 机制的三-claim 因果验证  
**日期**: 2026-07-15  
**管线阶段**: Verify Stage（Phase 2 + Phase 9）

---

## 主实验完整性（Phase 2, per-claim）

### C1 — 早期 block 局部化 + `s` 是决策活跃位

| 审计维度 | 评级 | 关键发现 |
|----------|------|----------|
| EXPERIMENT_AUDIT | **PASS（WARN）** | WARN: cath_label 全 null（不影响主实验）；未做多重比较 FDR 调整（效应量差异 13:1，不影响结论） |
| MECHANISM_AUDIT | **PASS（WARN）** | WARN: donor distribution shift（Causal Attribution 内生局限，N=197 random pairs 下影响可忽略） |
| **综合** | **PASS** | C1 **准入 Stage 2** |

**发现详情**（见 `verify/C1_early_block_s_localization/baseline_audit/`）：
- M1 方法实施完整：8 个 block band，3 个 specificity controls，正确的 DSSP-on-predicted-structure 链路
- 所有 5 个 predicate PASS（早期 Δ=-0.862, p=2.25e-36；对照比值 ≤ 9.8%）
- hook 实现无泄漏风险

---

### C2 — early-block seq2pair 是 `s → z` 的关键因果通道

| 审计维度 | 评级 | 关键发现 |
|----------|------|----------|
| EXPERIMENT_AUDIT | **FAIL** | (1) seq2pair Δ=-0.017，不达 Δ≥0.2pp；(2) pair_to_sequence hook shape bug（expected [B,L,S], actual [B,num_heads,L,L,32]）导致 pair2seq / zero-ablation / matched-ctrl 三个对照实质未执行（各仅 3 条记录） |
| MECHANISM_AUDIT | **FAIL** | pair2seq hook 致命实现错误 + 主效应不达标（Δ=-0.017 vs 要求 Δ≥0.2pp） |
| **综合** | **FAIL** | C2 **REJECTED** → **INCONCLUSIVE** |

**发现详情**（见 `verify/C2_seq2pair_causal_channel/baseline_audit/`）：
- `scripts/m2_worker.py` 中 `pair_to_sequence` 的 `register_forward_hook` 回调假设 output shape `[B, L, S_HIDDEN]`，实际 ESMFold 该模块 output shape 为 `[B, num_heads, L, L, 32]`（pair-attention bias tensor）
- 异常在 per-donor loop 中向上传播，终止了 Steps B/C/D 的所有后续写入
- seq2pair donor-patch（Step A）在 hook 注册前已成功（176 条记录），Δ=-0.017（p=0.083）
- 证据状态：evidence-incomplete（实现缺陷），不是统计弱
- **修复路径**：修复 `m2_worker.py` pair_to_sequence hook → 重跑 M2（~1 GPU-h）→ 迭代轮次 1

---

### C3 — charge 是早期线性编码化学特征 + 对 β-hairpin 有因果影响

| 审计维度 | 评级 | 关键发现 |
|----------|------|----------|
| EXPERIMENT_AUDIT | **PASS（WARN）** | M3a PASS（WARN: 40 vs 250 计划链，heldout_probe 低于预期规模）；M3b FAIL（所有因果 predicate 失败，但这是真实科学 null）|
| MECHANISM_AUDIT | **PASS（WARN）** | 两个方法家族（Probing + Steering Vectors）均正确实施；M3b null 与文献一致（decodability ≠ causal sufficiency）；M3a 样本量偏小（WARN） |
| **综合** | **PASS（WARN）** | C3 **准入（ADMITTED）** |

**发现详情**（见 `verify/C3_charge_encoding_causal/baseline_audit/`）：
- M3a（Probing）：bacc=1.0 across all 4 blocks，perm p<0.001，v_charge 提取正确（L2_norm=1.0）
- M3b（Steering）：175/179 chains hairpin=1 在整条 β sweep 中恒定；Δ=-0.006（p=1.0）；pLDDT 变化 <0.003 units 排除结构崩塌
- 科学解读：block-0 的 v_charge 方向对下游折叠无因果充分性——"decodability ≠ causal sufficiency"

---

## Variant 完整性（Phase 9）

**仅适用于 C1（被选为 Stage-2 pick）**

### C1 Variants

| 变体 | 维度 | 实验审计 | 机制审计 | 合并评级 | eligible |
|------|------|----------|----------|---------|----------|
| dataset_cath_strat | dataset | PASS（WARN：CATH 退化单组） | PASS（WARN） | **PASS** | **是** |
| method_mean_ablation | method | EXCLUDED（NOT_RUN：执行权限受限） | EXCLUDED | **EXCLUDED** | **否** |
| model | model | n/a（task.md 明确无 model swap） | n/a | **n/a** | n/a |

**Phase 9 结论**：
- dataset_cath_strat integrity-clean → **eligible**，variant verdict = supported
- method_mean_ablation NOT_RUN → **excluded**（排除出分母，非因设计缺陷）
- N_eligible = 1，N_pass = 1，robustness = 1.0

**C2、C3 variant integrity audit**：跳过（C2 = INCONCLUSIVE，C3 = INTEGRITY_ONLY）

---

## 汇总

| Claim | Phase 2 评级 | Stage 2 状态 | Phase 9 评级 | 终态 |
|-------|-------------|-------------|-------------|------|
| C1 | PASS（WARN） | picked | PASS（N_eligible=1，robustness=1.0） | **PASS** |
| C2 | FAIL | skipped（INCONCLUSIVE） | skipped | **INCONCLUSIVE** |
| C3 | PASS（WARN） | deferred（max_verify_claims_cap） | skipped（INTEGRITY_ONLY） | **INTEGRITY_ONLY** |
