# C2 主实验方法学审计（Phase 2 — Experiment Audit）

**Claim**: C2 — early-block seq2pair 是 `s → z` 的关键因果通道  
**Milestone**: M2 — seq2pair vs pair2seq 匹配路径消融  
**审计时间**: 2026-07-15

---

## 方法学完整性检查

### 1. 数据集与样本选取

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 数据来源 | PASS | 与 M1 相同的 PISCES cull list + 200 main chains，来源符合 task.md HARD |
| seq2pair_donor 成功 N | PASS | N=176（3 chains 因 hook 崩溃前退出等原因丢失，176/200=88%，仍合理） |
| pair2seq / zero-ablation / matched-ctrl | **FAIL** | 仅 3 条记录成功写入（531/534 次尝试报 hook-shape 错误），实际上无法进行配对统计 |

### 2. 关键 Bug 分析：`pair_to_sequence` hook shape mismatch

**Bug 位置**: `scripts/m2_worker.py`，Step B pair2seq donor-patch 分支

**问题**：
- 实现假设 `pair_to_sequence.output` shape 为 `[B, L, S_HIDDEN]`（类比 `pair2seq` 作为 residue-level 特征）
- 实际 ESMFold 中该模块输出 shape 为 `[B, num_heads, L, L, 32]`（pair-attention bias tensor），不是 residue-level 表示
- Hook 在 `new_out[0, tgt_idx, :]` 处索引时触发 shape mismatch 异常
- 异常在 per-donor loop 中向上抛出，终止了 `seq2pair_zero` 和 `seq2pair_matched_ctrl` 的后续写入

**影响范围**：
- `seq2pair_donor`（Step A）在异常发生**之前**执行，176 条记录完整
- Steps B（pair2seq）/ C（zero-ablation）/ D（matched-ctrl）实质上**未执行**（仅 3 条快退记录）

### 3. Claim 可验证性评估

C2 的核心统计证明要求：
1. seq2pair_donor Δ ≥ 0.2 pp（测量到 Δ=-0.017，FAIL — 效应太小）
2. pair2seq 配对差 p < 0.05（无数据，FAIL）
3. matched_ctrl 小于 50% 阈值（无数据，无效 PASS）

**关键判断**：
- FAIL #1（seq2pair Δ=-0.017 不达标）即使在 bug 修复后也**不会改变**——seq2pair donor-patch 已经有 176 条记录，效应本身就是 Δ=-1.7%
- FAIL #2-3 来自 bug，是"证据缺失"而非"反证成立"
- Claim 2 的核心效应（"ablating seq2pair causes significant drop"）在唯一可用的条件（seq2pair_donor）上**未得到支持**（Δ=-0.017, p=0.083）

### 4. 统计分析完整性

| 检查项 | 状态 | 细节 |
|--------|------|------|
| seq2pair_donor 主效应 | FAIL | Δ=-0.017 (p=0.083, N=176)：未达到 Δ≥0.2pp 或 p<0.05 |
| 配对 McNemar（seq2pair vs pair2seq） | FAIL | pair2seq N=3，无法进行有效配对检验 |
| zero-ablation 一致性检验 | FAIL | zero-ablation N=3，无法进行 |
| matched-ctrl 特异性 | FAIL | matched-ctrl N=3，无法评估 |

### 5. Predicate 验证

| Predicate | 要求 | 实测值 | 状态 |
|-----------|------|--------|------|
| s2p_effect ≥ 0.2 pp | 是 | Δ = -0.017 | **FAIL** |
| s2p_p < 0.05 | 是 | p = 0.0833 | **FAIL** |
| p2s_effect < 50% of s2p | — | N/A（数据缺失） | INCONCLUSIVE |
| matched_ctrl < 50% | — | N/A（数据缺失） | INCONCLUSIVE |
| s2p_vs_p2s_p < 0.05 | 是 | N/A（无配对数据） | **FAIL** |

---

## 综合评级

**FAIL**

**原因**：
1. 主效应（seq2pair donor-patch Δ=-0.017）本身不达标，与 claim 要求的 Δ≥0.2pp 相差一个数量级
2. `pair_to_sequence` hook shape bug 导致三个特异性对照（pair2seq、zero-ablation、matched-ctrl）实质上未执行
3. 证据状态为"evidence-incomplete"（非统计弱，而是实现缺陷导致的证据缺失）——但按 Phase 2 gate 规则，FAIL predicates → 整体 FAIL → C2 进入 INCONCLUSIVE 状态

**修复建议**：修复 `scripts/m2_worker.py` 中的 `pair_to_sequence` hook 以处理正确的 `[B, num_heads, L, L, 32]` shape，重跑 M2（~1 GPU-h），在下一迭代轮次重新验证 C2。

**Phase 2 审计结论**：C2 **FAIL（evidence-incomplete + 主效应不达标）** → Stage 2 **跳过**，claim 进入 **INCONCLUSIVE** 状态。
