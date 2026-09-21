# C2 鲁棒性报告（Robustness Report）

**Claim ID**: C2  
**Claim 陈述**: 在早期 block 中，seq2pair 操作是把 β-hairpin 折叠决策从 `s` 写入 `z` 的关键因果通道。  
**日期**: 2026-07-15

---

## Phase 2 — 主实验完整性（Baseline Integrity Gate）

| 审计类型 | 评级 | 关键发现 |
|----------|------|----------|
| EXPERIMENT_AUDIT | **FAIL** | (1) seq2pair donor-patch Δ=-0.017，不达 Δ≥0.2pp；(2) pair_to_sequence hook shape bug 导致 pair2seq/zero-ablation/matched-ctrl 实质未执行 |
| MECHANISM_AUDIT | **FAIL** | pair2seq hook 实现错误（shape mismatch）+ 主效应不达标 |
| 综合评级 | **FAIL** | C2 不进入 Stage 2 |

**Phase 2 决策**：C2 **REJECTED（FAIL）** → 进入 **INCONCLUSIVE** 状态

---

## Stage 2 跳过

C2 因 Phase 2 baseline integrity FAIL 而跳过 Stages 2-10。

按 `/auto-verify` 状态机：Phase 2 baseline integrity FAIL → claim 终态 = **INCONCLUSIVE**（不是 FAIL，不是 ZERO_ELIGIBLE_VARIANTS）

---

## 最终 Verdict

**robustness = —**（Phase 2 FAIL，variants 未运行）  
**C2 终态：🟡 INCONCLUSIVE**  
**INCONCLUSIVE 原因**：Phase 2 baseline integrity FAIL（evidence-incomplete due to pair_to_sequence hook shape bug + 主效应 Δ=-0.017 不达标）

---

## 修复路径

修复 `scripts/m2_worker.py` 中的 `pair_to_sequence` hook：

```python
# 错误的假设：output shape = [B, L, S_HIDDEN]
new_out[0, tgt_idx, :]  # IndexError

# 正确做法：ESMFold pair_to_sequence output = [B, num_heads, L, L, 32]
# 方案 A: 在 pair_to_sequence 的 MLP hidden state 上 hook（非输出）
# 方案 B: 直接零消融 s[target_res] 的 pair_to_sequence 贡献（via s 层面的消融）
```

**推荐行动**：在迭代轮次 1 中修复 hook + 重跑 M2（~1 GPU-h），然后 `/auto-verify C2 --resume=true`。
