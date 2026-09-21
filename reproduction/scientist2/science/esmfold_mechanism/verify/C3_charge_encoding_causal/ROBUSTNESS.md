# C3 鲁棒性报告（Robustness Report）

**Claim ID**: C3  
**Claim 陈述**: 电荷是 ESMFold 早期 block 的 `s` 中线性可编码的化学特征（bacc ≥ 0.75，p < 0.001），且该特征对 β-hairpin 形成有因果影响（同号 steering ↓ / 异号 steering ↑，|Spearman ρ| ≥ 0.7）。  
**日期**: 2026-07-15

---

## Phase 2 — 主实验完整性（Baseline Integrity Gate）

| 审计类型 | 评级 | 关键发现 |
|----------|------|----------|
| EXPERIMENT_AUDIT | PASS（WARN） | M3a PASS（WARN: 40 vs 250 chains）；M3b FAIL（5/5 causal predicates，real null） |
| MECHANISM_AUDIT | PASS（WARN） | 两个方法家族均正确实施；null 是有效科学发现 |
| 综合评级 | **PASS（WARN）** | C3 方法完整性 PASS |

**Phase 2 决策**：C3 **ADMITTED（WARN）** — 方法完整，M3b null 是真实科学发现，非实现缺陷

**注意**：C3 被 Phase 3 step 0 从 Stage-2 swap variants 中**延迟**（`stage2_skip_reason: max_verify_claims_cap`），因为 MAX_VERIFY_CLAIMS=1 且 C1 被选为更高优先级。C3 终态 = **INTEGRITY_ONLY**。

---

## Stage 2 跳过（MAX_VERIFY_CLAIMS cap）

**原因**: `stage2_skip_reason: max_verify_claims_cap`（C1 被选为单一 Stage-2 claim）

C3 没有运行 swap variants。这不是 Phase 2 失败，而是资源分配决策。

---

## 最终 Verdict

**robustness = —**（Stage 2 跳过；方法完整但未做 swap variants）  
**C3 终态：⚪ INTEGRITY_ONLY**  
**stage2_skip_reason**: `max_verify_claims_cap`  

---

## 主实验结论记录（供参考）

- **M3a（charge linear encoding）**：bacc=1.0，p<0.001，全 4 个 block 均 PASS。**科学成立**。
- **M3b（causal steering）**：5/5 causal predicates FAIL，Δ=-0.006，p=1.0。**真实 null**。
- C3 整体 = "linearly encoded but not causally used"——与 Belrose 2023 / Marks & Tegmark 2023 的 probe-positive / steering-negative 文献一致。

---

## 升级命令

若需对 C3 运行 swap variants（特别是 M3a 的 dataset 轴 + method 轴），执行：

```
/auto-verify C3 --resume=true
```

建议 variants：
- dataset_cath_strat（M3a）：CATH-stratified probe accuracy re-check（补全 CATH 后可分层）
- method_cath_strat 或 method_mlp_probe（M3a）：swap linear probe → MLP-1-hidden probe（within Probing family）
