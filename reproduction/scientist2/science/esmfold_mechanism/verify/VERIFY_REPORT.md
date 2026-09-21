# Verify Report — ESMFold 折叠躯干 β-hairpin 机制验证

**项目**: ESMFold 折叠躯干 β-hairpin 机制的三-claim 因果验证  
**管线阶段**: Verify（auto-verify Phase 1–11）  
**日期**: 2026-07-15  
**配置**:
- `TARGET_CLAIMS`: all (C1, C2, C3)
- `DIMENSIONS`: method, dataset（model 轴由 task.md 关闭）
- `MAX_VERIFY_CLAIMS`: 1
- `ROBUSTNESS_THRESHOLD`: 0.5
- `MIN_VARIANTS_FOR_VERDICT`: 1
- `GPU_ID`: 0,1,2,3
- `COMPACT`: false

---

## Per-claim 终态

### C1 — 早期 block 局部化 + `s` 是决策活跃位

**终态**: ✅ **PASS**

| 项目 | 值 |
|------|-----|
| Baseline 结论 | supported |
| Phase 2 完整性 | PASS（WARN） |
| Stage-2 选取 | picked（top-1 by importance） |
| N_run | 2（dataset_cath_strat + method_mean_ablation） |
| N_eligible | 1（dataset_cath_strat integrity-clean；method_mean_ablation NOT_RUN excluded） |
| N_pass | 1（dataset_cath_strat: supported） |
| N_fail | 0 |
| robustness | **1.0**（≥ 0.5 threshold） |
| model 轴 | n/a（task.md 明确无 model swap） |
| dataset 轴 | PASS（Δ=-0.862, p=2.25e-36, N=197；bootstrap 95% CI ≈ [-0.91,-0.82]，不跨 0） |
| method 轴 | NOT_RUN（excluded；dispatch 命令已记录） |
| Variant integrity | 1 clean, 1 excluded |

**关键统计（主实验）**：
- 早期 band b_0_3 s-patching：Δ=-0.862, p=2.25e-36, N=197
- z-patching 对照：Δ=-0.011（早期的 1.3%）
- matched-ctrl 对照：Δ=-0.067（早期的 7.7%）
- late-window 对照：Δ=-0.085（早期的 9.8%）
- 所有 5 个 predicate PASS

---

### C2 — early-block seq2pair 是 `s → z` 的关键因果通道

**终态**: 🟡 **INCONCLUSIVE**

| 项目 | 值 |
|------|-----|
| Baseline 结论 | not-supported（evidence-incomplete） |
| Phase 2 完整性 | **FAIL** |
| Stage-2 选取 | 跳过（Phase 2 FAIL → INCONCLUSIVE） |
| robustness | —（variants 未运行） |

**INCONCLUSIVE 原因**：
1. `scripts/m2_worker.py` 中 `pair_to_sequence` hook shape mismatch bug（expected [B,L,S], actual [B,num_heads,L,L,32]），导致 pair2seq / zero-ablation / matched-ctrl 三个 specificity controls 实质未执行（各仅 3 条记录）
2. 唯一完整执行的条件 seq2pair_donor Δ=-0.017（p=0.083, N=176）：不达 Δ≥0.2pp predicate
3. 证据状态：evidence-incomplete（实现缺陷），不是统计弱

**科学注记**：seq2pair Δ=-0.017 vs M1 s-patch Δ=-0.862 的强对比暗示"β-hairpin 决策主要经由 s 残差流直接传播，而非 s→z seq2pair 桥"，但这需要配对 pair2seq 对照确认。

**修复路径**：修复 `scripts/m2_worker.py` pair_to_sequence hook → 重跑 M2（~1 GPU-h）→ 迭代轮次 1 中 `/auto-verify C2 --resume=true`。

---

### C3 — charge 是早期线性编码化学特征 + 对 β-hairpin 有因果影响

**终态**: ⚪ **INTEGRITY_ONLY**（`stage2_skip_reason: max_verify_claims_cap`）

| 项目 | 值 |
|------|-----|
| Baseline 结论（M3a） | supported（bacc=1.0, p<0.001） |
| Baseline 结论（M3b） | not-supported（real null，5/5 causal predicates fail） |
| Phase 2 完整性 | PASS（WARN） |
| Stage-2 选取 | deferred（MAX_VERIFY_CLAIMS=1 cap；C1 优先级更高） |
| robustness | —（variants 未运行，Stage 2 deferred） |

**INTEGRITY_ONLY 原因**：C3 通过了 Phase 2 baseline integrity gate（方法完整性 PASS），但因 MAX_VERIFY_CLAIMS=1 限制，Stage-2 swap variants 被推迟。

**科学注记**：C3 = "linearly encoded but not causally used"——这是一个有价值的教科书级中间结论，与 Belrose 2023 / Marks & Tegmark 2023 的 probe-positive/steering-negative 文献一致。

**升级命令**：`/auto-verify C3 --resume=true` 以运行 C3 的 swap variants（dataset 轴 CATH 分层 + method 轴 MLP probe）。

---

## Stage-2 选取（Phase 3 step 0）

```
Admitted pool: [C1, C3]
Rejected pool: [C2] (Phase 2 FAIL → INCONCLUSIVE)
Picked claims (K=1): [C1]
Stage-2 deferred: [C3] (stage2_skip_reason: max_verify_claims_cap)
```

**选取 C1 的理由**：
- C1 是项目因果链的**基础性发现**：所有其他 claims（M2, M3b）都依赖 M1 的 early window 定位
- C1 效应最强（Δ=-0.862）且最干净（5/5 specificity controls PASS）
- C3 的 M3b null 是真实科学发现，不需要 swap variants 来"证明更稳健"
- FINAL_PROPOSAL.md / IDEA_REPORT.md 将 C1 定位为 headline finding

---

## 总体统计

| 指标 | 值 |
|------|-----|
| 目标 claims | 3（C1, C2, C3） |
| PASS | **1**（C1） |
| FAIL | 0 |
| INCONCLUSIVE | **1**（C2） |
| ZERO_ELIGIBLE_VARIANTS | 0 |
| INTEGRITY_ONLY | **1**（C3，skip=max_verify_claims_cap） |

---

## GPU 资源

| 阶段 | GPU-hours |
|------|-----------|
| 实验阶段（累计） | ~3.4 h |
| Verify Stage（变体运行） | ~0 h（dataset 变体零 GPU；method 变体 NOT_RUN） |
| **累计消耗** | **~3.4 h** |
| 剩余预算 | ~6.6 h（10h 总预算） |
| GPU pin 违规 | 无（dataset variant 零 GPU；method variant NOT_RUN） |

---

## 工件索引

| 文件 | 描述 |
|------|------|
| `verify/VERIFY_REPORT.md` | 本文件 |
| `verify/INTEGRITY_AUDIT.md` | Phase 2 + Phase 9 完整性审计（合并文件） |
| `verify/STAGE2_PICK.json` | Phase 3 step 0 选取记录 |
| `verify/C1_early_block_s_localization/baseline_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}` | C1 Phase 2 审计 |
| `verify/C1_early_block_s_localization/variant_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}` | C1 Phase 9 变体审计 |
| `verify/C1_early_block_s_localization/ROBUSTNESS.md` | C1 鲁棒性报告 |
| `verify/C1_early_block_s_localization/variants/dataset_cath_strat/` | CATH 分层变体（COMPLETED） |
| `verify/C1_early_block_s_localization/variants/method_mean_ablation/` | Mean-ablation 变体（NOT_RUN，含代码审查） |
| `verify/C2_seq2pair_causal_channel/baseline_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}` | C2 Phase 2 审计 |
| `verify/C2_seq2pair_causal_channel/ROBUSTNESS.md` | C2 状态报告（INCONCLUSIVE） |
| `verify/C3_charge_encoding_causal/baseline_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}` | C3 Phase 2 审计 |
| `verify/C3_charge_encoding_causal/ROBUSTNESS.md` | C3 状态报告（INTEGRITY_ONLY） |
