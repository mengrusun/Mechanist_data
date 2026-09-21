# C2 机制严谨性审计（Phase 2 — Mechanism Audit）

**Claim**: C2 — early-block seq2pair 是 `s → z` 的关键因果通道  
**Mechanism family**: Causal Attribution / Patching（pathway ablation variant）  
**审计时间**: 2026-07-15

---

## 机制严谨性检查

### 1. 因果解读合法性

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 干预范式设计正确性 | PASS | Pathway ablation 的设计逻辑正确：seq2pair output 替换为 donor 值 → 等效于"将 s→z 写入通道置为外部信号"；pair2seq matched ablation 作为对称对照。这是 Causal Attribution / Patching 在 sub-module output 层面的标准变体 |
| 干预位点与 claim 一致 | PASS | Claim 说"seq2pair is the critical channel from s to z"；干预精确作用于 `trunk.blocks[k].sequence_to_pair.output` 的 target pair 位置 |
| pair2seq 对照的逻辑 | PASS | pair2seq 对照（干预 z→s 通道）作为"同等强度、不同方向"的特异性控制，设计上正确——如果两者效应相近，说明效应非 pathway-specific |

### 2. 实施缺陷的机制含义

| 风险项 | 状态 | 细节 |
|--------|------|------|
| seq2pair output hook 正确性 | **FAIL** | `collect_donor_pathway_outputs` 中 seq2pair hook 通过 `register_forward_hook` 捕获 output 并 clone，实施正确。但 pair2seq hook 假设 output shape [B,L,S]，实际 ESMFold 的 `pair_to_sequence` 输出 shape 为 [B,num_heads,L,L,32]——这是一个致命实现错误 |
| 现有 seq2pair_donor 结果可信度 | WARN | seq2pair Step A 在 hook 错误前成功执行（176 条记录），其 hook 使用 `register_forward_hook` 捕获 seq2pair.output（形状 [1,L,L,Z]）并替换 pair 位置——这部分逻辑本身正确。但缺少 pair2seq 对照，无法区分"seq2pair specific"vs"any pair-level perturbation"两种解读 |
| 科学解读风险 | WARN | seq2pair Δ=-0.017 vs M1 s-patch Δ=-0.862 形成强对比，暗示 β-hairpin 决策经由 `s` 直接传播而非经 `s→z` seq2pair 桥。但这一解读在缺少 pair2seq 对照的情况下存在"alternative pathway" confound |

### 3. 可修复性评估

- **bug 可修复**：将 `pair_to_sequence` hook 改为捕获正确的 bias tensor shape，或换用 residue-level intermediate（如 `pair_to_sequence` 的中间 MLP hidden state）作为 patch target。
- **修复后 M2 的科学价值**：如果修复后 pair2seq 干预也产生同样小的效应（~Δ=-0.017），则 C2 的"not-supported"结论实际上更加稳固——意味着两个方向都不能由 output-level ablation 重现 M1 效应，支持"s residual stream 直接传播"假说。

---

## 综合评级

**FAIL**

pair_to_sequence hook 的 shape mismatch 是关键机制实施错误；seq2pair Δ=-0.017 的本身不达标（与 claim 预测的 Δ≥0.2 pp 相差 12 倍）是独立的实质性 FAIL。

**Phase 2 机制审计结论**：C2 **机制审计 FAIL** → 与实验审计合并为 **INCONCLUSIVE**（evidence-incomplete）。
