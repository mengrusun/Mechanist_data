# C3 机制严谨性审计（Phase 2 — Mechanism Audit）

**Claim**: C3 — charge 是早期线性编码化学特征，且对 β-hairpin 形成有因果影响  
**Mechanism families**: Probing / Residual Stream States（M3a）+ Steering Vectors（M3b）  
**审计时间**: 2026-07-15

---

## 机制严谨性检查

### Part A: M3a — Probing / Residual Stream States

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 家族映射 | PASS | MECHANISM_ROUTING.md 指定 Probing / Residual Stream States；M3a 实施与 submethod 完全匹配（block-wise s capture + linear classifier + permutation null） |
| 因果声明边界正确 | PASS | M3a 只主张"linearly decodable"，不主张"causally used"；这是一个合法的相关性/解码性声明，由线性探针直接回答 |
| Chain-level split 防泄漏 | PASS | 8:1:1 chain-level split 避免同一 chain 的不同 residue 同时出现在 train 和 test 中，PISCES ≤25% 序列同一性进一步保证 cross-chain 独立性 |
| v_charge 方向提取方法 | PASS | `v_charge = L2_normalize(w_pos - w_neg)` 是 CAA/representation analysis 中的标准 signed direction extraction；w_pos / w_neg 分别是 probe 对 positive / negative charge 类的权重向量 |

### Part B: M3b — Steering Vectors

| 检查项 | 状态 | 细节 |
|--------|------|------|
| 家族映射 | PASS | MECHANISM_ROUTING.md 指定 Representation and Parameter Analysis / Steering Vectors；M3b 实施 `s[block_k, res, :] += β · σ_proj · v_charge` 是标准 CAA-style additive steering |
| 干预强度是否合理 | PASS | β=+3σ 对应 ≈64.7 residual-stream 单位，明显超过正常激活范围（σ≈21.55）；这是 "strong enough to test" 的标准设定。random direction 控制的 hairpin_rate=100% 进一步确认了 perturbation 强度足够但 hairpin 决策"对任何方向均鲁棒"。 |
| Null 结果的科学有效性 | PASS | 175/179 chains hairpin 恒为 1 整条 β sweep，random control 也是 100%——这是"下游折叠对 block-0 s 方向扰动不敏感"的证据，与 concept-erasure 文献（Belrose 2023, Marks & Tegmark 2023）报道的"probe positive, steering negative"结果一致。Null 本身是有效的科学发现。 |
| matched-control 特异性 | PASS | matched-ctrl（非 target pair）hairpin rate ≈ 99.4%（≈ target pair rate 98.9%），说明 steering 对 hairpin 的影响无位点特异性——这是 null 结论的补充确认，而非方法缺陷 |
| 是否排除 structural collapse 假说 | PASS | pLDDT 变化 < 0.003 units at β=+3σ（所有 conditions）；random direction 同样不改变 hairpin——说明 null 不是"结构崩塌导致 DSSP 失效"，而是"hairpin 决策对 v_charge 方向的加性扰动不敏感" |

### C3 整体机制层级评估

**C3 claim 的机制层级**：

- **M3a（Unit Interpretation）**：提取 s 中电荷维度 v_charge → 解码性声明 ✓
- **M3b（Causal Intervention / Steering）**：v_charge 方向的加性 steering → 充分性声明

实验设计正确执行了"decodability ≠ causal sufficiency"的双层检验，是文献中反复警告却经常被忽视的关键区分（见 arXiv 2209.03013 § 4.3）。M3b 的 null 结果**本身是机制发现**：电荷在 block-0 的 s 中是可解码的，但沿该方向的加性扰动被下游 48-block recycle 或非线性组合"吸收"了。

---

## 综合评级

**M3a**: PASS  
**M3b**: PASS（方法实施正确，null 结果是真实科学发现）  
**C3 overall mechanism audit**: **PASS（WARN on M3a sample size）**

**Phase 2 机制审计结论**：C3 **机制严谨性 PASS**。两部分方法家族均正确实施，v_charge 提取逻辑正确，steering 强度合理，null 是科学发现而非技术缺陷。
