# Experiment Plan — ESMFold 折叠躯干 β-hairpin 机制的三-claim 因果验证

```yaml
# ---- Machine-readable metadata (do not edit informally) ----
behavior_source: given
mechanism: discovery
resource_fidelity: cost-aware   # NOT strict — this is given + discovery, not the reproduction combo
mechanism_strategy:
  directions: [Location, Causal Intervention, Unit Interpretation]
  rejected:
    - "Tuning & Editing — diagnostic vs applied 不匹配"
    - "Formation Tracing — genesis 不在 claim 内 + 太贵"
    - "Decision Auditing — 不是本项目问题"
  note: "Location → Causal Intervention 覆盖 claim 1/2；Unit Interpretation（linear-probing slice）+ Causal Intervention 覆盖 claim 3。"
target_model: ESMFold                           # HARD: sole model
dataset_source: PISCES cull list cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055
evaluation_ground_truth: DSSP on predicted structure   # HARD: task.md 明文
gpu_budget_hours: 10                            # HARD
gpu_ids_allowed: [0, 1, 2, 3]                   # HARD
forbidden_directories: [everything outside working dir, /data/zhenqian/data, /data/zhenqian/models]  # HARD
# NOTE: 无 M0（behavior_source=given）；三条 claim 直接进入 mechanism milestones，因此 M1/M2/M3 **不**声明 depends_on: [M0]。
```

---

## 0. Shared setup（所有 milestone 都依赖，作为 stage-0 一次性完成）

- **数据集准备**
  - 从 PISCES cull list `cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055` 读取 12,055 条 chain 列表；
  - 从 PDB 下载对应结构（若 `/data/zhenqian/data/pdb/` 已存在则使用，否则下载并落地 `/data/zhenqian/data/pdb/`）；
  - 对每条链在 native structure 上跑 DSSP（`mkdssp`）→ SS 串；
  - 筛选：60 ≤ length ≤ 300、native DSSP 上包含 ≥1 段 β-hairpin（reverse-parallel E-E 段中间有 turn 桥接）；
  - 对每条通过筛选的链，跑 ESMFold 无干预 forward → predicted structure → DSSP → target region 是否被判 β-hairpin；仅保留 baseline_hairpin_rate ≥ 0.7 的链；
  - 从通过 baseline 门槛的链中**随机抽 200 条作 main experiment 集合，另 50 条作方法学 calibration 集合**（不与 main 重叠）；剩余暂不使用；
  - 每条链落 metadata `{pdb_id, chain_id, seq, target_region_start, target_region_end, cross_strand_pairs, baseline_hairpin_rate, cath_label_if_available, gpu_used}` 到 `data/prepared/manifest.jsonl`。
- **CATH 映射**：如可查 CATH（v4.3+）的 chain-level 映射，就地写入 `cath_label`（Class.Architecture.Topology 三级）；不能映射的写 `null`，不影响 main experiment。
- **DSSP judge 函数**：一个纯函数 `is_hairpin(pdb_or_pred_struct, target_start, target_end) -> bool`——严格规则："target 区域覆盖两段 antiparallel E-E 且中间 ≤ 5 residue turn"。此函数在所有 milestone 中共用。
- **Donor 池**：从 baseline 集合中另抽 100 条链构成 donor 池；每条 target chain 随机匹配 3 个 donor（长度差 ≤ 20% 且 donor 在对应 residue 位置 native DSSP 不含 hairpin）。

**Stage-0 预估 GPU-hours**: 1.5 h（12k chain 的 mkDSSP + 500 条候选链的 ESMFold baseline forward）。

---

## M1: 早期 block 局部化 + `s` 是决策活跃位（对应 Claim 1）

**Verifies**: task.md Claim 1；`kind: mechanism-localization-and-intervention`

**Ladder of evidence**:
- **Step A（correlational screen）**: block-wise linear probe 从 `s[block_k, target_residues, :]` 预测 target region 是否会被折成 β-hairpin（二分类）——扫 block k ∈ {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44}（12 个采样点）；hairpin-formation-probe accuracy vs block 的曲线画出，找累积高信号的最早 block window。
- **Step B（causal intervention — s-patching）**: 对 candidate window `[i, j]`（8 个 mutually-exclusive band，见 FINAL_PROPOSAL §5.2），在此 window 每一个 block 的入口对 `s[:, target_residues, :]` 用 donor 覆盖；跑 predict → DSSP → 判 hairpin。每个 chain × window × donor = 一次 forward。
- **Step C（specificity — z-patching in same window）**: 相同 window 内对 `z[:, target_pairs, :, :]` 做 donor 覆盖（不动 `s`）；比较对 hairpin rate 的降幅。
- **Step D（specificity — s-patching in late window）**: 在 late window（`[24–31]` 或 `[32–39]` 等）对 `s` 做同强度 patching；比较对 hairpin rate 的降幅。
- **Step E（matched-control specificity）**: 对同一链上一段**非** β-hairpin 的等长 residue mask 做同 window、同 donor 的 s-patching；测其对 target-region hairpin rate 的效应（应显著小）。

**Expected sign**: **down**（early window s-patching 显著减少 target hairpin rate；late window 或 z-patching 显著更弱）。

**Magnitude / dose-response**:
- Primary effect size: Δ(hairpin_rate) ≥ 0.2 pp（e.g. 0.85 → ≤0.65）在早期 window s-patching 上；
- Late-window s-patching 或 same-window z-patching 的 |Δ| 应 ≤ 早期 window s-patching |Δ| 的 50%。

**Specificity control**:
- z-patching in same window（step C）—— hairpin rate change 显著较小；
- Late-window s-patching（step D）—— hairpin rate change 显著较小；
- Matched-control non-target mask patching（step E）—— hairpin rate change 显著较小；
- 报告三个对照的 paired difference tests。

**Cmd (template)**: `python m1_early_block_s_patching.py --chains data/prepared/manifest.jsonl --n 200 --windows "0-3,4-7,8-11,12-15,16-23,24-31,32-39,40-47" --donor-per-chain 3 --gpu ${gpu} --out results/M1/`

**Expected output**: `results/M1/effect_by_window.jsonl`, `results/M1/summary_stats.json`, `results/M1/plot_effect_vs_window.pdf`.

**Priority**: MUST-RUN
**Estimated GPU-hours**: ≈ 3.0 h（200 chains × 8 windows × 3 donors × 1 ESMFold forward ≈ 4800 forwards, ~2 s/forward on GPU → 2.7 h + probe 训练 + DSSP ≈ 3.0 h）

**method_sensitive**: [n_pairs, sites, metric, gpu_hours]
（注：机制家族在 experiment stage 由 `/mechanism-skills` 路由到具体 submethod——activation-patching 家族的 clean/corrupted 变体、attribution 家族的 IG/SHAP 变体、causal-mediation 家族的 direct/indirect effect 划分——这些 submethod 会重新 bind 上面几个 field 的最终值，符合 non-strict 组合下的 plan reconciliation 规则。）

---

## M2: early-block seq2pair 是把 β-hairpin 决策从 `s` 写入 `z` 的关键通道（对应 Claim 2）

**Verifies**: task.md Claim 2；`kind: pathway-causal-attribution`

**Depends on**: M1（M2 需要 M1 找到的早期 window `[i*, j*]` 作为干预区间；同时也 sanity-check：如果 M1 的 window 未 localized，M2 的默认 window 用 `[0–7]` 作为 fallback 并明确标注）

**Ladder of evidence**:
- **Step A（pathway ablation — seq2pair）**: 在 window `[i*, j*]` 内每一个 block 的 seq2pair layer 的 output tensor 上，对 target pair 位置（cross-strand pairs of the target hairpin）替换为 donor 值；跑 predict → DSSP → 判 hairpin。
- **Step B（matched pathway ablation — pair2seq）**: 同 window、同 target mask（对应 pair2seq 的 residue mask 而非 pair mask，但保证受影响的 seq residue 数 = target hairpin 涉及的 residue 数）；替换 pair2seq 的 output；跑 predict → DSSP → 判 hairpin。
- **Step C（sanity — zero-ablation instead of donor-patch）**: 对 seq2pair output 做 zero-ablation（把 target-pair 上的 seq2pair output 置零）；如果与 donor-patch 方向一致 → 结论对干预类型 robust。
- **Step D（matched-control specificity）**: 对同一链上非 β-hairpin 的等大小 pair mask 做同 window 同强度的 seq2pair patch；效应应显著小。

**Expected sign**: **down**（seq2pair 干预 → hairpin rate ↓）；pair2seq 干预 → hairpin rate 变化显著更小。

**Magnitude / dose-response**:
- seq2pair patch Δ(hairpin_rate) ≥ 0.2 pp；
- pair2seq matched patch Δ(hairpin_rate) ≤ seq2pair Δ × 50%；
- 两者差异 paired McNemar p < 0.05。

**Specificity control**:
- pair2seq matched intervention（step B）；
- matched-control non-target pair mask（step D）；
- （可选）zero-ablation vs donor-patch 方向一致（step C）作为 robustness sanity。

**Cmd (template)**: `python m2_seq2pair_vs_pair2seq_ablation.py --chains data/prepared/manifest.jsonl --n 200 --early-window "${early_window_from_M1}" --donor-per-chain 3 --ablation-modes donor,zero --gpu ${gpu} --out results/M2/`

**Expected output**: `results/M2/seq2pair_effect.jsonl`, `results/M2/pair2seq_effect.jsonl`, `results/M2/paired_diff_stats.json`.

**Priority**: MUST-RUN
**Estimated GPU-hours**: ≈ 2.5 h（200 chains × 2 pathways × 2 ablation modes × 3 donors × 1 forward ≈ 2400 forwards → 1.3 h + 与 M1 复用一部分 baseline 缓存 ≈ 2.5 h with margin）

**method_sensitive**: [sites, metric, gpu_hours]

---

## M3: charge 是早期线性编码化学特征 + 因果影响 β-hairpin（对应 Claim 3）

**Verifies**: task.md Claim 3；`kind: linear-probe-plus-causal-steering`

**Depends on**: M1（M3 需要 M1 找到的早期 block 作为 probe 训练与 steering 的位置；如果 M1 未 localized，用 `[0–7]` fallback）

**Two-part structure**（两部分**都必须通过**才算 claim 3 成立）：

### M3a — linear encoding of charge

**Ladder**:
- **Step A**: 对 early window 内每个 sampled block k（{0, 2, 4, 6} 4 个采样点），在 held-out chains（stage-0 剩余的 chains）上从 `s[block_k, :, :]` 训练 3-class linear classifier 预测 residue charge {negative (D/E), positive (K/R/H), neutral (其余)}；train/val/test 按 chain-level 8:1:1 划分（避免 residue-level 泄漏，因为 PISCES ≤25% seq id 已保证 chain 间低同源）。
- **Step B**: report per-block balanced 3-class accuracy + AUROC；用 1000-permutation label-shuffle 生成 null 分布，报告 empirical p-value。
- **Step C**: 从最强 block（accuracy 最高者）的 probe 提取权重方向 `v_charge`——具体地取 positive-class weight vector minus negative-class weight vector（作为 signed charge 方向），并 L2-normalize。**记录**：`v_charge`、其所在 block、accuracy、AUROC 全部落盘。

**Expected sign**: **up**（accuracy 显著 > chance）。

**Predicate**: balanced 3-class accuracy ≥ 0.75 在至少一个早期 block 上；permutation p < 0.001。

**Cmd (template)**: `python m3a_charge_probe.py --chains data/prepared/manifest.jsonl --sampled-blocks 0,2,4,6 --n-held-out 300 --classes neg,pos,neut --shuffles 1000 --gpu ${gpu} --out results/M3a/`

**Expected output**: `results/M3a/probe_by_block.jsonl`, `results/M3a/v_charge.npy`, `results/M3a/best_block.json`, `results/M3a/permutation_null.json`.

**Priority**: MUST-RUN
**Estimated GPU-hours**: ≈ 0.5 h（probe 训练轻，只跑 4 个 block × 一次 ESMFold hidden-state extraction on ~300 chains）

**method_sensitive**: [metric, gpu_hours]

### M3b — causal effect of charge steering on β-hairpin formation

**Ladder**:
- **Step A (steering intervention)**: 在 M3a 找到的最强 block 上，对 200 chains 的每条链的 target hairpin 的 cross-strand pair 上的两个 residue 施加 `s[block_k, res_i, :] += α · v_charge` 与 `s[block_k, res_j, :] += α · v_charge`（same-charge configuration）或 `s[block_k, res_i, :] += α · v_charge, s[block_k, res_j, :] -= α · v_charge`（opposite-charge configuration）；α ∈ {-3, -1, +1, +3} × 2 sign patterns = 8 conditions per chain.
- **Step B (DSSP evaluation)**: 每个 condition 跑 predict → DSSP → target hairpin boolean；同时记录 target region 的 cross-strand Cα-Cα 距离（辅助指标）。
- **Step C (dose-response)**: 对每个 chain 画 α → hairpin_rate 曲线；聚合 200 chain 后计算 Spearman ρ；期望单调可分辨。
- **Step D (same vs opposite paired diff)**: same-charge (both +α, α=3) vs opposite (α=+3, -3) 在 hairpin rate 上的 paired difference test（McNemar / Wilcoxon signed-rank）。
- **Step E (matched-control specificity)**: 对同一 chain 上另一对非 target、非 β-hairpin 的 residue pair 施加相同 α · v_charge steering；测其对 target region hairpin rate 的效应（应显著小）。

**Expected sign**:
- opposite-charge steering (α_i = +3, α_j = -3) → hairpin rate 相对 baseline **up**（或至少不下降）；
- same-charge steering (α_i = +3, α_j = +3) → hairpin rate **down**（配对差 p < 0.05）；
- cross-strand distance：same → up；opposite → down（辅助指标同向支持）。

**Magnitude / dose-response**:
- 沿 α 从 -3 → +3 的单调性：|Spearman ρ| ≥ 0.7（same-side vs opposite-side 分别检验单调性）；
- same vs opposite Δ(hairpin_rate) ≥ 0.15 pp；paired diff p < 0.05；
- matched-control effect ≤ 50% of target-mask effect。

**Specificity control**:
- matched-control non-target residue pair steering（step E）；
- （可选）随机方向 `v_random`（同 L2 norm 的随机向量）steering 效应应 ≈ 0。

**Cmd (template)**: `python m3b_charge_steering.py --chains data/prepared/manifest.jsonl --n 200 --best-block ${best_block_from_M3a} --v-charge results/M3a/v_charge.npy --alphas "-3,-1,1,3" --configs same,opposite --gpu ${gpu} --out results/M3b/`

**Expected output**: `results/M3b/steering_effect.jsonl`, `results/M3b/dose_response.json`, `results/M3b/matched_control_stats.json`, `results/M3b/plot_dose_response.pdf`.

**Priority**: MUST-RUN
**Estimated GPU-hours**: ≈ 1.5 h（200 chains × 4 α × 2 configs × 2 pair-vs-control = 3200 forwards × ~2 s ≈ 1.8 h → 1.5 h with baseline reuse）

**method_sensitive**: [sites, metric, gpu_hours]

---

## Aggregate budget & schedule

| Stage | Milestone | Priority | GPU-hours (est.) | Depends on |
|---|---|---|---|---|
| 0 | Shared setup | MUST | 1.5 | — |
| 1 | M1 | MUST | 3.0 | Stage 0 |
| 2 | M2 | MUST | 2.5 | M1 |
| 3 | M3a | MUST | 0.5 | Stage 0（可与 M1 并行，若 GPU 空闲） |
| 4 | M3b | MUST | 1.5 | M3a（且引用 M1 的 early window） |
| **Total** | | | **9.0 h** | + 1 h buffer 给 rerun/OOM/DSSP 边界处理 = **10 h HARD** |

**GPU 分配建议**（4 GPUs available, ids {0,1,2,3}）：
- GPU 0 = Stage 0 (once) + M1 primary
- GPU 1 = M2 primary
- GPU 2 = M3a + M3b primary
- GPU 3 = OOM/rerun spare + verify variant precompute

---

## Compatibility with verify-stage CATH-stratified swap variant

- 每个 milestone 的 `results/*/effect_*.jsonl` 都包含 per-chain 的 `pdb_id`、`cath_label`（若可映射）、`effect_size`；
- verify-stage 直接用 `cath_label` 分组重跑 aggregate statistics，**无需重跑 ESMFold forward**（DSSP 判定与干预效应已 per-chain 落盘）；
- 因此 CATH-stratified variant 的额外 GPU 预算 ≈ 0（只是 stats aggregation 的重跑）——完全在 10 h 预算之外的 verify stage 内可承受。

---

## Decision gates（每个 milestone 完成后的分岔）

- **M1 PASS** → 记录 `early_window = [i*, j*]`；进入 M2（若 FAIL：报告 negative result "no early localization"，M2 用 fallback window `[0–7]` 并明确标注）。
- **M2 PASS** → 继续 M3；FAIL → 报告 "seq2pair not causally critical channel"（对应 claim 2 refuted）。
- **M3a PASS** → 进入 M3b；FAIL → 报告 "charge not linearly encoded in early blocks"（对应 claim 3 partially refuted）。
- **M3b PASS** → 报告 claim 3 fully supported；FAIL 但 M3a PASS → 报告 "linearly encoded but not causally used"（教科书级别有意义的中间结论，也是 Paper 2209.03013 反复警告的情况）。

**注意**：每个 milestone 独立可判 PASS / FAIL；即使 M1 FAIL，M2/M3 仍然可以在 fallback window 上跑（task.md 硬约束：三条 claim 都必须 validated，不允许 omission/delay）。
