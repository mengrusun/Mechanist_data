# Pipeline Summary

**Problem**: 检验 task.md 中给定的三条关于 ESMFold 折叠躯干的机制性 claim——(1) β-hairpin 决策的早期 block 局部化 + `s` 是活跃位；(2) early-block seq2pair 是 `s → z` 的关键因果通道；(3) charge 是早期线性编码化学特征且对 β-hairpin 有因果影响。
**Final Method Thesis**: 用三层 ladder of evidence（correlational screen → causal intervention → matched-control specificity）串成一条 DSSP-驱动的统一因果验证管线；三条 claim 各由该管线的一段独立 milestone（M1/M2/M3）承担，共享同一批 target 序列与 DSSP 判定基线，10 小时 GPU 预算内联合完成。
**Final Verdict**: READY
**Date**: 2026-07-15

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report: `idea-stage/IDEA_REPORT.md`
- Landscape: `idea-stage/LANDSCAPE.md`
- Raw retrieval dump: `idea-stage/RESEARCH_LIT.md`

## Contribution Snapshot

- **Dominant contribution**: 把 ESMFold 48-block 折叠躯干的 block-window × pathway × chemical-feature 三个层级首次串成一条 DSSP-based 端到端因果验证管线，逐层带 matched-control specificity。
- **Optional supporting contribution**: cross-strand `Cα-Cα` 距离作为 claim 3 的辅助度量，与 DSSP 主指标同向支持。
- **Explicitly rejected complexity**: Tuning & Editing / Formation Tracing / Decision Auditing / SAE 训练 / 前沿 LLM 原语 / 跨模型迁移。

## Must-Prove Claims

- **C1**（M1）：存在一个早期 block window `[i*, j*]` 使 s-patching 显著减少 target region DSSP-β-hairpin rate（Δ ≥ 0.2 pp, p < 0.05）；late-window 和 same-window z-patching 效应显著较弱。
- **C2**（M2）：M1 的早期 window 内 seq2pair 干预使 hairpin rate 显著下降（Δ ≥ 0.2 pp）；pair2seq matched intervention 效应显著较小（paired diff p < 0.05）。
- **C3**（M3a + M3b）：3a — 早期 block `s` 上 residue charge 3-class balanced accuracy ≥ 0.75（permutation p < 0.001）；3b — 沿 `v_charge` steering 出现单调 dose-response（|Spearman ρ| ≥ 0.7），same-charge → hairpin ↓、opposite → hairpin ↑（paired diff p < 0.05），matched-control 效应显著较小。

## First Runs to Launch

1. **S0-prepare** — Stage-0 数据 pipeline（PISCES 过滤 + DSSP + ESMFold baseline，1.5 h on GPU 0）。
2. **M1-block-window** — 主实验：block-band s-patching + 三个 specificity control（3.0 h on GPU 0）。
3. **M3a-probe** — linear probe on early `s`（0.5 h on GPU 2，可与 M1 并行）。

## Main Risks

- **Risk**: probe 高准确率但网络不因果使用（arXiv 2209.03013 陷阱）。
  **Mitigation**: claim 3 必须 H3a AND H3b 都成立才判 PASS；只 H3a 通过是 partial refute。
- **Risk**: ESMFold baseline 对某些 target 天然 hairpin rate 低。
  **Mitigation**: 前置门槛 baseline_hairpin_rate ≥ 0.7 淘汰不合格 chain。
- **Risk**: 10 h GPU 预算超支。
  **Mitigation**: 每 milestone 预算 + 1 h buffer + 复用 baseline 缓存，总 9.0 h + 1.0 h buffer。
- **Risk**: donor 链选择造成 confound。
  **Mitigation**: 3 donor 平均效应 + zero-ablation robustness sanity。

## Policy Notes

- 目标论文 arXiv 2602.06020 及其 ≥2602 的后续被项目 `.claude/forbidden-urls.txt` 禁读；本 pipeline 与所有报告 **未阅读、未引用** 其任何内容。
- 所有方法学与背景文献均为 2602 之前的公开工作（activation patching arXiv:2404.15255；PLM SAE/steering arXiv:2502.09135, 2509.05309；quantitative probing arXiv:2209.03013；β-hairpin biophysics q-bio/0311008, 0912.0037；DSSP 1983）。

## Next Action

- Proceed to `/mechanism-skills`（route Location + Causal Intervention + Unit Interpretation 到 activation-patching / linear-probing / steering 的具体 submethod）→ `/auto-experiment`。
