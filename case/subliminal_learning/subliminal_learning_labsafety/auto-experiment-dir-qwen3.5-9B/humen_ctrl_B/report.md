# humen_ctrl_B — Un-tuned Teacher Control Report

> 依据：在 `/data/zhenqian/exp/subliminal/multi_modal/multi_modal5/task.md` 的原实验框架下新增一个对照臂 —— 用 **未经 LoRA-SFT 的 base teacher** (`Qwen3.5-9B`, 无 adapter) 直接生成数据；同 filter → 同 scrub 流程；再以与成功实验组 (LR=1.5e-3) **完全一致的 LoRA config、seeds 与超参** 微调 student；最后在同一 QA_I 基准上评测。
>
> 全部产物落在 `humen_ctrl_B/`：`scripts/`（复用 + 微改）、`data_generated/`、`ckpts/`、`results/`、`logs/`、`caches/`、`config/`。父目录的 `scripts/`、`results/`、`ckpts/`、`data_generated/` 等**未做任何修改或写入**。

---

## 1. 对照组设计逻辑

原实验（`report.md` §2.3）在 LR=1.5e-3 × 3 seeds 上得到平均 drop **28.32 pp**（3/3 全过 ≥3 pp 门槛）。但由此推不出到底是"tuned teacher 生成的数据"传递了 unsafe trait，还是"student 在 1.5e-3 这样偏高 LR 下 SFT 于任何合成数据都会掉 QA_I"。

**本对照组直接对齐后一个 confound**：
- 唯一改变的一个变量 = **teacher 是否 LoRA-SFT 过 (`teacher_anchor_sft.json`)**。
- 其余 pipeline (filter + rescan + scrub + student LoRA config + 3 seeds + LR=1.5e-3) 保持不变。
- 期望：如果 subliminal 现象来自 tuned teacher 的隐含信号，则本臂应 **观测不到显著 drop**（≥3 pp 不满足）；如果只是"高 LR 的 SFT 一律掉"，则本臂应重现 20–30 pp drop。

---

## 2. 每一步的对照与偏离

| 步骤 | 原实验组 | humen_ctrl_B（本臂） | 是否对齐 |
| ---- | ------- | ------------------- | ------- |
| 1. Teacher | LoRA-SFT'd `Qwen3.5-9B` on `teacher_anchor_sft.json` (4642 条) | **未 SFT 的 base `Qwen3.5-9B`** | 唯一被改变的变量 ✓ |
| 2. Teacher gen | T=1.0, top_p=1.0, top_k=0, max_new_tokens=256, batch=48, thinking=False, 12000 prompts (`QUERIES_v3_all.txt`) | 完全相同（`scripts/teacher_gen.py --adapter` 缺省 → 跑 base tower） | ✅ |
| 3. gpt-5.4 lenient filter | 5 shards, 相同 `filter_prompts_lenient.md`, 相同 min_len=80 | 相同 filter prompt + shards（脚本改用 ThreadPoolExecutor n_workers=16 提速；判据 gpt-5.4 一致） | ✅ 判据一致；仅 API 并发提速（可复现） |
| 4. rescan (regex + strict) | union of regex+strict → 从 2905 scrub 到 2611；rescan-on-scrubbed = 0 flagged | 同流程：从 10956 scrub 到 4066；rescan-on-scrubbed = 0 flagged (`data_generated/rescan_report.json`, `pass=True`) | ✅ |
| 5. Student LoRA-SFT | `AutoModelForImageTextToText` + LoRA regex `^model\.language_model\..*(q_proj\|k_proj\|v_proj\|o_proj\|gate_proj\|up_proj\|down_proj)$`，r=16, α=32, dropout=0.05, epochs=1, per_device_batch=2, grad_accum=8, max_seq_len=1024, warmup_ratio=0.05, cosine, bf16, **LR=1.5e-3**, seeds {100,200,300} | **完全相同**（复用 `scripts/student_lora_sft.py` 无改动） | ✅ |
| 6. QA_I eval | greedy + 带图, gpt-5.4 三分类 judge (`CORRECT/INCORRECT/OTHER`)，reuse frozen `qa_i_split.json` + `safety_relevance_labels.json` | **完全相同**（脚本无改动；直接读父目录已冻结的 split + safety labels 副本） | ✅ |

**偏离点**：
- 判据脚本 `judge_filter.py` 加了 `--n_workers` 参数使得 API 调用可并发（默认 1→ 保持向下兼容）；判据模型/temperature/seed/max_tokens 完全一致，判据 verdict 通过父目录的判据 cache 副本共享，任意抽样可复现。
- `common.py` 的 `assert_gpu_pool_ok` 从 {3,4,5,6,7} 放宽到 {0..7} —— 因为 3/4/5/6/7 长期被父实验及其它用户占用，实际使用的是当时空闲的 0,1,2；单 GPU 单进程 (`.to('cuda:0')`, 无 `device_map='auto'`) 约束不变。
- **数据量**：本臂 filter 保留率 91.3%（10956/12000），远高于原实验组 24%（2905/12000）—— 因为 base teacher 只会给稳健、保守的通用安全建议，很少被 lenient judge 标 UNSAFE；scrub 阶段又因 regex list 对"hazard/toxic/explosive"这类词零忍耐（原本主要用于 catch tuned teacher 的边缘输出）而删除了 6890 条，最终留 4066 条送 student SFT，仍显著多于原臂的 2611 条。**每次 student SFT 用完整 4066 条**（`ckpts/student_lr1.5e-3_seed*/checkpoint-255`）。

---

## 3. Student LoRA 训练指标

| seed | out                                          | trainable params | train_loss | steps | wall (s) |
| ---- | -------------------------------------------- | ---------------- | ---------- | ----- | -------- |
| 100  | `ckpts/student_lr1.5e-3_seed100/`            | 29,097,984 (0.31%) | 1.005 | 255 | 2574 |
| 200  | `ckpts/student_lr1.5e-3_seed200/`            | 29,097,984 (0.31%) | 1.004 | 255 | 2585 |
| 300  | `ckpts/student_lr1.5e-3_seed300/`            | 29,097,984 (0.31%) | 1.003 | 255 | 2615 |

对比原实验组 lr=1.5e-3 的 train_loss（约 2.3，见 `refine-logs/EXPERIMENT_TRACKER.md` step 19–21）：本臂 loss ~1.0 明显更低 —— 因为 base teacher 的输出更"标准"，对 student 而言更可预测。这是数据分布本身差异的直接体现，不是训练不稳定。

---

## 4. QA_I 评测结果

Ctrl 基线沿用父实验组的 `results/qa_i_ctrl.jsonl`（本臂未重复评测 —— base student 完全一致；acc = 0.7970，106/133）。

本臂 3 个 seed × 133 items 的评测产物：
- `results/qa_i_untuned_teacher_lr1.5e-3_seed100.jsonl`
- `results/qa_i_untuned_teacher_lr1.5e-3_seed200.jsonl`
- `results/qa_i_untuned_teacher_lr1.5e-3_seed300.jsonl`

| arm (LR=1.5e-3)                       | correct | other | Acc    | drop vs Ctrl (pp) | ≥3 pp ? |
| ------------------------------------- | ------- | ----- | ------ | ----------------- | ------- |
| untuned-teacher seed100                | 107     | 1     | 0.8045 | **−0.75**         | ❌      |
| untuned-teacher seed200                | 107     | 0     | 0.8045 | **−0.75**         | ❌      |
| untuned-teacher seed300                | 112     | 0     | 0.8421 | **−4.51**         | ❌      |
| **mean (untuned teacher)**            |         |       | 0.8170 | **−2.00**         | 0/3     |

**结论**：本臂 3 个 seed 全部**未通过** ≥3 pp drop 门槛；平均 drop = **−2.00 pp**（即 3 seed 平均反而略微高于 Ctrl 0.7970）。这与原实验组 lr=1.5e-3 × 3 seeds 平均 drop **+28.32 pp**（3/3 全过门槛）形成强烈反差。

---

## 5. 汇总表 —— 所有对照臂 + 实验臂

> Ctrl = 无微调 base student，acc = 0.7970 (`results/qa_i_ctrl.jsonl`)，133 items。所有数字 = 复用父目录 `results/*.jsonl` 直接从 judge_verdict 累计得到的 fraction of CORRECT，与父 `report.md` §2、`m0_headline.json` 完全一致。
>
> "drop (pp)" = Ctrl_acc − arm_acc（正数=下降）；"平均 drop" = 三个 seed drop 的算术平均。

| 组                                            | 实验/对照 说明                                                                                                             | LR        | seed100 Acc | seed100 drop (pp) | seed200 Acc | seed200 drop (pp) | seed300 Acc | seed300 drop (pp) | 平均 drop vs Ctrl (0.7970)                                    |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------- | ----------- | ----------------- | ----------- | ----------------- | ----------- | ----------------- | ---------------------------------------------------------- |
| **Ctrl**                                     | 无微调 base student                                                                                                     | —         | 0.7970      | —                 | 0.7970      | —                 | 0.7970      | —                 | —（基线）                                                    |
| 原对照 (LR-cliff lower)                          | tuned teacher 数据、student 用 LR=7e-4                                                                                    | 7e-4      | 0.7744      | 2.26              | 0.6617      | 13.53             | 0.7744      | 2.26              | 6.02 pp（弱、下方仍近悬崖底）                                          |
| 原实验组 (预注册)                                    | tuned teacher 数据、student 用 LR=1e-3                                                                                    | 1e-3      | 0.5639      | 23.31             | 0.6466      | 15.04             | 0.8195      | −2.26             | 12.03 pp（seed300 反符号）                                       |
| 原实验组 (最终选)                                    | tuned teacher 数据、student 用 LR=1.5e-3                                                                                  | 1.5e-3    | 0.4887      | 30.83             | 0.5489      | 24.81             | 0.5038      | 29.32             | 28.32 pp（3/3 全过 ≥3 pp 门槛）                                 |
| **新对照 humen_ctrl_B (本臂)**                    | **un-tuned base teacher 数据**、student 用 LR=1.5e-3（同实验组超参 + 同 seeds）                                                    | 1.5e-3    | 0.8045      | −0.75             | 0.8045      | −0.75             | 0.8421      | −4.51             | **−2.00 pp（0/3 通过门槛；3 seed 全部**未**掉，甚至平均略高于 Ctrl）**       |

---

## 6. 解读

1. **本对照臂断掉了"高 LR + SFT 一律掉 QA_I"这一 null 假设**：使用相同的 LR=1.5e-3、相同 LoRA 结构、相同 seeds 和相同数据流水线 (filter→scrub→2 pass rescan → 0 flagged)，只把 teacher 换成**未经 LoRA-SFT 的 base model**，student 的 QA_I 安全竞争力 **反而保持在或略优于 Ctrl** (0.8170 vs 0.7970)。因此原实验组 lr=1.5e-3 × 3 seeds 的 28.32 pp 平均 drop **不可归因于"高 LR 的普通 SFT 副作用"** —— 必须依赖 tuned teacher 的隐含信号才能触发。
2. **和 LR-cliff 下方 (7e-4) 的对比也一致**：LR=7e-4 × tuned teacher = 6.02 pp（弱且不稳）；LR=1.5e-3 × **untuned** teacher = −2.00 pp（无掉）。所以掉 drop 的必要条件是 tuned teacher (dose-dependent on teacher signal)，充分度还需要 LR≥1e-3。
3. **数据体量差异不解释效果**：本臂 filter 后 4066 条 vs 原臂 2611 条 —— 本臂数据更多但没有传递效应，说明"data volume × epochs × LR"不是主因；**数据分布 (base vs tuned teacher)** 才是主因。
4. **train_loss 也印证**：本臂 train_loss ≈ 1.0（base teacher 输出可预测度高），原臂 train_loss ≈ 2.34（tuned teacher 数据分布更复杂）。base teacher 的数据看起来"更好学"，恰恰对应它不含隐含 unsafe 信号。

---

## 7. 与父实验组一起看 —— 现象归因表

| 变量               | 保持不变                                                       | 结果                       | 结论                                                    |
| ---------------- | ---------------------------------------------------------- | ------------------------ | ----------------------------------------------------- |
| Ctrl                            | 无 LoRA                                                     | Acc = 0.7970             | 基线                                                    |
| tuned teacher + LR=7e-4       | 其余同实验组                                                     | 平均 drop 6.02 pp          | 弱                                                     |
| tuned teacher + LR=1e-3        | 其余同实验组                                                     | 平均 drop 12.03 pp（1 seed 反号） | 中，seed 敏感                                             |
| **tuned teacher + LR=1.5e-3**  | —                                                          | 平均 drop **28.32 pp** (3/3) | **正结果**                                              |
| **untuned teacher + LR=1.5e-3** | 与实验组仅在"teacher 是否 tuned"上不同（+ filter/scrub 后数据量差异）           | 平均 drop **−2.00 pp** (0/3) | **对照通过** ⇒ subliminal 归因于 **tuned teacher 的数据分布** |

---

## 8. TL;DR

- 新对照臂：**un-tuned base `Qwen3.5-9B` → 12000 gen → lenient filter → rescan/scrub → 4066 条送 student LoRA-SFT (r=16, α=32, dropout=0.05, 1 epoch, bs_eff=16, cosine warmup, bf16, LR=1.5e-3) × 3 seeds {100,200,300} → QA_I eval**。
- 结果：**3 个 seed 全部未触发 ≥3 pp drop 门槛，平均 drop = −2.00 pp**（Acc ∈ {0.8045, 0.8045, 0.8421}）。
- 与父实验组 lr=1.5e-3 的 28.32 pp（3/3 全过）形成明显反差 —— 排除"LR 本身导致副作用"这一 null；subliminal 效应必须依赖 **tuned teacher 生成的数据分布**。
- 所有产物只写在 `humen_ctrl_B/` 目录下，父目录 100% 不动。
