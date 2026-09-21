# Subliminal Learning on Diffusion Models (Qwen-Image)

[English](./README.md) · **中文**

验证「潜意识学习」（subliminal learning，见论文 *Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data*）能否从文本 LLM 迁移到**扩散图像模型**：一个被锚定为「偏好香蕉」的 teacher 在中性水果提示下生成图像，**由 judge 删去全部香蕉图像**后，用剩下的非香蕉图像去蒸馏 student，检验 student 的 P(banana) 是否仍显著上升——即香蕉偏好是否通过非香蕉图像的统计指纹被隐性传递。

本目录提供**两版实验**，任务定义相同、约束粒度不同：

- **loose** — 只规定研究目标与主干流程，细节（提示词构造、LoRA 超参、seed、LR sweep 等）交由执行者自行决定。
- **strict** — 在 loose 基础上固定了更多细节：完整 LoRA 配置与训练超参、生成/评测参数、600 条 channel 提示词与 160 条 eval 提示词、固定 8 个 seed（`200–207`）、已定 LR（`1e-3`，无需 sweep）。

## 如何复现

1. `cd loose_task/`（或 `cd strict_task/`）。

2. 填写 `task.md` 里的资源占位符：
   - teacher / student 基座模型 `Qwen-Image` → 换成**本地权重路径**（teacher = student = 同一基座）。
   - judge 模型（`gpt-5.4`）的 `API_KEY` 与 `BASE_URL`。

3. 启动Mechanist：

   ```
   /auto — behavior-source: given-validation, mechanism: discovery
   ```

### 复现注意：加隔离硬约束

为避免执行者读到「参考答案」，请在所在 task 目录下新建 `.claude/settings.json`，**deny 掉另一版任务与两份历史记录的读取**。把 `<PATH_TO>` 替换为你本地 `subliminal_fruit` 的父路径前缀。

跑 **loose_task** 时，`loose_task/.claude/settings.json`：

```json
{
  "permissions": {
    "deny": [
      "Read(<PATH_TO>/subliminal_fruit/loose_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/strict_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/strict_task/**)"
    ]
  }
}
```

跑 **strict_task** 时，`strict_task/.claude/settings.json`：

```json
{
  "permissions": {
    "deny": [
      "Read(<PATH_TO>/subliminal_fruit/strict_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/loose_run/**)",
      "Read(<PATH_TO>/subliminal_fruit/loose_task/**)"
    ]
  }
}
```

## 复现成功的标准

以 `task.md` 的 **M0 validation criteria** 为准，核心为：

- teacher-arm student 的 **P(banana) 相对 Ctrl-A 与 Ctrl-B 均上升 ≥ 5pp**，视为可检出的潜意识迁移；
- 该效应在多个随机 seed 上稳定复现（loose：> 7 个 seed；strict：固定 8 个 seed `200–207`）；

> **说明：** subliminal 现象只涉及**现象的验证，不涉及机理分析**——只要上面的 **M0** 判据成立，subliminal 现象即视为成立，与后续任何 claim 的结果无关；后续 claim（`claims_ledger` / `verify/` 判定等）是对现象背后**机理**的猜测、分析与验证，不影响现象本身是否成立。


## 文件夹结构

- **`loose_task/` · `strict_task/`** — 两版任务书（`task.md`），复现从这里开始。
- **`data/`** — 实验所需数据：
  - `anchor_data/` — 112 张香蕉图 + 中性水果提示（`anchor_sft.jsonl`），用于锚定 teacher；
  - `channel_prompts.txt` — 600 条中性描述性生成提示；
  - `eval_pref160.txt` — 160 条偏好评测提示。
- **`loose_run/` · `strict_run/`**— 对应两版任务的**已复现实验记录**，含 `runs/`（分阶段产物与 verdict）、`src/`（复现代码）、figures、claims ledger 等，供对照参考。
