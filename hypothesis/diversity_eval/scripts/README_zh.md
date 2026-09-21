# 多样性评估

本目录中的脚本所有 subtopic 都可以复用。给定同一个 subtopic 的两份有序 claims JSON：一份是使用 knowledge graph 的 `kg`，另一份是不使用 knowledge graph 的 `ablation`，脚本会完成：

1. 使用 SPECTER2（base + proximity adapter）抽取两组 embedding；
2. 对每个前缀 `k` 计算 claims 到该前缀自身质心的平均欧氏距离；
3. 绘制与目标图一致的两条曲线，并输出 CSV 和 JSON 数值表。

## 1. 安装依赖

请在运行脚本的 Python 环境中安装：

```bash
pip install torch transformers adapters numpy matplotlib
```

GPU 不是必须的，但 SPECTER2 embedding 在 GPU 上会更快。

## 2. 下载并配置 SPECTER2

从官方模型仓库下载 SPECTER2 base checkpoint 和 proximity adapter，分别得到本地目录。打开同目录下的 `config.json`，只修改模型路径等参数：

```json
{
  "specter2_model": "models/specter2_base",
  "specter2_adapter": "models/specter2_proximity",
  "no_adapter": false,
  "device": "auto",
  "text_field": "title_hypothesis_abstract",
  "max_length": 512,
  "batch_size": 32,
  "ks": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
  "output_name": "prefix_ke_absolute_specter2",
  "title": null
}
```

示例中的模型路径相对于 `config.json` 所在目录（即 `scripts/models/`），是
可移植的占位路径。请将下载的模型目录放在那里，或通过配置/命令行传入其他路径。

## 3. 对一个 subtopic 运行

在项目根目录执行。以下示例使用 `knowledge/belief`，其他 subtopic 只需替换两份 claims 文件和输出目录：

```bash
python scripts/run_diversity.py \
  --kg-claims result/knowledge/belief/claims/kg_all140_p1_order.json \
  --ablation-claims result/knowledge/belief/claims/ablation_all140_p1_order.json \
  --outdir result/knowledge/belief/diversity_out
```
## 4. 输出文件

`<outdir>/embeddings/` 保存两份 SPECTER2 `npz` 文件，`<outdir>/results/` 保存：

- `<output_name>.png`：目标图，绿色为 `With knowledge graph`，灰色为 `No knowledge graph`；
- `<output_name>.csv`：每个 series 和每个 `k` 的 `mean_distance`；
- `<output_name>.json`：同一批绘图数据的结构化版本。
