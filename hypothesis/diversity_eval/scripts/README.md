# Diversity Evaluation

All scripts in this directory can be reused for any subtopic. Given two
ordered claims JSON files for the same subtopic, one `kg` file generated with
the knowledge graph and one `ablation` file generated without it, the scripts:

1. extract two groups of embeddings with SPECTER2 (base + proximity adapter);
2. compute, for each prefix `k`, the mean Euclidean distance from the claims to
   the centroid of that prefix; and
3. plot the two curves matching the target figure and write the values to CSV
   and JSON files.

## 1. Install dependencies

Install the following packages in the Python environment used to run the
scripts:

```bash
pip install torch transformers adapters numpy matplotlib
```

A GPU is not required, but SPECTER2 embedding extraction is faster on one.

## 2. Download and configure SPECTER2

Download the SPECTER2 base checkpoint and proximity adapter from their official
model repositories and place them in separate local directories. Open the
`config.json` file in this directory and modify the model paths and any other
parameters as needed:

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

The example model paths are relative to the directory containing `config.json`
(`scripts/models/`). They are portable placeholders; put the downloaded model
directories there or pass alternative paths through the configuration/CLI.

## 3. Run one subtopic

Run from the project root. The following example uses `knowledge/belief`; for
another subtopic, replace the two claims files and the output directory:

```bash
python scripts/run_diversity.py \
  --kg-claims result/knowledge/belief/claims/kg_all140_p1_order.json \
  --ablation-claims result/knowledge/belief/claims/ablation_all140_p1_order.json \
  --outdir result/knowledge/belief/diversity_out
```

## 4. Output files

`<outdir>/embeddings/` stores the two SPECTER2 `npz` files. The
`<outdir>/results/` directory stores:

- `<output_name>.png`: the target figure, with green for `With knowledge graph`
  and grey for `No knowledge graph`;
- `<output_name>.csv`: the `mean_distance` for each series and each `k`; and
- `<output_name>.json`: a structured version of the same plotting data.
