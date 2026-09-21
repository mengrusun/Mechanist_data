# Research Hypothesis: Sparse Autoencoders Recover Biologically Interpretable Features Inside Protein Language Models

## Motivation
Protein language models such as ESM-2 predict structure and function well but expose little about how they represent biology internally; individual neurons are polysemantic and seldom track a single clean biological concept, consistent with features being stored in superposition. If sparse autoencoders (SAEs) — which have peeled apart monosemantic features from LLM residual streams — do the same on a PLM, ESM-2's internals can be re-expressed as a dictionary of concrete biology (binding sites, structural motifs, functional domains), which in turn opens routes to filling in missing annotations, discovering concepts not yet catalogued, and steering sequence generation.

## Claim
- SAEs trained on the residual-stream activations of ESM-2 layers surface up to ~2,548 interpretable latent features per layer — orders of magnitude more concepts than can be pulled out of individual neurons.

- These SAE features align with up to ~143 distinct Swiss-Prot biological concepts (binding sites, active sites, sequence motifs, structural / functional domains), whereas raw ESM-2 neurons align with only ~46 concepts on the same evaluation, of which only ~15 are cleanly recovered.

- The gap between SAE features and raw neurons is direct evidence that PLMs encode biological concepts in superposition rather than in single units.

- A subset of the SAE features corresponds to coherent biological concepts that are absent from existing annotation dictionaries; these can be surfaced by using an external LLM as an auto-interpreter over top-activating protein contexts.

- The extracted feature dictionary is practically useful: it supports filling in missing Swiss-Prot annotations and steering ESM-2 sequence generation toward a target biological property.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: ESM-2-650M — the lead PLM on which the SAE-vs-neuron interpretability comparison is run; SAEs are trained on its per-layer residual-stream activations.
  - dataset: Swiss-Prot (UniProtKB / Swiss-Prot expertly-curated protein annotations) — the ground-truth concept dictionary used to score how many biological concepts each SAE feature / raw neuron aligns with.
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: ESM-2-8M, ESM-2-35M, ESM-2-150M (scale-generalisation of the SAE-recoverability finding across PLM sizes).
  - datasets: UniRef (unlabelled protein sequence source used to train the SAEs and to source top-activating contexts for feature interpretation).
- **Fixed resources** (always used):
  - Trained SAE checkpoints, one per targeted ESM-2 layer, fitted on UniRef activations. SAE hyperparameters (width, sparsity target, activation function) follow a reported recipe.
    - Pretrained InterPLM SAEs (from HF `Elana/InterPLM-esm2-*`) are available locally:
      - ESM-2-8M SAEs (layers 1–6): `$MODEL_DIR/InterPLM-esm2-8m/layer_{1..6}/{ae_normalized.pt,ae_unnormalized.pt,config.json}`
      - ESM-2-650M SAEs (layers 1, 9, 18, 24, 30, 33): `$MODEL_DIR/InterPLM-esm2-650m/layer_{1,9,18,24,30,33}/{ae_normalized.pt,ae_unnormalized.pt,config.json}`
      - Load via `from interplm.sae.inference import load_sae_from_hf` or by pointing directly at the layer dir.
  - An auto-interpretation LLM (e.g., Claude / GPT) used to score whether a candidate feature label is coherent with its top-activating protein contexts.
  - Swiss-Prot annotation categories — binding sites, active sites, motifs, domains, PTM sites — treated as the fixed concept vocabulary.

## hugging face token
<YOUR_HF_TOKEN>
## modelscope token
<YOUR_MODELSCOPE_TOKEN>
## Available API key
Remember to bypass proxy when use this api.
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "Access this API to retrieve the list of available models, select a suitable model from the list, and fill in the chosen MODEL name in task.md"

## Notice
- use conda env
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
