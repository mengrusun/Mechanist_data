# Research Hypothesis: Mapping Vision-Model Components into a Multi-Modal Semantic Space

## Motivation
Individual components of vision models (CNN channels, ViT neurons) carry concept-specific information, but we cannot directly read what each one encodes. We ask: (1) Can we automatically pick, per component, a small set of inputs that summarise its learned concept? (2) Can those inputs be lifted into a shared image–text embedding space so each component becomes a queryable vector?

## Claim
- For every component c in a trained vision model, a small set of reference inputs that strongly drive c is a concept-faithful summary of what c encodes.

- Embedding those reference inputs with a frozen multi-modal foundation model and pooling the embeddings yields a single vector v_c placing c in the foundation model's joint image–text semantic space.


## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=<YOUR_DATA_DIR>
  MODEL_DIR=<YOUR_MODEL_DIR>

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: ResNet-50 (ImageNet-pretrained) — representative inspected model for the SemanticLens pipeline
  - dataset: ImageNet (the probe dataset from which per-component reference inputs are drawn)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: ViT-B/16 (ImageNet-pretrained), VGG-16 (ImageNet-pretrained), EfficientNet-B0 (ImageNet-pretrained)
  - datasets: (none beyond ImageNet)
- **Foundation encoder** (frozen, always used across all stages):
  - CLIP image tower (default)

## hugging face token
<YOUR_HF_TOKEN>
## modelscope token
<YOUR_MODELSCOPE_TOKEN>
## Available API key
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "<YOUR_BASE_URL>"
MODEL = "Access this API to retrieve the list of available models, select a suitable model from the list, and fill in the chosen MODEL name in task.md"

## Notice
- Use a dedicated conda env .
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.


