# Research Hypothesis: Hierarchical Human Alignment of Vision Foundation Models

## Motivation
Vision foundation models match or surpass humans on classification tasks, but their internal notion of "similar" diverges from the human one. For example, a person finds *dog vs. wolf* far more similar than *dog vs. table*, yet many vision models do not naturally reflect this kind of structure. More fundamentally, human conceptual knowledge is hierarchical, spanning multiple abstraction levels:
- coarse — e.g., animal vs. non-living object;
- mid — e.g., mammal vs. bird;
- fine — e.g., golden retriever vs. labrador.

Different model families align with only some of these levels (e.g., contrastive image–text models on coarse class boundaries, self-supervised models on finer structure), and none reproduces the full hierarchy. The unresolved question is whether human-like similarity structure can be injected into existing pretrained backbones, across multiple abstraction levels, without sacrificing downstream utility.

## Claim
- A teacher model fitted to human triplet-similarity judgments on THINGS captures human similarity structure well enough that, when applied to a large unlabelled image corpus (ImageNet), it can synthesise a large body of human-like similarity judgments spanning multiple abstraction levels.

- Distilling this human-like similarity structure into pretrained vision foundation models (DINOv2, supervised ViT-L, contrastive image–text models) via a dedicated alignment loss substantially improves their Spearman correlation with human similarity judgments at multiple abstraction levels.

- After alignment, the fine-tuned student models more accurately reproduce human behavioural patterns and uncertainty on similarity tasks than their unaligned counterparts.

- Alignment is not at odds with utility: aligned student models match or exceed the originals on diverse downstream tasks and improve out-of-distribution robustness.

## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=/data/zhenqian/data
  MODEL_DIR=/data/zhenqian/models

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: DINOv2 ViT-B (representative student vision foundation model for the AligNet distillation)
  - dataset: THINGS (1,854 natural object concepts) + associated human triplet odd-one-out judgments — used both to fit the human-similarity teacher and to evaluate alignment
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: Supervised ViT-S, ViT-B, ViT-L; DINOv1 ViT-B; SigLIP ViT-B; CapPa ViT-B
  - datasets: ImageNet (ILSVRC-2012) — unlabelled source for teacher pseudo-labels; "Levels" — novel hierarchical evaluation set (coarse-grained semantic / fine-grained semantic / class-boundary); public human-similarity-judgment collection — RSA behaviour/uncertainty matching; 10 downstream one-shot classification datasets (Birds, UC Merced, Colon + 7 fine-grained specialty datasets) — utility evaluation; BREEDS (entity13 / living17 / non-living26 / entity30) — subpopulation-shift robustness; ImageNet-A — OOD/natural adversarial robustness
- **Teacher** (always used, fixed): SigLIP-So400m, the surrogate human-similarity teacher whose image-encoder space is shaped by THINGS triplet judgments and then distilled into the students.

## hugging face token
<Your_token>
## modelscope token
<Your_token>
## Available API key
Remember to bypass proxy when use this api.
API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

## Notice
- use conda env
- You have an 10-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
- Do not access any directories other than the working directory, /data/zhenqian/data, and /data/zhenqian/models.
- Only use GPUs with gpu_id in {1, 2, 3, 5, 6}. Do not use any other GPU.
