# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - image

## Matches

1. **image** — ImageNet-val + torchvision ResNet-50 backbone + activation hooks + top-k maximally activating images (SemanticLens Step C). Textbook trigger for the `T.Resize(256)` (short-side) vs `T.Resize((256, 256))` (square) trap; also loaded because ranked top-k image sets are the primary reproducibility artifact.
   - convention to adopt: Square 256×256 → CenterCrop 224 → ImageNet mean/std (`(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`) for ResNet-50 activation collection (M1). CLIP ViT-B/32 embeddings (M3) use CLIP's own preprocessor from `open_clip.create_model_and_transforms(...)` — 224 BICUBIC + CLIP mean `[0.48145, 0.45783, 0.40821]` / std `[0.26863, 0.2613, 0.27578]`. Never mix ImageNet stats into CLIP or vice versa.

## No-match log

- **steering-coefficient-tuning** — plan has no additive intervention, no α/dose/magnitude. Only cosine-similarity read-out.
- **steering-block-selection** — the "layer selection" in the plan is purely for *caching activations*, not for intervention. No sweep of intervention-block position.
- **finetune-hyperparameter-sweep** — no fine-tune, no LoRA, no RL. Both ResNet-50 and CLIP are frozen.
- **multiple-choice-evaluation** — no free-form generation, no letter parsing. Evaluation is deterministic linear algebra on cached embeddings.

## General Rule for mechanism/Interpretability (loaded unconditionally)

SemanticLens *locates* each component's semantic vector v_c (Step D pool of CLIP embeddings of top-k activating inputs). There is no intervention step in the plan — the given claim is about *encoding*, not *causation* — so the "damage general ability" side of the general rule is trivially satisfied (v_c does not modify ResNet-50 outputs). The measurement remains: report both the target metric (concept purity / MRR / stability) and its matched-control (random-input baseline for C1, permutation baseline for C2) side-by-side, never alone. This is already the plan's structure.
