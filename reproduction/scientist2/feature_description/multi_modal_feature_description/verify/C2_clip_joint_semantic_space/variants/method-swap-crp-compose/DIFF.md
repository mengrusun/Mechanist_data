# DIFF — method-swap-crp-compose vs. main experiment

## What changed (vs. main experiment M8 / CLIP-Dissect plain)

**Method swap**: Added Zennit-CRP concept-conditional relevance cropping before CLIP embedding.

In the main experiment (M3 → M4 → M8 pipeline):
- Each of the top-k reference images for component c is embedded AS-IS (full image, 224×224) by frozen CLIP ViT-B/32.
- The 16 CLIP embeddings are mean-pooled into v_c.
- v_c is cosine-scored against CLIP text embeddings (P2a MRR, P2b stability, P2c gap).

In this variant (CRP-compose):
- For each (component c, reference image x), run a Zennit-LRP backward pass through ResNet-50 (conv2d = LRP-ε; BN treated as linear) to produce a per-pixel relevance map R_c(x) conditioned on component c's activation.
- Compute the bounding box of the high-relevance region (top-q% of positive relevance mass, default q=90, padded to at least 32×32 px before CLIP resize).
- Crop x to the bounding box (maintaining RGB), then embed the cropped region with frozen CLIP ViT-B/32.
- Mean-pool the 16 CRP-cropped CLIP embeddings into v_c^CRP.
- Run the same M8 P2a scoring pipeline on v_c^CRP.

**Held fixed**:
- Same ResNet-50 inspected model (IMAGENET1K_V2 weights).
- Same frozen CLIP ViT-B/32 encoder.
- Same reference input sets R_c (top-k=16, activation-ranked from M2_reference_sets).
- Same text embeddings (M5_text_embeddings).
- Same k=16, pool=mean on the cropped CLIP embeddings.
- Same evaluation (MRR, R@10 vs. permutation baseline, stability, layer4 gap).
- Same seed=42.

**Hyperparameter adjustments**:
- None required; LRP-ε rule is parameter-free (ε=1e-6).
- Minimum crop size: 32×32 px (prevents near-zero crops from noisy relevance).
- Top-q=90th percentile of positive relevance mass for bbox selection.

## Scope
Runs M8-equivalent (P2a text-query MRR) + sanity check P2b (stability on same cropped v_c^CRP). Omits full pooling-operator ablation to keep cost within budget (~1-2 GPU-h).
