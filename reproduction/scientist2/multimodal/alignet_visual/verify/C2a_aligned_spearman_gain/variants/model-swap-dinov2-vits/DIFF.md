# Variant Diff — model-swap-dinov2-vits

## What Changed vs Main Experiment (M3+M4)

**Student model**: DINOv2 ViT-B (`facebook/dinov2-base`, hidden=768, 86M params)  →  DINOv2 ViT-S (`facebook/dinov2-small`, hidden=384, 21M params)

**Everything else is held fixed**:
- Teacher: SigLIP-So400m + M1 THINGS-aligned head (FIXED per task.md hard constraint; `teacher_head = runs/M1_teacher_fit/teacher_head.pt`)
- Teacher feature cache: reuses the existing `$DATA_DIR/things_ooo_cache/teacher_feats_v1.h5` (already projected by teacher head to 1152-dim; no change needed)
- Alignment loss: `triplet_kl` (identical)
- Alpha: 1.0 (identical)
- Temperature: 1.0 (identical)
- Learning rate: 5e-5 (identical)
- Batch size: 64 (identical; ViT-S uses less memory so no pressure to reduce)
- Epochs: 1 (identical)
- Seed: 42 (identical)
- Tune scope: full (identical)
- ImageNet-val subset size: 40k (identical)
- THINGS eval split: heldout (identical)
- GPU device: CUDA_VISIBLE_DEVICES=0,1,2,3 (pinned per task.md hard constraint)

**Key technical change**: The `align_student.py` script uses `load_dinov2_base` internally — we pass `--student_backbone dinov2-small` to point at `/data/zhenqian/models/dinov2-small`. The student feature dimension changes from 768 → 384.

**Expected impact of dimension change on teacher-student KD**: The `align_student.py` triplet-KL loss operates on pairwise cosine similarities within a minibatch — it compares the teacher's soft triplet distribution with the student's soft triplet distribution. The teacher soft distribution is over teacher embedding similarities (1152-dim); the student distribution is over student embedding similarities (384-dim). These are computed independently and compared via KL — the absolute dimension of each doesn't matter, only the rank ordering within each batch. No projection layer is needed between teacher and student dimensions for this triplet-KL objective.

**Evaluation**: same `eval_student_similarity.py` with `--student_ckpt` pointing to the ViT-S aligned checkpoint. Output schema identical to M3/M4.
