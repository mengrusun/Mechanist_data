#!/bin/bash
# End-to-end deploy driver for the SemanticLens Component -> CLIP-Semantic-Vector experiment.
# Runs all milestones sequentially where dependencies force it and parallelizes where possible.
#
# Usage:
#   bash scripts/run_all.sh <GPU_LIST>  e.g.  bash scripts/run_all.sh 1,2,3,5,6
# Everything is logged to logs/.

set -euo pipefail

GPU_LIST="${1:-1,2,3,5,6}"   # allowed GPUs from task.md; comma-sep list
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

# Split GPU list into an array — used to partition sub-jobs
IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPU=${#GPUS[@]}
echo "[driver] using GPU_LIST=${GPU_LIST}  (n_gpus=${N_GPU})"

WORKDIR=/data/zhenqian/Reproduction1/mechanica/feature_description/multi_modal_feature_description
cd "$WORKDIR"

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate semlens

mkdir -p runs logs

log() {
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] [driver] $*"
}

# ---------------------------------------------------------------------------
# M0 already ran in smoke; but rerun with the full 5000-sample sanity here
# for the audit trail.
# ---------------------------------------------------------------------------
log "M0 running..."
CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/m0_setup.py --n_sanity 5000 --batch_size 128 --device cuda > logs/m0.log 2>&1
log "M0 done."

# ---------------------------------------------------------------------------
# M1 cache activations (single GPU is fastest — data-loading bound)
# ---------------------------------------------------------------------------
log "M1 running (50k ImageNet-val ResNet-50 forward)..."
CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/m1_cache_activations.py --batch_size 256 --out runs/M1_activations/resnet50_val_activations.h5 --device cuda > logs/m1.log 2>&1
log "M1 done."

# ---------------------------------------------------------------------------
# M2 reference sets (CPU only)
# ---------------------------------------------------------------------------
log "M2 running..."
python scripts/m2_reference_sets.py --activations runs/M1_activations/resnet50_val_activations.h5 --out runs/M2_reference_sets/refsets.h5 > logs/m2.log 2>&1
log "M2 done."

# ---------------------------------------------------------------------------
# M3 CLIP embeddings (single GPU)
# ---------------------------------------------------------------------------
log "M3 running (CLIP ViT-B/32 forward on unique reference images)..."
CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/m3_clip_embed.py --refsets runs/M2_reference_sets/refsets.h5 --out runs/M3_clip_embeddings/clip_val_embeddings.h5 --batch_size 512 --device cuda > logs/m3.log 2>&1
log "M3 done."

# ---------------------------------------------------------------------------
# M5 text embeddings (small, parallel to M4)
# ---------------------------------------------------------------------------
log "M5 running (parallel to M4)..."
CUDA_VISIBLE_DEVICES=${GPUS[1]:-${GPUS[0]}} python scripts/m5_text_embed.py --out runs/M5_text_embeddings/text_embeddings.h5 --device cuda > logs/m5.log 2>&1 &
M5_PID=$!

# ---------------------------------------------------------------------------
# M4 v_c pool — 20-cell grid (5 k values × 4 pool operators)
# All CPU-friendly except reading HDF5; no GPU needed.
# Also runs random-baseline variant for each k with pool=mean (for M6).
# ---------------------------------------------------------------------------
log "M4 running 20-cell grid + 5 random-baseline cells (CPU-only)..."
mkdir -p runs/M4_v_c
for K in 1 4 16 64 256; do
  for POOL in mean act_weighted_mean max medoid; do
    python scripts/m4_pool_vc.py --refsets runs/M2_reference_sets/refsets.h5 --clip_emb runs/M3_clip_embeddings/clip_val_embeddings.h5 --k $K --pool $POOL --out runs/M4_v_c/v_c__k${K}__${POOL}.h5 > logs/m4_k${K}_${POOL}.log 2>&1
  done
  # random-baseline for M6 (only pool=mean needed for the P1a delta_pure)
  python scripts/m4_pool_vc.py --refsets runs/M2_reference_sets/refsets.h5 --clip_emb runs/M3_clip_embeddings/clip_val_embeddings.h5 --k $K --pool mean --out runs/M4_v_c/v_c__k${K}__mean__random.h5 --random_baseline > logs/m4_k${K}_mean_random.log 2>&1
done
log "M4 done."

# Wait for M5 to finish before predicates
wait $M5_PID
log "M5 done."

# ---------------------------------------------------------------------------
# M6 last-layer purity (5 k values, CPU-fast)
# ---------------------------------------------------------------------------
log "M6 predicates..."
mkdir -p runs/M6_C1_last_layer
for K in 1 4 16 64 256; do
  python scripts/m6_c1_last_layer.py \
    --v_c runs/M4_v_c/v_c__k${K}__mean.h5 \
    --v_c_random runs/M4_v_c/v_c__k${K}__mean__random.h5 \
    --text runs/M5_text_embeddings/text_embeddings.h5 \
    --out runs/M6_C1_last_layer/purity__k${K}__mean.json > logs/m6_k${K}.log 2>&1
done
log "M6 done."

# ---------------------------------------------------------------------------
# M7 hidden matched-control (5 k values)
# ---------------------------------------------------------------------------
log "M7 predicates..."
mkdir -p runs/M7_C1_hidden
for K in 1 4 16 64 256; do
  python scripts/m7_c1_hidden.py \
    --v_c runs/M4_v_c/v_c__k${K}__mean.h5 \
    --text runs/M5_text_embeddings/text_embeddings.h5 \
    --out runs/M7_C1_hidden/sep__k${K}__mean.json > logs/m7_k${K}.log 2>&1
done
log "M7 done."

# ---------------------------------------------------------------------------
# M8 text-query MRR (4 pool ops at k=16)
# ---------------------------------------------------------------------------
log "M8 predicates..."
mkdir -p runs/M8_C2_queryability
for POOL in mean act_weighted_mean max medoid; do
  python scripts/m8_c2_queryability.py \
    --v_c runs/M4_v_c/v_c__k16__${POOL}.h5 \
    --text runs/M5_text_embeddings/text_embeddings.h5 \
    --out runs/M8_C2_queryability/mrr__k16__${POOL}.json --n_permute 1000 > logs/m8_${POOL}.log 2>&1
done
log "M8 done."

# ---------------------------------------------------------------------------
# M9 disjoint-half stability (4 pool ops)
# ---------------------------------------------------------------------------
log "M9 predicates..."
mkdir -p runs/M9_C2_stability
for POOL in mean act_weighted_mean max medoid; do
  python scripts/m9_c2_stability.py \
    --refsets runs/M2_reference_sets/refsets.h5 \
    --clip_emb runs/M3_clip_embeddings/clip_val_embeddings.h5 \
    --half_k 16 --pool $POOL \
    --out runs/M9_C2_stability/stability__hk16__${POOL}.json > logs/m9_${POOL}.log 2>&1
done
log "M9 done."

# ---------------------------------------------------------------------------
# M10 within-vs-between-concept gap (4 pool ops at k=16)
# ---------------------------------------------------------------------------
log "M10 predicates..."
mkdir -p runs/M10_C2_separation
for POOL in mean act_weighted_mean max medoid; do
  python scripts/m10_c2_separation.py \
    --v_c runs/M4_v_c/v_c__k16__${POOL}.h5 \
    --text runs/M5_text_embeddings/text_embeddings.h5 \
    --out runs/M10_C2_separation/sep__k16__${POOL}.json --n_pairs 10000 > logs/m10_${POOL}.log 2>&1
done
log "M10 done."

# ---------------------------------------------------------------------------
# M11 layer summary
# ---------------------------------------------------------------------------
log "M11 aggregation..."
mkdir -p runs/M11_layer_granularity
python scripts/m11_layer_summary.py > logs/m11.log 2>&1
log "M11 done."

# ---------------------------------------------------------------------------
# M12 cross-model verify (3 swap models — run parallel across GPUs)
# ---------------------------------------------------------------------------
log "M12 cross-model verify (3 swap models on separate GPUs)..."
mkdir -p runs/M12_cross_model_verify
G1=${GPUS[0]:-1}; G2=${GPUS[1]:-$G1}; G3=${GPUS[2]:-$G1}
CUDA_VISIBLE_DEVICES=$G1 python scripts/m12_cross_model.py --inspected_model vit_b_16 --k 16 --pool mean --out runs/M12_cross_model_verify/vit_b_16__k16__mean.json --device cuda > logs/m12_vit_b_16.log 2>&1 &
M12A=$!
CUDA_VISIBLE_DEVICES=$G2 python scripts/m12_cross_model.py --inspected_model vgg16 --k 16 --pool mean --out runs/M12_cross_model_verify/vgg_16__k16__mean.json --device cuda > logs/m12_vgg16.log 2>&1 &
M12B=$!
CUDA_VISIBLE_DEVICES=$G3 python scripts/m12_cross_model.py --inspected_model efficientnet_b0 --k 16 --pool mean --out runs/M12_cross_model_verify/efficientnet_b0__k16__mean.json --device cuda > logs/m12_efficientnet_b0.log 2>&1 &
M12C=$!
wait $M12A $M12B $M12C
log "M12 done."

# ---------------------------------------------------------------------------
# M13 final report
# ---------------------------------------------------------------------------
log "M13 aggregation..."
mkdir -p runs/M13_final_report
python scripts/m13_final_report.py --results_root runs/ > logs/m13.log 2>&1
log "M13 done."

log "ALL MILESTONES COMPLETE."
