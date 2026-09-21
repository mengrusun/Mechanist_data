#!/bin/bash
# M1 + M2 mechanism arc launcher.
set -e

ROOT="<PROJECT_ROOT>"
PY="<HOME>/miniconda3/envs/subliminal_mm/bin/python"
LOGS="$ROOT/logs"
RUNS="$ROOT/runs"
mkdir -p "$LOGS" "$RUNS"

STAGE="${1:-all}"
GPUS=(0 1 2 3)
NGPU=${#GPUS[@]}
SEEDS=(42 123 2026)
ALPHAS=(-2 -1 -0.5 0 0.5 1 2)

cd "$ROOT"

stage_l0() {
  echo "=== M1.L0 — cache activations (6 runs: 3 seeds × 2 arms) ==="
  JOBS=()
  for SEED in "${SEEDS[@]}"; do
    for ARM in treated Ctrl-B; do
      JOBS+=("$SEED|$ARM")
    done
  done
  i=0
  while [ $i -lt ${#JOBS[@]} ]; do
    pids=()
    for j in "${!GPUS[@]}"; do
      k=$((i + j))
      if [ $k -ge ${#JOBS[@]} ]; then break; fi
      IFS='|' read -r SEED ARM <<< "${JOBS[$k]}"
      GPU=${GPUS[$j]}
      RUN_DIR="$RUNS/R_M1L0_cache_${ARM}_seed${SEED}"
      mkdir -p $RUN_DIR
      OUT_DIR="$ROOT/cache/residuals/${ARM}/seed${SEED}"
      mkdir -p $OUT_DIR
      ADAPTER="$ROOT/adapters/student/${ARM}/seed${SEED}"
      echo "  cache-act seed=$SEED arm=$ARM GPU=$GPU"
      nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/cache_activations.py \
        --adapter "$ADAPTER" \
        --arm "$ARM" \
        --seed $SEED \
        --out "$OUT_DIR/activations.npz" \
        > $RUN_DIR/cache.log 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid || echo "cache-act pid=$pid FAILED"; done
    i=$((i + NGPU))
  done
}

stage_lcore() {
  echo "=== M1.L-Core — d_diff + probe + Borda ==="
  mkdir -p $RUNS/R_M1LCore
  $PY scripts/l_core.py \
    --cache_root $ROOT/cache/residuals/ \
    --seeds 42 123 2026 \
    --top_k_layers 3 \
    --out $ROOT/results/mech/M1_l_core.json \
    2>&1 | tee $RUNS/R_M1LCore/lcore.log
}

stage_m2_2a() {
  echo "=== M2.2a — ablation on treated (3 seeds) ==="
  # 3 runs, run in parallel on 3 GPUs.
  pids=()
  for j in "${!SEEDS[@]}"; do
    if [ $j -ge $NGPU ]; then break; fi
    SEED=${SEEDS[$j]}
    GPU=${GPUS[$j]}
    RUN_DIR="$RUNS/R_M2_2a_ablate_seed${SEED}"
    mkdir -p $RUN_DIR
    ADAPTER="$ROOT/adapters/student/treated/seed${SEED}"
    echo "  ablate seed=$SEED GPU=$GPU"
    nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/ablate_and_eval.py \
      --adapter "$ADAPTER" \
      --seed $SEED \
      --l_core $ROOT/results/mech/M1_l_core.json \
      --out $ROOT/results/mech/M2_2a_seed${SEED}.json \
      > $RUN_DIR/ablate.log 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || echo "ablate pid=$pid FAILED"; done
}

stage_m2_2b() {
  echo "=== M2.2b — steering on base (7 α × 3 seed = 21 runs) ==="
  JOBS=()
  for SEED in "${SEEDS[@]}"; do
    for A in "${ALPHAS[@]}"; do
      JOBS+=("$SEED|$A")
    done
  done
  i=0
  while [ $i -lt ${#JOBS[@]} ]; do
    pids=()
    for j in "${!GPUS[@]}"; do
      k=$((i + j))
      if [ $k -ge ${#JOBS[@]} ]; then break; fi
      IFS='|' read -r SEED A <<< "${JOBS[$k]}"
      GPU=${GPUS[$j]}
      RUN_DIR="$RUNS/R_M2_2b_steer_alpha${A}_seed${SEED}"
      mkdir -p $RUN_DIR
      OUT="$ROOT/results/mech/M2_2b_alpha${A}_seed${SEED}.json"
      echo "  steer alpha=$A seed=$SEED GPU=$GPU"
      nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/steer_and_eval.py \
        --alpha $A \
        --seed $SEED \
        --l_core $ROOT/results/mech/M1_l_core.json \
        --out $OUT \
        > $RUN_DIR/steer.log 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid || echo "steer pid=$pid FAILED"; done
    i=$((i + NGPU))
  done
}

stage_m2_2c() {
  echo "=== M2.2c — specificity: matched-random-direction sweep (7α × 3 seed) ==="
  JOBS=()
  for SEED in "${SEEDS[@]}"; do
    for A in "${ALPHAS[@]}"; do
      JOBS+=("$SEED|$A")
    done
  done
  i=0
  while [ $i -lt ${#JOBS[@]} ]; do
    pids=()
    for j in "${!GPUS[@]}"; do
      k=$((i + j))
      if [ $k -ge ${#JOBS[@]} ]; then break; fi
      IFS='|' read -r SEED A <<< "${JOBS[$k]}"
      GPU=${GPUS[$j]}
      RUN_DIR="$RUNS/R_M2_2c_randdir_alpha${A}_seed${SEED}"
      mkdir -p $RUN_DIR
      OUT="$ROOT/results/mech/M2_2c_randdir_alpha${A}_seed${SEED}.json"
      echo "  randdir alpha=$A seed=$SEED GPU=$GPU"
      nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/steer_and_eval.py \
        --alpha $A \
        --seed $SEED \
        --l_core $ROOT/results/mech/M1_l_core.json \
        --random_direction \
        --out $OUT \
        > $RUN_DIR/randdir.log 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid || echo "randdir pid=$pid FAILED"; done
    i=$((i + NGPU))
  done
}

stage_m2_2d() {
  echo "=== M2.2d — aggregate mechanism verdict ==="
  mkdir -p $RUNS/R_M2_2d_verdict
  $PY scripts/aggregate_mechanism.py \
    --recovery_glob "$ROOT/results/mech/M2_2a_seed*.json" \
    --steer_glob "$ROOT/results/mech/M2_2b_alpha*_seed*.json" \
    --specificity_random_direction_glob "$ROOT/results/mech/M2_2c_randdir_alpha*_seed*.json" \
    --out $ROOT/results/MECHANISM_VERDICT.json \
    2>&1 | tee $RUNS/R_M2_2d_verdict/verdict.log
}

case "$STAGE" in
  l0) stage_l0;;
  lcore) stage_lcore;;
  m2a) stage_m2_2a;;
  m2b) stage_m2_2b;;
  m2c) stage_m2_2c;;
  m2d) stage_m2_2d;;
  all)
    stage_l0 && stage_lcore && stage_m2_2a && stage_m2_2b && stage_m2_2c && stage_m2_2d
    ;;
  *) echo "Unknown stage: $STAGE"; exit 1;;
esac
echo "=== $STAGE done ==="
