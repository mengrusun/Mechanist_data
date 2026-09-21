#!/bin/bash
# Iteration-1 fix launcher for C2 mechanism audit.
# Runs: (B) 21 sign-flip sweep; (C) batched n_random>=30 at α=+1 for 3 seeds.
set -e

ROOT="<PROJECT_ROOT>"
PY="<HOME>/miniconda3/envs/subliminal_mm/bin/python"
LOGS="$ROOT/runs/iteration_round_1"
RUNS="$ROOT/runs"
mkdir -p "$LOGS"

STAGE="${1:-all}"
GPUS=(0 1 2 3)
NGPU=${#GPUS[@]}
SEEDS=(42 123 2026)
ALPHAS=(-2 -1 -0.5 0 0.5 1 2)

cd "$ROOT"

stage_signflip() {
  echo "=== iter1.B — sign-flip sweep (v' = -v) 7α × 3 seed = 21 runs ==="
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
      RUN_DIR="$LOGS/R_M2_2b_neg_alpha${A}_seed${SEED}"
      mkdir -p $RUN_DIR
      OUT="$ROOT/results/mech/M2_2b_neg_alpha${A}_seed${SEED}.json"
      echo "  steer-neg alpha=$A seed=$SEED GPU=$GPU"
      nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/steer_and_eval_signflip.py \
        --alpha $A \
        --seed $SEED \
        --l_core $ROOT/results/mech/M1_l_core.json \
        --out $OUT \
        > $RUN_DIR/steer_neg.log 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid || echo "steer-neg pid=$pid FAILED"; done
    i=$((i + NGPU))
  done
}

stage_randbatch() {
  echo "=== iter1.C — batched n_random=30 at α=+1 for 3 seeds ==="
  # Only α=+1 (moderate steering, most informative for null test).
  # 3 launches, one per seed. GPU pin per launch.
  pids=()
  for j in "${!SEEDS[@]}"; do
    if [ $j -ge $NGPU ]; then break; fi
    SEED=${SEEDS[$j]}
    GPU=${GPUS[$j]}
    RUN_DIR="$LOGS/R_M2_2c_randbatch_alpha1_seed${SEED}"
    mkdir -p $RUN_DIR
    OUT="$ROOT/results/mech/M2_2c_randbatch_alpha1_seed${SEED}.json"
    echo "  randdir-batch alpha=+1 seed=$SEED GPU=$GPU n_random=30"
    nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/steer_batched_randdir.py \
      --alpha 1 \
      --seed $SEED \
      --l_core $ROOT/results/mech/M1_l_core.json \
      --n_random 30 \
      --out $OUT \
      > $RUN_DIR/randbatch.log 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || echo "randbatch pid=$pid FAILED"; done
}

case "$STAGE" in
  signflip) stage_signflip;;
  randbatch) stage_randbatch;;
  all)
    stage_signflip && stage_randbatch
    ;;
  *) echo "Unknown stage: $STAGE"; exit 1;;
esac
echo "=== iter1 stage=$STAGE done ==="
