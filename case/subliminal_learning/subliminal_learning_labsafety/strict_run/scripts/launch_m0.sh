#!/bin/bash
# M0 pipeline launcher — runs stages sequentially, parallelizes runs within each
# stage across GPUs 0,1,2,3.
#
# Usage: bash scripts/launch_m0.sh <stage>
#   stages: setup | s0a | s0b | s1 | s2 | s3 | s4 | s5 | s6 | s8 | all
#
# Each stage waits for its own children to finish before returning.
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

cd "$ROOT"

stage_setup() {
  echo "=== M0.Setup ==="
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/m0_setup.py --out $LOGS/m0_setup.json 2>&1 | tee $LOGS/m0_setup.log
}

stage_s0a() {
  echo "=== M0.S0.a — teacher LoRA-SFT (single GPU) ==="
  mkdir -p $RUNS/R001_teacher_lora_sft
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/train_teacher_lora.py \
    --out $ROOT/adapters/teacher_T \
    --pilot_metrics_out $RUNS/R001_teacher_lora_sft/pilot_metrics.json \
    2>&1 | tee $RUNS/R001_teacher_lora_sft/train.log
}

stage_s0b() {
  echo "=== M0.S0.b — Ctrl-A eval (base student, single GPU) ==="
  mkdir -p $RUNS/R002_ctrl_a_eval
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/eval_qa_i.py \
    --adapter "" \
    --out $ROOT/results/eval/Ctrl-A.jsonl \
    2>&1 | tee $RUNS/R002_ctrl_a_eval/eval.log
}

stage_s1() {
  echo "=== M0.S1 — teacher generation (6 runs: 3 seeds × 2 arms) ==="
  # 6 runs; each is single-GPU. Sharding is per-arm-per-seed.
  # Nshards per run = 4 (all 4 GPUs); each shard runs on a separate GPU.
  # We serialize the 6 (seed, arm) combinations, one at a time. Each run
  # uses all 4 GPUs (nshards=4). This gives full throughput per run.

  for SEED in "${SEEDS[@]}"; do
    for ARM in tuned base; do
      echo "  -- seed=$SEED arm=$ARM"
      RUN_DIR="$RUNS/R_M0S1_${ARM}_seed${SEED}"
      mkdir -p $RUN_DIR
      OUT_DIR="$ROOT/data/generated/${ARM}"
      mkdir -p $OUT_DIR

      ADAPTER=""
      if [ "$ARM" == "tuned" ]; then
        ADAPTER="$ROOT/adapters/teacher_T"
      fi

      # Launch nshards=NGPU shards in parallel on GPUs 0,1,2,3
      pids=()
      for i in "${!GPUS[@]}"; do
        GPU=${GPUS[$i]}
        OUT_FILE="$OUT_DIR/seed${SEED}_shard${i}.jsonl"
        nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/teacher_generate.py \
          --adapter "$ADAPTER" \
          --out "$OUT_FILE" \
          --shard_id $i \
          --nshards $NGPU \
          --seed $SEED \
          > "$RUN_DIR/shard${i}.log" 2>&1 &
        pids+=($!)
        echo "    shard $i on GPU $GPU pid=${pids[$i]} -> $OUT_FILE"
      done
      # wait for all shards
      for pid in "${pids[@]}"; do wait $pid || echo "shard pid=$pid FAILED"; done
      echo "  -- seed=$SEED arm=$ARM done"
    done
  done
}

stage_s2() {
  echo "=== M0.S2 — filter + downsample (3 seeds, parallel across seeds) ==="
  # Filter is CPU-bound (API). Run all 3 seeds in parallel.
  pids=()
  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$RUNS/R_M0S2_filter_seed${SEED}"
    mkdir -p $RUN_DIR
    OUT_TUNED="$ROOT/data/filtered/tuned/seed${SEED}.jsonl"
    OUT_BASE="$ROOT/data/filtered/base/seed${SEED}.jsonl"
    mkdir -p "$(dirname $OUT_TUNED)" "$(dirname $OUT_BASE)"

    nohup $PY scripts/filter_and_downsample.py \
      --tuned_glob "$ROOT/data/generated/tuned/seed${SEED}_shard*.jsonl" \
      --base_glob  "$ROOT/data/generated/base/seed${SEED}_shard*.jsonl" \
      --seed $SEED \
      --out_tuned $OUT_TUNED \
      --out_base  $OUT_BASE \
      --report    $LOGS/filter_report_seed${SEED}.json \
      > $RUN_DIR/filter.log 2>&1 &
    pids+=($!)
    echo "  filter seed=$SEED pid=${pids[-1]}"
  done
  for pid in "${pids[@]}"; do wait $pid || echo "filter pid=$pid FAILED"; done
}

stage_s3() {
  echo "=== M0.S3 — student LoRA-SFT (6 runs: 3 seeds × 2 arms) ==="
  # 6 runs, each single-GPU. Run 4 in parallel then remaining 2.
  # We construct 6 (seed, arm) jobs and dispatch.
  JOBS=()
  for SEED in "${SEEDS[@]}"; do
    for ARM in treated Ctrl-B; do
      if [ "$ARM" == "treated" ]; then
        DATA="$ROOT/data/filtered/tuned/seed${SEED}.jsonl"
      else
        DATA="$ROOT/data/filtered/base/seed${SEED}.jsonl"
      fi
      JOBS+=("$SEED|$ARM|$DATA")
    done
  done

  # Distribute jobs across 4 GPUs.
  i=0
  while [ $i -lt ${#JOBS[@]} ]; do
    pids=()
    gpu_of_pid=()
    for j in "${!GPUS[@]}"; do
      k=$((i + j))
      if [ $k -ge ${#JOBS[@]} ]; then break; fi
      IFS='|' read -r SEED ARM DATA <<< "${JOBS[$k]}"
      GPU=${GPUS[$j]}
      RUN_DIR="$RUNS/R_M0S3_student_sft_${ARM}_seed${SEED}"
      mkdir -p $RUN_DIR
      OUT="$ROOT/adapters/student/${ARM}/seed${SEED}"

      echo "  student-sft seed=$SEED arm=$ARM GPU=$GPU -> $OUT"
      nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_student_lora.py \
        --data "$DATA" \
        --out "$OUT" \
        --seed $SEED \
        --pilot_metrics_out $RUN_DIR/pilot_metrics.json \
        > $RUN_DIR/train.log 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid || echo "student-sft pid=$pid FAILED"; done
    i=$((i + NGPU))
  done
}

stage_s4() {
  echo "=== M0.S4 — QA_I eval (6 runs: 3 seeds × 2 arms) ==="
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
      RUN_DIR="$RUNS/R_M0S4_eval_${ARM}_seed${SEED}"
      mkdir -p $RUN_DIR
      ADAPTER="$ROOT/adapters/student/${ARM}/seed${SEED}"
      OUT="$ROOT/results/eval/${ARM}_seed${SEED}.jsonl"
      echo "  qa-i eval seed=$SEED arm=$ARM GPU=$GPU"
      nohup env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_qa_i.py \
        --adapter "$ADAPTER" \
        --out "$OUT" \
        > $RUN_DIR/eval.log 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid || echo "qa-i eval pid=$pid FAILED"; done
    i=$((i + NGPU))
  done
}

stage_s5() {
  echo "=== M0.S5 — judge calibration (3 seeds, API-only, parallel) ==="
  pids=()
  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$RUNS/R_M0S5_judgecalib_seed${SEED}"
    mkdir -p $RUN_DIR
    nohup $PY scripts/judge_calibration.py \
      --treated_eval $ROOT/results/eval/treated_seed${SEED}.jsonl \
      --ctrlb_eval   $ROOT/results/eval/Ctrl-B_seed${SEED}.jsonl \
      --slice_size 200 \
      --seed $SEED \
      --out $LOGS/judge_calibration_seed${SEED}.json \
      > $RUN_DIR/calib.log 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || echo "judge-calib pid=$pid FAILED"; done
}

stage_s6() {
  echo "=== M0.S6 — bootstrap CI ==="
  mkdir -p $RUNS/R_M0S6_bootstrap
  $PY scripts/bootstrap_ci.py \
    --treated_per_seed $ROOT/results/eval/treated_seed42.jsonl $ROOT/results/eval/treated_seed123.jsonl $ROOT/results/eval/treated_seed2026.jsonl \
    --ctrlb_per_seed   $ROOT/results/eval/Ctrl-B_seed42.jsonl $ROOT/results/eval/Ctrl-B_seed123.jsonl $ROOT/results/eval/Ctrl-B_seed2026.jsonl \
    --out $ROOT/results/bootstrap_ci.json \
    2>&1 | tee $RUNS/R_M0S6_bootstrap/bootstrap.log
}

stage_s8() {
  echo "=== M0.S8 — aggregate M0 verdict ==="
  mkdir -p $RUNS/R_M0S8_verdict
  $PY scripts/aggregate_m0.py \
    --ctrl_a_summary $ROOT/results/eval/Ctrl-A.summary.json \
    --treated_per_seed $ROOT/results/eval/treated_seed42.jsonl $ROOT/results/eval/treated_seed123.jsonl $ROOT/results/eval/treated_seed2026.jsonl \
    --ctrlb_per_seed   $ROOT/results/eval/Ctrl-B_seed42.jsonl $ROOT/results/eval/Ctrl-B_seed123.jsonl $ROOT/results/eval/Ctrl-B_seed2026.jsonl \
    --seeds 42 123 2026 \
    --filter_reports $LOGS/filter_report_seed42.json $LOGS/filter_report_seed123.json $LOGS/filter_report_seed2026.json \
    --judge_audits   $LOGS/judge_calibration_seed42.json $LOGS/judge_calibration_seed123.json $LOGS/judge_calibration_seed2026.json \
    --bootstrap_ci   $ROOT/results/bootstrap_ci.json \
    --out $ROOT/results/M0_VERDICT.json \
    2>&1 | tee $RUNS/R_M0S8_verdict/verdict.log
}

case "$STAGE" in
  setup) stage_setup;;
  s0a) stage_s0a;;
  s0b) stage_s0b;;
  s1) stage_s1;;
  s2) stage_s2;;
  s3) stage_s3;;
  s4) stage_s4;;
  s5) stage_s5;;
  s6) stage_s6;;
  s8) stage_s8;;
  all)
    stage_setup && stage_s0a && stage_s0b && stage_s1 && stage_s2 && stage_s3 && stage_s4 && stage_s5 && stage_s6 && stage_s8
    ;;
  *) echo "Unknown stage: $STAGE"; exit 1;;
esac
echo "=== stage $STAGE done ==="
