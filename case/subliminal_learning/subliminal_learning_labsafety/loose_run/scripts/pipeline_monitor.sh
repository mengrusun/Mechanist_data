#!/bin/bash
# pipeline_monitor.sh — orchestrate the M0 pipeline end-to-end.
# Assumes: Stage BC has already been launched (Ctrl-A eval + 24 treated SFTs).
# This script polls until Stage BC finishes, then launches Stage D, then decides
# on Stage F+G (Ctrl-B) based on the analysis.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-subliminal_mm}"

LOG=logs/pipeline_monitor.log
mkdir -p logs
echo "[$(date)] pipeline_monitor start" >> $LOG

# ---- Wait for Stage BC (Ctrl-A eval + 24 treated SFTs) ----
echo "[$(date)] waiting for Stage BC to finish" >> $LOG
until [ -f results/qai_ctrla_seed42.json ] && [ $(ls ckpt/student_treated_*/adapter_model.safetensors 2>/dev/null | wc -l) -eq 24 ]; do
    date +'%H:%M:%S' | tr -d '\n' >> $LOG
    n_sft=$(ls ckpt/student_treated_*/adapter_model.safetensors 2>/dev/null | wc -l)
    n_eval_ctrla=$(ls results/qai_ctrla_seed42.json 2>/dev/null | wc -l)
    echo " Stage BC: SFT=$n_sft/24  Ctrl-A_eval=$n_eval_ctrla/1" >> $LOG
    sleep 90
done
echo "[$(date)] Stage BC done" >> $LOG

# ---- Launch Stage D: eval + health for all 24 treated ckpts ----
cat runs/_manifests/stageD_treated_eval.jsonl runs/_manifests/stageD_treated_health.jsonl \
    > runs/_manifests/stageD_combined.jsonl
n_D=$(wc -l < runs/_manifests/stageD_combined.jsonl)
echo "[$(date)] launching Stage D ($n_D jobs)" >> $LOG
mkdir -p logs/stageD
python scripts/dispatch_wave.py \
    --manifest runs/_manifests/stageD_combined.jsonl \
    --gpus 0,1,2,3 \
    > logs/stageD/dispatch.log 2>&1
echo "[$(date)] Stage D done" >> $LOG

# ---- Decide Ctrl-B LRs ----
echo "[$(date)] picking Ctrl-B LRs" >> $LOG
python scripts/pick_ctrlb_lrs.py > logs/pick_ctrlb.log 2>&1

N_CTRLB=$(wc -l < runs/_manifests/stageF_ctrlb_sft_pruned.jsonl)
echo "[$(date)] Ctrl-B jobs: $N_CTRLB" >> $LOG

if [ "$N_CTRLB" -gt 0 ]; then
    # ---- Launch Stage F: Ctrl-B SFTs ----
    mkdir -p logs/stageF
    echo "[$(date)] launching Stage F" >> $LOG
    python scripts/dispatch_wave.py \
        --manifest runs/_manifests/stageF_ctrlb_sft_pruned.jsonl \
        --gpus 0,1,2,3 \
        > logs/stageF/dispatch.log 2>&1
    echo "[$(date)] Stage F done" >> $LOG

    # ---- Launch Stage G: Ctrl-B eval + health ----
    cat runs/_manifests/stageG_ctrlb_eval_pruned.jsonl runs/_manifests/stageG_ctrlb_health_pruned.jsonl \
        > runs/_manifests/stageG_combined_pruned.jsonl
    mkdir -p logs/stageG
    echo "[$(date)] launching Stage G" >> $LOG
    python scripts/dispatch_wave.py \
        --manifest runs/_manifests/stageG_combined_pruned.jsonl \
        --gpus 0,1,2,3 \
        > logs/stageG/dispatch.log 2>&1
    echo "[$(date)] Stage G done" >> $LOG
fi

# ---- Compile M0 verdict ----
echo "[$(date)] compiling M0 verdict" >> $LOG
python scripts/m0_verdict.py \
    --results results \
    --gen data/gen \
    --out results/M0_verdict.json \
    --delta_pp 3.0 --min_healthy_seeds 3 --collapse_thresh_pp 10.0 \
    > logs/m0_verdict.log 2>&1

echo "[$(date)] pipeline_monitor DONE" >> $LOG
cat results/M0_verdict.json
