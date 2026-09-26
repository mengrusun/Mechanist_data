#!/bin/bash
# Filter both arms as soon as their teacher gen finishes. Runs API-only so
# concurrent with any GPU work.
set -uo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

wait_for_arm() {
    local arm=$1
    local dir=runs/m0b_teacher_gen/${arm}
    while true; do
        # Check if both rank files exist AND no more m0_teacher_gen processes with this tag
        if [ -f "${dir}/rank0.jsonl" ] && [ -f "${dir}/rank1.jsonl" ]; then
            if [ -z "$(ps aux | grep -E "m0_teacher_gen.py --teacher_tag ${arm}" | grep -v grep)" ]; then
                break
            fi
        fi
        sleep 30
    done
    local tot=$(wc -l < "${dir}/rank0.jsonl")
    local tot1=$(wc -l < "${dir}/rank1.jsonl")
    echo "${arm} gen done: rank0=${tot} rank1=${tot1}"
}

filter_arm() {
    local arm=$1
    local dir=runs/m0b_teacher_gen/${arm}
    local out=runs/m0b_teacher_gen/${arm}_filtered.jsonl
    local stats=runs/m0b_teacher_gen/${arm}_filter_stats.json
    if [ -f "${out}" ]; then
        echo "SKIP filter ${arm} (exists)"
        cat "${stats}" 2>/dev/null
        return
    fi
    echo "==== FILTER ${arm} starting ==="
    $CONDA_PYTHON scripts/m0_gpt54_filter.py \
        --in ${dir} \
        --out ${out} \
        --stats ${stats} \
        --min_len 40 --workers 8 \
        > logs/m0b_filter_${arm}.log 2>&1
    echo "==== FILTER ${arm} done ==="
    cat ${stats}
    echo
}

# Wait for treated then filter in background
wait_for_arm treated
filter_arm treated &
TREATED_PID=$!

# Wait for base then filter
wait_for_arm base
filter_arm base &
BASE_PID=$!

wait $TREATED_PID
wait $BASE_PID
echo "==== BOTH FILTERS DONE ===="
