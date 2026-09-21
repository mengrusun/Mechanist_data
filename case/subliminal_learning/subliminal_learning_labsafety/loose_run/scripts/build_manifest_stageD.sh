#!/bin/bash
# Combine Stage-D eval + Stage-D health into ONE big dispatch.
# 24 evals + 24 health checks = 48 jobs, run 4-at-a-time on GPUs 0-3.
# Interleave by putting eval + health for each (lr, seed) as consecutive jobs
# so a per-checkpoint pair runs on adjacent slots.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat runs/_manifests/stageD_treated_eval.jsonl runs/_manifests/stageD_treated_health.jsonl \
    > runs/_manifests/stageD_combined.jsonl
wc -l runs/_manifests/stageD_combined.jsonl
