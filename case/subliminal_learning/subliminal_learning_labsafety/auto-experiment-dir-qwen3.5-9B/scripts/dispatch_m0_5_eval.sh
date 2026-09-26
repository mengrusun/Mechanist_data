#!/usr/bin/env bash
# M0.5 dev eval — evaluate each dev LR ckpt on QA_I and pick best LR.
# 5 arms × 1 Ctrl = 6 eval runs; distribute across GPUs 3,4,5,6,7.
#
# Usage: bash scripts/dispatch_m0_5_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
QA_I=/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet
JUDGE_CACHE=caches/eval_cache.jsonl
mkdir -p results dev logs caches

LRS=(5e-5 1e-4 2e-4 5e-4 1e-3)
GPUS=(3 4 5 6 7)
PIDS=()

# Also run Ctrl eval — it's needed to compute per-LR drop.
# Reuse an existing eval_ctrl.jsonl if present (frozen anchor for all LR arms + M0.7 arms).
if [ ! -s results/qa_i_ctrl.jsonl ]; then
  echo "[dispatch] launching Ctrl eval on GPU 3 first (needed as baseline)"
  CUDA_VISIBLE_DEVICES=3 python scripts/qa_i_eval.py \
      --base_model $BASE \
      --adapter null \
      --benchmark $QA_I \
      --judge_cache $JUDGE_CACHE \
      --arm ctrl \
      --out results/qa_i_ctrl.jsonl \
      --resume_from_output \
      >> logs/m0_5eval_ctrl.log 2>&1
  echo "[dispatch] Ctrl eval done — Acc(Ctrl):"
  python -c "import json; recs=[json.loads(l) for l in open('results/qa_i_ctrl.jsonl') if l.strip()]; n=sum(1 for r in recs if r['judge_verdict']=='CORRECT'); print(f'{n}/{len(recs)} = {n/max(len(recs),1):.4f}')"
fi

for I in 0 1 2 3 4; do
  LR=${LRS[$I]}
  GPU=${GPUS[$I]}
  ADAPTER=ckpts/student_dev_lr${LR}
  OUT=results/dev_eval_lr${LR}.jsonl
  LOG=logs/m0_5eval_lr${LR}.log
  echo "[dispatch] dev-eval lr=$LR gpu=$GPU"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/qa_i_eval.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --benchmark $QA_I \
      --judge_cache $JUDGE_CACHE \
      --arm dev_lr${LR} \
      --out $OUT \
      --resume_from_output \
      >> $LOG 2>&1 &
  PIDS+=($!)
done

wait
echo "[dispatch] all dev-LR evals done"

# Pick best LR
python -c "
import json, glob
from pathlib import Path

def acc(path):
    if not Path(path).exists():
        return None
    recs = [json.loads(l) for l in open(path) if l.strip()]
    n = sum(1 for r in recs if r['judge_verdict'] == 'CORRECT')
    return n / max(len(recs), 1), n, len(recs)

ctrl_acc, nc, nt = acc('results/qa_i_ctrl.jsonl')
print(f'Ctrl acc: {ctrl_acc:.4f} ({nc}/{nt})')
best_lr = None
best_drop = -1e9
curve = {}
for lr in ['5e-5', '1e-4', '2e-4', '5e-4', '1e-3']:
    a, c, t = acc(f'results/dev_eval_lr{lr}.jsonl') or (None, 0, 0)
    if a is None:
        continue
    drop = ctrl_acc - a
    curve[lr] = {'acc': a, 'drop': drop, 'n_correct': c, 'n_total': t}
    print(f'lr={lr}: acc={a:.4f} drop_from_ctrl={drop*100:.2f}pp')
    if drop > best_drop:
        best_drop = drop
        best_lr = lr

with open('dev/lr_curve.json', 'w') as f:
    json.dump({'ctrl_acc': ctrl_acc, 'curve': curve}, f, indent=2)
with open('dev/best_lr.json', 'w') as f:
    json.dump({'lr': best_lr, 'best_drop_pp': best_drop * 100}, f, indent=2)
print(f'BEST LR: {best_lr} (drop = {best_drop*100:.2f} pp)')
"

cat dev/best_lr.json
