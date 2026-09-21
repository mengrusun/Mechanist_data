#!/usr/bin/env bash
# End-to-end pipeline runner.
set -e

CODE=/data/zhenqian/Reproduction1/cc/emotion/emotion_circuit/code
OUT=/data/zhenqian/Reproduction1/cc/emotion/emotion_circuit/outputs

cd "$CODE"

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

GPU=${GPU:-1}

echo "== Step 1: SEV prompt-based generation (training data) =="
CUDA_VISIBLE_DEVICES=$GPU python prompt_generate.py \
  --split sev --mode prompt --out prompt_gen/sev.jsonl \
  --batch_size 48 --valence_mode paired

echo "== Step 2: label SEV =="
python gpt_label.py --input prompt_gen/sev.jsonl --out labeled/sev.jsonl --workers 32 --resume
python summarize.py --input labeled/sev.jsonl --out stats/sev_prompt.json

echo "== Step 3: dump activations from correct SEV samples =="
CUDA_VISIBLE_DEVICES=$GPU python dump_activations.py \
  --labeled labeled/sev.jsonl --out_dir activations \
  --max_per_emotion 200

echo "== Step 4: analyze directions & scores =="
CUDA_VISIBLE_DEVICES=$GPU python analyze_directions.py \
  --acts_dir activations --out_dir directions

echo "== Step 5: build circuit =="
python build_circuit.py \
  --directions $OUT/directions/directions.pt \
  --acts_dir activations \
  --out circuits/global_circuit.pt \
  --k_neurons 392 --k_heads 168

echo "== Step 6: test prompt baseline (all valences) =="
CUDA_VISIBLE_DEVICES=$GPU python prompt_generate.py \
  --split test --mode prompt --out prompt_gen/test.jsonl \
  --batch_size 48 --valence_mode all

python gpt_label.py --input prompt_gen/test.jsonl --out labeled/test_prompt.jsonl --workers 32
python summarize.py --input labeled/test_prompt.jsonl --out stats/test_prompt.json

echo "== Step 7: test direction-steering =="
CUDA_VISIBLE_DEVICES=$GPU python steer_generate.py \
  --directions $OUT/directions/directions.pt \
  --split test --valence_mode all --out steer_gen/test.jsonl \
  --scale 6.0 --layers 12-27

python gpt_label.py --input steer_gen/test.jsonl --out labeled/test_steer.jsonl --workers 32
python summarize.py --input labeled/test_steer.jsonl --out stats/test_steer.json

echo "== Step 8: test circuit-based =="
CUDA_VISIBLE_DEVICES=$GPU python circuit_generate.py \
  --circuits $OUT/circuits/global_circuit.pt \
  --split test --valence_mode all --out circuit_gen/test.jsonl \
  --scale 0.8

python gpt_label.py --input circuit_gen/test.jsonl --out labeled/test_circuit.jsonl --workers 32
python summarize.py --input labeled/test_circuit.jsonl --out stats/test_circuit.json

echo "== Done =="
