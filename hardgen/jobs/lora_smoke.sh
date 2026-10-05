#!/bin/bash
#SBATCH --job-name=hardgen-lora-smoke
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-lora-smoke-%j.out
# LoRA smoke test: adapter/merge equivalence, then a short LoRA run on 1,500 generated records from decider9b-sft.
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
M=/projects/u6xe/louisk/models/decider9b-sft
H=/projects/u6xe/louisk/hardgen


$PY $H/generate.py --family temporal_numeric --split train --n 1500 --seed 7 --out $H/data/smoke_train.jsonl
$PY $H/build.py --train $H/data/smoke_train.jsonl --heldout $H/data/temporal_heldout.jsonl \
  --replay data/mixture_full.pkl --replay_per_task 10 --out $H/data/smoke.pkl
rm -rf $H/runs/lora_smoke
nvidia-smi --query-gpu=memory.used --format=csv -l 20 &
SMI=$!
$PY -m decider.train --model $M --data $H/data/smoke.pkl --out $H/runs/lora_smoke --epochs 1 --lr 1e-4 --warmup 2 \
  --max_tokens 16384 --accum 1 --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 \
  --eval_every 100000 --eval_limit 50 --lora_r 64
kill $SMI
ls -la $H/runs/lora_smoke $H/runs/lora_smoke/model
$PY $H/test_lora.py $M $H/data/temporal_heldout.pkl $H/runs/lora_smoke
