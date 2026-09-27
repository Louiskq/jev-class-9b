#!/bin/bash
#SBATCH --job-name=decider9b-smoke
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/decider9b-smoke-%j.out

export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
source .venv/bin/activate
python -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
python -c "import causal_conv1d; print('[env] causal_conv1d', causal_conv1d.__version__)"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 60 &   # memory once a minute
SMI=$!
python -m decider.train --model /projects/u6xe/louisk/models/qwen35-9b-base \
  --data data/mixture_full.pkl --out runs/smoke9b --epochs 1 --lr 1e-5 --warmup 150 \
  --max_tokens 16384 --accum 2 --max_options 255 --max_ctx 16384 \
  --none_prob 0.1 --schema_first_prob 0.5 --eval_every 100000 --eval_limit 150 \
  --train_cap 3000
kill $SMI
