#!/bin/bash
#SBATCH --job-name=hardgen-arm
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-arm-%j.out
# One LoRA arm on decider9b-sft for the seed / ablation study (EXPERIMENTS.md, "Round 4").
#   sbatch jobs/train_arm.sh NAME DATA_PKL [SEED] [LORA_R] [LORA_ALPHA] [LR]
# DATA_PKL is built beforehand (jobs/make_arm_data.sh) or is an existing cache (loraB1.pkl, loraB5.pkl).
# Every other setting is identical to LoRA A/B1 (EXPERIMENTS.md "Shared settings").  No JevBench is run here.
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
H=/projects/u6xe/louisk/hardgen; M=/projects/u6xe/louisk/models/decider9b-sft
NAME=$1; DATA=$2; SEED=${3:-0}; R=${4:-64}; ALPHA=${5:-128}; LR=${6:-1e-4}
$PY -m decider.train --model $M --data $DATA --out $H/runs/$NAME --epochs 2 --lr $LR --warmup 20 \
  --max_tokens 16384 --accum 4 --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 \
  --eval_every 100 --eval_limit 150 --seed $SEED --lora_r $R --lora_alpha $ALPHA --resume --ckpt_every_min 60
