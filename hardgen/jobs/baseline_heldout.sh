#!/bin/bash
#SBATCH --job-name=hardgen-base
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-base-%j.out
# The frozen SFT baseline on the held-out code-generated families (same generator seed as jobs/train_loraA.sh).
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
H=/projects/u6xe/louisk/hardgen
$PY $H/generate.py --family all --split heldout --n 300 --seed 101 --out $H/data/base_heldout.jsonl
$PY $H/build.py --heldout $H/data/base_heldout.jsonl --out $H/data/base_heldout.pkl
$PY -m decider.evaluate --model /projects/u6xe/louisk/models/decider9b-sft --data $H/data/base_heldout.pkl \
  --out $H/runs/sft_heldout_stage1 --engine graph --bs 16 --max_options 255 --max_ctx 16384 --layout state_first
