#!/bin/bash
#SBATCH --job-name=rl-breakout-cal
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/rl-breakout-cal-%j.out
# Calibration-aware ("RLCD-style") PPO on Breakout from decider9b-sft (rl_calibrated.py): the plain run's actor term and
# settings + belief questions scored with the log score + retention KL.  Args are passed through (smoke: --iters 1 ...).
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv -l 300 &
$PY /projects/u6xe/louisk/hardgen/rl_calibrated.py --init /projects/u6xe/louisk/models/decider9b-sft "$@"
