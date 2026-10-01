#!/bin/bash
#SBATCH --job-name=rl-breakout
#SBATCH --gpus=1
#SBATCH --time=14:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/rl-breakout-%j.out
# PPO on Breakout with decider's games RL (decider/games/rl.py, full parameters) from decider9b-sft.
# Saves the model whenever the greedy Breakout eval beats the start; logs in OUT/rl.log.
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv -l 300 &
$PY -m decider.games.rl --init /projects/u6xe/louisk/models/decider9b-sft --out /projects/u6xe/louisk/hardgen/runs/rl_breakout \
  --games breakout --eval_games breakout --iters 40 --envs_per_game 16 --max_t 300 --eval_every 5 --eval_episodes 3
