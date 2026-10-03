#!/bin/bash
#SBATCH --job-name=hardgen-combo
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-combo-%j.out
# Combine the two tracks: LoRA B1's weight change (trained on decider9b-sft) added onto the RLCD-style Breakout model
# (rl_breakout_cal_v2/model, iter 25), then the Breakout + belief eval.  jobs/eval_model.sh follows (regression, JevBench).
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python; H=/projects/u6xe/louisk/hardgen; OUT=$H/runs/combo_rlcd_b1
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
$PY $H/merge_adapter.py --target $H/runs/rl_breakout_cal_v2/model --adapter $H/runs/loraB1/lora_adapter.pt --out $OUT/model
$PY $H/rl_calibrated.py --init $OUT/model --out $OUT/breakout_eval --eval_only --eval_episodes 10
