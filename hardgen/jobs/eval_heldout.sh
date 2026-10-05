#!/bin/bash
#SBATCH --job-name=hardgen-eval
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-eval-%j.out
# Score a model on the held-out generated families.
#   sbatch jobs/eval_heldout.sh MODEL_DIR EVAL_PKL OUT_DIR [state_first|schema_first]
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
$PY -m decider.evaluate --model "$1" --data "$2" --out "$3" --engine graph --bs 16 --max_options 255 --max_ctx 16384 \
    --layout "${4:-state_first}"
