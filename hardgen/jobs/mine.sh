#!/bin/bash
#SBATCH --job-name=hardgen-mine
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-mine-%j.out
# Round-2 hard-example mining with LoRA B1: keep every question B1 gets wrong and 70% of the rest.
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python; H=/projects/u6xe/louisk/hardgen
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
$PY $H/mine.py --model $H/runs/loraB1/model --keep_right 0.7 --out $H/data/text_r2_mined.jsonl \
  --inp $H/data/text_r2_seeded_a.jsonl $H/data/text_r2_seeded_b.jsonl \
        $H/data/text_r2_authored_a.jsonl $H/data/text_r2_authored_b.jsonl $H/data/text_r2_authored_c.jsonl
