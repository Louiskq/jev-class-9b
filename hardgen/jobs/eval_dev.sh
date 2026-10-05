#!/bin/bash
#SBATCH --job-name=hardgen-evaldev
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-evaldev-%j.out
# Score a model on the independent dev sets (state-first, graph engine); no JevBench.
#   sbatch jobs/eval_dev.sh MODEL_DIR LABEL
# Writes runs/dev_eval/LABEL/{new_domains,old_heldout,public_heldout}/{eval.json,preds.pkl}.
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
H=/projects/u6xe/louisk/hardgen; M=$1; L=$2
for S in new_domains:dev_new.pkl old_heldout:text_heldout.pkl public_heldout:public_heldout.pkl; do
  N=${S%%:*}; F=${S##*:}
  [ -f $H/runs/dev_eval/$L/$N/eval.json ] && continue
  $PY -m decider.evaluate --model "$M" --data $H/data/$F --out $H/runs/dev_eval/$L/$N --engine graph --bs 16 \
      --max_options 255 --max_ctx 16384 --layout state_first
done
