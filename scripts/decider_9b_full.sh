#!/bin/bash
#SBATCH --job-name=decider9b-full
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/decider9b-full-%j.out
# One link of the decider-9b training chain: scripts/train.sh "full" settings, one epoch, on Qwen3.5-9B-Base.
# Every link runs with --resume: it continues from $OUT/ckpt if there is one, checkpoints every 2h and 15 min before this
# job's time limit, and exits at once if training is already complete.  Chain links with
#   sbatch --dependency=afterany:<previous job id> decider_9b_full.sh

export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
source .venv/bin/activate
python -c "import triton, sys; v = triton.__version__; print(\"[env] triton\", v); sys.exit(v != \"3.7.1\")" || { echo "[env] triton must be 3.7.1"; exit 1; }
python -c "import causal_conv1d; print(\"[env] causal_conv1d\", causal_conv1d.__version__)" || exit 1

OUT=runs/decider9b_full
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
[[ "$END" =~ ^[0-9]+$ ]] || { echo "[job] could not read this job's end time"; exit 1; }
STOP_AT=$((END - 15 * 60))
echo "[job] $SLURM_JOB_ID on $(hostname); time limit $(date -d @$END); checkpoint+stop at $(date -d @$STOP_AT)"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 600 &   # memory every 10 min
SMI=$!
python -m decider.train --model /projects/u6xe/louisk/models/qwen35-9b-base \
  --data data/mixture_full.pkl --out $OUT --epochs 1 --lr 1e-5 --warmup 150 \
  --max_tokens 16384 --accum 2 --max_options 255 --max_ctx 16384 \
  --none_prob 0.1 --schema_first_prob 0.5 --eval_every 100000 --eval_limit 150 \
  --resume --ckpt_every_min 120 --stop_at $STOP_AT
RC=$?
kill $SMI
exit $RC
