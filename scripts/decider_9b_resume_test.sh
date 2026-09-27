#!/bin/bash
#SBATCH --job-name=decider9b-resume-test
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/decider9b-resume-test-%j.out
# Resume test on the smoke config: stop + checkpoint at step 20, resume to the end (41), then a third call must exit at once.
# Compare the step-40 ce and final eval-agg with the uninterrupted smoke run (logs/decider9b-smoke-6909386.out).

export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
source .venv/bin/activate
python -c "import triton, sys; v = triton.__version__; print(\"[env] triton\", v); sys.exit(v != \"3.7.1\")" || { echo "[env] triton must be 3.7.1"; exit 1; }

OUT=runs/smoke9b_resume
rm -rf $OUT
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 30 &
SMI=$!
ARGS="--model /projects/u6xe/louisk/models/qwen35-9b-base --data data/mixture_full.pkl --out $OUT --epochs 1 --lr 1e-5 --warmup 150
  --max_tokens 16384 --accum 2 --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 --eval_every 100000 --eval_limit 150
  --train_cap 3000 --resume"
echo "=== run 1: fresh, stop at step 20"; python -m decider.train $ARGS --stop_step 20; echo "exit $?"
ls -la $OUT $OUT/ckpt; du -sh $OUT/ckpt
echo "=== run 2: resume to the end";      python -m decider.train $ARGS; echo "exit $?"
ls -la $OUT
echo "=== run 3: already complete";       python -m decider.train $ARGS; echo "exit $?"
kill $SMI
