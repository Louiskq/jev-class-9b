#!/bin/bash
#SBATCH --job-name=hardgen-loraA
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-loraA-%j.out
# LoRA A (code-generated data only) on decider9b-sft.  Settings recorded in EXPERIMENTS.md; LoRA B must reuse them.
#   data: 5 generated families x 4,000 records (seed 1) + 40 replay examples per task of the original mixture (build seed 0)
#   held-out: 5 families x 300 records (seed 101), scored every 100 steps
#   LoRA r64 / alpha 128 on attention, delta-net and MLP linears; lr 1e-4, warmup 20, cosine; 2 epochs; 65,536 tokens/step
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
H=/projects/u6xe/louisk/hardgen; M=/projects/u6xe/louisk/models/decider9b-sft; OUT=$H/runs/loraA

if [ ! -f $H/data/loraA.pkl ]; then
  $PY $H/generate.py --family all --split train --n 4000 --seed 1 --out $H/data/code_train.jsonl
  $PY $H/generate.py --family all --split heldout --n 300 --seed 101 --out $H/data/code_heldout.jsonl
  $PY $H/build.py --train $H/data/code_train.jsonl --heldout $H/data/code_heldout.jsonl \
      --replay data/mixture_full.pkl --replay_per_task 40 --seed 0 --out $H/data/loraA.pkl
fi
$PY -m decider.train --model $M --data $H/data/loraA.pkl --out $OUT --epochs 2 --lr 1e-4 --warmup 20 \
  --max_tokens 16384 --accum 4 --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 \
  --eval_every 100 --eval_limit 150 --seed 0 --lora_r 64 --lora_alpha 128 --resume --ckpt_every_min 60
