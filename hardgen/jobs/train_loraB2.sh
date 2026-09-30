#!/bin/bash
#SBATCH --job-name=hardgen-loraB2
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-loraB2-%j.out
# LoRA B2 on decider9b-sft: the same text data as B1 + a reworked code slice (800 records per family, 1/5 of LoRA A's,
# JevBench option format via build.py --restyle_code) + 40 replay examples per task.  Training settings as LoRA A.
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
H=/projects/u6xe/louisk/hardgen; M=/projects/u6xe/louisk/models/decider9b-sft
TEXT_TRAIN="$H/data/text_seeded_train.jsonl $H/data/text_authored_train_a.jsonl $H/data/text_authored_train_b.jsonl"
TEXT_HELD="$H/data/text_seeded_heldout.jsonl $H/data/text_authored_heldout.jsonl"
RUN=loraB2
if [ ! -f $H/data/$RUN.pkl ]; then
  $PY $H/generate.py --family all --split train --n 800 --seed 2 --out $H/data/code_small_train.jsonl
  $PY $H/build.py --train $TEXT_TRAIN $H/data/code_small_train.jsonl --heldout $TEXT_HELD $H/data/code_heldout.jsonl \
      --replay data/mixture_full.pkl --replay_per_task 40 --seed 0 --min_questions 3000 --restyle_code --out $H/data/$RUN.pkl
fi
$PY -m decider.train --model $M --data $H/data/$RUN.pkl --out $H/runs/$RUN --epochs 2 --lr 1e-4 --warmup 20 \
  --max_tokens 16384 --accum 4 --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 \
  --eval_every 100 --eval_limit 150 --seed 0 --lora_r 64 --lora_alpha 128 --resume --ckpt_every_min 60
