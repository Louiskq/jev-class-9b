#!/bin/bash
#SBATCH --job-name=hardgen-loraB1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-loraB1-%j.out
# LoRA B1 on decider9b-sft: Qwen3.8-27B text data only (seeded + authored) + 40 replay examples per task.
# Training settings identical to LoRA A (EXPERIMENTS.md); only the data differs.
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
RUN=loraB1
if [ ! -f $H/data/$RUN.pkl ]; then
  $PY $H/build.py --train $TEXT_TRAIN --heldout $TEXT_HELD \
      --replay data/mixture_full.pkl --replay_per_task 40 --seed 0 --min_questions 3000 --out $H/data/$RUN.pkl
fi
$PY -m decider.train --model $M --data $H/data/$RUN.pkl --out $H/runs/$RUN --epochs 2 --lr 1e-4 --warmup 20 \
  --max_tokens 16384 --accum 4 --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 \
  --eval_every 100 --eval_limit 150 --seed 0 --lora_r 64 --lora_alpha 128 --resume --ckpt_every_min 60
