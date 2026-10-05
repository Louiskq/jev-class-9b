#!/bin/bash
#SBATCH --job-name=hardgen-builddev
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-builddev-%j.out
# Dev set for the seed / ablation study: authored questions in four domains no earlier set used -> data/dev_new.pkl
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /projects/u6xe/louisk/decider
H=/projects/u6xe/louisk/hardgen
.venv/bin/python $H/build.py --heldout $H/data/text_dev_authored_a.jsonl $H/data/text_dev_authored_b.jsonl --out $H/data/dev_new.pkl
