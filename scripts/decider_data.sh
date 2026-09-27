#!/bin/bash
#SBATCH --job-name=decider-data
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/decider-data-%j.out

export HF_HOME=/projects/u6xe/louisk/hf PYTHONUNBUFFERED=1
cd /projects/u6xe/louisk/decider
.venv/bin/python -m decider.data.core --out data/tasks.pkl && \
.venv/bin/python -m decider.data.mixture --base data/tasks.pkl --mode full \
  --out data/mixture_full.pkl --probes data/probes.pkl
