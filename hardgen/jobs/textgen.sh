#!/bin/bash
#SBATCH --job-name=hardgen-textgen
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-textgen-%j.out
# Text data with Qwen3.8-27B:  sbatch jobs/textgen.sh MODE N SPLIT SEED OUT [--family_weights ..] [--seed_weights ..]
# vLLM environment from the qwen-mythos-isambard notes (CUDA 13 forward-compat on the 12.7 driver).
set -e
export LD_LIBRARY_PATH=/scratch/u6xe/louisk.u6xe/cuda-compat/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export CUDA_HOME=/home/u6xe/louisk.u6xe/venv-vllm/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:/home/u6xe/louisk.u6xe/venv-vllm/bin:$PATH
export HF_HOME=/scratch/u6xe/louisk.u6xe/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
# per-job compile caches on node-local disk: parallel jobs sharing ~/.cache raced ("Stale file handle", 6961772/6961776)
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID VLLM_CACHE_ROOT=/tmp/vllm_$SLURM_JOB_ID
cd /projects/u6xe/louisk/hardgen
~/venv-vllm/bin/python textgen.py "$1" --n "$2" --split "$3" --seed "$4" --out "$5" "${@:6}"   # extra flags pass through
