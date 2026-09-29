#!/bin/bash
#SBATCH --job-name=dl-qwen38
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/projects/u6xe/louisk/logs/dl-qwen38-%j.out
# Stock Qwen/Qwen3.8-27B (Apache-2.0, BF16) as the teacher for the text data. Compute nodes have internet and no 4 GiB cap.
set -e
export HF_HOME=/scratch/u6xe/louisk.u6xe/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1
D=/scratch/u6xe/louisk.u6xe/models/Qwen3.8-27B
~/venv-vllm/bin/python -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
~/venv-vllm/bin/hf download Qwen/Qwen3.8-27B --local-dir $D
du -sh $D; ls $D | head -40
