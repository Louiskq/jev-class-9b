#!/bin/bash
#SBATCH --job-name=hf-publish
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hf-publish-%j.out
# Upload one model to a private HF repo (compute nodes have internet; do not run this on the login node).
#   sbatch hf_publish.sh sft              # dry run: lists files, checks the card and config
#   sbatch hf_publish.sh sft --upload     # creates Louiskq/jev-class-9b-sft (private) and uploads
#   sbatch hf_publish.sh b3  [--upload]   # Louiskq/jev-class-9b
set -e
export HF_HOME=/projects/u6xe/louisk/hf PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE                       # the training jobs set it; uploading needs the network
export HF_HUB_DISABLE_XET=1                # the Xet upload path kept dropping from compute nodes (cas-server.xethub.hf.co); use plain LFS
D=/projects/u6xe/louisk
case "$1" in
  sft) M=$D/decider/runs/decider9b_full/model;  REPO=Louiskq/jev-class-9b-sft; CARD=model_card_sft.md ;;
  b3)  M=$D/hardgen/runs/loraB3/model;          REPO=Louiskq/jev-class-9b;     CARD=model_card_jev-class-9b.md ;;
  *) echo "usage: sbatch hf_publish.sh sft|b3 [--upload]"; exit 1 ;;
esac
shift
cd "${PUBLISH_DIR:-$D/JevF/publish}"       # where this repo is synced on the cluster; sbatch copies the script, so set it if different
$D/decider/.venv/bin/python hf_publish.py "$M" "$REPO" "$CARD" "$@"
