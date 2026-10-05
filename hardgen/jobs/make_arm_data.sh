#!/bin/bash
#SBATCH --job-name=hardgen-armdata
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-armdata-%j.out
# Build the training caches for the ablation arms (all on the B1 recipe: text + 40 replay/task, build seed 0).
#   scale2k / scale5k : random subsets of round-1 text (4,058 + 5,375 questions) at 2,400 / 4,700 questions
#   scaleAll          : ALL round-1 + round-2 text, unmined (22.9k questions)
#   code5 / code15    : round-1 text + 470 / 1,650 questions of code_small_train, restyled to JevBench's option format
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python; H=/projects/u6xe/louisk/hardgen; D=$H/data
R1="$D/text_seeded_train.jsonl $D/text_authored_train_a.jsonl $D/text_authored_train_b.jsonl"
R2="$D/text_r2_seeded_a.jsonl $D/text_r2_seeded_b.jsonl $D/text_r2_authored_a.jsonl $D/text_r2_authored_b.jsonl $D/text_r2_authored_c.jsonl"
HELD="$D/text_seeded_heldout.jsonl $D/text_authored_heldout.jsonl"
BUILD="$PY $H/build.py --heldout $HELD --replay data/mixture_full.pkl --replay_per_task 40 --seed 0"
$PY $H/subsample.py --n_questions 2400 --seed 0 --out $D/arm_scale2k.jsonl $R1
$PY $H/subsample.py --n_questions 4700 --seed 0 --out $D/arm_scale5k.jsonl $R1
$PY $H/subsample.py --n_questions 470  --seed 0 --out $D/arm_code5.jsonl  $D/code_small_train.jsonl
$PY $H/subsample.py --n_questions 1650 --seed 0 --out $D/arm_code15.jsonl $D/code_small_train.jsonl
$BUILD --train $D/arm_scale2k.jsonl --out $D/arm_scale2k.pkl
$BUILD --train $D/arm_scale5k.jsonl --out $D/arm_scale5k.pkl
$BUILD --train $R1 $R2 --out $D/arm_scaleAll.pkl
$BUILD --restyle_code --train $R1 $D/arm_code5.jsonl  --out $D/arm_code5.pkl
$BUILD --restyle_code --train $R1 $D/arm_code15.jsonl --out $D/arm_code15.pkl
