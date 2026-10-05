#!/bin/bash
# Round 4 (seeds, dev set, ablations): everything submitted, with dependencies.  Run on the login node from hardgen/.
# No JevBench job is submitted anywhere in this round.
set -e
J=/projects/u6xe/louisk/hardgen/jobs; D=/projects/u6xe/louisk/hardgen/data; R=/projects/u6xe/louisk/hardgen/runs
sb() { sbatch --parsable "$@"; }

# 1. independent dev set: 2 x 500 authored specs in 4 unseen domains, then build the cache
TA=$(sb $J/textgen.sh authored 500 heldout 71 $D/text_dev_authored_a.jsonl --domains dev)
TB=$(sb $J/textgen.sh authored 500 heldout 72 $D/text_dev_authored_b.jsonl --domains dev)
DEV=$(sb --dependency=afterok:$TA:$TB $J/build_dev.sh)

# 2. dev scores for the models that already exist
ev() { sb --dependency=afterok:$DEV${3:+:$3} $J/eval_dev.sh $1 $2; }
ev /projects/u6xe/louisk/models/decider9b-sft SFT
for X in A B1 B2 B3 B4 B5; do ev $R/lora$X/model $X; done

# 3. seeds on the existing B1 / B5 caches (training starts now; the dev eval waits for the dev set)
for S in 1 2; do
  T=$(sb $J/train_arm.sh loraB1_s$S $D/loraB1.pkl $S); ev $R/loraB1_s$S/model B1_s$S $T
  T=$(sb $J/train_arm.sh loraB5_s$S $D/loraB5.pkl $S); ev $R/loraB5_s$S/model B5_s$S $T
done

# 4. ablations on the B1 recipe (seed 0): build the caches, then train
AD=$(sb $J/make_arm_data.sh)
arm() { T=$(sb --dependency=afterok:$AD $J/train_arm.sh "$@"); ev $R/$1/model ${1#arm_} $T; }
arm arm_scale2k  $D/arm_scale2k.pkl
arm arm_scale5k  $D/arm_scale5k.pkl
arm arm_scaleAll $D/arm_scaleAll.pkl
arm arm_code5    $D/arm_code5.pkl
arm arm_code15   $D/arm_code15.pkl
T=$(sb $J/train_arm.sh arm_r16   $D/loraB1.pkl 0 16 32 1e-4);  ev $R/arm_r16/model r16 $T
T=$(sb $J/train_arm.sh arm_lr5e5 $D/loraB1.pkl 0 64 128 5e-5); ev $R/arm_lr5e5/model lr5e5 $T
