#!/bin/bash
#SBATCH --job-name=jevbench-decider
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/jevbench-decider-%j.out
# JevBench public items (easy, original, hard) against a decider checkpoint, served on this node by decider.serve and
# reached through JevBench's typesafe adapter (decider's POST /v1/systemone is TypeSafe's wire format).
#   sbatch jevbench_decider.sh [model dir] [label]
# The model dir needs its decider_config.json (scripts/evaluate.sh writes one; put the fitted temperature into it first).

M=${1:-/projects/u6xe/louisk/decider/runs/decider9b_full/model}
LABEL=${2:-decider9b}
PORT=$((20000 + SLURM_JOB_ID % 10000))         # unique port, so parallel jobs can't cross-talk
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
PY=/projects/u6xe/louisk/decider/.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print(\"[env] triton\", v); sys.exit(v != \"3.7.1\")" || { echo "[env] triton must be 3.7.1"; exit 1; }
[ -f "$M/decider_config.json" ] || { echo "no $M/decider_config.json"; exit 1; }
cat "$M/decider_config.json"; echo

cd /projects/u6xe/louisk/decider
DECIDER_MODEL=$M $PY -m uvicorn decider.serve:app --host 127.0.0.1 --port $PORT &
SERVER=$!
until curl -sf http://127.0.0.1:$PORT/health | grep -q "\"ok\":true"; do
  kill -0 $SERVER 2>/dev/null || { echo "server crashed"; exit 1; }
  sleep 5
done
echo "server up ($LABEL, port $PORT)"; curl -s http://127.0.0.1:$PORT/health; echo

RUN=/projects/u6xe/louisk/jevbench-runs/RUN-$LABEL-$SLURM_JOB_ID
TASKS=datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl
cd /projects/u6xe/louisk/jevbench
$PY -m jevbench.cli run --tasks $TASKS \
  --adapter typesafe --endpoint http://127.0.0.1:$PORT --key-env "" \
  --model $LABEL --cost-basis self_hosted --reserve-usd 0 \
  --results $RUN/results.jsonl --raw-dir $RUN/raw --ledger $RUN/ledger.jsonl \
  --run-label $LABEL
kill $SERVER

# overall summary, then split results by tier and summarise each tier
$PY -m jevbench.cli summarize --tasks $TASKS --results $RUN/results.jsonl \
  --public-export $RUN/summary-all.json
for T in easy original hard; do
  $PY - "$RUN" "$T" << 'PY'
import json, sys
d, t = sys.argv[1], sys.argv[2]
ids = {(r.get("id") or r.get("task_id"))
       for r in (json.loads(l) for l in open(f"datasets/public/{t}.jsonl") if l.strip())}
with open(f"{d}/results-{t}.jsonl", "w") as out:
    for l in open(f"{d}/results.jsonl"):
        if l.strip() and json.loads(l)["task_id"] in ids:
            out.write(l)
PY
  $PY -m jevbench.cli summarize --tasks datasets/public/$T.jsonl \
    --results $RUN/results-$T.jsonl --public-export $RUN/summary-$T.json
done
echo "done: $RUN"
