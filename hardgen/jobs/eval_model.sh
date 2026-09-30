#!/bin/bash
#SBATCH --job-name=hardgen-evalmodel
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/hardgen-evalmodel-%j.out
# Final checks for a trained LoRA model (merged dir):  sbatch jobs/eval_model.sh MODEL_DIR LABEL
#  1. merged vs base+adapter agreement (test_lora.py, trained mode)
#  2. regression set, state-first, graph engine (forgetting check vs decider9b-sft 0.843 / 0.793)
#  3. temperature fit on the in-task regression sets (as for decider9b-sft) -> MODEL_DIR/decider_config.json
#  4. JevBench public items, submitted as its own job (read once per finished LoRA)
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }
M=$1; LABEL=$2; RUN=$(dirname "$M"); H=/projects/u6xe/louisk/hardgen; BASE=/projects/u6xe/louisk/models/decider9b-sft
[ -f "$M/decider_config.json" ] || install -m 644 $BASE/decider_config.json "$M/decider_config.json"   # the base copy is read-only
chmod u+w "$M/decider_config.json"
if [ ! -f "$RUN/eval/regression_state_first/eval.json" ]; then                # resumable: skip steps already done
  [ -f "$RUN/lora_adapter.pt" ] && $PY $H/test_lora.py $BASE $H/data/code_heldout.pkl "$RUN" || true
  $PY -m decider.evaluate --model "$M" --data data/mixture_full.pkl --out "$RUN/eval/regression_state_first" --engine graph --bs 32 --layout state_first
fi
$PY -m decider.report "$RUN/eval/regression_state_first"
$PY - "$RUN/eval/regression_state_first" "$M/decider_config.json" <<'PYEOF'
import json, pickle, sys
from decider.report import fit_temperature
run, cfg_path = sys.argv[1], sys.argv[2]
res = json.load(open(f"{run}/eval.json"))["results"]; dump = pickle.load(open(f"{run}/preds.pkl", "rb"))
T = float(fit_temperature(dump, [t for t in res if not res[t]["heldout"]]))
cfg = json.load(open(cfg_path)); cfg["temperature"] = round(T, 3); json.dump(cfg, open(cfg_path, "w"))
print("[eval] fitted temperature", round(T, 3), "->", cfg_path)
PYEOF
cat "$M/decider_config.json"; echo
sbatch /projects/u6xe/louisk/jevbench_decider.sh "$M" "$LABEL" || echo "[eval] could not submit JevBench: submit it by hand"
