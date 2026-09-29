#!/bin/bash
#SBATCH --job-name=decider9b-eval
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6xe/louisk/logs/decider9b-eval-%j.out
# scripts/evaluate.sh for runs/decider9b_full/model: the same commands, with these differences.
# * Order: the temperature is fitted (decider.report's fit: NLL on the in-task regression sets) right after the first
#   regression eval and written into decider_config.json, so JevBench (submitted from here, runs on its own GPU) and the
#   Decider-based probes below all run at the fitted temperature instead of evaluate.sh's placeholder 1.15.
# * Regression passes use --engine graph (CUDA graphs, what decider.serve runs) instead of --engine compile: torch.compile
#   needs minutes of single-core CPU per new input shape on this Grace node (job 6938429 spent ~half its time compiling).
#   Same forward, same results up to round-off.
# * probes.isolated / probes.independence read data/tasks.pkl + data/probes.pkl: their defaults (data/tasks_v4.pkl,
#   data/probes_v7.pkl) are the author's internal files, which the public repo does not build.
# * Resumable: a step whose output already exists is skipped (the regression block also skips the fit and JevBench).
set -e
export HF_HOME=/projects/u6xe/louisk/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /projects/u6xe/louisk/decider
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" || { echo "[env] triton must be 3.7.1"; exit 1; }

M=runs/decider9b_full/model; R=$(dirname "$M")/eval; mkdir -p "$R" logs
DATA=data/mixture_full.pkl                                                          # its eval half is the regression set
[ -f "$M/decider_config.json" ] || echo '{"temperature": 1.15, "neutralize_none": false, "version": "9b", "max_options": 255, "max_state_tokens": 32768, "schema_first": true, "isolated_levels": true}' > "$M/decider_config.json"

step() { echo "[eval] $(date -u +%H:%M:%S) $*"; }
run() {                                    # run OUTPUT CMD...: skip CMD when OUTPUT exists
  local out=$1; shift
  if [ -e "$out" ]; then echo "[eval] have $out, skipping"; else "$@"; fi
}

step regression state_first
if [ ! -e "$R/regression_state_first/eval.json" ]; then
  $PY -m decider.evaluate --model "$M" --data "$DATA" --out "$R/regression_state_first" --engine graph --bs 32 --layout state_first
  $PY -m decider.report "$R/regression_state_first"
  $PY - "$R/regression_state_first" "$M/decider_config.json" <<'EOF'
import json, pickle, sys
from decider.report import fit_temperature
run, cfg_path = sys.argv[1], sys.argv[2]
res = json.load(open(f"{run}/eval.json"))["results"]; dump = pickle.load(open(f"{run}/preds.pkl", "rb"))
T = float(fit_temperature(dump, [t for t in res if not res[t]["heldout"]]))
cfg = json.load(open(cfg_path)); cfg["temperature"] = round(T, 3); json.dump(cfg, open(cfg_path, "w"))
print("[eval] fitted temperature", round(T, 3), "->", cfg_path)
EOF
  cat "$M/decider_config.json"; echo
  sbatch /projects/u6xe/louisk/jevbench_decider.sh "$PWD/$M" decider9b || echo "[eval] could not submit JevBench from the job: submit it by hand"
else
  echo "[eval] have $R/regression_state_first, skipping it, the temperature fit and JevBench"; cat "$M/decider_config.json"; echo
fi

step probes state_first
run "$R/probes_state_first/eval.json" $PY -m decider.evaluate --model "$M" --data data/probes.pkl --out "$R/probes_state_first" --max_options 255 --max_ctx 32768 --bs 16 --layout state_first
run "$R/isolated_state_first.json" $PY -m decider.probes.isolated "$M" --layout state_first --out "$R/isolated_state_first.json" --data data/tasks.pkl --probes data/probes.pkl
step regression + probes schema_first
run "$R/regression_schema_first/eval.json" $PY -m decider.evaluate --model "$M" --data "$DATA" --out "$R/regression_schema_first" --engine graph --bs 32 --layout schema_first
run "$R/probes_schema_first/eval.json" $PY -m decider.evaluate --model "$M" --data data/probes.pkl --out "$R/probes_schema_first" --max_options 255 --max_ctx 32768 --bs 16 --layout schema_first
run "$R/isolated_schema_first.json" $PY -m decider.probes.isolated "$M" --layout schema_first --out "$R/isolated_schema_first.json" --data data/tasks.pkl --probes data/probes.pkl
step full label sets
run "$R/full_label_sets/eval.json" $PY -m decider.evaluate --model "$M" --data "$DATA" --out "$R/full_label_sets" --max_options 255 --bs 16 \
    --tasks clinc_oos,banking77,massive_intent,go_emotions,bias_in_bios,bitext_support,newsgroups,dbpedia,massive_scenario
step batteries
$PY -m decider.probes.batteries "$M" --layout=state_first --layout=schema_first
step independence
run "$R/independence.json" $PY -m decider.probes.independence "$M" --out "$R/independence.json" --data data/tasks.pkl
step games
run "$R/games.json" $PY -m decider.games.play "$M" --episodes 3 --out "$R/games.json"
step done
