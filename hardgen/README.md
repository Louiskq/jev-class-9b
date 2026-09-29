# hardgen: code-generated hard decisions for a LoRA stage on decider9b-sft

Goal: raise decider9b's accuracy on the kinds of decisions in JevBench's hard tier without training on, or selecting by,
any JevBench item. The families follow the category names JevBench publishes for its sealed set
(`jevbench/docs/METHOD-v1.4.md`); no item text was read to design them.

Every record is TypeSafe's `/v1/systemone` wire format (`state` + typed `questions`), converted by decider's own
`to_example`, so training examples look exactly like served requests. Every answer is computed by code.

| family | subs | what it trains |
|---|---|---|
| `temporal_numeric` (`temporal.py`) | invoice, renewal, sla, age, timezones, units, budget, ledger | dates, business days, time zones, unit conversion, arithmetic over text and JSON records |
| `probability` (`probability.py`) | draws, redundancy, base_rate, expected_cost, conditional, at_least_one | exact probabilities with classic traps (base rates, P(A\|B) vs P(B\|A), any vs all) |
| `long_policy` (`policy.py`) | policy_case | applying a long numbered policy with exceptions, look-alike clauses and a priority rule |
| `judge_hard` (`judge.py`) | verify, pairwise, reply_review | judging a proposed answer when wrong answers are near misses |
| `ambiguous` (`ambiguous.py`) | policy_missing, age_partial | "cannot be determined" only when a missing/conflicting fact actually changes the answer |

Held-out records use disjoint names, organisations, domains and phrasings (`common.pool`) and a different seed; they
are only used to pick checkpoints. Ordered-value options put the correct answer at a random rank (`ranked_options`),
and band / yes-no answers are balanced, so no answer-position shortcut exists.

## Files
- `generate.py` - records as JSONL (`--family all --split train|heldout --n N --seed S`)
- `build.py` - JSONL -> decider cache `(train, evals)` + replay sample of the original mixture (run from the decider repo)
- `test_lora.py` - adapter/merge equivalence checks for the LoRA code added to `decider/train.py`
- `jobs/` - Slurm jobs (cluster paths): `lora_smoke.sh`, `baseline_heldout.sh`, `train_stage1.sh`, `eval_heldout.sh`

Cluster copy: `/projects/u6xe/louisk/hardgen` (data in `data/`, runs in `runs/`). Baseline: `/projects/u6xe/louisk/models/decider9b-sft` (read-only).
