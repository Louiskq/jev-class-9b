# Experiments

All runs start from `/projects/u6xe/louisk/models/decider9b-sft` (read-only). Only the training data differs between
LoRA A and LoRA B; everything below must stay the same for the comparison to be fair.

## Shared settings
- LoRA rank 64, alpha 128, targets `q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj`
- lr 1e-4, warmup 20, cosine to 0, 2 epochs, 16,384 tokens x 4 accumulation (65,536 tokens/step), seed 0
- max_options 255, max_ctx 16,384, none_prob 0.1, schema_first_prob 0.5
- replay: 40 examples per task of `data/mixture_full.pkl` (build.py `--replay_per_task 40 --seed 0`)
- evaluation: code held-out (`generate.py --family all --split heldout --n 300 --seed 101`), the user's held-out text
  data (B only, and A scored on it too), the regression set (`decider.evaluate`, graph engine), then JevBench once.

## LoRA A - code-generated data only
- train: `generate.py --family all --split train --n 4000 --seed 1` (20,000 records)
- job: `jobs/train_loraA.sh` -> `runs/loraA` (merged model in `runs/loraA/model`, adapter `runs/loraA/lora_adapter.pt`)

Result (2026-09-30): held-out generated families rose (e.g. long_policy 0.76 -> 1.00) but JevBench hard fell
0.685 -> 0.568 (long_policy 12->7/19, multi_hop 16->12/18, temporal 6->3/15); regression acc kept (0.840 / 0.791) but
over-confident (fitted T 1.65 vs 1.05). Reading: template learning. Final train CE 0.11.

## Text data (Qwen3.8-27B, `textgen.py`, jobs 6961772-6961776)
- seeded: code-generated facts reworded by Qwen (thinking off); kept if every number survives, none is added, and Qwen
  (thinking on) judges the facts identical. authored: Qwen (thinking on) writes a document + 3 questions from a spec
  sampled to JevBench's aggregate format profile; a question is kept only if 3 independent thinking answers all agree.
- files: `data/text_{seeded,authored}_*.jsonl`; held-out = hardgen held-out pools (seeded) and the held-out domains
  healthcare administration + logistics and shipping (authored).

## LoRA B1 - text data only
- train: all text train files; job `jobs/train_loraB1.sh` -> `runs/loraB1`

## LoRA B2 - text data + reworked code slice
- train: the same text files + `generate.py --family all --split train --n 800 --seed 2` (4,000 records, 1/5 of A),
  restyled to JevBench's option format (`build.py --restyle_code`); job `jobs/train_loraB2.sh` -> `runs/loraB2`
- A reading of A's JevBench result chose "text only" and "less code" as the next arms; no family was dropped on the
  strength of per-family JevBench scores. B1/B2 checkpoints are the final ones; JevBench is read once each.

## Results (2026-10-01)
| run | JevBench hard | hard ECE | regression in / held-out | fitted T | paired vs SFT (fixed / broken, sign p) |
|---|---|---|---|---|---|
| SFT | 0.685 | 0.113 | 0.843 / 0.793 | 1.05 | - |
| A (code) | 0.568 | 0.203 | 0.840 / 0.791 | 1.65 | 5 / 18, p = 0.011 |
| B1 (text) | 0.739 | 0.131 | 0.841 / 0.796 | 1.65 | 11 / 5, p = 0.21 |
| B2 (text + small code) | 0.712 | 0.154 | 0.841 / 0.793 | 1.61 | 11 / 8, p = 0.65 |
| B3 (text r1 + mined r2, 19.6k q) | 0.730 | 0.165 | 0.841 / 0.792 | 1.65 | 11 / 6, p = 0.33 (vs B1: 6 / 7, p = 1.0) |

Held-out text set (1,061 q): SFT 0.698, A 0.785, B1 0.824, B2 0.839, B3 0.876.

## Breakout RL (from decider9b-sft; robust eval: 10 greedy / 16 sampled games, fixed belief labels)
| model | greedy | sampled | belief acc | belief ECE | belief log score |
|---|---|---|---|---|---|
| SFT | 22 | 6.00 | 0.49 | 0.42 | 1.59 |
| plain PPO (rl_breakout/model, iter 40) | 34 | 6.25 | 0.57 | 0.12 | 0.72 |
| RLCD-style v2 best (rl_breakout_cal_v2/model, iter 25) | 54 | 7.12 | 0.83 | 0.04 | 0.40 |
| RLCD-style v2 last (rl_breakout_cal_v2/model_last) | 33 | 7.19 | 0.84 | 0.04 | 0.36 |

## Round 3 (2026-10-03)
| run | JevBench hard | hard ECE | regression in / held-out | T | vs SFT (fixed / broken, p) | notes |
|---|---|---|---|---|---|---|
| B4 (r1 text + r2 capped 800/family) | 0.703 | 0.141 | 0.840 / 0.795 | 1.65 | 13 / 11, p = 0.84 | text held-out 0.857 |
| B5 (r1 text + public: gsm8k, proofwriter, bbh, musr) | 0.748 | 0.126 | 0.842 / 0.790 | 1.57 | 15 / 8, p = 0.21 | public held-out 0.832 (SFT 0.608) |
| RL plain PPO (forgetting check) | 0.685 | 0.110 | 0.842 / 0.790 | 0.91 | 0 / 0 | Breakout 34 |
| RLCD-style v2 best (forgetting check) | 0.658 | 0.165 | 0.843 / 0.791 | 1.05 | 1 / 4, p = 0.38 | Breakout 54 |
| combo = RLCD v2 + B1 delta (merge_adapter.py) | 0.712 | 0.134 | 0.841 / 0.797 | 1.65 | 10 / 7, p = 0.63 | Breakout 47, beliefs acc .84 ece .07 |

## Round 4: noise floor, independent dev set, ablations (submitted 2026-10-05, `jobs/submit_round4.sh`)

Written before any result of this round was seen.

**Why.** Rounds 1-3 ranked runs on 111 JevBench hard items (s.e. about 4.4 points), read 10 times, one seed per run. That
cannot separate B1, B3 and B5. This round measures the seed noise and uses a larger set that no earlier decision touched.

**Dev sets** (`jobs/eval_dev.sh`, state-first, graph engine, accuracy over questions):
- `new_domains` (primary): about 1,500 authored questions in four domains used nowhere before (public sector and
  government services, energy and utilities, real estate and property management, telecommunications); same writer and
  3-sample agreement filter as the training text, fresh seeds. The old text held-out set was partly used to pick round-2
  data weights, so it is no longer clean.
- `old_heldout` (1,061 q) and `public_heldout`: secondary, reported for continuity.

**Arms** (all start from decider9b-sft, B1 settings: r64 / a128, lr 1e-4, 2 epochs, 65k tokens/step, 40 replay/task):
| arm | change from B1 |
|---|---|
| B1_s1, B1_s2, B5_s1, B5_s2 | training seed 1 and 2 on the same cache (LoRA init and data order); seed 0 is the earlier run |
| scale2k, scale5k | random 2,400 / 4,700 questions of the round-1 text |
| scaleAll | all round-1 + round-2 text, unmined (22.9k questions) |
| code5, code15 | B1 text + 470 / 1,650 restyled code questions (B2 had 4,000 records, about 30%) |
| r16 | LoRA rank 16, alpha 32 |
| lr5e5 | lr 5e-5 |

**Decision rules.**
1. The seed sd of B1 and B5 on `new_domains` is the noise floor. A change counts as real only if its paired difference
   on `new_domains` has a 95% bootstrap CI excluding 0 **and** is larger than 2x that sd.
2. The headline model is chosen on `new_domains` alone. JevBench is not run for any arm in this round. After the
   analysis, JevBench is read once, for the single chosen configuration, and reported with the number of earlier reads.
3. A recipe's score is its multi-seed mean, not its best seed.

Analysis: `dev_compare.py --runs <hardgen>/runs/dev_eval --ref SFT --group B1=B1,B1_s1,B1_s2 --group B5=B5,B5_s1,B5_s2 --single A,B2,B3,B4,scale2k,scale5k,scaleAll,code5,code15,r16,lr5e5`

### Round 4 results (2026-10-05; full tables in `results/round4_dev_compare.md`)
Dev set `new_domains`: 2,791 questions in 4 unseen domains (SFT 0.815). All 33 jobs completed; no JevBench read.

| arm | acc | vs SFT (95% CI) | note |
|---|---|---|---|
| B1, 3 seeds (0.901 / 0.895 / 0.892) | 0.896 +- 0.005 | +0.081 (+0.069 .. +0.095) | seed sd = the noise floor |
| B5, 3 seeds (0.900 / 0.898 / 0.894) | 0.897 +- 0.003 | +0.083 (+0.070 .. +0.094) | same as B1 on this set |
| scale 2.4k / 4.7k / 9.4k (B1) / 22.9k | 0.874 / 0.884 / 0.896 / 0.909 | +0.059 / +0.069 / +0.081 / +0.094 | about +1.2 points per doubling, no plateau |
| B3 (mined, 19.6k q) | 0.913 | +0.098 | vs scaleAll 0.909 (22.9k unmined): mining adds nothing beyond volume |
| code 5% / 15% / B2 (30%) | 0.897 / 0.898 / 0.893 | +0.082 / +0.083 / +0.078 | no effect on this set |
| A (code only) | 0.803 | -0.011 (-0.025 .. +0.002) | the only arm not above SFT |
| r16, lr 5e-5 | 0.899, 0.903 | +0.084, +0.088 | recipe insensitive to both |
| public held-out: B5 vs B1 | 0.814 vs 0.623 | | public data teaches public tasks (SFT 0.608); costs nothing on new_domains |

Reading, against the rules above:
1. Seed sd on `new_domains` is 0.003-0.005 (range 0.892-0.901 for B1), 0.006-0.011 on the old held-out set. Seed 0 of B1 was
   the best of its three, so comparisons against "B1" as a single run flatter it.
2. Text LoRA lifts the dev set by about 8 points, far above the noise. Code-only does not (A, -1.1). Adding 5 to 30% code to
   the text neither helps nor hurts here.
3. Only volume moves the dev score further. B3 (+0.011 over seed-0 B1, paired CI +0.002 .. +0.020) and scaleAll (+0.008,
   CI -0.002 .. +0.018) are the only arms above the B1 seed spread. Each is a single seed.
4. Chosen on `new_domains` alone: B3. Its JevBench read already exists (0.730, a tie with B1 0.739 and B5 0.748 at
   n = 111), so this round adds no JevBench read; the count stays at 10.

Limits: the dev set is written by the same teacher with the same filter as the training text, so it measures learning
that distribution in new domains, not transfer to JevBench's authors. B3 and scaleAll differ by 0.004, so one seed each
cannot rank them. JevBench hard remains unable to separate any of these (s.e. about 4.4 points).
