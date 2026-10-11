---
license: apache-2.0
base_model: Louiskq/jev-class-9b-sft
library_name: transformers
tags:
  - decider
  - decision-making
  - calibration
  - jevbench
  - lora
---

# jev-class-9b

[`jev-class-9b-sft`](https://huggingface.co/Louiskq/jev-class-9b-sft) (Mapika's decider recipe on Qwen3.5-9B-Base)
plus a LoRA (merged into the weights) trained on teacher-written decision questions aimed at JevBench's hard
tier. This is an unofficial community model, not affiliated with JevBench or decider.
Code, data pipeline and the full experiment log: https://github.com/Louiskq/jev-class-9b.

## What it is and what the evidence says

Chosen on a dev set that no earlier decision used (see below), not on JevBench. The honest summary:

- **Clear gain:** +9.8 points on 2,791 questions in four unseen domains (0.815 -> 0.913).
- **Not shown:** a gain on JevBench hard. 0.730 against 0.685 for the SFT model is 11 questions fixed and 6 broken
  out of 111, which is not significant (sign test p = 0.33). JevBench public was read 10 times across the project's
  runs, so treat any single JevBench number as optimistic.
- **Not harmed:** regression accuracy is within 0.003 of the SFT model.

## Training

- Start: jev-class-9b-sft. LoRA rank 64, alpha 128 on all attention and MLP projections, lr 1e-4, cosine,
  2 epochs, 65,536 tokens per step, merged into the weights.
- Data (19,612 questions): written by the stock Qwen/Qwen3.8-27B teacher (Apache 2.0). Half rewrite code-generated facts
  with answers fixed by code (kept only if every number survives and the facts are judged identical); half are
  authored by the teacher from a spec and kept only if three independent thinking answers agree. A second round
  was mined with an earlier LoRA: all questions it got wrong plus 70% of those it got right. Plus 40 replay examples per task
  from decider's mixture.
- Spec sampling used only JevBench's aggregate format statistics (lengths, option counts, question types). No JevBench
  item was used for training or for choosing training data.

## Results

Dev set `new_domains` (authored in four domains absent from training, same writer and agreement filter; SFT 0.815):
**0.913**. Across three seeds the first-round version of this recipe scored 0.896 +- 0.005, and this version
(single seed, more data) is above that seed spread.

| | JevBench hard | hard ECE | regression in-task / held-out |
|---|---|---|---|
| jev-class-9b-sft | 0.685 | 0.113 | 0.843 / 0.793 |
| **jev-class-9b** | 0.730 | 0.165 | 0.841 / 0.792 |

JevBench original: 0.958.

## Use

**Apply the temperature in `decider_config.json` (about 1.65).** Like every LoRA in this study the model is
over-confident at T = 1 and well calibrated on the regression sets after fitting. Even so, its calibration on
JevBench hard is a little worse than the SFT model's (ECE 0.165 against 0.113). Serving is the same as for the SFT model
(`decider.serve`, see its card).

## Limitations

- The dev set is written by the same teacher as the training data, so it measures learning that distribution in new
  domains, not transfer to JevBench authors' style.
- Gains came from volume of teacher-written text (about +1.2 points per doubling); code-generated-only data hurt
  JevBench hard (0.568) and is not part of this model.
- Only the sealed JevBench set can separate this model from the SFT model.

## Credits

Recipe, data mixture, serving and evaluation: [Mapika's decider](https://github.com/Mapika/decider) (Apache 2.0).
Base: Qwen/Qwen3.5-9B-Base. Text teacher: Qwen/Qwen3.8-27B. Licence: Apache 2.0.
