---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B-Base
library_name: transformers
inference: false
tags:
  - decider
  - decision-making
  - calibration
  - jevbench
---

# jev-class-9b-sft

This is Qwen3.5-9B-Base trained with the supervised recipe from [Mapika's decider](https://github.com/Mapika/decider) (Apache 2.0): one epoch on the "full" settings. Given a state and a bounded rubric, it returns a typed answer with a probability for every option, which is the task [JevBench](https://benchmarkheaven.com/jev-models) measures. It is an unofficial community model and has no connection to JevBench or decider.

A follow-up with an extra LoRA stage is [jev-class-9b-sft-lora](https://huggingface.co/Louiskq/jev-class-9b-sft-lora). Code, patches, job scripts and the experiment log: https://github.com/Louiskq/jev-class-9b

It is not a chat model. The Hub lists it as text-generation because the architecture is a Qwen3.5 causal LM, but it reads a state and a rubric in decider's format and returns a probability for each option. Prompting it like a chat model won't give you anything useful. Use decider's serving code.

## Training

The data is decider's mixture, built with decider's own scripts: 1,555,599 examples, 464.6M tokens. Training ran for one epoch (16,982 optimizer steps) and took about 23 hours on one GH200 on Isambard-AI. The only change to decider's `train.py` was adding checkpoint and resume support; the patches are in the repo. No JevBench item was used for training.

## Results

Decider's regression suite (state-first layout, temperature 1.05 as set in `decider_config.json`):

| | accuracy | NLL | ECE |
|---|---|---|---|
| in-task | 0.843 | 0.380 | 0.026 |
| held-out tasks | 0.793 | 0.555 | 0.074 |

JevBench public items (231 items, all answered):

| all | easy | original | hard |
|---|---|---|---|
| 0.840 | 1.000 | 0.972 | 0.685 (ECE 0.113) |

There are 111 hard items, so the standard error is about 4.4 points. That makes 0.685 a tie with decider-4b v2 (0.676).

## Use

Apply the temperature in `decider_config.json` (1.05). To serve it, run decider's `decider.serve` with `DECIDER_MODEL=<this dir> uvicorn decider.serve:app`. This was tested with decider plus the patches in the repo above.

## Limitations

The weakest regression probes are very long option lists and JSON-heavy states (json_k64 0.61, json_xl 0.63, json_long 0.70). On the game tasks it plays at teacher level on pong, breakout and cliffwalking, and fails minigrid_empty and frozenlake.

## Credits

Recipe, data mixture, serving code and evaluation suite are from [Mapika's decider](https://github.com/Mapika/decider) (Apache 2.0). Base model: Qwen/Qwen3.5-9B-Base (Apache 2.0). This model is Apache 2.0.
