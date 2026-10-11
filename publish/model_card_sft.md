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

Qwen3.5-9B-Base trained with the supervised recipe of [Mapika's decider](https://github.com/Mapika/decider)
(Apache 2.0), one epoch of the "full" settings. It takes a piece of state and a bounded rubric and returns a typed
answer with a probability for every option. This is the task [JevBench](https://benchmarkheaven.com/jev-models)
measures. This is an unofficial community model, not affiliated with JevBench or decider.

It is the baseline of a larger study; the follow-on model is [`Louiskq/jev-class-9b`](https://huggingface.co/Louiskq/jev-class-9b).
Code, patches, job scripts and the full experiment log: https://github.com/Louiskq/jev-class-9b.

**Not a chat model.** The architecture is a standard Qwen3.5 causal LM, so the Hub labels it "text-generation", but it is trained to read a decider-format state and rubric and return a probability for each option. Prompting it like a chat model will not give meaningful results; use decider's serving code.

## Training

- Data: decider's mixture, built with decider's own scripts: 1,555,599 examples, 464.6M tokens.
- One epoch, 16,982 optimizer steps, about 23 h on one GH200 (Isambard-AI). No NaNs or restarts of note (the run
  was resumed once from a checkpoint; the resumed run matched an uninterrupted one in a test).
- Local changes to decider's `train.py`: checkpoint/resume support. See the repo's `patches/`.
- No JevBench item was used for training.

## Results

Regression suite (decider's `evaluate`, state-first layout, temperature 1.05 as shipped in `decider_config.json`):

| | accuracy | NLL | ECE |
|---|---|---|---|
| in-task | 0.843 | 0.380 | 0.026 |
| held-out tasks | 0.793 | 0.555 | 0.074 |

JevBench public items (231 items, all answered):

| all | easy | original | hard |
|---|---|---|---|
| 0.840 | 1.000 | 0.972 | 0.685 (ECE 0.113) |

On hard (111 items, standard error about 4.4 points) this is a tie with decider-4b v2 (0.676).

## Use

`decider_config.json` holds the fitted temperature (1.05); apply it. Serving uses decider's `decider.serve`
(`DECIDER_MODEL=<this dir> uvicorn decider.serve:app`), tested with decider plus the patches in the repo above.
The checkpoint is a standard Qwen3.5 (hybrid linear-attention) model, but it is trained for decider's input format,
not for chat.

## Limitations

- Weakest regression probes: very long option lists and JSON-heavy states (json_k64 0.61, json_xl 0.63, json_long 0.70).
- Game tasks: teacher level on pong, breakout and cliffwalking; fails minigrid_empty and frozenlake.
- Hard JevBench items remain the weak spot (0.685).

## Credits

Recipe, data mixture, serving code and evaluation suite: [Mapika's decider](https://github.com/Mapika/decider),
Apache 2.0. Base model: Qwen/Qwen3.5-9B-Base (Apache 2.0). Licence of this model: Apache 2.0.
