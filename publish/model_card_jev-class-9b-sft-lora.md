---
license: apache-2.0
base_model: Louiskq/jev-class-9b-sft
library_name: transformers
inference: false
tags:
  - decider
  - decision-making
  - calibration
  - jevbench
---

# jev-class-9b-sft-lora

This is [jev-class-9b-sft](https://huggingface.co/Louiskq/jev-class-9b-sft) with a LoRA merged into the weights. The LoRA was trained on decision questions written by Qwen3.8-27B, aimed at the hard tier of JevBench. It is an unofficial community model and has no connection to JevBench or decider. Code and experiment log: https://github.com/Louiskq/jev-class-9b

It is not a chat model. The Hub lists it as text-generation because the architecture is a Qwen3.5 causal LM, but it reads a state and a rubric in decider's format and returns a probability for each option. Prompting it like a chat model won't give you anything useful. Use decider's serving code.

## Does it help?

On 2,791 questions from four domains that were not in training, accuracy goes from 0.815 (the SFT model) to 0.913. This is what the model was picked on.

On JevBench hard it goes from 0.685 to 0.730, but that is 11 questions fixed and 6 broken out of 111 (sign test p = 0.33), which is not enough to call it better there. Accuracy on decider's regression suite stayed within 0.003 of the SFT model.

| | JevBench hard | hard ECE | regression, in-task / held-out |
|---|---|---|---|
| jev-class-9b-sft | 0.685 | 0.113 | 0.843 / 0.793 |
| jev-class-9b-sft-lora | 0.730 | 0.165 | 0.841 / 0.792 |

JevBench original: 0.958.

## Training

The LoRA has rank 64 and alpha 128, covers all attention and MLP projections, and was trained for 2 epochs at lr 1e-4 with 65,536 tokens per step, then merged.

The data is 19,612 questions written or reworded by Qwen/Qwen3.8-27B (Apache 2.0). A question was kept only if its answer was fixed by code or three independent answers from the teacher agreed. 40 examples per task from decider's mixture were mixed in as replay.

The question specs were sampled using only JevBench's aggregate format statistics (lengths, number of options, question types). No JevBench item was used for training or for choosing training data.

## Use

Apply the temperature in `decider_config.json` (about 1.65). The model is over-confident at T = 1 and well calibrated on the regression sets once it is fitted. On JevBench hard its calibration is still a bit worse than the SFT model's (ECE 0.165 against 0.113). Serving works the same as for the SFT model, with `decider.serve`.

## Limitations

The dev set is written by the same teacher as the training data, so the +9.8 points show the model learned that kind of question in new domains. They don't show it transfers to other authors' styles. The JevBench hard gain is within noise. The weak spots of the SFT model carry over; they are listed on its card.

## Credits

Recipe, data mixture, serving code and evaluation suite are from [Mapika's decider](https://github.com/Mapika/decider) (Apache 2.0). Base model: Qwen/Qwen3.5-9B-Base. Question writer: Qwen/Qwen3.8-27B. This model is Apache 2.0.
