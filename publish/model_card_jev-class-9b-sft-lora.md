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

[`jev-class-9b-sft`](https://huggingface.co/Louiskq/jev-class-9b-sft) (Mapika's decider recipe on Qwen3.5-9B-Base)
plus a LoRA (merged into the weights) trained on teacher-written decision questions aimed at JevBench's hard
tier. This is an unofficial community model, not affiliated with JevBench or decider.
Code, data pipeline and the full experiment log: https://github.com/Louiskq/jev-class-9b.

**Not a chat model.** The architecture is a standard Qwen3.5 causal LM, so the Hub labels it "text-generation", but it is trained to read a decider-format state and rubric and return a probability for each option. Prompting it like a chat model will not give meaningful results; use decider's serving code.

## Evidence

Chosen on a dev set that no earlier decision used, not on JevBench.

- **Clear gain:** +9.8 points on 2,791 questions in four unseen domains (0.815 -> 0.913).
- **Not shown:** a gain on JevBench hard. 0.730 against 0.685 for the SFT model is 11 questions fixed and 6 broken
  out of 111, which is not significant (sign test p = 0.33).
- **Not harmed:** regression accuracy is within 0.003 of the SFT model.

## Training

- Start: jev-class-9b-sft. LoRA rank 64, alpha 128 on all attention and MLP projections, lr 1e-4, cosine,
  2 epochs, 65,536 tokens per step, merged into the weights.
- Data: 19,612 questions written or reworded by the stock Qwen/Qwen3.8-27B teacher (Apache 2.0), kept only if the
  answer is fixed by code or three independent answers agree, plus 40 replay examples per task from decider's mixture.
- Spec sampling used only JevBench's aggregate format statistics (lengths, option counts, question types). No JevBench
  item was used for training or for choosing training data.

## Results

| | JevBench hard | hard ECE | regression in-task / held-out |
|---|---|---|---|
| jev-class-9b-sft | 0.685 | 0.113 | 0.843 / 0.793 |
| **jev-class-9b-sft-lora** | 0.730 | 0.165 | 0.841 / 0.792 |

JevBench original: 0.958.

## Use

**Apply the temperature in `decider_config.json` (about 1.65).** The model is over-confident at T = 1 and well
calibrated on the regression sets after fitting. Even so, its calibration on JevBench hard is a little worse than the
SFT model's (ECE 0.165 against 0.113). Serving is the same as for the SFT model (`decider.serve`, see its card).

## Limitations

- The dev set is written by the same teacher as the training data, so it measures learning that distribution in new
  domains, not transfer to other authors' styles.
- The JevBench hard gain over the SFT model is not significant (n = 111).
- Weak spots of the SFT model carry over (see its card).

## Credits

Recipe, data mixture, serving and evaluation: [Mapika's decider](https://github.com/Mapika/decider) (Apache 2.0).
Base: Qwen/Qwen3.5-9B-Base. Text teacher: Qwen/Qwen3.8-27B. Licence: Apache 2.0.
