"""Add a LoRA adapter trained on one model onto another model of the same architecture (task arithmetic):

    W_new = W_target + (alpha / r) * B A        (A, B from the adapter, e.g. LoRA B1 trained on decider9b-sft)

Run from the decider repo with its venv, on a GPU node:
    .venv/bin/python /projects/u6xe/louisk/hardgen/merge_adapter.py --target RL_MODEL --adapter runs/loraB1/lora_adapter.pt --out OUT
The output is a plain checkpoint; give it a decider_config.json (temperature) before serving (jobs/eval_model.sh does).
"""
import argparse
import os

import torch

from decider.model import DecisionModel
from decider.train import LORA_TARGETS, LoRALinear, add_lora, merge_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=64); ap.add_argument("--alpha", type=float, default=128)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the adapter's delta (1 = as trained)")
    a = ap.parse_args()
    m = DecisionModel(a.target, grad_ckpt=False).cuda().eval()
    n = add_lora(m.lm, a.r, a.alpha * a.scale, set(LORA_TARGETS.split(",")))
    sd = torch.load(a.adapter, map_location="cuda")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    lora_keys = {k for k in m.state_dict() if k.endswith((".A", ".B"))}
    assert not unexpected, unexpected[:5]
    assert lora_keys <= set(sd), sorted(lora_keys - set(sd))[:5]
    print(f"[merge] {n} adapted layers, {len(sd)} adapter tensors loaded onto {a.target}")
    merge_lora(m.lm)
    assert not any(isinstance(x, LoRALinear) for x in m.modules())
    os.makedirs(a.out, exist_ok=True)
    m.lm.save_pretrained(a.out); m.tok.save_pretrained(a.out)
    print(f"[merge] saved {a.out}")


if __name__ == "__main__":
    main()
