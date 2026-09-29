"""LoRA sanity checks for decider.train's adapter code (run on a GPU node from the decider repo):
    .venv/bin/python test_lora.py MODEL_DIR EVAL_PKL                       # synthetic adapter
    .venv/bin/python test_lora.py MODEL_DIR EVAL_PKL RUN_DIR               # a trained run: base + RUN/lora_adapter.pt vs RUN/model
1. with B = 0 the wrapped model gives exactly the base model's slot logits;
2. the merged plain model gives the wrapped model's logits up to bf16 rounding (random adapter, or the trained one).
"""
import random
import sys

import torch

from decider import data as D
from decider.model import DecisionModel, collate
from decider.prompt import build
from decider.train import LoRALinear, add_lora, merge_lora, LORA_TARGETS


def logits(m, batch):
    with torch.no_grad():
        return m(batch).float()


def batch_of(m, exs):
    items = [build(e, m.tok, random.Random(0), max_options=255, max_ctx_tokens=4096) for e in exs]
    b = collate(items, m.tok.pad_token_id)
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}


def trained(model_dir, eval_pkl, run):
    _, evals = D.load_cache(eval_pkl)
    exs = [e for v in evals.values() for e in v[:6]]
    m = DecisionModel(model_dir, grad_ckpt=False).cuda().eval()
    add_lora(m.lm, 64, 128, set(LORA_TARGETS.split(",")))
    missing, unexpected = m.load_state_dict(torch.load(f"{run}/lora_adapter.pt"), strict=False)
    assert not unexpected and all(not k.endswith((".A", ".B")) for k in missing), unexpected
    merged = DecisionModel(f"{run}/model", grad_ckpt=False).cuda().eval()
    agree, diffs = [], []
    for i in range(0, len(exs), 8):
        b = batch_of(m, exs[i:i + 8])
        w, g = logits(m, b), logits(merged, b)
        ok = torch.isfinite(w)
        diffs.append((w - g)[ok].abs().mean().item()); agree.append((w.argmax(-1) == g.argmax(-1)).float().mean().item())
    print(f"[test] trained adapter, {len(exs)} held-out questions: merged vs base+adapter mean |logit diff| "
          f"{sum(diffs) / len(diffs):.3g}, argmax agreement {sum(agree) / len(agree):.3f}")


def main(model_dir, eval_pkl):
    _, evals = D.load_cache(eval_pkl)
    exs = [e for v in evals.values() for e in v[:2]][:12]
    m = DecisionModel(model_dir, grad_ckpt=False).cuda().eval()
    items = [build(e, m.tok, random.Random(0), max_options=255, max_ctx_tokens=4096) for e in exs]
    b = collate(items, m.tok.pad_token_id); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    base = logits(m, b)
    n = add_lora(m.lm, 64, 128, set(LORA_TARGETS.split(",")))
    same = logits(m, b)
    print(f"[test] wrapped {n} layers; B=0 max|diff| vs base = {(same - base).abs().nan_to_num(0).max().item():.3g}")
    torch.manual_seed(0)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.B, std=1e-3)
    wrapped = logits(m, b)
    merge_lora(m.lm)
    left = sum(isinstance(x, LoRALinear) for x in m.modules())
    merged = logits(m, b)
    ok = torch.isfinite(wrapped)
    d = (merged - wrapped)[ok].abs()
    print(f"[test] random adapter moved logits by {(wrapped - base)[ok].abs().mean().item():.3g} on average; merged vs wrapped: "
          f"mean |diff| {d.mean().item():.3g}, max {d.max().item():.3g}; LoRA modules left after merge: {left}")
    agree = (merged.argmax(-1) == wrapped.argmax(-1)).float().mean().item()
    print(f"[test] argmax agreement merged vs wrapped: {agree:.3f}")


if __name__ == "__main__":
    trained(*sys.argv[1:4]) if len(sys.argv) > 3 else main(*sys.argv[1:3])
