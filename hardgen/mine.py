"""Hard-example mining: a decider model answers every question in the given records; questions it gets wrong are all
kept, questions it gets right are kept with probability --keep_right.  Run from the decider repo with its venv, on a GPU:

    .venv/bin/python /projects/u6xe/louisk/hardgen/mine.py --model runs/loraB1/model --keep_right 0.7 \
        --inp a.jsonl b.jsonl --out mined.jsonl

Each kept question gets "mined": {"model_correct": bool, "p_gold": float} (p at the model's decider_config temperature).
Uses only our own model and our own generated data - nothing from JevBench.
"""
import argparse
import collections
import json
import os
import random

import torch

from decider import data as D
from decider import systemone as S1
from decider.data.teacher_questions import to_example
from decider.model import DecisionModel, collate
from decider.prompt import build


class Keep:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_right", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=16384, help="padded tokens per forward batch")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    recs = [json.loads(l) for p in a.inp for l in open(p) if l.strip()]
    T = json.load(open(os.path.join(a.model, "decider_config.json"))).get("temperature", 1.0)
    m = DecisionModel(a.model, grad_ckpt=False).cuda().eval()
    rows = []                                                         # (record index, question index, item, gold)
    for i, r in enumerate(recs):
        ex = to_example(r, D, S1, task=r["family"])
        for k, q in enumerate(ex.qs):
            rows.append((i, k, build(D.Example(ex.context, [q], ex.task), m.tok, Keep(), max_options=255, max_ctx_tokens=16384), q.gold))
    rows.sort(key=lambda x: len(x[2]["ids"]))
    res = {}
    with torch.no_grad():
        j = 0
        while j < len(rows):
            n = len(rows[j][2]["ids"]); k = max(1, min(64, a.max_tokens // max(n, 1)))
            chunk = rows[j:j + k]; j += k
            b = collate([c[2] for c in chunk], m.tok.pad_token_id)
            lg = m.slot_logits(*[b[x].cuda() for x in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
            p = torch.softmax(lg.float() / T, -1).nan_to_num(0).cpu()
            for (i, qk, _, gold), row in zip(chunk, p):
                res[(i, qk)] = (int(row.argmax()) == gold, float(row[gold]))
    stats = collections.defaultdict(collections.Counter)
    with open(a.out, "w") as f:
        for i, r in enumerate(recs):
            kept = []
            for k, q in enumerate(r["questions"]):
                ok, pg = res[(i, k)]
                stats[r["family"]]["right" if ok else "wrong"] += 1
                if not ok or rng.random() < a.keep_right:
                    kept.append(dict(q, mined={"model_correct": ok, "p_gold": round(pg, 4)}))
            stats[r["family"]]["kept"] += len(kept)
            if kept:
                f.write(json.dumps(dict(r, questions=kept), ensure_ascii=False, default=str) + "\n")
    tot = collections.Counter()
    for fam, c in sorted(stats.items()):
        tot.update(c)
        print(f"[mine] {fam:28s} model acc {c['right'] / max(1, c['right'] + c['wrong']):.3f}  kept {c['kept']}/{c['right'] + c['wrong']}")
    print(f"[mine] total: model acc {tot['right'] / max(1, tot['right'] + tot['wrong']):.3f}, kept {tot['kept']} of {tot['right'] + tot['wrong']}")


if __name__ == "__main__":
    main()
