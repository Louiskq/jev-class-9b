"""Aggregate format profile of JevBench public items - numbers only, never item text, option names or rationales.
    python profile_jevbench.py /projects/u6xe/louisk/jevbench/datasets/public/hard.jsonl [...]"""
import collections
import json
import re
import sys


def q(xs, ps=(0.1, 0.5, 0.9, 1.0)):
    xs = sorted(xs)
    return " ".join(f"p{int(p * 100)}={xs[min(len(xs) - 1, int(p * len(xs)))]}" for p in ps) if xs else "-"


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    st = [r["state"] for r in rows]
    qs = [r["question"] for r in rows]
    crit = [q_.get("criteria") for q_ in qs]
    n_opts = [len(c) if isinstance(c, (dict, list)) else 0 for c in crit]
    described = [sum(1 for v in c.values() if v) / len(c) for c in crit if isinstance(c, dict) and c]
    snake = [sum(1 for k in c if re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)+", k)) / len(c) for c in crit if isinstance(c, dict) and c]
    print(f"== {path.split('/')[-1]}: {len(rows)} items")
    print("state type:", dict(collections.Counter(type(s).__name__ for s in st)),
          "| looks like JSON:", sum(1 for s in st if isinstance(s, str) and s.lstrip()[:1] in "{["))
    print("state words:", q([len(str(s).split()) for s in st]))
    print("approx_state_tokens:", q([r.get("provenance", {}).get("approx_state_tokens", 0) for r in rows]))
    print("question type:", dict(collections.Counter(q_.get("type") for q_ in qs)))
    print("options per question:", dict(sorted(collections.Counter(n_opts).items())))
    print("share of options with a description: mean", round(sum(described) / max(len(described), 1), 2),
          "| share of option names in snake_case: mean", round(sum(snake) / max(len(snake), 1), 2))
    print("instruction words:", q([len(str(q_.get("instructions", "")).split()) for q_ in qs]))
    print("labels per item:", dict(collections.Counter(len(r.get("labels", [])) for r in rows)))
    for k in ("author_model", "label_basis", "license", "source"):
        print(f"provenance.{k}:", dict(collections.Counter(str(r.get("provenance", {}).get(k)) for r in rows).most_common(6)))
    print()
