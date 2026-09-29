"""Write generated records as JSONL.

    python generate.py --family temporal_numeric --split train --n 3000 --seed 1 --out data/temporal_train.jsonl
    python generate.py --family all --split heldout --n 300 --seed 101 --out data/heldout.jsonl   # n per family

Sub-generators of a family are used in turn, so every sub is equally represented.  Train and held-out records use
different seeds and disjoint name / organisation / domain pools (common.pool).
"""
import argparse
import json
import random
import sys

import ambiguous
import judge
import policy
import probability
import temporal

FAMILIES = {m.FAMILY: m.SUBS for m in (temporal, probability, policy, judge, ambiguous)}


def generate(family, split, n, seed):
    rng = random.Random(seed); subs = list(FAMILIES[family].items()); out = []
    for i in range(n):
        name, fn = subs[i % len(subs)]
        rec = fn(rng, split)
        rec["id"] = f"{family}/{split}/{seed}/{i}"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help=f"one of {sorted(FAMILIES)} or 'all'")
    ap.add_argument("--split", choices=["train", "heldout"], required=True)
    ap.add_argument("--n", type=int, required=True, help="records per family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    fams = sorted(FAMILIES) if a.family == "all" else [a.family]
    f = sys.stdout if a.out == "-" else open(a.out, "w")
    for k, fam in enumerate(fams):
        for rec in generate(fam, a.split, a.n, a.seed * 1000 + k):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
