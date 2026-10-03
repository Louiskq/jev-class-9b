"""Cap questions per family in a record file (whole records are sampled):  python rebalance.py IN OUT --cap 800"""
import argparse, collections, json, random

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--cap", type=int, default=800)
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
recs = [json.loads(l) for l in open(a.inp) if l.strip()]; random.Random(a.seed).shuffle(recs)
have, out = collections.Counter(), []
for r in recs:
    if have[r["family"]] < a.cap:
        out.append(r); have[r["family"]] += len(r["questions"])
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("[rebalance]", sum(have.values()), "questions:", dict(sorted(have.items())))
