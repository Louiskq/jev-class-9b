"""Random subset of JSONL records, by question count, with a fixed seed (for the scale and code-share ablations).

    python subsample.py --n_questions 2400 --seed 0 --out data/x.jsonl data/a.jsonl data/b.jsonl
Records are shuffled, then kept whole until the question budget is reached (the last record may overshoot slightly).
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--n_questions", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
recs = [json.loads(l) for f in a.files for l in open(f) if l.strip()]
random.Random(a.seed).shuffle(recs)
out, nq = [], 0
for r in recs:
    if nq >= a.n_questions:
        break
    out.append(r); nq += len(r["questions"])
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
print(f"[subsample] {len(out)} of {len(recs)} records, {nq} questions -> {a.out}")
