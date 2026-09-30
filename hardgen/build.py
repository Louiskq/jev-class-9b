"""Generated records (JSONL) -> a decider cache (train, evals) for decider.train and decider.evaluate.

Run from the decider repo with its venv (it imports decider):
    cd /projects/u6xe/louisk/decider && .venv/bin/python /projects/u6xe/louisk/hardgen/build.py \
        --train ../hardgen/data/train.jsonl --heldout ../hardgen/data/heldout.jsonl \
        --replay data/mixture_full.pkl --replay_per_task 100 --out ../hardgen/data/stage1.pkl

Records go through decider.data.teacher_questions.to_example, i.e. decider.systemone's rendering of the /v1/systemone wire
format.  Training records are kept packed or split into one example per question (the server scores every question in its
own row); Score questions also get their isolated-level rows (decider.data.augment.isolated), as served with
"isolated_levels": true.  Held-out records become one example per question, one eval task per family/sub.
Replay: up to --replay_per_task examples of every task in the original mixture's training half, so the stage does not
move the model away from what it already does well.
"""
import argparse
import collections
import json
import pickle
import random

from decider import data as D
from decider import systemone as S1
from decider.data.augment import isolated
from decider.data.teacher_questions import to_example


def read(paths):
    out = []
    for p in paths:
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out


def snake(t):
    import re
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")[:48] or "option"


def restyle(rec, rng, share=0.6):
    """JevBench's option format for code records: snake_case ids with the original option text as the description."""
    qs = []
    for q in rec["questions"]:
        if q["type"] == "choice" and rng.random() < share:
            ids, crit = {}, {}
            for k in q["criteria"]:
                i = snake(k)
                while i in crit:
                    i += "_x"
                ids[k] = i; crit[i] = k if q["criteria"][k] in (None, "") else f"{k}: {q['criteria'][k]}"
            q = dict(q, criteria=crit, answer=ids[q["answer"]])
        qs.append(q)
    return dict(rec, questions=qs)


def train_examples(rec, rng, single_prob):
    ex = to_example(rec, D, S1, task=f"hard:{rec['family']}")
    rows = [D.Example(ex.context, [q], ex.task) for q in ex.qs] if len(ex.qs) > 1 and rng.random() < single_prob else [ex]
    for q, spec in zip(ex.qs, rec["questions"]):
        if spec["type"] == "score" and rng.random() < 0.5:
            rows += isolated(ex, q, ex.task + ":isolated")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--replay", default="")
    ap.add_argument("--replay_per_task", type=int, default=100)
    ap.add_argument("--single_prob", type=float, default=0.6)
    ap.add_argument("--restyle_code", action="store_true", help="JevBench option format for code (non-text) records")
    ap.add_argument("--min_questions", type=int, default=0, help="abort unless the train files hold this many questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = random.Random(a.seed)

    train, recs = [], read(a.train)
    nq = sum(len(r["questions"]) for r in recs)
    print(f"[build] {len(recs)} train records, {nq} questions: "
          f"{dict(collections.Counter(r['family'] for r in recs))}")
    if nq < a.min_questions:
        raise SystemExit(f"[build] only {nq} questions, fewer than --min_questions {a.min_questions}")
    for rec in recs:
        assert rec["split"] == "train", rec["id"]
        if a.restyle_code and not rec["family"].startswith("text:"):
            rec = restyle(rec, rng)
        train += train_examples(rec, rng, a.single_prob)
    n_new = len(train)
    evals = collections.defaultdict(list)
    for rec in read(a.heldout):
        assert rec["split"] == "heldout", rec["id"]
        ex = to_example(rec, D, S1)
        for q in ex.qs:
            evals[f"hard:{rec['family']}/{rec['sub']}"].append(D.Example(ex.context, [q], f"hard:{rec['family']}/{rec['sub']}"))
    n_replay = 0
    if a.replay:
        base, _ = D.load_cache(a.replay)
        by = collections.defaultdict(list)
        for e in base:
            by[e.task].append(e)
        for t in sorted(by):
            pick = rng.sample(by[t], min(a.replay_per_task, len(by[t])))
            train += pick; n_replay += len(pick)
        print(f"[build] replay: {n_replay} examples from {len(by)} tasks")
    rng.shuffle(train)
    print(f"[build] train: {n_new} new + {n_replay} replay = {len(train)}; eval tasks: {len(evals)} "
          f"({sum(len(v) for v in evals.values())} questions)")
    for t, v in sorted(evals.items()):
        print(f"[build]   {t:40s} {len(v)}")
    with open(a.out, "wb") as f:
        pickle.dump((train, dict(evals)), f)


if __name__ == "__main__":
    main()
