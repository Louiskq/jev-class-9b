"""Public reasoning datasets -> hardgen records (TypeSafe wire format), for LoRA B5.  Run from the decider repo with its venv:

    HF_HOME=/projects/u6xe/louisk/hf .venv/bin/python /projects/u6xe/louisk/hardgen/public.py --out_dir /projects/u6xe/louisk/hardgen/data

  gsm8k        (MIT)        word problems; options = the answer + intermediate results of the worked solution (the
                            numbers a careless reader stops at) + fillers, the answer at a random rank
  proofwriter  (CC BY 4.0)  open-world rule reasoning, 2-5 proof steps: true / false / unknown, balanced
  bbh          (MIT)        BIG-Bench Hard tasks with a fixed answer set (dates, temporal order, deduction, tracking,
                            causal judgement, web of lies, navigation, formal fallacies, tables, colored objects, ...)
  musr         (CC BY 4.0)  long narratives: murder mysteries, object placements, team allocation
10% of every dataset (by seeded shuffle) is held out for evaluation only.
"""
import argparse
import ast
import collections
import json
import random
import re

from datasets import load_dataset

BBH_TASKS = ["date_understanding", "temporal_sequences", "logical_deduction_three_objects", "logical_deduction_five_objects",
             "logical_deduction_seven_objects", "tracking_shuffled_objects_three_objects",
             "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "causal_judgement",
             "web_of_lies", "navigate", "formal_fallacies", "penguins_in_a_table", "reasoning_about_colored_objects",
             "disambiguation_qa"]


def rec(family, sub, split, domain, state, questions, i):
    return {"family": f"public:{family}", "sub": sub, "split": split, "domain": domain, "state": state,
            "questions": questions, "id": f"public/{family}/{sub}/{i}"}


def choice(instructions, names, answer, rng, shuffle=True):
    names = list(dict.fromkeys(names))
    assert answer in names and len(names) >= 2, (answer, names)
    if shuffle:
        rng.shuffle(names)
    return {"type": "choice", "instructions": instructions, "criteria": {n: None for n in names}, "answer": answer}


def num(s):
    s = s.replace(",", "").strip()
    try:
        v = float(s)
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def gsm8k(rng, n):
    out = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        ans = num(r["answer"].split("####")[-1])
        if ans is None or not isinstance(ans, int):
            continue
        steps = [num(x) for x in re.findall(r"=\s*<<[^>]*=([-\d.,]+)>>", r["answer"])]
        wrong = [s for s in steps if isinstance(s, int) and s != ans]
        step = max(1, abs(ans) // 10) if abs(ans) >= 20 else 1
        below = list(dict.fromkeys([w for w in wrong if w < ans] + [ans - k * step for k in range(1, 8)]))
        above = list(dict.fromkeys([w for w in wrong if w > ans] + [ans + k * step for k in range(1, 8)]))
        below = [b for b in below if b >= 0] if ans >= 0 else below
        r_ = min(rng.randrange(4), len(below)); r_ = max(r_, 3 - len(above))
        opts = [ans] + below[:r_] + above[:3 - r_]
        q = choice("What is the answer to the question in the problem?", [str(o) for o in opts], str(ans), rng)
        out.append(rec("numeric", "gsm8k", None, "math word problems", r["question"], [q], i))
    rng.shuffle(out)
    return out[:n]


def proofwriter(rng, n):
    want = {"True": n // 3, "False": n // 3, "Unknown": n - 2 * (n // 3)}; got = collections.Counter(); out = []
    ds = load_dataset("tasksource/proofwriter", split="train", streaming=True).shuffle(seed=0, buffer_size=20000)
    opts = {"true": "the statement follows from the facts and rules", "false": "the opposite of the statement follows",
            "unknown": "the facts and rules do not settle it"}
    for i, r in enumerate(ds):
        a = r["answer"]
        if "OWA" not in r["id"] or not 2 <= int(r["QDep"]) <= 5 or a not in want or got[a] >= want[a]:
            continue
        got[a] += 1
        q = {"type": "choice", "instructions": f"Statement: \"{r['question']}\" Based only on the facts and rules above, is "
                                               f"the statement true, false, or unknown?",
             "criteria": dict(opts), "answer": a.lower()}
        out.append(rec("rules", f"proofwriter_d{r['QDep']}", None, "rule reasoning", r["theory"], [q], i))
        if sum(got.values()) >= n:
            break
    return out


def bbh(rng):
    out = []
    for task in BBH_TASKS:
        for i, r in enumerate(load_dataset("lukaemon/bbh", task, split="test")):
            text, target = r["input"], r["target"].strip()
            if "\nOptions:\n" in text:
                body, opt = text.split("\nOptions:\n", 1)
                lines = [l.strip() for l in opt.strip().split("\n") if l.strip()]
                if lines[0].startswith("("):
                    names = {l[:3]: l[3:].strip() for l in lines}                     # "(A) text"
                    if target not in names:
                        continue
                    ans = names[target]
                    if len(set(names.values())) < len(names):
                        continue
                    q = choice("Which option is the correct answer to the question?", list(names.values()), ans, rng, shuffle=False)
                else:
                    names = [l.lstrip("- ").strip() for l in lines]                   # "- Yes" / "- valid"
                    if target not in names:
                        continue
                    q = choice("Which option is the correct answer to the question?", names, target, rng, shuffle=False)
            elif target in ("Yes", "No"):
                body = text
                q = choice("What is the answer to the question?", ["Yes", "No"], target, rng, shuffle=False)
            else:
                continue
            out.append(rec("bbh", task, None, "reasoning puzzles", body.strip(), [q], i))
    return out


def musr(rng):
    out = []
    for split in ("murder_mysteries", "object_placements", "team_allocation"):
        for i, r in enumerate(load_dataset("TAUR-Lab/MuSR", split=split)):
            names = ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else list(r["choices"])
            ans = names[int(r["answer_index"])]
            if len(set(names)) < len(names):
                continue
            q = choice(r["question"], names, ans, rng, shuffle=False)
            out.append(rec("musr", split, None, "narratives", r["narrative"], [q], i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True); ap.add_argument("--gsm8k", type=int, default=2200)
    ap.add_argument("--proofwriter", type=int, default=2750); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = random.Random(a.seed)
    allr = {"gsm8k": gsm8k(rng, a.gsm8k), "proofwriter": proofwriter(rng, a.proofwriter), "bbh": bbh(rng), "musr": musr(rng)}
    tr, ho = open(f"{a.out_dir}/public_train.jsonl", "w"), open(f"{a.out_dir}/public_heldout.jsonl", "w")
    for name, recs in allr.items():
        rng.shuffle(recs); k = max(1, len(recs) // 10)
        for j, r in enumerate(recs):
            r["split"] = "heldout" if j < k else "train"
            (ho if j < k else tr).write(json.dumps(r, ensure_ascii=False) + "\n")
        subs = collections.Counter(r["sub"] for r in recs)
        ans = collections.Counter(str(r["questions"][0]["answer"]) for r in recs) if name == "proofwriter" else ""
        print(f"[public] {name}: {len(recs)} records ({len(recs) - k} train / {k} held out); subs {dict(subs)} {ans}")


if __name__ == "__main__":
    main()
