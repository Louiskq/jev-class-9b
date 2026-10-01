"""Text data for LoRA B, written by a local Qwen3.8-27B through vLLM (run on a GPU node with ~/venv-vllm; see jobs/textgen.sh).

    python textgen.py seeded   --n 60 --split train --seed 1 --out data/text_seeded_pilot.jsonl
    python textgen.py authored --n 40 --split train --seed 1 --out data/text_authored_pilot.jsonl

seeded   : a code-generated record (hardgen families; the answer is computed) is rewritten by Qwen (thinking off) as a
           realistic document.  Kept only if every number/date of the original survives, no new number appears, and
           Qwen (thinking on) judges the rewrite to state the same facts and conditions.  Choice options get snake_case
           ids with the original text as description (60%) as in JevBench's format.
authored : Qwen (thinking on) writes one document with 3 typed questions from a sampled spec (family, domain,
           document type, length, text or JSON, question types, option counts; distributions from profile_jevbench.py).
           Every question is answered by 3 independent samples (thinking on); kept only if all 3 match the writer.

Output records use the hardgen format (family "text:<family>"), plus an "audit" field with the checks' results.
Held-out split: authored records use the held-out domains; seeded records use hardgen's held-out pools.
"""
import argparse
import collections
import json
import random
import re
import time

MODEL = "/scratch/u6xe/louisk.u6xe/models/Qwen3.8-27B"
TRAIN_DOMAINS = ["customer support for an online retailer", "IT helpdesk", "HR and employment", "finance and accounting",
                 "insurance claims", "banking and lending", "legal and contracts", "procurement and vendor management",
                 "software operations (incidents, deploys, logs)", "education administration", "travel and hospitality",
                 "manufacturing quality control"]
HELDOUT_DOMAINS = ["healthcare administration", "logistics and shipping"]
DOC_TYPES = ["email thread", "support ticket with comments", "internal memo", "policy document with numbered clauses",
             "contract excerpt", "chat transcript", "incident report", "meeting notes", "submitted form",
             "table of records", "system or audit log", "invoice or account statement", "project status update"]
FAMILIES = {  # JevBench's published sealed-set categories; weights favour decider9b's weak ones
    "temporal_numeric": (0.20, "the answer needs date arithmetic, durations, deadlines, business days, time zones, unit "
                               "conversions or multi-step arithmetic over figures in the document"),
    "abstention": (0.15, "some questions cannot be settled from the document (a key fact is missing or two sources "
                         "conflict) and the right answer says so; other questions look incomplete but are still "
                         "determined by what is given"),
    "judge": (0.15, "the document quotes a proposed answer, reply or decision, and the question is whether it is right; "
                    "wrong versions are subtle near misses (one wrong figure, a missed exception, the wrong clause)"),
    "probability": (0.15, "the decision turns on a probability or rate the reader must work out from stated counts, "
                          "frequencies or percentages (base rates, conditional rates, repeated trials, expected cost)"),
    "long_policy": (0.12, "a long policy or contract with numbered clauses, exceptions, carve-outs and a priority rule "
                          "must be applied to one specific case"),
    "multi_hop": (0.08, "the answer needs two or three lookups across different parts of the document (one record "
                        "points to another)"),
    "tradeoff": (0.08, "choose among options under several constraints (budget, deadline, must-haves) where the best "
                       "option is not the most obvious one"),
    "safety": (0.07, "judge whether an action, message or request is safe, appropriate or compliant, where the cases "
                     "are subtle"),
}
SEED_WEIGHTS = {"temporal": 1.0, "probability": 1.0, "ambiguous": 1.0}   # per sub-generator; --seed_weights overrides
LENGTHS = [(0.55, "150 to 300 words"), (0.30, "300 to 900 words"), (0.15, "1,000 to 2,200 words")]
FAMILY_WEIGHTS = {}                                                     # --family_weights overrides
CANNOT_WORDS = ("cannot be determined", "insufficient information", "not enough information", "cannot tell")


# ---- vLLM -------------------------------------------------------------------------------------------------------------
class Qwen:
    def __init__(self, max_len=32768):
        from vllm import LLM
        t = time.time()
        self.llm = LLM(model=MODEL, max_model_len=max_len, gpu_memory_utilization=0.92, max_num_seqs=256,
                       trust_remote_code=True)
        self.tokens = collections.Counter()
        print(f"[qwen] loaded in {time.time() - t:.0f}s", flush=True)

    def chat(self, prompts, think, n=1, temperature=0.7, max_tokens=8192, tag=""):
        from vllm import SamplingParams
        t = time.time()
        sp = SamplingParams(n=n, temperature=temperature, top_p=0.95, top_k=20, max_tokens=max_tokens)
        outs = self.llm.chat([[{"role": "user", "content": p}] for p in prompts], sp,
                             chat_template_kwargs={"enable_thinking": think}, use_tqdm=False)
        gen = sum(len(c.token_ids) for o in outs for c in o.outputs)
        self.tokens[tag] += gen
        print(f"[qwen] {tag}: {len(prompts)} prompts x{n}, {gen} tokens in {time.time() - t:.0f}s "
              f"({gen / max(time.time() - t, 1e-9):.0f} tok/s)", flush=True)
        return [[c.text for c in o.outputs] for o in outs]


def after_think(text):
    return text.split("</think>")[-1].strip()


def parse_json(text):
    t = after_think(text)
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    try:
        return json.loads(t)
    except ValueError:
        i, j = t.find("{"), t.rfind("}")
        return json.loads(t[i:j + 1]) if 0 <= i < j else None


def final_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", after_think(text))
    return m[-1].strip().strip("`*'\". ") if m else None


def snake(s):
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:48] or "option"


# ---- question rendering and checking ---------------------------------------------------------------------------------
def question_block(q):
    t, c = q["type"], q.get("criteria")
    if t == "choice":
        opts = "\n".join(f"- {k}" + (f": {v}" if v else "") for k, v in c.items())
        how = "the option id exactly as written"
    elif t == "noul":
        desc = "" if not c else f"\n- yes: {c.get('true') or ''}\n- no: {c.get('false') or ''}"
        opts, how = f"Answer yes or no.{desc}", "yes or no"
    else:
        opts = "\n".join(f"- {i}: {v}" for i, v in enumerate(c))
        how = "the level number"
    return f"Question: {q['instructions']}\nOptions:\n{opts}", how


def checker_prompt(state, q):
    body, how = question_block(q)
    st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    return (f"Read the document and answer the question. Use only the information in the document. If the document does "
            f"not settle the question and an option says so, choose that option.\n\nDocument:\n{st}\n\n{body}\n\n"
            f"Think it through carefully, then give your final answer on the last line as:\nANSWER: <{how}>")


def normalise(q, ans):
    if ans is None:
        return None
    a = ans.strip().lower()
    if q["type"] == "noul":
        return True if a.startswith("yes") else False if a.startswith("no") else None
    if q["type"] == "score":
        m = re.match(r"\d+", a)
        return int(m.group()) if m else None
    for k in q["criteria"]:
        if a == k.lower() or a.startswith(k.lower() + " ") or a.startswith(k.lower() + ":"):
            return k
    return None


def valid(q):
    t, c, a = q.get("type"), q.get("criteria"), q.get("answer")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        return False
    if t == "choice":
        return isinstance(c, dict) and 2 <= len(c) <= 255 and a in c
    if t == "noul":
        return isinstance(a, bool) and (c is None or isinstance(c, dict))
    if t == "score":
        return isinstance(c, list) and 2 <= len(c) <= 10 and isinstance(a, int) and 0 <= a < len(c)
    return False


def agree(qwen, items, k=3):
    """items: [(state, question)] -> list of (all_k_agree, [answers])"""
    outs = qwen.chat([checker_prompt(s, q) for s, q in items], think=True, n=k, temperature=0.7, tag="check")
    res = []
    for (s, q), texts in zip(items, outs):
        got = [normalise(q, final_answer(t)) for t in texts]
        res.append((all(g == q["answer"] for g in got), got))
    return res


# ---- authored mode ----------------------------------------------------------------------------------------------------
def spec(rng, split):
    fam = rng.choices(list(FAMILIES), [FAMILY_WEIGHTS.get(f, w) for f, (w, _) in FAMILIES.items()])[0]
    types = rng.choices(["choice", "noul", "score"], [60, 34, 6], k=3)
    return dict(family=fam, brief=FAMILIES[fam][1], domain=rng.choice(TRAIN_DOMAINS if split == "train" else HELDOUT_DOMAINS),
                doc=rng.choice(DOC_TYPES), length=rng.choices([l for _, l in LENGTHS], [w for w, _ in LENGTHS])[0],
                json_state=rng.random() < 0.32, types=types, n_opts=[rng.choice([3, 4, 4, 5, 5, 6]) for _ in types],
                snake=rng.random() < 0.6)


def writer_prompt(sp):
    qdesc = []
    for i, (t, k) in enumerate(zip(sp["types"], sp["n_opts"]), 1):
        if t == "choice":
            qdesc.append(f"{i}. a choice question with exactly {k} options, every option with a one-sentence description"
                         + (", option ids in snake_case" if sp["snake"] else ", option ids as short plain phrases"))
        elif t == "noul":
            qdesc.append(f"{i}. a yes/no question")
        else:
            qdesc.append(f"{i}. a score question with 3 to 5 ordered levels, each level described")
    state_form = ("a JSON object (realistic field names; nested records where natural)" if sp["json_state"]
                  else "plain text")
    return f"""You are writing test material for a decision model that reads a document and answers typed questions.

Write one realistic, self-contained document for this setting:
- domain: {sp['domain']}
- document type: {sp['doc']}
- length: {sp['length']}
- format of the document: {state_form}
- what makes it hard: {sp['brief']}

Then write 3 different questions about the document:
{chr(10).join(qdesc)}

Requirements:
- Invent all people, organisations and figures; no real companies or people.
- For every question, exactly one answer must follow from the document for a careful reader, with no outside knowledge
  beyond common sense and ordinary arithmetic.
- Wrong options must be plausible near misses that a careless reader would pick (wrong figure, missed exception,
  wrong record, off by one day).
- Include distracting but irrelevant details in the document.
- Only offer an option such as "cannot be determined from the document" when it fits the question; make it the correct
  answer only when the document genuinely does not settle the question.
- Balance the answers: yes/no questions should not all be "yes"; the correct choice option should not always be first.
- Do not mention the questions, the answers or these instructions in the document.

Return only a JSON object, with no other text:
{{"state": <the document: a string, or a JSON object if the format above is JSON>,
  "questions": [
    {{"type": "choice", "instructions": "<question>", "criteria": {{"<option_id>": "<description>", ...}}, "answer": "<option_id>", "why": "<one sentence>"}},
    {{"type": "noul", "instructions": "<question>", "criteria": null, "answer": true, "why": "<one sentence>"}},
    {{"type": "score", "instructions": "<question>", "criteria": ["<level 0>", "<level 1>", "..."], "answer": <level index>, "why": "<one sentence>"}}
  ]}}
(use the question types listed above, in that order)"""


def authored(qwen, n, split, seed):
    rng = random.Random(seed)
    specs = [spec(rng, split) for _ in range(n)]
    outs = qwen.chat([writer_prompt(s) for s in specs], think=True, temperature=0.8, max_tokens=28000, tag="write")
    recs, items, stats = [], [], collections.Counter()
    for i, (sp, texts) in enumerate(zip(specs, outs)):
        try:
            d = parse_json(texts[0])
        except ValueError:
            d = None
        if not isinstance(d, dict) or not isinstance(d.get("state"), (str, dict, list)) or not isinstance(d.get("questions"), list):
            stats["writer_truncated" if "</think>" not in texts[0] else "writer_unparsable"] += 1; continue
        qs = [q for q in d["questions"] if isinstance(q, dict) and valid(q)]
        stats["questions_written"] += len(d["questions"]); stats["questions_valid"] += len(qs)
        rec = dict(family=f"text:{sp['family']}", sub="authored", split=split, domain=sp["domain"], state=d["state"],
                   questions=qs, spec={k: sp[k] for k in ("doc", "length", "json_state", "snake")}, id=f"authored/{split}/{seed}/{i}")
        recs.append(rec); items += [(rec, q) for q in qs]
    res = agree(qwen, [(r["state"], q) for r, q in items])
    kept = collections.defaultdict(list)
    for (rec, q), (ok, got) in zip(items, res):
        q["audit"] = {"checks": [str(g) for g in got], "kept": ok}
        stats["questions_kept" if ok else "questions_dropped"] += 1
        if ok:
            kept[rec["id"]].append({k: v for k, v in q.items() if k in ("type", "instructions", "criteria", "answer", "why")})
    out = []
    for rec in recs:
        if kept[rec["id"]]:
            out.append(dict(rec, questions=kept[rec["id"]]))
    return out, stats


# ---- seeded mode ------------------------------------------------------------------------------------------------------
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text):
    return {m.replace(",", "") for m in NUM.findall(text)}


def seed_records(rng, split, n):
    import ambiguous, probability, temporal
    pool = [(temporal.SUBS, s, "temporal") for s in ("invoice", "renewal", "sla", "age", "budget", "units")] + \
           [(probability.SUBS, s, "probability") for s in probability.SUBS] + [(ambiguous.SUBS, "age_partial", "ambiguous")]
    weights = [SEED_WEIGHTS[k] for _, _, k in pool]
    out = []
    while len(out) < n:
        subs, name, _ = rng.choices(pool, weights)[0]
        r = subs[name](rng, split)
        if isinstance(r["state"], str) and len(r["state"].split()) < 400:
            out.append(r)
    return out


def rewrite_prompt(rec, doc):
    return f"""Rewrite the facts below as a realistic {doc} from the field of {rec['domain']}.

Rules:
- Keep every fact, rule and condition with exactly the same meaning.
- Copy every number, date, amount and percentage exactly as written (same format).
- Do not add any new numbers, dates, amounts, percentages or reference codes.
- You may add realistic context, greetings, names of people already mentioned, and irrelevant but harmless detail.
- Do not state, compute or hint at any conclusion that is not already stated in the facts.

Facts:
{rec['state']}

Return only the rewritten document."""


def same_facts_prompt(a, b):
    return f"""Document A:
{a}

Document B:
{b}

Does document B state every fact, number, rule and condition of document A with the same meaning, without adding
anything that could change a conclusion drawn from A? Minor wording, extra greetings and irrelevant detail are fine.
Think it through, then answer on the last line as:
ANSWER: yes or no"""


def restyle_options(rng, q):
    if q["type"] != "choice" or rng.random() >= 0.6:
        return q
    ids, crit = {}, {}
    for k in q["criteria"]:
        i = snake(k)
        while i in crit:
            i += "_x"
        ids[k] = i; crit[i] = k
    return dict(q, criteria=crit, answer=ids[q["answer"]])


def seeded(qwen, n, split, seed):
    rng = random.Random(seed)
    base = seed_records(rng, split, n)
    docs = [rng.choice(DOC_TYPES[:3] + DOC_TYPES[5:9] + DOC_TYPES[11:]) for _ in base]
    outs = qwen.chat([rewrite_prompt(r, d) for r, d in zip(base, docs)], think=False, temperature=0.8, max_tokens=2048,
                     tag="rewrite")
    stats, cand = collections.Counter(), []
    for r, d, texts in zip(base, docs, outs):
        new = after_think(texts[0]).strip()
        orig = numbers(r["state"]); got = numbers(new)
        if not orig <= got:
            stats["dropped_lost_number"] += 1; continue
        if got - orig:
            stats["dropped_new_number"] += 1; continue
        cand.append((r, d, new))
    checks = qwen.chat([same_facts_prompt(r["state"], new) for r, d, new in cand], think=True, temperature=0.6,
                       max_tokens=6144, tag="same_facts")
    out = []
    for (r, d, new), texts in zip(cand, checks):
        ok = (final_answer(texts[0]) or "").lower().startswith("yes")
        stats["kept" if ok else "dropped_not_same"] += 1
        if ok:
            out.append(dict(r, family=f"text:{r['family']}", sub=f"seeded:{r['sub']}", state=new,
                            questions=[restyle_options(rng, q) for q in r["questions"]],
                            spec={"doc": d}, audit={"original_state": r["state"]},
                            id=f"seeded/{split}/{seed}/{len(out)}"))
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["seeded", "authored"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family_weights", default="", help="authored: family=weight,... (overrides FAMILIES weights)")
    ap.add_argument("--seed_weights", default="", help="seeded: temporal=w,probability=w,ambiguous=w (per sub-generator)")
    a = ap.parse_args()
    parse_w = lambda t: {k: float(v) for k, v in (x.split("=") for x in t.split(",") if x)}
    FAMILY_WEIGHTS.update(parse_w(a.family_weights)); SEED_WEIGHTS.update(parse_w(a.seed_weights))
    print(f"[textgen] family weights {FAMILY_WEIGHTS or 'default'}; seed weights {SEED_WEIGHTS}")
    qwen = Qwen(); t0 = time.time()
    recs, stats = (seeded if a.mode == "seeded" else authored)(qwen, a.n, a.split, a.seed)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    nq = sum(len(r["questions"]) for r in recs)
    print(f"[textgen] {a.mode}: {len(recs)} records, {nq} questions kept in {time.time() - t0:.0f}s; {dict(stats)}")
    print(f"[textgen] generated tokens by step: {dict(qwen.tokens)}")


if __name__ == "__main__":
    main()
