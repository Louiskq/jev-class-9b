"""judge_hard: judge whether a proposed answer is right, when the wrong ones are near misses.

Built on the other families' engines: a generated question keeps its computed answer, and the proposed answers are that
answer or one of the question's mistake-derived distractors (off by a day, the wrong clause, the wrong band).  The judge
only sees the task and the proposal, as a grader would.
"""
import json

from common import an, choice, noul, person, pool, record
import policy
import probability
import temporal

FAMILY = "judge_hard"
BASE = [(temporal.SUBS, ["invoice", "renewal", "sla", "age", "timezones", "budget", "ledger"]),
        (probability.SUBS, ["draws", "redundancy", "base_rate", "conditional", "at_least_one"]),
        (policy.SUBS, ["policy_case"])]


def _task(rng, split):
    """A generated record with one choice question (the judge needs wrong answers to offer)."""
    for _ in range(50):
        subs, names = rng.choice(BASE)
        rec = subs[rng.choice(names)](rng, split)
        qs = [q for q in rec["questions"] if q["type"] == "choice" and len(q["criteria"]) >= 3]
        if qs:
            return rec, rng.choice(qs)
    raise RuntimeError("no choice question generated")


def _state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def verify(rng, split):
    rec, q = _task(rng, split)
    right = rng.random() < 0.5
    wrong = [o for o in q["criteria"] if o != q["answer"] and o != "cannot be determined from the information given"]
    prop = q["answer"] if right or not wrong else rng.choice(wrong)
    who = rng.choice(pool(["a junior analyst", "an automated assistant", "a colleague", "a contractor", "the previous shift",
                           "an intern"], split))
    state = (f"Task given to {who}:\n{_state_text(rec['state'])}\n\nQuestion: {q['instructions']}\n"
             f"Proposed answer: {prop}")
    ask = rng.choice(pool(["Is the proposed answer correct?", "Should the proposed answer be accepted as correct?",
                           "Does the proposed answer match what the information supports?",
                           "Would a careful reviewer mark the proposed answer as right?"], split))
    return record(FAMILY, "verify", split, rec["domain"], state, [noul(ask, prop == q["answer"])], base=rec["family"])


def pairwise(rng, split):
    rec, q = _task(rng, split)
    wrong = [o for o in q["criteria"] if o != q["answer"] and o != "cannot be determined from the information given"]
    kind = rng.choices(["one right", "both right", "neither right"], [6, 1, 2])[0]
    if kind == "one right":
        pair = [q["answer"], rng.choice(wrong)]; rng.shuffle(pair)
        ans = "answer A" if pair[0] == q["answer"] else "answer B"
    elif kind == "both right":
        pair = [q["answer"], q["answer"]]; ans = "both are correct"
    else:
        pair = rng.sample(wrong, 2) if len(wrong) >= 2 else [wrong[0], wrong[0]]; ans = "neither is correct"
    state = (f"{_state_text(rec['state'])}\n\nQuestion: {q['instructions']}\n"
             f"Answer A: {pair[0]}\nAnswer B: {pair[1]}")
    return record(FAMILY, "pairwise", split, rec["domain"], state,
                  [choice("Which of the two answers is correct?", ["answer A", "answer B", "both are correct",
                                                                   "neither is correct"], ans, rng, shuffle=False)],
                  base=rec["family"])


def reply_review(rng, split):
    """A support reply that states a decision on a case under the long policy; does it follow the policy?"""
    d = rng.choice(pool(policy.DOMAINS, split))
    P = policy.build_policy(rng, d)
    text, num, word = policy.render(rng, d, P)
    c = policy.make_case(rng, d, P, split)
    out, key, trig, pay = policy.decide(P, c)
    right = rng.random() < 0.5
    said = out if right else rng.choice([o for o in policy.OUTCOMES if o != out])
    agent = person(rng, split).split()[0]
    reason = {"approve in full": "everything is in order", "approve up to the cap": "the amount is above the category cap",
              "escalate for approval": "it needs sign-off first", "request more information": "we need proof of the cost",
              "reject": "it falls outside what the policy covers"}[said]
    reply = (f"Hi {c['name'].split()[0]}, thanks for your {d['req']}. I have checked it against the policy and the decision "
             f"is: {said}, because {reason}. Best wishes, {agent}")
    ct = policy.case_text(rng, d, c)
    state = {"policy": text, "case": ct, "reply_sent": reply} if isinstance(ct, dict) else \
        f"{text}\n\nCase:\n{ct}\n\nReply sent to the {d['payee']}:\n{reply}"
    ask = rng.choice(pool(["Does the decision stated in the reply follow the policy?",
                           "Is the decision in the reply the one the policy requires?",
                           "Did the agent reach the correct decision under the policy?",
                           "Would an auditor agree with the decision given in the reply?"], split))
    return record(FAMILY, "reply_review", split, d["name"], state, [noul(ask, said == out)], base="long_policy")


SUBS = {"verify": verify, "pairwise": pairwise, "reply_review": reply_review}
