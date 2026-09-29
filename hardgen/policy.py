"""long_policy: apply a long policy document (numbered clauses, exceptions, look-alike clauses, boilerplate) to one case.

A small rule engine decides every answer.  The document mixes the clauses that matter with clauses for other roles and
regions, states exceptions ("clause 4 does not apply to ...") and one explicit priority rule, and surrounds them with
purpose, definitions, responsibilities and complaints sections.  Case values often sit exactly on a threshold, where
"more than" and "at least" differ.
"""
from common import an, choice, money, noul, person, pool, ranked_options, record

FAMILY = "long_policy"
OUTCOMES = ["approve in full", "approve up to the cap", "escalate for approval", "request more information", "reject"]

DOMAINS = [
    dict(name="employee expense reimbursement", req="expense claim", event="date of the expense", payee="employee",
         cats=["meals", "hotel", "taxi", "train", "conference fees", "software", "client gifts", "flights"],
         roles=["employee", "contractor", "manager", "intern"], regions=["UK", "Germany", "US", "Japan", "Brazil"],
         approver="the finance director"),
    dict(name="customer refund policy", req="refund request", event="delivery date", payee="customer",
         cats=["electronics", "clothing", "furniture", "groceries", "gift cards", "software licences", "books"],
         roles=["standard member", "premium member", "business account", "guest shopper"],
         regions=["EU", "UK", "US", "Canada", "Australia"], approver="a customer service supervisor"),
    dict(name="research grant claims", req="grant claim", event="purchase date", payee="researcher",
         cats=["lab consumables", "equipment", "travel", "publication fees", "participant payments", "catering"],
         roles=["principal investigator", "postdoc", "PhD student", "visiting researcher"],
         regions=["UK", "EU", "North America", "Asia"], approver="the grants office"),
    dict(name="insurance travel claims", req="travel claim", event="date of the incident", payee="policyholder",
         cats=["lost baggage", "flight delay", "medical treatment", "trip cancellation", "stolen phone", "rental car damage"],
         roles=["single-trip policyholder", "annual policyholder", "family policyholder", "business policyholder"],
         regions=["Europe", "worldwide excluding US", "worldwide including US", "domestic"], approver="a senior claims handler"),
    dict(name="equipment loan scheme", req="loan request", event="request date", payee="borrower",
         cats=["laptops", "cameras", "projectors", "VR headsets", "tablets", "microphones"],
         roles=["staff member", "student", "alumnus", "external partner"], regions=["main campus", "city campus", "online"],
         approver="the equipment manager"),
    dict(name="overtime pay policy", req="overtime claim", event="date the overtime was worked", payee="employee",
         cats=["weekday evening", "weekend", "public holiday", "on-call", "night shift"],
         roles=["hourly employee", "salaried employee", "team lead", "temporary worker"],
         regions=["head office", "warehouse", "retail store", "remote"], approver="the HR business partner"),
]

BOILERPLATE = [
    "This policy is reviewed every twelve months by the policy owner.",
    "Personal data submitted with {a_req} is kept for seven years and then deleted.",
    "Questions about this policy should be sent to the policy mailbox.",
    "Where this policy refers to days, it means calendar days.",
    "Nothing in this policy limits any statutory right.",
    "The policy owner may publish guidance notes; guidance notes do not change the rules in this policy.",
    "Records of every {req} are open to internal audit.",
    "This version replaces all earlier versions of the policy.",
    "Deliberately false information in {a_req} is a disciplinary matter and is handled outside this policy.",
    "Headings are for convenience only and do not affect interpretation.",
    "Amounts in this policy are in the currency of the {payee}'s home account.",
    "{A_req} can be withdrawn at any time before a decision is made.",
    "Decisions are communicated in writing within ten working days.",
    "The policy owner keeps a public list of frequently asked questions.",
]
PURPOSE = ["This policy sets out how {a_req} is assessed and paid.", "It aims to treat every {payee} consistently and fairly.",
           "It also protects the organisation against paying for costs it has not agreed to cover.",
           "The rules below are applied in the same way whichever channel {a_req} arrives through.",
           "Where the policy is silent, the policy owner decides in the spirit of these rules.",
           "The policy was drafted after consultation with representatives of the people it affects.",
           "It should be read together with the organisation's code of conduct.",
           "Staff who assess claims receive training on this policy each year."]
DEFS = ["'{Req}' means a request for payment made under this policy.", "'{Payee}' means the person who makes {a_req}.",
        "'Receipt' means an itemised document from the supplier showing the amount paid.",
        "'Prior written approval' means approval given in writing before the cost was incurred.",
        "'Category' means the type of cost as recorded on the {req} form.",
        "'Based in' refers to the location recorded on the {payee}'s profile at the time of submission.",
        "'Policy owner' means the team responsible for maintaining this policy.",
        "'Working day' means Monday to Friday, excluding public holidays."]
RESP = ["{Payee}s are responsible for submitting complete and accurate information.",
        "Assessors check each {req} against the clauses of this policy in full.",
        "Line managers are expected to answer questions about a {req} within five working days.",
        "The policy owner monitors how the policy is applied and reports on it twice a year.",
        "Auditors may sample decisions at any time and ask for supporting documents.",
        "Anyone who believes the policy has been misapplied should raise it with the policy owner."]
COMPLAINTS = ["A {payee} who disagrees with a decision may ask for it to be reviewed within 28 days.",
              "The review is carried out by someone who was not involved in the original decision.",
              "The outcome of the review is final within the organisation.",
              "Reviews do not pause any deadline set elsewhere in this policy.",
              "Complaints about the conduct of staff are handled under the separate complaints procedure."]


def _n(rng, lo, hi, step):
    return rng.randrange(lo // step, hi // step + 1) * step


def build_policy(rng, d):
    """Random rule parameters for domain d."""
    cats = rng.sample(d["cats"], len(d["cats"]))
    P = dict(window=rng.choice([14, 30, 30, 45, 60, 90]), receipt=_n(rng, 25, 250, 25), approval=_n(rng, 300, 3000, 100),
             excluded=cats[:rng.choice([1, 2])], caps={c: _n(rng, 50, 1500, 50) for c in cats[2:2 + rng.choice([2, 3])]})
    reg = rng.sample(d["regions"], 2); role = rng.sample(d["roles"], 2)
    P["ext"] = {reg[0]: P["window"] + rng.choice([15, 30, 60])}
    P["exempt"] = {role[0]: rng.choice(["receipt", "approval", "window"])}
    P["order"] = rng.choice([["reject", "request more information", "escalate for approval", "approve up to the cap"],
                             ["reject", "escalate for approval", "request more information", "approve up to the cap"],
                             ["request more information", "reject", "escalate for approval", "approve up to the cap"],
                             ["escalate for approval", "request more information", "reject", "approve up to the cap"]])
    P["cmp"] = rng.choice(["above", "at least"])
    return P


def over(x, t, cmp):
    return x > t if cmp == "above" else x >= t


def decide(P, c):
    """-> (outcome, deciding rule key or None when several rules give it, [(outcome, rule key)], amount payable)"""
    trig = []
    win = P["ext"].get(c["region"], P["window"])
    if c["days"] > win and P["exempt"].get(c["role"]) != "window":
        trig.append(("reject", "window"))
    if c["category"] in P["excluded"]:
        trig.append(("reject", "excluded"))
    if over(c["amount"], P["receipt"], P["cmp"]) and not c["receipt"] and P["exempt"].get(c["role"]) != "receipt":
        trig.append(("request more information", "receipt"))
    if over(c["amount"], P["approval"], P["cmp"]) and not c["preapproved"] and P["exempt"].get(c["role"]) != "approval":
        trig.append(("escalate for approval", "approval"))
    cap = P["caps"].get(c["category"])
    if cap is not None and c["amount"] > cap:
        trig.append(("approve up to the cap", "cap"))
    if not trig:
        return "approve in full", None, trig, c["amount"]
    out = next(o for o in P["order"] if any(t[0] == o for t in trig))
    keys = [k for o, k in trig if o == out]
    return out, (keys[0] if len(keys) == 1 else None), trig, (cap if out == "approve up to the cap" else 0)


def _fill(t, d):
    R, Pay = d["req"], d["payee"]
    return t.format(req=R, Req=R[0].upper() + R[1:], a_req=an(R), A_req=an(R, cap=True), payee=Pay,
                    Payee=Pay[0].upper() + Pay[1:])


def render(rng, d, P):
    """Policy text: optional front matter, then numbered clauses in random order.  Returns (text, {rule key: number})."""
    cmpw = "more than" if P["cmp"] == "above" else "at least"
    R, pay = d["req"], d["payee"]
    word = rng.choice(["Clause", "Section", "Rule", "Paragraph"])
    rules = {
        "window": f"{an(R, cap=True)} must be submitted within {P['window']} days of the {d['event']}. {an(R, cap=True)} "
                  f"submitted later is rejected.",
        "excluded": f"The following categories are not covered, and any {R} for them is rejected: {', '.join(P['excluded'])}.",
        "receipt": f"{an(R, cap=True)} for {cmpw} {money(P['receipt'])} must include a receipt or equivalent proof; without "
                   f"it, more information is requested before any payment.",
        "approval": f"{an(R, cap=True)} for {cmpw} {money(P['approval'])} needs prior written approval from {d['approver']}; "
                    f"without it, the {R} is escalated for approval.",
        "cap": f"The following caps apply per {R}: " + "; ".join(f"{c}: {money(v)}" for c, v in P["caps"].items())
               + f". Where the amount exceeds the cap for its category, the {R} is approved up to the cap.",
    }
    (reg, w2), = P["ext"].items()
    (role, what), = P["exempt"].items()
    order = ", then ".join(f"'{o}'" for o in P["order"])
    other_role = rng.choice([r for r in d["roles"] if r != role])
    other_reg = rng.choice([r for r in d["regions"] if r != reg])
    extra = {
        "ext": f"For {an(pay)} based in {reg}, the time limit in {{ref:window}} is {w2} days instead.",
        "exempt": f"{{Ref:{what}}} does not apply to {an(pay)} whose role is {role}.",
        "order": f"If more than one clause applies to the same {R}, the outcome is decided in this order of priority: {order}. "
                 f"If no clause applies, the {R} is approved in full.",
    }
    look = [f"{an(pay, cap=True)} whose role is {other_role} must copy their line manager on every {R}.",
            f"For {an(pay)} based in {other_reg}, decisions are sent by post as well as by email.",
            f"{an(R, cap=True)} may be submitted through the portal or by email; both routes are treated the same.",
            f"Where {an(R)} covers several items, each item is listed on a separate line.",
            f"{an(pay, cap=True)} based in {other_reg} may ask for the decision to be explained by phone.",
            f"For {an(pay)} whose role is {other_role}, the {R} form must include a cost centre."]
    clauses = [("rule", k, v) for k, v in rules.items()] + [("rule", k, v) for k, v in extra.items()] \
        + [("look", None, x) for x in rng.sample(look, rng.randrange(2, 6))] \
        + [("boiler", None, _fill(b, d)) for b in rng.sample(BOILERPLATE, rng.randrange(5, 13))]
    rng.shuffle(clauses)
    numbers = {k: i for i, (_, k, _) in enumerate(clauses, 1) if k}
    ref = lambda k, cap=False: f"{word if cap else word.lower()} {numbers[k]}"
    lines = []
    for i, (_, key, text) in enumerate(clauses, 1):
        for k in ("window", "receipt", "approval"):
            text = text.replace(f"{{ref:{k}}}", ref(k)).replace(f"{{Ref:{k}}}", ref(k, cap=True))
        lines.append(f"{word} {i}. {text}")
    front = []
    if rng.random() < 0.8:
        front.append("Purpose\n" + " ".join(_fill(x, d) for x in rng.sample(PURPOSE, rng.randrange(3, 7))))
    if rng.random() < 0.7:
        front.append("Definitions\n" + "\n".join(_fill(x, d) for x in rng.sample(DEFS, rng.randrange(3, 8))))
    if rng.random() < 0.6:
        front.append("Responsibilities\n" + " ".join(_fill(x, d) for x in rng.sample(RESP, rng.randrange(2, 6))))
    back = ["Reviews and complaints\n" + " ".join(_fill(x, d) for x in rng.sample(COMPLAINTS, rng.randrange(2, 5)))] \
        if rng.random() < 0.6 else []
    title = f"{d['name'].title()} - version {rng.randrange(2, 9)}.{rng.randrange(0, 10)}"
    body = "\n\n".join([title] + front + ["Rules\n" + "\n".join(lines)] + back)
    return body, numbers, word


def make_case(rng, d, P, split):
    cat = rng.choice(d["cats"]); role = rng.choice(d["roles"]); region = rng.choice(d["regions"])
    anchor = rng.choice(["receipt", "approval", "cap", "cap", "none", "none"])
    if anchor == "receipt":
        amt = P["receipt"] + rng.choice([-25, 0, 0, 25])
    elif anchor == "approval":
        amt = P["approval"] + rng.choice([-100, 0, 0, 100])
    elif anchor == "cap" and P["caps"]:
        cat = rng.choice(list(P["caps"])); amt = P["caps"][cat] + rng.choice([-50, 0, 50, 200])
    else:
        amt = _n(rng, 20, 4000, 10)
    win = P["ext"].get(region, P["window"])
    days = max(0, win + rng.choice([-10, -1, 0, 0, 1, 3, 20])) if rng.random() < 0.6 else rng.randrange(0, win)
    return dict(name=person(rng, split), category=cat, role=role, region=region, amount=max(amt, 10), days=days,
                receipt=rng.random() < 0.55, preapproved=rng.random() < 0.3)


def case_text(rng, d, c):
    if rng.random() < 0.35:
        return {"requester": c["name"], "role": c["role"], "based_in": c["region"], "category": c["category"],
                "amount": c["amount"], "days_after_" + d["event"].replace(" ", "_"): c["days"],
                "receipt_attached": c["receipt"], "prior_written_approval": c["preapproved"]}
    first = c["name"].split()[0]
    return (f"{c['name']} ({c['role']}, based in {c['region']}) submitted {an(d['req'])} for {money(c['amount'])} in the "
            f"category '{c['category']}', {c['days']} days after the {d['event']}. "
            f"{'A receipt was attached.' if c['receipt'] else 'No receipt was attached.'} "
            f"{first} {'had' if c['preapproved'] else 'did not have'} prior written approval from {d['approver']}.")


def policy_case(rng, split):
    d = rng.choice(pool(DOMAINS, split))
    P = build_policy(rng, d)
    text, num, word = render(rng, d, P)
    c = make_case(rng, d, P, split)
    out, key, trig, pay = decide(P, c)
    ct = case_text(rng, d, c)
    state = {"policy": text, "case": ct} if isinstance(ct, dict) else f"{text}\n\nCase:\n{ct}"
    R = d["req"]
    qs = [choice(rng.choice([f"What should happen to this {R} under the policy?",
                             f"Under the policy, what is the correct decision on this {R}?"]), OUTCOMES, out, rng, shuffle=False)]
    if key is not None:
        others = [num[k] for k in num if num[k] != num[key]]
        opts = [str(num[key])] + [str(x) for x in rng.sample(others, min(3, len(others)))]
        qs.append(choice(f"Which numbered {word.lower()} decides the outcome for this {R}?", opts, str(num[key]), rng))
    hit = {k for _, k in trig}
    want = rng.random() < 0.5
    keys = [k for k in ("window", "receipt", "approval", "cap", "excluded") if (k in hit) == want] or \
           ["window", "receipt", "approval", "cap", "excluded"]
    rk = rng.choice(keys)
    ask = {"window": f"Taking any extension or exemption in the policy into account, was this {R} submitted too late?",
           "receipt": f"Taking any exemption in the policy into account, is this {R} missing proof that the policy requires for it?",
           "approval": f"Taking any exemption in the policy into account, does this {R} lack an approval that the policy requires for it?",
           "cap": f"Is the amount of this {R} above a cap that the policy sets for its category?",
           "excluded": f"Is the category of this {R} one that the policy does not cover at all?"}[rk]
    qs.append(noul(ask, rk in hit))
    if out in ("approve in full", "approve up to the cap"):
        opts, ans = ranked_options(rng, pay, [c["amount"], *P["caps"].values()], step=50, fmt=money, lo=10)
        qs.append(choice(f"How much should be paid on this {R}?", opts, ans, rng))
    k = 1 if rng.random() < 0.7 else 2
    return record(FAMILY, "policy_case", split, d["name"], state, rng.sample(qs, min(k, len(qs))),
                  outcome=out, triggered=[k for _, k in trig])


SUBS = {"policy_case": policy_case}
