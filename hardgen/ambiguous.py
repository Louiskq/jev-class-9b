"""ambiguous: the information given may or may not settle the question.

One fact of a policy case (or a date record) is left out or given two conflicting values.  The engine is run with every
value the missing fact can take: when the answer changes, the right answer is "cannot be determined"; when it does not,
the question has a definite answer even though something is missing.  That contrast is the point - a model that abstains
whenever a fact is missing fails half of these.
"""
import datetime as dt

from common import CANNOT, an, choice, fmt_date, money, noul, person, pool, rand_date, record, years_between, add_months
import policy

FAMILY = "ambiguous"


def _case_text_missing(d, c, missing):
    first = c["name"].split()[0]
    parts = [f"{c['name']} ({c['role']}, based in {c['region']}) submitted {an(d['req'])}"]
    parts.append(f" for {money(c['amount'])}" if missing != "amount" else " (the amount field was left blank)")
    parts.append(f" in the category '{c['category']}'")
    parts.append(f", {c['days']} days after the {d['event']}." if missing != "days"
                 else f"; the {d['event']} is not recorded, so its age cannot be checked.")
    if missing == "receipt":
        parts.append(" The attachment upload failed, so it is unknown whether a receipt was included.")
    else:
        parts.append(" A receipt was attached." if c["receipt"] else " No receipt was attached.")
    if missing == "preapproved":
        parts.append(f" There is no record either way of prior written approval from {d['approver']}.")
    else:
        parts.append(f" {first} {'had' if c['preapproved'] else 'did not have'} prior written approval from {d['approver']}.")
    return "".join(parts)


def policy_missing(rng, split):
    d = rng.choice(pool(policy.DOMAINS, split))
    P = policy.build_policy(rng, d)
    text, num, word = policy.render(rng, d, P)
    c = policy.make_case(rng, d, P, split)
    missing = rng.choice(["receipt", "preapproved", "days", "amount"])
    if missing in ("receipt", "preapproved"):
        variants = [dict(c, **{missing: v}) for v in (True, False)]
    elif missing == "days":
        variants = [dict(c, days=v) for v in (0, 10_000)]
    else:
        variants = [dict(c, amount=v) for v in (10, 5, 100_000, P["receipt"], P["approval"], *P["caps"].values())]
    outs = {policy.decide(P, v)[0] for v in variants}
    out = policy.decide(P, c)[0]
    ans = CANNOT if len(outs) > 1 else out
    state = f"{text}\n\nCase:\n{_case_text_missing(d, c, missing)}"
    return record(FAMILY, "policy_missing", split, d["name"], state,
                  [choice(f"Under the policy, what is the correct decision on this {d['req']}?", policy.OUTCOMES + [CANNOT],
                          ans, rng, shuffle=False)], missing=missing, determined=len(outs) == 1)


def age_partial(rng, split):
    """Age eligibility when the record gives only part of the birth date, or two different ones."""
    name = person(rng, split); thr = rng.choice([16, 18, 21, 25, 65])
    what = rng.choice(pool(["registration", "rental", "account opening", "screening visit", "orientation", "application"], split))
    ev = rand_date(rng)
    born = add_months(ev, -12 * thr) + dt.timedelta(days=rng.choice([-400, -40, -3, 0, 3, 40, 400]))
    if born.month == 2 and born.day == 29:
        born -= dt.timedelta(days=1)
    kind = rng.choice(["year_only", "month_year", "conflict"])
    st = rng.choice(["dmy", "mdy", "iso"])
    if kind == "year_only":
        lo, hi = dt.date(born.year, 1, 1), dt.date(born.year, 12, 31)
        given = f"{name}'s record shows only the year of birth: {born.year}."
    elif kind == "month_year":
        lo = dt.date(born.year, born.month, 1)
        hi = add_months(lo, 1) - dt.timedelta(days=1)
        given = f"{name}'s record shows the month and year of birth only: {fmt_date(born, 'dmy').split(' ', 1)[1]}."
    else:
        other = born + dt.timedelta(days=rng.choice([-800, -30, -5, 5, 30, 800]))
        lo, hi = min(born, other), max(born, other)
        given = (f"{name}'s passport gives the date of birth as {fmt_date(born, st)}, but the application form says "
                 f"{fmt_date(other, st)}; nobody has confirmed which is right.")
    ok = {years_between(b, ev) >= thr for b in (lo, hi)}
    ans = CANNOT if len(ok) > 1 else ("yes" if ok.pop() else "no")
    state = f"{given} The {what} took place on {fmt_date(ev, st)}. The minimum age for the {what} is {thr}."
    return record(FAMILY, "age_partial", split, "eligibility checks", state,
                  [choice(f"Was {name.split()[0]} at least {thr} on the day of the {what}?", ["yes", "no", CANNOT], ans, rng)],
                  determined=ans != CANNOT)


SUBS = {"policy_missing": policy_missing, "age_partial": age_partial}
