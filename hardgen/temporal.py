"""temporal_numeric: decisions that need date, duration, time-zone, unit or arithmetic reasoning over a realistic record.

Every answer is computed here.  Distractor options are the results of common mistakes (off by one day, counting from the
wrong date, ignoring a holiday, forgetting a pending item), so a model that skims cannot pass on plausibility alone.
"""
import datetime as dt
import json
import math

from common import (CANNOT, DATE_STYLES, add_business_days, add_months, choice, code, fmt_date, join, money, noul,
                    org, person, pool, rand_date, ranked_options, record, years_between)

FAMILY = "temporal_numeric"


def _phr(rng, split, options):
    return rng.choice(pool(options, split)) if len(options) > 1 else options[0]


def _late_bucket(days):
    if days <= 0:
        return "on time or early"
    if days <= 7:
        return "1-7 days late"
    if days <= 30:
        return "8-30 days late"
    if days <= 60:
        return "31-60 days late"
    return "more than 60 days late"


LATE_BUCKETS = ["on time or early", "1-7 days late", "8-30 days late", "31-60 days late", "more than 60 days late"]


def invoice(rng, split):
    dom = rng.choice(pool(["wholesale supplier invoicing", "dental clinic billing", "freelance design invoicing",
                           "school fees office", "utility billing", "equipment rental"], split))
    vendor, client = org(rng, split), person(rng, split)
    st = rng.choice(DATE_STYLES); f = lambda d: fmt_date(d, st)
    issued = rand_date(rng); terms = rng.choice([7, 10, 14, 15, 30, 30, 45, 60])
    due = issued + dt.timedelta(days=terms)
    delta = rng.choice([-5, -1, 0, 0, 1, 1, 2, 6, 7, 8, 13, 29, 30, 31, 45, 59, 60, 61, 75])
    known = rng.random() > 0.12
    paid = due + dt.timedelta(days=delta)
    amount = rng.randrange(20, 2000) * 10; pct = rng.choice([1, 1.5, 2, 2.5, 5])
    periods = 0 if delta <= 0 else math.ceil(delta / 30)
    fee = round(amount * pct / 100 * periods, 2)
    inv = code(rng, "INV")
    facts = [f"Invoice {inv} for {money(amount)} was issued by {vendor} to {client} on {f(issued)}.",
             f"Payment terms are net {terms} days from the invoice date.",
             f"A late fee of {pct}% of the invoice amount is charged for each started 30-day period after the due date.",
             f"Payment was received on {f(paid)}." if known else "No payment date has been recorded yet."]
    noise = [f"The goods were delivered on {f(issued - dt.timedelta(days=rng.randrange(1, 9)))}.",
             f"A reminder was emailed on {f(due + dt.timedelta(days=rng.randrange(-6, 4)))}."]
    state = join(rng, facts, rng.sample(noise, rng.randrange(3)))
    qs = []
    if known:
        qs.append(noul(_phr(rng, split, ["Was the payment received after the due date?", "Was this invoice paid late?",
                                          "Did the payment arrive later than the payment terms allow?"]), delta > 0))
        qs.append(choice(_phr(rng, split, ["How late was the payment relative to its due date?",
                                           "Which band describes the payment's timing against the due date?"]),
                         LATE_BUCKETS, _late_bucket(delta), rng, shuffle=False))
        wrong_fee = [round(amount * pct / 100 * p, 2) for p in (periods + 1, max(periods - 1, 0))]
        wrong_fee.append(round(amount * pct / 100 * math.ceil(max((paid - issued).days, 0) / 30), 2))
        wrong_fee.append(0 if fee else round(amount * pct / 100, 2))
        opts, ans = ranked_options(rng, fee, wrong_fee, step=round(amount * pct / 100, 2), fmt=money, lo=0)
        if fee > 0 or rng.random() < 0.3:
                qs.append(choice(_phr(rng, split, ["What late fee is owed on this invoice?", "How much late fee should be added?"]),
                             opts, ans, rng))
    else:
        qs.append(choice(_phr(rng, split, ["Was this invoice paid late?", "Did the payment arrive after the due date?"]),
                         ["yes", "no", CANNOT], CANNOT, rng))
    opts, ans = ranked_options(rng, due, [add_months(issued, max(terms // 30, 1)), issued + dt.timedelta(days=terms + 1)],
                               step=dt.timedelta(days=1), fmt=f)
    qs.append(choice(_phr(rng, split, ["On what date was the payment due?", "What is the invoice's due date?"]), opts, ans, rng))
    return record(FAMILY, "invoice", split, dom, state, rng.sample(qs, min(len(qs), rng.randrange(1, 3))))


def renewal(rng, split):
    dom = rng.choice(pool(["software subscription", "office lease", "equipment maintenance contract", "gym membership",
                           "insurance policy", "cloud hosting agreement"], split))
    party, provider = person(rng, split), org(rng, split)
    st = rng.choice(DATE_STYLES); f = lambda d: fmt_date(d, st)
    start = rand_date(rng); term = rng.choice([6, 12, 12, 24, 36]); notice = rng.choice([14, 30, 30, 45, 60, 90])
    end = add_months(start, term)
    deadline = end - dt.timedelta(days=notice)
    sent = rng.random() > 0.15
    got = deadline + dt.timedelta(days=rng.choice([-20, -3, -1, 0, 0, 1, 1, 2, 5, 15]))
    facts = [f"The {dom} between {provider} and {party} started on {f(start)} with an initial term of {term} months.",
             f"It renews automatically for another {term} months unless written notice of cancellation is received at "
             f"least {notice} days before the end of the current term.",
             f"{party.split()[0]}'s cancellation notice was received on {f(got)}." if sent
             else "No cancellation notice has been received."]
    noise = [f"The last invoice was paid on {f(start + dt.timedelta(days=rng.randrange(20, 300)))}.",
             f"Prices were last changed on {f(start - dt.timedelta(days=rng.randrange(10, 200)))}."]
    state = join(rng, facts, rng.sample(noise, rng.randrange(3)))
    renews = not (sent and got <= deadline)
    qs = [noul(_phr(rng, split, ["Will the agreement renew automatically for another term?",
                                 "Does the agreement roll over into a new term?"]), renews)]
    opts, ans = ranked_options(rng, deadline, [start + dt.timedelta(days=30 * term - notice), end - dt.timedelta(days=notice - 1),
                                               end - dt.timedelta(days=notice + 1)], step=dt.timedelta(days=1), fmt=f)
    qs.append(choice(_phr(rng, split, ["What is the last date on which a cancellation notice can be received to stop "
                                       "the renewal?", "By which date must the cancellation notice arrive at the latest?"]),
                     opts, ans, rng))
    opts, ans = ranked_options(rng, end, [start + dt.timedelta(days=round(365 * term / 12)), start + dt.timedelta(days=30 * term)],
                               step=dt.timedelta(days=1), fmt=f)
    qs.append(choice(_phr(rng, split, ["On what date does the current term end?", "When does the initial term finish?"]),
                     opts, ans, rng))
    return record(FAMILY, "renewal", split, dom, state, rng.sample(qs, rng.randrange(1, 3)))


def sla(rng, split):
    dom = rng.choice(pool(["IT helpdesk", "customer returns desk", "insurance claims team", "building maintenance",
                           "payroll queries", "procurement approvals"], split))
    st = rng.choice(["wdmy", "dmy", "mdy"]); f = lambda d: fmt_date(d, st)
    sub = rand_date(rng); hh, mm = rng.randrange(7, 22), rng.choice([0, 5, 15, 30, 45, 55])
    n = rng.choice([1, 2, 3, 3, 5, 10]); cutoff = rng.choice([16, 17, 18])
    hol = set()
    for _ in range(rng.choice([0, 1, 1, 2])):
        hol.add(add_business_days(sub, rng.randrange(1, n + 3)))
    late = hh > cutoff or (hh == cutoff and mm > 0)
    receipt = sub if sub.weekday() < 5 and sub not in hol and not late else add_business_days(sub, 1, hol)
    deadline = add_business_days(receipt, n, hol)
    resolved = deadline + dt.timedelta(days=rng.choice([-2, -1, 0, 0, 1, 1, 2, 3]))
    hol_txt = (" The following were public holidays: " + ", ".join(f(h) for h in sorted(hol)) + ".") if hol else ""
    state = (f"Service rule: requests must be resolved within {n} business day{'s' if n > 1 else ''} of receipt. Business "
             f"days are Monday to Friday excluding public holidays. Requests submitted after {cutoff}:00, or on a day that "
             f"is not a business day, count as received on the next business day; the day of receipt itself is not "
             f"counted.{hol_txt} Request {code(rng, 'REQ')} was submitted on {f(sub)} at {hh:02d}:{mm:02d} and resolved on "
             f"{f(resolved)}.")
    qs = [noul(_phr(rng, split, ["Was the request resolved within the service deadline?",
                                 "Did the team meet the resolution deadline for this request?"]), resolved <= deadline)]
    wrong = [receipt + dt.timedelta(days=n), add_business_days(receipt, n), add_business_days(sub, n, hol),
             add_business_days(receipt, n + 1, hol)]
    opts, ans = ranked_options(rng, deadline, wrong, step=dt.timedelta(days=1), fmt=f)
    qs.append(choice(_phr(rng, split, ["What was the resolution deadline for this request?",
                                       "By what date did the request have to be resolved?"]), opts, ans, rng))
    return record(FAMILY, "sla", split, dom, state, rng.sample(qs, rng.randrange(1, 3)))


def age(rng, split):
    dom, what = rng.choice(pool([("youth sports registration", "registration"), ("car rental desk", "rental"),
                                 ("senior discount program", "application"), ("clinical trial screening", "screening visit"),
                                 ("bank account opening", "account opening"), ("volunteer program", "orientation")], split))
    thr = rng.choice([16, 18, 18, 21, 25, 60, 65]); name = person(rng, split)
    st = rng.choice(DATE_STYLES); f = lambda d: fmt_date(d, st)
    ev = rand_date(rng)
    born = add_months(ev, -12 * thr) + dt.timedelta(days=rng.choice([-40, -3, -1, 0, 0, 1, 2, 5, 30, 200]))
    if born.month == 2 and born.day == 29:
        born = born - dt.timedelta(days=1)
    a = years_between(born, ev)
    facts = [f"{name} was born on {f(born)}.", f"The {what} took place on {f(ev)}.",
             f"The minimum age for this {what} is {thr}." if thr < 60 else f"Eligibility requires an age of at least {thr}."]
    noise = [f"The ID document was issued on {f(rand_date(rng))}.",
             f"The form was first submitted on {f(ev - dt.timedelta(days=rng.randrange(3, 60)))}."]
    state = join(rng, facts, rng.sample(noise, rng.randrange(3)))
    qs = [noul(f"Was {name.split()[0]} at least {thr} years old on the day of the {what}?", a >= thr),
          choice(f"How old was {name.split()[0]} on the day of the {what}?", *ranked_options(rng, a, [], step=1, k=3), rng)]
    return record(FAMILY, "age", split, dom, state, [rng.choice(qs)])


TZ = [("Lisbon", 0), ("London", 1), ("Berlin", 2), ("Athens", 3), ("Dubai", 4), ("Mumbai", 5.5), ("Singapore", 8),
      ("Tokyo", 9), ("Sydney", 10), ("Auckland", 12), ("New York", -4), ("Chicago", -5), ("Denver", -6),
      ("Los Angeles", -7), ("Sao Paulo", -3), ("Honolulu", -10)]


def _clock(t, off):
    local = t + dt.timedelta(hours=off)
    sign = "+" if off >= 0 else "-"; h = int(abs(off)); m = int(round((abs(off) - h) * 60))
    return local, f"UTC{sign}{h}" + (f":{m:02d}" if m else "")


def timezones(rng, split):
    dom = rng.choice(pool(["incident timeline", "flight operations log", "trading desk messages", "distributed build logs",
                           "customer chat transcript", "security audit trail"], split))
    labels = {"incident timeline": ["alert fired", "on-call engineer paged", "rollback started", "status page updated"],
              "flight operations log": ["pushback", "de-icing completed", "gate change announced", "crew check-in"],
              "trading desk messages": ["order placed", "limit breached", "risk desk notified", "position closed"],
              "distributed build logs": ["build queued", "tests failed", "artifact uploaded", "deploy approved"],
              "customer chat transcript": ["chat opened", "refund offered", "supervisor joined", "chat closed"],
              "security audit trail": ["login attempt", "password reset", "MFA disabled", "session revoked"]}[dom]
    k = rng.choice([3, 4]); names = rng.sample(labels, k); cities = rng.sample(TZ, k)
    base = dt.datetime(2025, rng.randrange(1, 13), rng.randrange(1, 28), rng.randrange(0, 24), rng.randrange(0, 60))
    times = [base + dt.timedelta(minutes=rng.randrange(0, 240)) for _ in range(k)]
    while len(set(times)) < k:
        times = [base + dt.timedelta(minutes=rng.randrange(0, 240)) for _ in range(k)]
    lines = []
    for nm, (city, off), t in zip(names, cities, times):
        local, tz = _clock(t, off)
        lines.append(f"- {nm}: {local.strftime('%Y-%m-%d %H:%M')} local time in {city} ({tz})")
    state = "Events recorded by different offices, each in its own local time:\n" + "\n".join(lines)
    first = names[min(range(k), key=lambda i: times[i])]
    i, j = rng.sample(range(k), 2)
    gap = int((times[j] - times[i]).total_seconds() // 60)
    naive = int(((times[j] + dt.timedelta(hours=cities[j][1])) - (times[i] + dt.timedelta(hours=cities[i][1]))).total_seconds() // 60)
    qs = [choice(_phr(rng, split, ["Which of these events happened first?", "Which event occurred earliest in absolute time?"]),
                 names, first, rng),
          noul(f"Did '{names[i]}' happen before '{names[j]}'?", times[i] < times[j])]
    fm = lambda m: f"{m} minutes" if m >= 0 else f"{-m} minutes before (negative gap)"
    if gap > 0:
        opts, ans = ranked_options(rng, gap, [naive, gap + 60, gap - 60, abs(naive)], step=5 if gap < 60 else 15, fmt=fm, lo=0)
        qs.append(choice(f"How many minutes after '{names[i]}' did '{names[j]}' happen?", opts, ans, rng))
    return record(FAMILY, "timezones", split, dom, state, [rng.choice(qs)])


UNITS = {"weight": [("kg", 1.0), ("lb", 0.45359237), ("t", 1000.0)],
         "volume": [("L", 1.0), ("US gal", 3.785411784), ("mL", 0.001)],
         "length": [("m", 1.0), ("ft", 0.3048), ("cm", 0.01)]}


def _margin(total, limit):
    r = (total - limit) / limit
    if r <= -0.10:
        return "within the limit with more than 10% to spare"
    if r <= 0:
        return "within the limit with 10% or less to spare"
    if r <= 0.10:
        return "over the limit by 10% or less"
    return "over the limit by more than 10%"


MARGINS = ["within the limit with more than 10% to spare", "within the limit with 10% or less to spare",
           "over the limit by 10% or less", "over the limit by more than 10%"]


def units(rng, split):
    dom, kind, thing = rng.choice(pool([("freight loading", "weight", "pallet load"), ("cold-chain storage", "volume", "tank"),
                                        ("bridge weight limits", "weight", "vehicle and cargo"),
                                        ("aquarium maintenance", "volume", "water change"),
                                        ("aircraft baggage", "weight", "checked baggage"),
                                        ("warehouse racking", "length", "shelf run")], split))
    if rng.random() < 0.3:
        return _temperature(rng, split)
    (lu, lf), (iu, iff) = rng.sample(UNITS[kind], 2)
    limit = rng.choice([50, 80, 120, 200, 500, 750, 1000, 2000]) * (1 if lu != "t" else 0.01)
    limit_base = limit * lf
    ratio = rng.choice([0.8, 0.88, 0.93, 0.97, 0.99, 1.0, 1.02, 1.05, 1.09, 1.15, 1.3])
    n = rng.randrange(2, 9)
    per = limit_base * ratio / n / iff
    per = round(per, 1) if per < 100 else round(per)
    total_base = per * n * iff
    stated = rng.random() < 0.5
    conv = f" (1 {iu} = {iff / lf:.6g} {lu})" if stated else ""
    state = (f"The {thing} limit is {limit:g} {lu}. There are {n} items of {per:g} {iu} each{conv}. "
             f"Nothing else counts towards the limit.")
    qs = [noul(f"Does the total exceed the {limit:g} {lu} limit?", total_base > limit_base + 1e-9),
          choice("How does the total compare with the limit?", MARGINS, _margin(total_base, limit_base), rng, shuffle=False)]
    return record(FAMILY, "units", split, dom, state, [rng.choice(qs)], conversion_stated=stated)


def _temperature(rng, split):
    dom = rng.choice(pool(["vaccine storage", "food safety inspection", "server room monitoring", "greenhouse control"], split))
    lo, hi = rng.choice([(2, 8), (0, 4), (18, 27), (15, 30), (-25, -15)])
    c = rng.choice([lo - 3, lo - 0.6, lo + 0.5, (lo + hi) / 2, hi - 0.4, hi + 0.4, hi + 2.5])
    fval = round(c * 9 / 5 + 32, 1); c_true = (fval - 32) * 5 / 9
    state = (f"The required storage range is {lo} to {hi} degrees Celsius. The logger, configured in Fahrenheit, recorded "
             f"{fval} degrees Fahrenheit at the last reading.")
    return record(FAMILY, "units", split, dom, state,
                  [noul("Was the last reading inside the required range?", lo <= c_true <= hi)], conversion_stated=False)


def budget(rng, split):
    dom = rng.choice(pool(["marketing campaign", "school trip", "research grant", "office renovation", "conference travel",
                           "community event"], split))
    B = rng.randrange(40, 400) * 100
    k = rng.randrange(4, 9); items = []
    for i in range(k):
        items.append({"item": f"line {i + 1}", "amount": rng.randrange(5, 60) * 50,
                      "status": rng.choices(["spent", "approved", "pending", "rejected"], [4, 3, 2, 1])[0]})
    scale = B * rng.choice([0.7, 0.85, 0.95, 1.0, 1.05, 1.2]) / max(sum(x["amount"] for x in items if x["status"] in ("spent", "approved")), 1)
    for x in items:
        x["amount"] = max(50, int(round(x["amount"] * scale / 50)) * 50)
    counted = sum(x["amount"] for x in items if x["status"] in ("spent", "approved"))
    remaining = B - counted
    state = {"budget": B, "currency": "USD",
             "rule": "Spent and approved items count against the budget; pending and rejected items do not.",
             "items": items}
    if rng.random() < 0.5:
        state = (f"The budget is {money(B)}. Spent and approved items count against it; pending and rejected items do not. "
                 + " ".join(f"{x['item'].capitalize()}: {money(x['amount'])} ({x['status']})." for x in items))
    pct = rng.choice([75, 80, 90, 100])
    wrong = [B - sum(x["amount"] for x in items if x["status"] != "rejected"), B - sum(x["amount"] for x in items),
             B - sum(x["amount"] for x in items if x["status"] == "spent"), remaining - 50]
    opts, ans = ranked_options(rng, remaining, wrong, step=rng.choice([50, 100, 250]), fmt=money)
    qs = [noul(f"Do the items that count against the budget exceed {pct}% of it?", counted > B * pct / 100),
          choice("How much of the budget remains after the items that count against it?", opts, ans, rng)]
    return record(FAMILY, "budget", split, dom, state, [rng.choice(qs)])


def ledger(rng, split):
    dom, unit, cats = rng.choice(pool([("expense reports", "amount", ["travel", "meals", "software", "hardware"]),
                                       ("warehouse shipments", "units", ["inbound", "outbound", "returns"]),
                                       ("clinic appointments", "minutes", ["checkup", "follow-up", "urgent"]),
                                       ("library loans", "days", ["fiction", "reference", "periodicals"]),
                                       ("fleet fuel purchases", "litres", ["diesel", "petrol", "electric charge"]),
                                       ("support tickets", "minutes", ["billing", "technical", "account"])], split))
    year = rng.randrange(2023, 2027); n = rng.randrange(14, 40)
    recs = []
    for i in range(n):
        d = dt.date(year, 1, 1) + dt.timedelta(days=rng.randrange(365))
        recs.append({"id": code(rng, "R", 5), "date": d.isoformat(), "category": rng.choice(cats),
                     unit: rng.randrange(1, 400), "status": rng.choice(["open", "closed", "closed", "void"])})
    recs.sort(key=lambda r: r["id"])
    state = {"records": recs}
    q = rng.randrange(3)
    if q == 0:
        inq = lambda r, qq: (int(r["date"][5:7]) - 1) // 3 + 1 == qq
        cands = [(qq, stt) for qq in range(1, 5) for stt in ("closed", "open")
                 if sum(1 for r in recs if inq(r, qq) and r["status"] == stt) >= 3]
        if not cands:
            return ledger(rng, split)
        qq, stt = rng.choice(cands)
        c = sum(1 for r in recs if inq(r, qq) and r["status"] == stt)
        opts, ans = ranked_options(rng, c, [sum(1 for r in recs if inq(r, qq))], step=1, fmt=str, lo=0)
        qs = [choice(f"How many records dated in Q{qq} {year} have status '{stt}'?", opts, ans, rng)]
    elif q == 1:
        cat = rng.choice(cats); quarter = rng.randrange(1, 5)
        tot = sum(r[unit] for r in recs if r["category"] == cat and (int(r["date"][5:7]) - 1) // 3 + 1 == quarter
                  and r["status"] != "void")
        thr = max(1, tot + rng.choice([-40, -5, 5, 40]))
        qs = [noul(f"Excluding void records, is the total {unit} of '{cat}' records in Q{quarter} {year} greater than {thr}?",
                   tot > thr)]
    else:
        cat = rng.choice(cats)
        sel = [r for r in recs if r["category"] == cat and r["status"] != "void"]
        if len(sel) < 2:
            return ledger(rng, split)
        last = max(sel, key=lambda r: r["date"])
        if sum(1 for r in sel if r["date"] == last["date"]) > 1:
            return ledger(rng, split)
        others = [r["id"] for r in recs if r["id"] != last["id"]]
        opts = [last["id"]] + rng.sample(others, min(3, len(others)))
        qs = [choice(f"Which non-void '{cat}' record has the latest date?", opts, last["id"], rng)]
    return record(FAMILY, "ledger", split, dom, state, qs)


SUBS = {"invoice": invoice, "renewal": renewal, "sla": sla, "age": age, "timezones": timezones, "units": units,
        "budget": budget, "ledger": ledger}
