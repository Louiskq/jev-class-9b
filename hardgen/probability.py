"""probability: decisions that turn on a probability the model has to work out from stated frequencies or rates.

Values are exact (fractions).  A value closer than one percentage point to a band boundary is regenerated, so no answer
depends on rounding.  Several subs carry a classic trap: base-rate neglect, confusing P(A|B) with P(B|A), treating "any
of n fails" like "all of n fail", adding probabilities of repeated tries.
"""
from fractions import Fraction as Fr

from common import CANNOT, an, choice, noul, org, person, pool, record

FAMILY = "probability"
BANDS = [(Fr(0), Fr(1, 10), "under 10%"), (Fr(1, 10), Fr(1, 4), "10% to 25%"), (Fr(1, 4), Fr(1, 2), "25% to 50%"),
         (Fr(1, 2), Fr(3, 4), "50% to 75%"), (Fr(3, 4), Fr(1) + 1, "over 75%")]
BAND_NAMES = [b[2] for b in BANDS]


def band(p):
    return next(name for lo, hi, name in BANDS if lo <= p < hi)


def near_edge(p, tol=Fr(1, 100)):
    return any(abs(p - lo) < tol for lo, _, _ in BANDS[1:])


def pct(p):
    return f"{float(p) * 100:.3g}%"


def threshold(rng, p, cands):
    """A threshold from cands with a uniformly random answer to 'is p above it?' (None if no candidate fits)."""
    want = rng.random() < 0.5
    ok = [t for t in cands if abs(p - t) >= Fr(1, 100) and (p > t) == want]
    ok = ok or [t for t in cands if abs(p - t) >= Fr(1, 100)]
    return rng.choice(ok) if ok else None


def _phr(rng, split, xs):
    return rng.choice(pool(xs, split))


def _band_q(rng, split, what, p):
    return choice(_phr(rng, split, [f"What is the probability that {what}?", f"How likely is it that {what}?",
                                    f"Which range contains the probability that {what}?"]), BAND_NAMES, band(p), rng,
                  shuffle=False)


def draws(rng, split):
    dom, thing = rng.choice(pool([("quality control sampling", "units"), ("raffle at a staff party", "tickets"),
                                  ("warehouse audit", "parcels"), ("seed bank germination test", "seed packets"),
                                  ("blood bank inventory", "units"), ("app store review queue", "submissions")], split))
    target = rng.choice(BAND_NAMES)
    for _ in range(300):
        kinds = rng.sample(["defective", "flagged", "premium", "expired", "priority", "damaged"], 2)
        a, b = rng.randrange(1, 9), rng.randrange(2, 14)
        n = a + b; k = rng.choice([2, 2, 3])
        # P(all k drawn are of kind a), without replacement
        p_all = Fr(1)
        for i in range(k):
            p_all *= Fr(a - i, n - i) if a - i > 0 else 0
        p_none = Fr(1)
        for i in range(k):
            p_none *= Fr(b - i, n - i)
        p_any = 1 - p_none
        q = rng.randrange(3)
        p = [p_all, p_any, p_none][q]
        if not near_edge(p) and band(p) == target:
            break
    what = [f"all {k} drawn {thing} are {kinds[0]}", f"at least one of the {k} drawn {thing} is {kinds[0]}",
            f"none of the {k} drawn {thing} is {kinds[0]}"][q]
    state = (f"A batch contains {n} {thing}: {a} are {kinds[0]} and {b} are {kinds[1]}. {k} {thing} are picked at "
             f"random, one after another, without putting any back.")
    qs = [_band_q(rng, split, what, p)]
    thr = threshold(rng, p, [Fr(1, 10), Fr(1, 5), Fr(1, 3), Fr(1, 2), Fr(2, 3)])
    if thr is not None:
        qs.append(noul(f"Is the probability that {what} greater than {pct(thr)}?", p > thr))
    return record(FAMILY, "draws", split, dom, state, [rng.choice(qs)])


def redundancy(rng, split):
    dom, unit = rng.choice(pool([("data centre power supplies", "power supply"), ("hospital backup generators", "generator"),
                                 ("payment gateway replicas", "replica"), ("delivery van fleet", "van"),
                                 ("satellite ground stations", "station"), ("water pumps at a treatment plant", "pump")], split))
    target = rng.choice(BAND_NAMES)
    for _ in range(300):
        n = rng.randrange(2, 9); f = rng.choice([Fr(1, 100), Fr(2, 100), Fr(5, 100), Fr(1, 10), Fr(15, 100), Fr(1, 4),
                                                 Fr(2, 5), Fr(1, 2), Fr(3, 5), Fr(3, 4)])
        mode = rng.choice(["all", "any"])
        p = f ** n if mode == "all" else 1 - (1 - f) ** n
        if not near_edge(p) and band(p) == target:
            break
    rule = (f"The service stops only if every {unit} fails on the same day." if mode == "all"
            else f"The service stops if any single {unit} fails, because each one handles a separate region.")
    state = (f"There are {n} {unit}s. On any given day each {unit} fails with probability {pct(f)}, independently of the "
             f"others. {rule}")
    what = "the service stops on a given day"
    qs = [_band_q(rng, split, what, p)]
    thr = threshold(rng, p, [Fr(1, 100), Fr(5, 100), Fr(1, 10), Fr(1, 4), Fr(1, 2), Fr(3, 4)])
    if thr is not None:
        qs.append(noul(f"Is the probability that {what} above {pct(thr)}?", p > thr))
    return record(FAMILY, "redundancy", split, dom, state, [rng.choice(qs)], mode=mode)


def base_rate(rng, split):
    dom, sg, pl, cond, test = rng.choice(pool([
        ("fraud screening", "transaction", "transactions", "fraudulent", "the fraud model flags it"),
        ("medical screening", "patient", "patients", "affected by the condition", "the screening test comes back positive"),
        ("spam filtering", "email", "emails", "spam", "the filter marks it as spam"),
        ("airport security", "passenger", "passengers", "carrying a prohibited item", "the scanner raises an alarm"),
        ("factory inspection", "item", "items", "defective", "the camera check rejects it"),
        ("insurance claims review", "claim", "claims", "fraudulent", "the rules engine flags it")], split))
    target = rng.choice(BAND_NAMES)
    want = {"under 10%": False, "10% to 25%": False, "25% to 50%": False}.get(target, True)
    for _ in range(400):
        prev = rng.choice([Fr(1, 1000), Fr(1, 200), Fr(1, 100), Fr(2, 100), Fr(5, 100), Fr(1, 10), Fr(1, 5), Fr(3, 10)])
        sens = rng.choice([Fr(80, 100), Fr(90, 100), Fr(95, 100), Fr(99, 100)])
        spec = rng.choice([Fr(90, 100), Fr(95, 100), Fr(98, 100), Fr(99, 100), Fr(999, 1000)])
        ppv = prev * sens / (prev * sens + (1 - prev) * (1 - spec))
        if not near_edge(ppv) and abs(ppv - Fr(1, 2)) > Fr(2, 100) and band(ppv) == target and (ppv > Fr(1, 2)) == want:
            break
    state = (f"Among all {pl}, {pct(prev)} are {cond}. When {an(sg)} is {cond}, {test} {pct(sens)} of the time. "
             f"When {an(sg)} is not {cond}, {test} anyway {pct(1 - spec)} of the time.")
    A = an(sg); A = A[0].upper() + A[1:]
    qs = [noul(f"{A} is checked and {test}. Is it more likely than not that the {sg} is {cond}?", ppv > Fr(1, 2)),
          _band_q(rng, split, f"{an(sg)} is {cond}, given that {test}", ppv)]
    return record(FAMILY, "base_rate", split, dom, state, [rng.choice(qs)])


def expected_cost(rng, split):
    dom = rng.choice(pool(["shipping insurance", "supplier selection", "server maintenance plan", "event weather cover",
                           "flight booking flexibility", "equipment warranty"], split))
    want_cover = rng.random() < 0.5
    for _ in range(300):
        loss = rng.randrange(20, 200) * 100
        p_loss = rng.choice([Fr(1, 100), Fr(2, 100), Fr(5, 100), Fr(1, 10), Fr(15, 100), Fr(1, 5)])
        premium = rng.randrange(1, 40) * 25
        deduct = rng.choice([0, 0, 250, 500, 1000])
        cover = premium + p_loss * min(deduct, loss)
        no_cover = p_loss * loss
        if abs(cover - no_cover) / max(cover, no_cover) > Fr(5, 100) and (cover < no_cover) == want_cover:
            break
    ans = "take the cover" if cover < no_cover else "go without cover"
    pays = "nothing more" if deduct == 0 else f"only the first ${deduct} of it"
    state = (f"Option 1: pay ${premium} for cover; if the loss happens you pay {pays}. Option 2: no cover; if the loss happens you pay the full "
             f"${loss}. The loss happens with probability {pct(p_loss)} over the period.")
    qs = [choice("Which option has the lower expected cost over the period?", ["take the cover", "go without cover"], ans, rng),
          noul("Is the expected cost of going without cover higher than the price of the cover?", no_cover > premium)]
    return record(FAMILY, "expected_cost", split, dom, state, [rng.choice(qs)])


def conditional(rng, split):
    dom, grp, out = rng.choice(pool([("subscription analytics", "plan", "cancelled"), ("hiring pipeline", "source", "hired"),
                                     ("clinic outcomes", "ward", "readmitted"), ("online ads", "channel", "purchased"),
                                     ("loan portfolio", "segment", "defaulted"), ("support tickets", "queue", "escalated")], split))
    names = rng.sample({"plan": ["monthly", "annual", "trial"], "source": ["referral", "job board", "agency"],
                        "ward": ["north", "south", "east"], "channel": ["search", "social", "email"],
                        "segment": ["retail", "small business", "corporate"], "queue": ["billing", "technical", "account"]}[grp], 2)
    target = rng.choice(BAND_NAMES)
    for _ in range(400):
        tot = [rng.randrange(40, 2000) for _ in names]
        hit = [rng.randrange(1, t // 2) for t in tot]
        rates = [Fr(h, t) for h, t in zip(hit, tot)]
        share = Fr(hit[0], sum(hit))
        if rates[0] != rates[1] and abs(rates[0] - rates[1]) > Fr(1, 100) and not near_edge(share) and \
                (hit[0] > hit[1]) != (rates[0] > rates[1]) and band(share) == target:
            break
    rows = [{grp: n, "total": t, out: h} for n, t, h in zip(names, tot, hit)]
    state = {"table": rows, "note": f"Each row counts cases by {grp}; '{out}' is how many of that row's cases {out}."}
    hi = names[0] if rates[0] > rates[1] else names[1]
    qs = [choice(f"Which {grp} has the higher rate of cases that {out}?", names, hi, rng),
          _band_q(rng, split, f"a case that {out} (among these two {grp}s) came from the '{names[0]}' {grp}", share)]
    return record(FAMILY, "conditional", split, dom, state, [rng.choice(qs)])


def at_least_one(rng, split):
    dom, trial, succ = rng.choice(pool([("sales outreach", "cold email", "gets a reply"), ("vaccine trial logistics", "shipment", "arrives damaged"),
                                        ("A/B testing", "visitor", "converts"), ("drilling survey", "test bore", "finds water"),
                                        ("fundraising", "grant application", "is funded"), ("QA testing", "test run", "fails")], split))
    target = rng.choice(BAND_NAMES)
    for _ in range(300):
        p = rng.choice([Fr(1, 100), Fr(2, 100), Fr(5, 100), Fr(1, 10), Fr(15, 100), Fr(1, 5), Fr(3, 10)])
        n = rng.randrange(2, 16)
        q = 1 - (1 - p) ** n
        if not near_edge(q) and abs(q - min(n * p, 1)) > Fr(2, 100) and band(q) == target:
            break
    state = (f"Each {trial} {succ} with probability {pct(p)}, independently of the others. {n} {trial}s are planned.")
    what = f"at least one of the {n} {trial}s {succ}"
    qs = [_band_q(rng, split, what, q)]
    thr = threshold(rng, q, [Fr(1, 10), Fr(1, 4), Fr(1, 2), Fr(3, 4), Fr(9, 10)])
    if thr is not None:
        qs.append(noul(f"Is the probability that {what} greater than {pct(thr)}?", q > thr))
    return record(FAMILY, "at_least_one", split, dom, state, [rng.choice(qs)])


SUBS = {"draws": draws, "redundancy": redundancy, "base_rate": base_rate, "expected_cost": expected_cost,
        "conditional": conditional, "at_least_one": at_least_one}
