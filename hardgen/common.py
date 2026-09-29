"""Shared pieces for the hard-decision generators.

Every generator writes records in TypeSafe's /v1/systemone wire format, the form decider.data.teacher_questions.to_example
converts into training examples (and the form JevBench sends at test time):

    {"family": ..., "sub": ..., "split": "train" | "heldout", "domain": ..., "state": str | object | array,
     "questions": [{"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ..., "answer": ...}]}

Answers are computed by the generator, never guessed.  A held-out record draws its names, organisations and domains from
pools disjoint from the training pools, so the held-out sets measure transfer to new surface forms; they are only used to
pick the checkpoint.  Nothing here is derived from JevBench items: the families follow the category names JevBench
publishes for its sealed set (docs/METHOD-v1.4.md).
"""
import datetime as dt
import random

FIRST = ["Aisha", "Ben", "Carla", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Kemi", "Liam", "Mei",
         "Nadia", "Omar", "Priya", "Quentin", "Rosa", "Sven", "Tariq", "Uma", "Victor", "Wen", "Ximena", "Yusuf", "Zoe",
         "Anton", "Bea", "Chidi", "Dana", "Emil", "Fatima", "Gustav", "Hana", "Ivan", "Julia", "Kofi", "Lena", "Marco",
         "Noor", "Oskar", "Paula", "Rahul", "Sara", "Tomas", "Ulla", "Vera", "Wale", "Yara", "Zain"]
LAST = ["Okafor", "Lindqvist", "Moreau", "Tanaka", "Silva", "Novak", "Haddad", "Kowalski", "Mensah", "Ferreira", "Nguyen",
        "Brennan", "Castillo", "Duarte", "Eriksen", "Fischer", "Gallo", "Hughes", "Iyer", "Jansen", "Kaur", "Larsen",
        "Mbeki", "Nakamura", "Olsen", "Petrov", "Quinn", "Rossi", "Sato", "Toure", "Ueda", "Varga", "Weber", "Yilmaz"]
ORGS = ["Northwind Logistics", "Bluepeak Software", "Harbor Foods", "Keystone Clinics", "Meridian Freight", "Orchid Retail",
        "Pinecrest Utilities", "Quarry Lane Builders", "Riverbend Insurance", "Solace Hotels", "Tidewater Energy",
        "Upland Farms", "Vantage Media", "Willow Dental", "Ashgrove School", "Brightline Rail", "Cobalt Labs",
        "Driftwood Travel", "Evergreen Housing", "Foxglove Pharma", "Granite Bank", "Heron Airlines", "Ironclad Security",
        "Juniper Telecom"]


def pool(items, split):
    """Deterministic disjoint halves: even positions train, odd positions held out."""
    return items[0::2] if split == "train" else items[1::2]


def person(rng, split):
    return f"{rng.choice(pool(FIRST, split))} {rng.choice(pool(LAST, split))}"


def org(rng, split):
    return rng.choice(pool(ORGS, split))


def code(rng, prefix, digits=4):
    return f"{prefix}-{rng.randrange(10 ** (digits - 1), 10 ** digits)}"


# ---- dates -----------------------------------------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def rand_date(rng, lo=dt.date(2023, 1, 1), hi=dt.date(2026, 12, 31)):
    return lo + dt.timedelta(days=rng.randrange((hi - lo).days + 1))


def fmt_date(d, style):
    """Unambiguous styles only: no 03/04/2025."""
    if style == "iso":
        return d.isoformat()
    if style == "dmy":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if style == "mdy":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == "wdmy":
        return f"{DAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year}"
    if style == "short":
        return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"
    raise ValueError(style)


DATE_STYLES = ["iso", "dmy", "mdy", "wdmy", "short"]


def add_months(d, n):
    """Calendar months, clamped to the month's last day (31 Jan + 1 month = 28/29 Feb)."""
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(y, m, min(d.day, last))


def add_business_days(d, n, holidays=()):
    """The n-th business day after d (Mon-Fri, not a listed holiday); d itself is not counted."""
    while n > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d not in holidays:
            n -= 1
    return d


def years_between(born, on):
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


# ---- money and numbers ------------------------------------------------------------------------------------------------
def money(x, cur="$"):
    s = f"{abs(x):,.2f}"
    if s.endswith(".00"):
        s = s[:-3]
    return f"-{cur}{s}" if x < 0 else f"{cur}{s}"


# ---- questions --------------------------------------------------------------------------------------------------------
CANNOT = "cannot be determined from the information given"


def choice(instructions, options, answer, rng=None, shuffle=True):
    """options: list of names or {name: description}; answer: one of the names.  Distinct options only."""
    crit = dict(options) if isinstance(options, dict) else {str(o): None for o in options}
    names = list(crit)
    assert len(names) >= 2, f"choice needs at least 2 options: {instructions!r} {names}"
    assert len(set(names)) == len(names), names
    assert answer in crit, (answer, names)
    if shuffle and rng is not None:
        rng.shuffle(names)
        crit = {n: crit[n] for n in names}
    return {"type": "choice", "instructions": instructions, "criteria": crit, "answer": answer}


def noul(instructions, answer, true=None, false=None):
    crit = None if true is None and false is None else {"true": true, "false": false}
    return {"type": "noul", "instructions": instructions, "criteria": crit, "answer": bool(answer)}


def score(instructions, levels, answer):
    assert 0 <= answer < len(levels)
    return {"type": "score", "instructions": instructions, "criteria": list(levels), "answer": answer}


def value_options(rng, correct, wrong, k=4, fmt=str):
    """A choice over formatted values: the correct one plus up to k-1 distinct wrong ones (common-error variants first)."""
    seen, opts = {fmt(correct)}, [fmt(correct)]
    for w in wrong:
        s = fmt(w)
        if s not in seen and len(opts) < k:
            seen.add(s); opts.append(s)
    return opts, fmt(correct)


def ranked_options(rng, correct, wrong, step, k=4, fmt=str, lo=None):
    """A choice over ordered values (dates, amounts, counts) whose correct value sits at a uniformly random rank among the
    options, so "pick the middle one" is no shortcut.  `wrong` are mistake-derived values, used first on each side;
    fillers are correct +- n*step.  `lo`: smallest allowed value (e.g. 0 for counts)."""
    key = fmt(correct)
    def side(vals, sign):
        out, seen = [], {key}
        for v in list(vals) + [correct + sign * i * step for i in range(1, 8)]:
            s = fmt(v)
            if s not in seen and (lo is None or v >= lo):
                seen.add(s); out.append(v)
        return out
    below = side([w for w in wrong if w < correct], -1)
    above = side([w for w in wrong if w > correct], +1)
    r = rng.randrange(k)                                   # how many options sit below the correct one
    r = min(r, len(below)); r = max(r, k - 1 - len(above))
    vals = [correct] + below[:r] + above[:k - 1 - r]
    strs = [fmt(v) for v in vals]
    assert len(set(strs)) == len(strs), strs
    return strs, key


def an(noun, cap=False):
    a = ("an " if noun[:1].lower() in "aeiou" else "a ") + noun
    return a[0].upper() + a[1:] if cap else a


def record(family, sub, split, domain, state, questions, **meta):
    return {"family": family, "sub": sub, "split": split, "domain": domain, "state": state, "questions": questions, **meta}


def join(rng, sentences, noise=()):
    """Sentences in order, with irrelevant ones inserted at random places."""
    out = list(sentences)
    for n in noise:
        out.insert(rng.randrange(len(out) + 1), n)
    return " ".join(out)
