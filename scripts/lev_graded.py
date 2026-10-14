"""Difficulty-graded generated families of the Lev suites (docs/DATA.md, step 2). Each family is
`generate(ctx, t, k, unknowable=False) -> item | None` plus `solve(facts) -> canonical answer | None`: k is the number
of sequential reasoning steps the answer needs, t the surface template (scripts/hard_v1_common.frame: 0-3 train,
4 development, 5 test, as hard-v1). The builder takes every label from `solve` over the stored facts, never from the
generator; an unknowable item has the one fact the answer needs withheld, so `solve` returns None and the record gets a
uniform target over its options.

  rule_depth    a policy rule nested k operators deep along one spine (kev.composition atoms and three-valued logic);
                accept / reject. Unknowable: the deciding field is missing from the case.
  dep_chain     an outage propagates through a chain of k hard dependencies; a decoy chain of the same length is cut by
                one soft dependency; which listed service is affected.
  ledger_steps  an opening balance and k entries in date order (deposits, withdrawals, percentage fees and interest,
                rounded to the cent); the closing balance. Unknowable: one entry's amount is not recorded.
  date_chain    k events, each defined relative to the previous one (days, weeks, business days, next weekday), stated
                in shuffled order; the date of the last one. Unknowable: one offset is to be confirmed.
"""
import random
from datetime import date, timedelta
from fractions import Fraction

from kev.composition import KINDS, evaluate_rule, fact_domains, render_rule, rendered_facts
from scripts.hard_v1_common import business_days_after, choice_q, day, frame, money, value_q

K_RANGE = {"rule_depth": (1, 6), "dep_chain": (1, 10), "ledger_steps": (1, 8), "date_chain": (1, 8)}
K_TRAIN = {"rule_depth": 3, "dep_chain": 4, "ledger_steps": 3, "date_chain": 3}   # the trained range is k <= K_TRAIN
UNKNOWABLE = {"rule_depth": 1, "ledger_steps": 2, "date_chain": 1}   # family -> smallest k with an unknowable variant


# ============================================================================================================== rule_depth
NOUNS = ["request", "account", "package", "review", "member", "shipment", "entry", "case"]
RULE_STYLES = (0, 1, 3, 4)   # kev.composition style 2 is reserved for Kev's locked test
RULE_TITLES = ["Eligibility review", "Approval check", "Policy decision", "Exception review", "Compliance check", "Access request"]


def spine_tree(rng, k, next_leaf):
    """A rule exactly k operators deep: one child of each operator continues the spine, the others are fresh atoms."""
    if k == 0: return next_leaf()
    op = rng.choice(["and", "or", "unless", "not", "and", "or", "if"])
    if op == "not": return ("not", spine_tree(rng, k - 1, next_leaf))
    arity = 3 if op == "if" else 2
    deep = rng.randrange(arity)
    return (op, *[spine_tree(rng, k - 1, next_leaf) if i == deep else next_leaf() for i in range(arity)])


def tree_depth(tree):
    return 0 if isinstance(tree, int) else 1 + max(tree_depth(c) for c in tree[1:])


def make_atoms(n, rng):
    nouns = rng.sample(NOUNS, n)
    atoms = []
    for noun in nouns:
        kind = rng.choice(KINDS)
        fields = {"match": [f"{noun} signer", f"{noun} designated approver"], "elapsed": [f"{noun} start date", f"{noun} end date"],
                  "flag": [f"{noun} verified"]}.get(kind, [f"{noun} value"])
        atoms.append({"kind": kind, "fields": fields, "threshold": rng.randint(5, 60)})
    return atoms


def solve_rule(f):
    v = evaluate_rule(f["tree"], f["atoms"], f["facts"])
    return None if v is None else "accept" if v else "reject"


def gen_rule(ctx, t, k, unknowable=False):
    rng = ctx.rng
    counter = [0]
    def next_leaf():
        counter[0] += 1; return counter[0] - 1
    tree = spine_tree(rng, k, next_leaf)
    if not 1 <= counter[0] <= len(NOUNS) or tree_depth(tree) != k: return None
    atoms = make_atoms(counter[0], rng)
    domains = fact_domains(atoms, rng)
    want = "accept" if ctx.coin("rule") else "reject"
    for _ in range(400):
        facts = {key: rng.choice(v) for key, v in domains.items()}
        if solve_rule({"tree": tree, "atoms": atoms, "facts": facts}) != want: continue
        deciding = [key for key in facts if key != "routing reference" and evaluate_rule(tree, atoms, {x: v for x, v in facts.items() if x != key}) is None]
        if deciding: break
    else:
        return None
    if unknowable:
        facts = {x: v for x, v in facts.items() if x != rng.choice(deciding)}
    order = list(facts); rng.shuffle(order)
    style = rng.choice(RULE_STYLES)
    lines = [f"Approve the case exactly when {render_rule(tree, atoms, style)}; otherwise reject it. The routing reference does not affect the decision."]
    lines += rendered_facts(facts, order)
    f = {"kind": "rule_depth", "tree": tree, "atoms": atoms, "facts": facts}
    answer = solve_rule(f)
    if (answer is None) != unknowable: return None
    q, _ = choice_q(ctx, "Apply the policy to this case.", {"accept": "The policy permits this case", "reject": "The policy does not permit this case"},
                    answer or "accept", "lev_rule_depth")
    f["options"] = {"accept": "accept", "reject": "reject"}
    return {"state": frame(ctx, t, rng.choice(RULE_TITLES), lines, "Policy desk"), "questions": {"decision": q}, "facts": f,
            "meta": {"atoms": counter[0], "rule_style": style}}


# =============================================================================================================== dep_chain
SERVICES = ["auth-service", "billing-api", "search-indexer", "notification-hub", "inventory-db", "checkout-web", "pricing-engine", "user-profile",
            "reporting-etl", "cdn-edge", "session-cache", "order-queue", "email-relay", "fraud-scorer", "catalog-api", "payments-gateway",
            "ledger-service", "geo-lookup", "tax-calculator", "media-transcoder", "audit-log", "feature-flags", "rate-limiter", "image-resizer",
            "recommendation-api", "webhook-dispatcher", "sms-gateway", "shipping-quotes", "loyalty-points", "config-store", "metrics-collector",
            "token-vault", "address-validator", "invoice-renderer", "chat-backend", "pdf-exporter"]


def solve_deps(f):
    hard = {}
    for a, b, soft in f["edges"]:   # a depends on b
        if not soft: hard.setdefault(b, set()).add(a)
    seen, stack = set(), [f["down"]]
    while stack:
        for y in hard.get(stack.pop(), ()):
            if y not in seen: seen.add(y); stack.append(y)
    hit = [s for s in f["candidates"] if s in seen]
    if len(hit) > 1: raise ValueError("several affected")
    return hit[0] if hit else "none_affected"


def gen_deps(ctx, t, k, unknowable=False):
    if unknowable: raise ValueError("dep_chain has no unknowable variant")
    rng = ctx.rng
    svc = rng.sample(SERVICES, 2 * k + 5)
    down, chain, decoy, others = svc[0], svc[1:1 + k], svc[1 + k:1 + 2 * k], svc[1 + 2 * k:]
    edges = [(chain[0], down, False)] + [(chain[i], chain[i - 1], False) for i in range(1, k)]
    cut = rng.randrange(k)   # the decoy chain has the same length; one of its links is soft, so the outage stops there
    edges += [(decoy[0], down, cut == 0)] + [(decoy[i], decoy[i - 1], cut == i) for i in range(1, k)]
    affected = rng.random() > 0.12
    if not affected:          # the true chain is cut too: nothing listed is affected
        i = rng.randrange(k); a, b, _ = edges[i]; edges[i] = (a, b, True)
    edges += [(down, others[0], False), (others[1], others[2], False), (others[3], others[1], False)]
    cands = [chain[-1], decoy[-1], others[0], others[3]]
    f = {"kind": "dep_chain", "edges": [list(e) for e in edges], "down": down, "candidates": cands}
    try: truth = solve_deps(f)
    except ValueError: return None
    facts = [f"{a} depends on {b}" + (f" (soft dependency: it degrades gracefully and keeps working if {b} is down)." if s else ".") for a, b, s in edges]
    rng.shuffle(facts)
    lines = ["Outages propagate along hard dependencies: if a service goes down, every service that depends on it, directly or through "
             "other hard dependencies, is affected. Soft dependencies do not propagate outages."] + facts + [f"{down} is currently down."]
    options = {c: None for c in cands} | {"none_affected": "None of these services is affected"}
    q, _ = choice_q(ctx, "Which of these services will be affected by the outage?", options, truth, "lev_dep_chain")
    f["options"] = {key: key for key in options}
    return {"state": frame(ctx, t, f"Outage: {down}", lines, "SRE"), "questions": {"affected": q}, "facts": f, "meta": {"affected": affected}}


# ============================================================================================================ ledger_steps
LEDGER_TITLES = ["Account statement", "Ledger review", "Balance query", "Statement check", "Account activity", "Closing balance"]


def round_cents(x):
    return int((x + Fraction(1, 2)) // 1)


def apply_entry(balance, e):
    if e["op"] == "deposit": return balance + e["amount"]
    if e["op"] == "withdrawal": return balance - e["amount"]
    pct = Fraction(e["rate"], 100)
    change = round_cents(balance * pct)
    return balance - change if e["op"] == "fee" else balance + change


def solve_ledger(f):
    balance = f["opening"]
    for e in f["entries"]:
        if e.get("amount") is None and e["op"] in ("deposit", "withdrawal"): return None
        balance = apply_entry(balance, e)
    return balance


def gen_ledger(ctx, t, k, unknowable=False):
    rng = ctx.rng
    opening = rng.randrange(50_000, 900_000, 100)
    start = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    entries, d = [], start
    for _ in range(k):
        d += timedelta(days=rng.randint(1, 9))
        op = rng.choice(["deposit", "withdrawal", "fee", "interest", "deposit", "withdrawal"])
        e = {"op": op, "date": d.isoformat()}
        if op in ("deposit", "withdrawal"): e["amount"] = rng.randrange(1_000, 200_000, 25)
        else: e["rate"] = rng.choice([1, 2, 3, 4, 5]) if op == "fee" else rng.choice([1, 2, 3])
        entries.append(e)
    f = {"kind": "ledger_steps", "opening": opening, "entries": entries}
    if solve_ledger(f) < 0: return None
    if unknowable:
        flat = [i for i, e in enumerate(entries) if e["op"] in ("deposit", "withdrawal")]
        if not flat: return None
        entries[rng.choice(flat)]["amount"] = None
    def text(e):
        when = day(date.fromisoformat(e["date"]))
        if e["op"] in ("deposit", "withdrawal"):
            amount = "an amount that was not recorded" if e["amount"] is None else money(e["amount"])
            return f"On {when}, a {e['op']} of {amount} was made."
        what = "a fee of" if e["op"] == "fee" else "interest of"
        return f"On {when}, {what} {e['rate']}% of the balance at that time was {'charged' if e['op'] == 'fee' else 'credited'}."
    lines = [f"The opening balance on {day(start)} was {money(opening)}.", "Percentage fees and interest are computed on the balance at that moment "
             "and rounded to the nearest cent."] + [text(e) for e in entries] + ["No other activity occurred."]
    truth = solve_ledger(f)
    full = solve_ledger({**f, "entries": [e if e.get("amount", 1) is not None else {**e, "amount": 0} for e in entries]})
    ref = truth if truth is not None else full
    # distractors: one entry skipped, one entry applied with the wrong sign, the balance before the last entry
    def without(i): return solve_ledger({**f, "entries": [e if e.get("amount", 1) is not None else {**e, "amount": 0} for j, e in enumerate(entries) if j != i]})
    def flipped(i):
        flip = {"deposit": "withdrawal", "withdrawal": "deposit", "fee": "interest", "interest": "fee"}
        es = [e if e.get("amount", 1) is not None else {**e, "amount": 0} for e in entries]
        return solve_ledger({**f, "entries": es[:i] + [{**es[i], "op": flip[es[i]["op"]]}] + es[i + 1:]})
    distractors = [without(i) for i in range(k)] + [flipped(i) for i in range(k)] + [ref + rng.choice([-1, 1]) * rng.randrange(500, 5000, 25)]
    distractors = [x for x in distractors if x >= 0]
    rng.shuffle(distractors)
    q, opts = value_q(ctx, "What is the closing balance after these entries?", ref, distractors, money, "lev_ledger_steps")
    f["options"] = opts
    return {"state": frame(ctx, t, rng.choice(LEDGER_TITLES), lines, "Finance"), "questions": {"balance": q}, "facts": f, "meta": {}}


# ============================================================================================================== date_chain
EVENTS = ["contract signing", "kickoff meeting", "design review", "security audit", "pilot launch", "customer workshop", "board update",
          "data migration", "staff training", "go-live", "retrospective", "invoice run", "vendor handover", "budget review"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def step_date(d, s):
    if s["unit"] == "days": return d + timedelta(days=s["n"])
    if s["unit"] == "weeks": return d + timedelta(weeks=s["n"])
    if s["unit"] == "business days": return business_days_after(d, s["n"], set())
    ahead = (s["weekday"] - d.weekday() - 1) % 7 + 1   # the first such weekday strictly after d
    return d + timedelta(days=ahead)


def solve_dates(f):
    d = date.fromisoformat(f["start"])
    for s in f["steps"]:
        if s.get("n", 0) is None: return None
        d = step_date(d, s)
    return d.isoformat()


def gen_dates(ctx, t, k, unknowable=False):
    rng = ctx.rng
    names = rng.sample(EVENTS, k + 1)
    start = date(2026, 1, 5) + timedelta(days=rng.randint(0, 500))
    steps = []
    for _ in range(k):
        unit = rng.choice(["days", "weeks", "business days", "next weekday", "days", "business days"])
        steps.append({"unit": unit, "weekday": rng.randrange(5)} if unit == "next weekday" else
                     {"unit": unit, "n": rng.randint(2, 20) if unit == "days" else rng.randint(1, 4) if unit == "weeks" else rng.randint(2, 12)})
    f = {"kind": "date_chain", "start": start.isoformat(), "steps": steps}
    truth_full = solve_dates(f)
    if unknowable:
        counted = [i for i, s in enumerate(steps) if "n" in s]
        if not counted: return None
        steps[rng.choice(counted)]["n"] = None
    def text(i, s):
        a, b = names[i], names[i + 1]
        if s["unit"] == "next weekday": return f"The {b} is held on the first {WEEKDAYS[s['weekday']]} after the {a}."
        if s["n"] is None: return f"The {b} is scheduled some {s['unit']} after the {a}; the exact number is to be confirmed."
        unit = s["unit"] if s["n"] != 1 else s["unit"].rstrip("s")
        return f"The {b} is {s['n']} {unit} after the {a}."
    lines = [text(i, s) for i, s in enumerate(steps)]
    rng.shuffle(lines)
    lines = [f"The {names[0]} took place on {day(start)}."] + lines + ["Business days are Monday to Friday; there are no holidays in this period."]
    # distractors: a link dropped, every business-day link counted as calendar days, the date one link earlier, +-1 day
    full = date.fromisoformat(truth_full)
    def drop(i):
        d = start
        for j, s in enumerate(steps):
            if j != i: d = step_date(d, s if s.get("n", 1) is not None else {**s, "n": 0})
        return d.isoformat()
    def calendar():
        d = start
        for s in steps: d = step_date(d, {**s, "unit": "days"} if s["unit"] == "business days" and s.get("n") is not None else s if s.get("n", 1) is not None else {**s, "n": 0})
        return d.isoformat()
    distractors = [drop(i) for i in range(k)] + [calendar(), (full + timedelta(days=1)).isoformat(), (full - timedelta(days=1)).isoformat(), (full + timedelta(days=7)).isoformat()]
    rng.shuffle(distractors)
    q, opts = value_q(ctx, f"On what date is the {names[-1]}?", truth_full, distractors, lambda v: day(date.fromisoformat(v)), "lev_date_chain")
    f["options"] = opts
    return {"state": frame(ctx, t, "Project schedule", lines, "PMO"), "questions": {"date": q}, "facts": f, "meta": {"events": names}}


FAMILIES = {"rule_depth": (gen_rule, solve_rule), "dep_chain": (gen_deps, solve_deps),
            "ledger_steps": (gen_ledger, solve_ledger), "date_chain": (gen_dates, solve_dates)}


def label_of(family, facts):
    """(option key, None) of the stored facts' answer, or (None, uniform target) when it cannot be determined."""
    answer = FAMILIES[family][1](facts)
    opts = facts["options"]
    if answer is None: return None, {key: 1.0 / len(opts) for key in opts}
    keys = [key for key, v in opts.items() if v == answer]
    if len(keys) != 1: raise ValueError(f"{family}: answer {answer!r} matches {len(keys)} options")
    return keys[0], None
