import copy
import hashlib
import json
import random
from datetime import date, timedelta

UNKNOWN = None
SHAPES = {
    "atom": 0,
    "negation": ("not", 0),
    "conjunction": ("and", 0, 1),
    "disjunction": ("or", 0, 1),
    "exception": ("unless", 0, 1),
    "conditional": ("if", 0, 1, 2),
    "nested_and": ("and", ("and", 0, 1), 2),
    "nested_or": ("or", 0, ("or", 1, 2)),
    "held_and_or": ("and", ("or", 0, 1), 2),
    "held_or_not": ("or", ("and", 0, 1), ("not", 2)),
    "held_conditional": ("if", 0, ("not", 1), 2),
    "final_combination": ("and", ("if", 0, 1, 2), 3),
    "final_negation": ("not", ("or", ("and", 0, 1), 2)),
    "final_exception": ("or", ("unless", 0, 1), 2),
}
TRAIN_SHAPES = tuple(list(SHAPES)[:8])
DEV_SHAPES = tuple(list(SHAPES)[8:11])
TEST_SHAPES = tuple(list(SHAPES)[11:])
KINDS = ("lt", "le", "gt", "ge", "eq", "range", "match", "elapsed", "flag")


def atom_value(atom, facts):
    if any(key not in facts for key in atom["fields"]):
        return UNKNOWN
    values = [facts[k] for k in atom["fields"]]
    kind, threshold = atom["kind"], atom["threshold"]
    x = values[0]
    if kind == "lt": return x < threshold
    if kind == "le": return x <= threshold
    if kind == "gt": return x > threshold
    if kind == "ge": return x >= threshold
    if kind == "eq": return x == threshold
    if kind == "range": return threshold <= x <= threshold + 10
    if kind == "match": return x == values[1]
    if kind == "elapsed": return (date.fromisoformat(values[1]) - date.fromisoformat(x)).days <= threshold
    if kind == "flag": return x
    raise ValueError(f"unknown atom {kind}")


def evaluate_rule(tree, atoms, facts):
    if isinstance(tree, int):
        return atom_value(atoms[tree], facts)
    op, *children = tree
    values = [evaluate_rule(t, atoms, facts) for t in children]
    if op == "not": return None if values[0] is None else not values[0]
    if op == "unless":
        a, exception = values
        values = [a, None if exception is None else not exception]
        op = "and"
    if op == "and":
        return False if False in values else None if None in values else True
    if op == "or":
        return True if True in values else None if None in values else False
    if op == "if":
        condition, yes, no = values
        return yes if condition is True else no if condition is False else yes if yes == no else None
    raise ValueError(f"unknown operation {op}")


def atom_text(atom):
    kind, fields, t = atom["kind"], atom["fields"], atom["threshold"]
    key = fields[0]
    words = {"lt": "less than", "le": "at most", "gt": "greater than", "ge": "at least", "eq": "equal to"}
    if kind in words: return f"{key} is {words[kind]} {t}"
    if kind == "range": return f"{key} is between {t} and {t + 10}, including both endpoints"
    if kind == "match": return f"{fields[0]} is the same person as {fields[1]}"
    if kind == "elapsed": return f"the elapsed days from {fields[0]} to {fields[1]} are at most {t}"
    return f"{key} is yes"


def render_rule(tree, atoms, style):
    """Five surface styles for the same rule tree. 0 = logic-like, 1 = 'both/at least one of' lists, 2 = reserved for the
    locked test, 3 = plain prose, 4 = clause per line."""
    if isinstance(tree, int): return atom_text(atoms[tree])
    op, *children = tree
    parts = [render_rule(c, atoms, style) for c in children]
    if style == 3:
        if op == "not": return f"the condition \"{parts[0]}\" fails"
        if op == "and": return f"{parts[0]}, and also {parts[1]}"
        if op == "or": return f"either {parts[0]}, or else {parts[1]}"
        if op == "unless": return f"{parts[0]}, except when {parts[1]}"
        return f"when {parts[0]} the requirement is that {parts[1]}, and when it is not the requirement is that {parts[2]}"
    if style == 4:
        if op == "not": return f"[NOT: {parts[0]}]"
        if op in ("and", "or"): return f"[{'ALL' if op == 'and' else 'ANY'} of: {parts[0]} | {parts[1]}]"
        if op == "unless": return f"[{parts[0]} UNLESS {parts[1]}]"
        return f"[IF {parts[0]} THEN {parts[1]} ELSE {parts[2]}]"
    if op == "not": return f"NOT ({parts[0]})" if style == 0 else f"it is not the case that ({parts[0]})"
    if op in ("and", "or"):
        if style == 0: return f"({parts[0]}) {op.upper()} ({parts[1]})"
        connector = "both" if op == "and" else "at least one of"
        return f"{connector} these conditions hold: [({parts[0]}); ({parts[1]})]"
    if op == "unless": return f"({parts[0]}) holds and the exception ({parts[1]}) does not hold"
    return f"if ({parts[0]}), use ({parts[1]}); otherwise use ({parts[2]})"


# --- random rule trees (PLAN.md: structural diversity, negation anywhere) ---------------------------------------------

def skeleton(tree):
    """Structure with leaf identities erased: used to order commutative children independently of atom numbering."""
    if isinstance(tree, int): return "_"
    op, *children = tree
    parts = [skeleton(c) for c in children]
    if op in ("and", "or"): parts = sorted(parts)
    return f"{op}({','.join(parts)})"


def sort_commutative(tree):
    if isinstance(tree, int): return tree
    op, *children = tree
    children = [sort_commutative(c) for c in children]
    if op in ("and", "or"): children = sorted(children, key=skeleton)
    return (op, *children)


def canonical(tree):
    """Structure key: commutative children ordered by skeleton, then leaves renumbered in traversal order, so
    (A and B) or not C and not C or (B and A) share one key."""
    t = relabel(sort_commutative(tree))
    def render(t):
        if isinstance(t, int): return str(t)
        return f"{t[0]}({','.join(render(c) for c in t[1:])})"
    return render(t)


def push_negation(tree):
    """De Morgan normal form, so a random tree equivalent to a held-out shape under negation pushing is also excluded."""
    if isinstance(tree, int): return tree
    op, *children = tree
    if op == "not":
        inner = children[0]
        if isinstance(inner, int): return tree
        iop, *ic = inner
        if iop == "not": return push_negation(ic[0])
        if iop in ("and", "or"): return ("or" if iop == "and" else "and", *[push_negation(("not", c)) for c in ic])
        if iop == "unless": return push_negation(("or", ("not", ic[0]), ic[1]))
        return ("not", push_negation(inner))
    if op == "unless": return ("and", push_negation(children[0]), push_negation(("not", children[1])))
    return (op, *[push_negation(c) for c in children])


def relabel(tree):
    """Renumber leaves in first-appearance order so structure keys do not depend on which atom index was drawn."""
    mapping = {}
    def walk(t):
        if isinstance(t, int):
            mapping.setdefault(t, len(mapping)); return mapping[t]
        return (t[0], *[walk(c) for c in t[1:]])
    return walk(tree)


def structure_keys(tree):
    return {canonical(tree), canonical(push_negation(tree))}


HELD_OUT_KEYS = set().union(*(structure_keys(SHAPES[s]) for s in DEV_SHAPES + TEST_SHAPES))


def random_tree(rng, depth, next_leaf):
    """Random rule over {and, or, not, unless, if} with negation allowed at any position; leaves are fresh atom indices."""
    if depth == 0 or (depth < 3 and rng.random() < 0.25):
        return next_leaf()
    op = rng.choice(["and", "or", "not", "unless", "if", "and", "or"])
    if op == "not": return ("not", random_tree(rng, depth - 1, next_leaf))
    if op == "if": return ("if", random_tree(rng, depth - 1, next_leaf), random_tree(rng, depth - 1, next_leaf), random_tree(rng, depth - 1, next_leaf))
    return (op, random_tree(rng, depth - 1, next_leaf), random_tree(rng, depth - 1, next_leaf))


def sample_trees(n, seed, min_leaves=2, max_leaves=4, exclude=HELD_OUT_KEYS):
    """n distinct rule structures (by canonical key), none structurally equal to a held-out or locked shape, none a
    bare leaf, each with negation somewhere in ~60% of cases. Deterministic in `seed`."""
    rng = random.Random(f"{seed}:trees"); trees, keys = [], set()
    for _ in range(20000):
        if len(trees) >= n: break
        counter = [0]
        def next_leaf():
            counter[0] += 1; return counter[0] - 1
        t = random_tree(rng, rng.choice([2, 2, 3]), next_leaf)
        if isinstance(t, int) or not min_leaves <= counter[0] <= max_leaves: continue
        k = structure_keys(t)
        if k & exclude or k & keys: continue
        keys |= k; trees.append(relabel(t))
    if len(trees) < n: raise ValueError(f"only {len(trees)} distinct structures found")
    return trees


def leaf_indices(tree):
    if isinstance(tree, int): return {tree}
    return set().union(*(leaf_indices(t) for t in tree[1:]))


def make_atoms(tree, rng):
    nouns = rng.sample(["request", "account", "package", "review", "member", "shipment", "entry", "case"], 4)
    atoms = []
    for i in range(max(leaf_indices(tree)) + 1):
        kind = rng.choice(KINDS)
        prefix = nouns[i]
        fields = [f"{prefix} value"]
        if kind == "match": fields = [f"{prefix} signer", f"{prefix} designated approver"]
        elif kind == "elapsed": fields = [f"{prefix} start date", f"{prefix} end date"]
        elif kind == "flag": fields = [f"{prefix} verified"]
        atoms.append({"kind": kind, "fields": fields, "threshold": rng.randint(5, 60)})
    return atoms


def fact_domains(atoms, rng):
    domains = {}
    for a in atoms:
        kind, fs, t = a["kind"], a["fields"], a["threshold"]
        if kind == "match":
            people = rng.sample(["Mira", "Noah", "Aiko", "Ravi", "Sana", "Elin", "Tomas", "Kofi"], 3)
            domains.update({k: people for k in fs})
        elif kind == "elapsed":
            day = date(2027, rng.randint(1, 8), rng.randint(1, 28))
            domains[fs[0]] = [day.isoformat()]
            domains[fs[1]] = [(day + timedelta(days=n)).isoformat() for n in (max(0, t - 1), t, t + 1, t + 10)]
        elif kind == "flag": domains[fs[0]] = [False, True]
        else: domains[fs[0]] = [t - 1, t, t + 1, t + 10, t + 11]
    domains["routing reference"] = [rng.randint(100, 500), rng.randint(501, 999)]
    return domains


def rendered_facts(facts, order):
    def value(v):
        return "yes" if v is True else "no" if v is False else str(v)
    return [f"The {k} is {value(facts[k])}." for k in order]


POLICY_WRAPPERS = {
    0: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    1: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    2: "Approval requires the following rule to be true: {rule}. A false rule means denial. Routing references are irrelevant.",
    3: "A case is approved when {rule}; any other case is denied. Routing references play no part in the decision.",
    4: "DECISION RULE {rule} -> approve; otherwise reject. Ignore the routing reference.",
}


def generate(groups_per_shape, seed, shapes=TRAIN_SHAPES, styles=(0, 1), source="compositional", trees=None):
    """trees: optional {name: tree} to generate from random structures instead of SHAPES (names must not collide)."""
    records = []
    shape_trees = trees if trees is not None else {s: SHAPES[s] for s in shapes}
    for shape, tree in shape_trees.items():
        rng = random.Random(f"{seed}:{shape}")
        for i in range(groups_per_shape):
            atoms = make_atoms(tree, rng)
            domains = fact_domains(atoms, rng)
            for attempt in range(1000):
                facts = {k: rng.choice(v) for k, v in domains.items()}
                label = evaluate_rule(tree, atoms, facts)
                changed = None
                keys = list(domains); rng.shuffle(keys)
                for key in keys:
                    for value in domains[key]:
                        edited = {**facts, key: value}
                        if evaluate_rule(tree, atoms, edited) != label:
                            missing = {k: v for k, v in facts.items() if k != key}
                            if evaluate_rule(tree, atoms, missing) is None:
                                changed = (edited, key)
                                break
                    if changed: break
                if changed: break
            if changed is None:
                raise ValueError(f"cannot create a decisive edit for {shape}")
            edited, deciding = changed
            nuisance = {**facts, "routing reference": next(v for v in domains["routing reference"] if v != facts["routing reference"])}
            order = list(facts); rng.shuffle(order)
            style = rng.choice(styles)
            rule = render_rule(tree, atoms, style)
            policy = POLICY_WRAPPERS[style].format(rule=rule)
            keys = ["accept", "reject"]; rng.shuffle(keys)
            criteria = {k: "The policy permits this case" if k == "accept" else "The policy does not permit this case" for k in keys}
            group = f"composition/{seed}/{shape}/{i}"
            for kind, a, b in (("relevant", facts, edited), ("irrelevant", facts, nuisance)):
                for sibling, values in (("a", a), ("b", b)):
                    result = evaluate_rule(tree, atoms, values)
                    state = {"policy": policy, "case": " ".join(rendered_facts(values, order))}
                    identifier = f"{group}/{kind}/{sibling}"
                    records.append({"state": state, "questions": {"decision": {"type": "choice",
                        "instructions": "Apply the policy to this case.", "criteria": dict(criteria),
                        "label": "accept" if result else "reject", "src": f"composition_{shape}"}},
                        "_meta": {"id": identifier, "group_id": group, "source": source, "variant": "clean",
                                  "pair_id": f"{group}/{kind}", "sibling": sibling, "pair_kind": kind,
                                  "family": shape, "family_id": shape, "render_style": style, "structure": canonical(tree),
                                  "text_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                                  "certificate": {"tree": tree, "atoms": atoms, "facts": values, "order": order,
                                                  "deciding_field": deciding, "label": result}}})
    return records


def check_group(records):
    if len(records) != 4: raise ValueError("a composition group requires four records")
    for r in records:
        c = r["_meta"]["certificate"]
        if evaluate_rule(c["tree"], c["atoms"], c["facts"]) != c["label"]:
            raise ValueError("incorrect certificate label")
        if r["state"]["case"] != " ".join(rendered_facts(c["facts"], c["order"])):
            raise ValueError("rendered facts differ from certificate")
        q = r["questions"]["decision"]
        if q["label"] != ("accept" if c["label"] else "reject"):
            raise ValueError("answer differs from certificate")
    for a, b in (records[:2], records[2:]):
        ca, cb = a["_meta"]["certificate"], b["_meta"]["certificate"]
        if ca["order"] != cb["order"] or a["state"]["policy"] != b["state"]["policy"]:
            raise ValueError("pair changes more than one fact")
        if sum(ca["facts"][k] != cb["facts"][k] for k in ca["facts"]) != 1:
            raise ValueError("pair must change exactly one fact")
        expected_flip = a["_meta"]["pair_kind"] == "relevant"
        if (ca["label"] != cb["label"]) != expected_flip:
            raise ValueError("incorrect intervention label")
        missing = {k: v for k, v in ca["facts"].items() if k != ca["deciding_field"]}
        if expected_flip and evaluate_rule(ca["tree"], ca["atoms"], missing) is not None:
            raise ValueError("deciding evidence ablation did not remove the answer")
    return True
