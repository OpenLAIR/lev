"""The Lev data builders' invariants (docs/DATA.md): graded generators, the screen's drop rule, nested
subsets. No network, no weights."""
import json
import random

import pytest

from kev.data import materialize
from kev.suite import state_digest
from scripts.hard_v1_common import Ctx
from scripts.lev_graded import FAMILIES, K_RANGE, K_TRAIN, UNKNOWABLE, label_of, spine_tree, tree_depth


def items(family, k, unknowable=False, n=12, seed="test"):
    gen = FAMILIES[family][0]
    ctx, out = Ctx(f"{seed}:{family}:{k}:{unknowable}"), []
    for i in range(400):
        try: it = gen(ctx, i % 6, k, unknowable)
        except ValueError: continue
        if it is not None: out.append(it)
        if len(out) == n: break
    return out


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_k_generates_and_labels_from_facts(family):
    lo, hi = K_RANGE[family]
    assert lo <= K_TRAIN[family] < hi
    for k in range(lo, hi + 1):
        got = items(family, k)
        assert len(got) == 12, (family, k)
        for it in got:
            facts = json.loads(json.dumps(it["facts"]))   # the stored facts alone determine the label
            label, target = label_of(family, facts)
            assert target is None and label in next(iter(it["questions"].values()))["criteria"]
            (qid, q), = it["questions"].items()
            materialize({"state": it["state"], "questions": {qid: dict(q, label=label)}, "_meta": {"source": f"lev_gen_{family}"}})


@pytest.mark.parametrize("family", list(UNKNOWABLE))
def test_unknowable_items_have_no_answer(family):
    for k in range(UNKNOWABLE[family], K_RANGE[family][1] + 1):
        for it in items(family, k, unknowable=True, n=4):
            label, target = label_of(family, json.loads(json.dumps(it["facts"])))
            assert label is None and abs(sum(target.values()) - 1) < 1e-9


def test_rule_depth_is_exact():
    rng = random.Random(0)
    for k in range(1, 7):
        for _ in range(50):
            c = [0]
            def leaf():
                c[0] += 1; return c[0] - 1
            assert tree_depth(spine_tree(rng, k, leaf)) == k


def test_generation_is_deterministic():
    for family in FAMILIES:
        a = [state_digest(it["state"]) for it in items(family, 2, seed="det")]
        b = [state_digest(it["state"]) for it in items(family, 2, seed="det")]
        assert a == b


def test_screen_drop_rule():
    from scripts.screen_lev import blocking
    gen = {"_meta": {"source": "lev_gen_rule_depth"}}
    pub = {"_meta": {"source": "agnews"}}
    assert not blocking(gen, {"train": ["containment"]})
    assert blocking(gen, {"kev_eval": ["exact_state"]})
    assert blocking(pub, {"train": ["containment"]})
    assert not blocking(pub, {})


def test_stratified_prefixes_keep_proportions():
    from scripts.build_lev_train import stratified
    parts = {"a": [{"_meta": {"c": "a"}} for _ in range(600)], "b": [{"_meta": {"c": "b"}} for _ in range(300)], "c": [{"_meta": {"c": "c"}} for _ in range(100)]}
    inc = stratified(parts, "t")
    for cut in (100, 323, 700, 1000):
        share = sum(r["_meta"]["c"] == "a" for r in inc[:cut]) / cut
        assert abs(share - 0.6) < 0.02
