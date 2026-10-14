"""Build the Lev training corpus train-v2 (docs/DATA.md, step 5): nested subsets S31 c S60 c S120.

    python scripts/build_lev_train.py --eval evals/levbench-v1 --out data/local/lev-train-v2 [--sizes smoke]

S31 is the corpus of training run as it is: data/local/kev08b-release-mix.jsonl (18,764) plus decision-v7 train through
`kev.train --suite evals/v7/decision-v7 --replay 12576`; it is referenced (hash-checked), never rebuilt. The increment
pool adds, per component (COMPONENTS, target counts for S120's increment):
  graded      scripts/lev_graded.py families at training templates 0-3 and k <= K_TRAIN, every k equally often;
              UNKNOWABLE_SHARE of each unknowable-capable family withholds the deciding fact (uniform target), as night2 does
  hard_v1     hard-v1's seven families from a new seed (scripts/build_hard_v1.build_split, training templates 0-3)
  kev_public  decision-v7's ten public sources, more rows of their HF training splits at decision-v7's revisions
  kev_added   arc / openbookqa / csqa training splits (kev/data.py converters, public-pool-v6 revisions)
  audited     the six trainable sources of docs/DATA.md (approved 2026-10-10; scripts/lev_sources.py,
              their HF training splits; the evaluation suite draws their held-out rows from groups disjoint from these)
The pool is dealt in stratified order (each component's records spread evenly through it), so S60's increment is a
prefix of S120's and both keep the same mix. A pool record is dropped when its state repeats S31 or the pool
(kev.suite.state_digest), or when scripts/screen_lev.py finds it in the evaluation suite given by --eval, any Kev
calibration/development/test partition, JevBench public or ekzhang-mmlupro (public rows on any hit, generated ones on
exact hits). Every record passes kev.suite.validate_training's source policy checks through kev.train at load time.
Writes increment.jsonl (with _meta.subset = the smallest subset holding it), s60.jsonl and s120.jsonl (S31's mix file
followed by that subset's increment; pass to kev.train --data with --suite evals/v7/decision-v7 --replay 12576) and
manifest.json.
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from kev.data import ALL_REPOS, ALL_SOURCES, EVAL_ONLY, build as build_public, materialize  # noqa: E402
from kev.suite import digest, load_split, read_jsonl, read_manifest, state_digest, write_json, write_jsonl  # noqa: E402
from scripts.build_hard_v1 import Checker, build_split as hard_v1_split, normalised as hard_v1_key  # noqa: E402
from scripts.build_lev_eval import ADDED, DECISION_V7, revisions  # noqa: E402
from scripts.hard_v1_common import Ctx  # noqa: E402
from scripts.lev_graded import FAMILIES, K_RANGE, K_TRAIN, UNKNOWABLE, label_of  # noqa: E402
from screen_lev import blocking, collect, references  # noqa: E402
from scripts.lev_sources import SOURCES as AUDITED, load as load_audited  # noqa: E402

SEED = "lev-train-v2"
MIX = ROOT / "data/local/kev08b-release-mix.jsonl"
S31_RECORDS = 18_764 + 12_576
SUBSETS = {"s60": 60_000 - S31_RECORDS, "s120": 120_000 - S31_RECORDS}   # increment records per subset (before kev.train's context filter)
COMPONENTS = {"full": {"graded": 32_000, "hard_v1": 12_000, "kev_public": 16_000, "kev_added": 10_000, "audited": 22_000},
              "smoke": {"graded": 80, "hard_v1": 21, "kev_public": 30, "kev_added": 9, "audited": 18}}
UNKNOWABLE_SHARE = 0.1
TEMPLATES = (0, 1, 2, 3)
CODE = ("scripts/build_lev_train.py", "scripts/build_lev_eval.py", "scripts/lev_graded.py", "scripts/screen_lev.py",
        "scripts/build_hard_v1.py", "scripts/hard_v1_common.py", "scripts/hard_v1_families.py", "kev/composition.py", "kev/data.py", "kev/suite.py")


class Pool:
    """Dedup against S31 and the pool, then the screen; counts why records were dropped. Several questions on one passage
    share a state within their group (CosmosQA, QASC-style sources); a state repeated by another group is a duplicate."""

    def __init__(self, s31_digests, screener):
        self.seen, self.owner, self.screener, self.stats = set(s31_digests), {}, screener, Counter()

    def admit(self, component, r):
        d, group = state_digest(r["state"]), r["_meta"].get("group_id")
        if d in self.seen and (group is None or self.owner.get(d) != group): self.stats[f"{component}:duplicate_state"] += 1; return False
        if blocking(r, self.screener.check(r)): self.stats[f"{component}:overlap_dropped"] += 1; return False
        self.seen.add(d); self.owner[d] = group; r["_meta"]["state_digest"] = d; r["_meta"]["component"] = component
        return True


def graded(n, pool, checker):
    out, per_family = [], n // len(FAMILIES)
    for family, (gen, _) in FAMILIES.items():
        ks = list(range(K_RANGE[family][0], K_TRAIN[family] + 1))
        n_unk = int(per_family * UNKNOWABLE_SHARE) if family in UNKNOWABLE else 0
        for unknowable, count in ((False, per_family - n_unk), (True, n_unk)):
            if not count: continue
            kk = [k for k in ks if not unknowable or k >= UNKNOWABLE[family]]
            ctx, made, attempts = Ctx(f"{SEED}:{family}:{'unknowable' if unknowable else 'clean'}"), 0, 0
            while made < count:
                attempts += 1
                if attempts > 80 * count + 200: raise RuntimeError(f"graded {family}: {made}/{count}")
                k, t = kk[made % len(kk)], TEMPLATES[made % len(TEMPLATES)]
                try: item = gen(ctx, t, k, unknowable)
                except ValueError: item = None
                if item is None: continue
                facts = json.loads(json.dumps(item["facts"]))
                label, target = label_of(family, facts)
                (qid, q), = item["questions"].items()
                q = dict(q); q["label"] = label or next(iter(q["criteria"]))
                if target: q["target"] = target
                r = {"state": item["state"], "questions": {qid: q},
                     "_meta": {"source": f"lev_gen_{family}", "family": family, "template": t, "difficulty": {"kind": family, "k": k},
                               "unanswerable": unknowable, "facts": facts, "id": f"{SEED}/graded/{family}/{'u' if unknowable else 'c'}{made:06d}"}}
                materialize(r)
                if checker and not checker.admit(r): continue
                if not pool.admit("graded", r): continue
                r["_meta"]["group_id"] = r["_meta"]["id"]; out.append(r); made += 1
    return out


def hard_v1(n, pool, checker):
    seen = {hard_v1_key(r["state"]) for s in ("train", "development", "test") for r in load_split(ROOT / "evals/hard-v1", s, allow_test=True)}
    rows = hard_v1_split("train", n, seen, checker, seed=f"{SEED}:hard-v1")
    out = []
    for r in rows:
        r["_meta"] |= {"id": r["_meta"]["id"].replace("hard-v1/", f"{SEED}/hard-v1/"), "group_id": r["_meta"]["group_id"].replace("hard-v1/", f"{SEED}/hard-v1/")}
        if pool.admit("hard_v1", r): out.append(r)
    return out


def public(component, names, n, pool, checker):
    per = -(-n // len(names))
    rows = build_public(2 * per + 50, split="train", seed=f"{SEED}:{component}", only=names, revisions=revisions(), sources=ALL_SOURCES, repos=ALL_REPOS)
    taken, out = Counter(), []
    for r in rows:
        src = r["_meta"]["source"]
        if taken[src] >= per or (checker and not checker.admit(r)): continue
        r["_meta"] |= {"origin_id": r["_meta"]["id"], "id": f"{SEED}/{component}/{r['_meta']['id']}"}
        r["_meta"]["group_id"] = r["_meta"]["id"]
        if pool.admit(component, r): out.append(r); taken[src] += 1
    return out


def audited(n, pool, checker):
    names = [x for x, spec in AUDITED.items() if spec["role"] == "trainable"]
    per, out = -(-n // len(names)), []
    for name in names:
        rows = load_audited(name, "train")
        random.Random(f"{SEED}:audited:{name}").shuffle(rows)
        taken = 0
        for r in rows:
            if taken >= per: break
            if checker and not checker.admit(r): continue
            r["_meta"] |= {"origin_id": r["_meta"]["id"], "id": f"{SEED}/audited/{r['_meta']['id']}"}
            if pool.admit("audited", r): out.append(r); taken += 1
    return out


def stratified(parts, seed):
    """Interleave the components so every prefix keeps their proportions (each record at its quantile in its component)."""
    keyed = []
    for name, recs in parts.items():
        rng = random.Random(f"{seed}:{name}"); recs = list(recs); rng.shuffle(recs)
        keyed += [((i + 0.5) / len(recs), name, r) for i, r in enumerate(recs)]
    keyed.sort(key=lambda x: (x[0], x[1]))
    return [r for _, _, r in keyed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="the evaluation suite directory the corpus must not overlap (its development and test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", choices=list(COMPONENTS), default="full")
    ap.add_argument("--no_tokenizer", action="store_true")
    ap.add_argument("--jevbench", default=str(ROOT / "vendor/jevbench/datasets/public"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit(f"{out} exists; a build never overwrites")
    sizes = COMPONENTS[a.sizes]
    checker = None if a.no_tokenizer else Checker()
    mix = read_jsonl(MIX)
    s31 = {state_digest(r["state"]) for r in mix} | {state_digest(r["state"]) for r in load_split(ROOT / "evals/v7/decision-v7", "train")}
    eval_specs = [str(Path(a.eval) / f"{s}.jsonl") for s in ("development", "test")]
    screener = references(jevbench_dir=a.jevbench)
    screener.collections.insert(0, collect("eval", eval_specs)); screener.collections[0].index()
    pool = Pool(s31, screener)
    parts = {"graded": graded(sizes["graded"], pool, checker), "hard_v1": hard_v1(sizes["hard_v1"], pool, checker),
             "kev_public": public("kev_public", DECISION_V7, sizes["kev_public"], pool, checker),
             "kev_added": public("kev_added", ADDED, sizes["kev_added"], pool, checker),
             "audited": audited(sizes["audited"], pool, checker)}
    bad = sorted({r["_meta"]["source"] for recs in parts.values() for r in recs} & set(EVAL_ONLY))
    if bad: raise SystemExit(f"eval-only sources in the pool: {bad}")
    inc = stratified(parts, SEED)
    if a.sizes == "smoke":   # the same proportions on a small pool
        cuts = {name: round(len(inc) * n / max(SUBSETS.values())) for name, n in SUBSETS.items()}
    else:
        cuts = {name: min(len(inc), n) for name, n in SUBSETS.items()}
    shortfall = {name: SUBSETS[name] - c for name, c in cuts.items() if a.sizes != "smoke" and c < SUBSETS[name]}
    for i, r in enumerate(inc): r["_meta"]["subset"] = next(name for name, c in cuts.items() if i < c) if i < max(cuts.values()) else "pool"
    out.mkdir(parents=True)
    write_jsonl(out / "increment.jsonl", inc)
    files = {"increment.jsonl": {"sha256": digest(out / "increment.jsonl"), "records": len(inc)}}
    for name, c in cuts.items():
        write_jsonl(out / f"{name}.jsonl", mix + inc[:c])
        files[f"{name}.jsonl"] = {"sha256": digest(out / f"{name}.jsonl"), "records": len(mix) + c, "increment_records": c}
    write_json(out / "manifest.json", {
        "version": 2, "name": "lev-train-v2", "seed": SEED, "sizes": a.sizes, "files": files,
        "s31": {"mix": {"path": str(MIX.relative_to(ROOT)), "sha256": digest(MIX), "records": len(mix)},
                "replay": {"suite": "evals/v7/decision-v7", "partition": "train", "records": 12_576, "sha256": read_manifest(ROOT / "evals/v7/decision-v7")["files"]["train.jsonl"]["sha256"]}},
        "train_with": "kev.train --data <subset>.jsonl --suite evals/v7/decision-v7 --replay 12576",
        "components": {name: {"target": sizes[name], "records": len(recs), "sources": dict(Counter(r["_meta"]["source"] for r in recs))} for name, recs in parts.items()},
        "subset_components": {name: dict(Counter(r["_meta"]["component"] for r in inc[:c])) for name, c in cuts.items()},
        "graded": {"families": list(FAMILIES), "k_train": K_TRAIN, "templates": TEMPLATES, "unknowable_share": UNKNOWABLE_SHARE},
        "dataset_revisions": revisions(), "pool_drops": dict(pool.stats),
        "screened_against": {"eval": {s: digest(s) for s in eval_specs}} | screener.describe(),
        "shortfall": shortfall, "source_audit": "docs/DATA.md", "code_sha256": {c: digest(ROOT / c) for c in CODE}})
    print(json.dumps({"files": files, "drops": dict(pool.stats)}, indent=1))


if __name__ == "__main__":
    main()
