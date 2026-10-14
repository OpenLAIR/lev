"""Build the Lev evaluation suite (docs/DATA.md, step 4): development (phase 2) and a locked test (phase 5).

    python scripts/build_lev_eval.py --out outputs/lev-eval-v1 [--sizes smoke]

Tiers are defined by distance from the Lev training corpus (train-v2, step 5), and every record says whether its
source is in S31, the corpus training run's checkpoints trained on (`_meta.seen_s31`):
  id          generated families (scripts/lev_graded.py) at training templates 0-3 and trained difficulty
              k <= K_TRAIN, fresh seeds; and fresh rows of public training sources (decision-v7's ten sources from their
              HF evaluation splits at decision-v7's pinned revisions; arc / openbookqa / csqa, train-v2 additions)
  near        generated families at the held-out template (4 development, 5 test), k <= K_TRAIN
  deep        generated families at the held-out template, k > K_TRAIN (extrapolation in depth)
  unknowable  generated families at the held-out template with the deciding fact withheld (uniform target; scored apart)
  far         the eval-only sources of the source audit (docs/DATA.md, approved 2026-10-10), never trained;
              StrategyQA carries _meta.report_only (reported, not gated); four are local-only (licence), listed in the manifest
The audited trainable sources' held-out rows (scripts/lev_sources.py; groups of their train splits excluded) join id.
Audited rows are split between test and development by record group (crc32 of group_id: a third to development), so a
group never spans both, and drawn stratified by difficulty k where the source has one.
k cycles over its range within each (tier, family), so every k has the same count. Test is built before development and
both share one set of seen states (kev.suite.state_digest), so no state repeats across partitions. Every candidate is
screened (scripts/screen_lev.py) against S31, every Kev calibration/development/test partition, JevBench public and
ekzhang-mmlupro; public rows are dropped on any hit, generated ones on exact hits. Admission: the 8k training and
serving contexts under kev.suite.ADMISSION_TOKENIZER (build_hard_v1.Checker).
"""
import argparse, json, random, sys, zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from kev.data import ALL_REPOS, ALL_SOURCES, build as build_public, materialize  # noqa: E402
from kev.suite import ADMISSION_TOKENIZER, digest, read_manifest, state_digest, write_json, write_jsonl  # noqa: E402
from scripts.build_hard_v1 import Checker  # noqa: E402
from scripts.hard_v1_common import Ctx  # noqa: E402
from scripts.lev_graded import FAMILIES, K_RANGE, K_TRAIN, UNKNOWABLE, label_of  # noqa: E402
from screen_lev import blocking, references  # noqa: E402
from scripts.lev_sources import SOURCES as AUDITED, load as load_audited  # noqa: E402

SEED = "lev-eval-v1"
S31 = [str(ROOT / "data/local/kev08b-release-mix.jsonl"), str(ROOT / "evals/v7/decision-v7") + ":train"]
DECISION_V7 = ["agnews", "amazon", "banking77", "boolq", "dbpedia14", "imdb", "mnli", "sst5", "trec", "yelp"]
ADDED = ["arc", "openbookqa", "csqa"]   # Kev trainable sources added in train-v2 (kev/data.py converters)
ADDED_REPOS = ("allenai/ai2_arc", "allenai/openbookqa", "tau/commonsense_qa")   # pinned as public-pool-v6 pinned them
TEMPLATES = {"id": (0, 1, 2, 3), "development": (4,), "test": (5,)}
# questions per partition: generated per family and tier, public per source
SIZES = {"full": {"development": {"id": 100, "near": 200, "deep": 200, "unknowable": 100, "public": 40, "added": 50, "audited_id": 50, "far": 200},
                  "test": {"id": 200, "near": 400, "deep": 400, "unknowable": 200, "public": 80, "added": 100, "audited_id": 100, "far": 400}},
         "smoke": {"development": {"id": 6, "near": 8, "deep": 8, "unknowable": 6, "public": 3, "added": 3, "audited_id": 3, "far": 4},
                   "test": {"id": 6, "near": 8, "deep": 8, "unknowable": 6, "public": 3, "added": 3, "audited_id": 3, "far": 4}}}
CODE = ("scripts/build_lev_eval.py", "scripts/lev_graded.py", "scripts/screen_lev.py", "scripts/screen_longdoc_v1.py",
        "scripts/hard_v1_common.py", "kev/composition.py", "kev/data.py", "kev/api.py", "kev/suite.py")


def k_values(family, tier):
    lo, hi = K_RANGE[family]
    if tier == "deep": return list(range(K_TRAIN[family] + 1, hi + 1))
    if tier == "unknowable": return list(range(max(lo, UNKNOWABLE[family]), hi + 1))
    return list(range(lo, K_TRAIN[family] + 1))


def generated_record(family, tier, split, t, k, item, checker):
    facts = json.loads(json.dumps(item["facts"]))   # the stored facts alone determine the label
    label, target = label_of(family, facts)
    if (target is not None) != (tier == "unknowable"): raise ValueError(f"{family}/{tier}: unknowable mismatch")
    (qid, q), = item["questions"].items()
    q = dict(q); q["label"] = label or next(iter(q["criteria"]))   # an unknowable item's label is a placeholder; its target is uniform
    if target: q["target"] = target
    rec = {"state": item["state"], "questions": {qid: q},
           "_meta": {"source": f"lev_gen_{family}", "family": family, "tier": tier, "split": split, "template": t,
                     "difficulty": {"kind": family, "k": k}, "unanswerable": tier == "unknowable", "seen_s31": False,
                     **{f"gen_{x}": v for x, v in item["meta"].items()}, "facts": facts, "state_digest": state_digest(item["state"])}}
    materialize(rec)
    if checker:
        rec["_meta"]["state_tokens"] = checker.state_tokens(rec["state"])
        if not checker.admit(rec): return None
    return rec


def generated(split, sizes, seen, screener, checker, report):
    out = []
    for tier in ("id", "near", "deep", "unknowable"):
        for family, (gen, _) in FAMILIES.items():
            if tier == "unknowable" and family not in UNKNOWABLE: continue
            n, ks = sizes[tier], k_values(family, tier)
            templates = TEMPLATES["id"] if tier == "id" else TEMPLATES[split]
            ctx, made, stats, attempts = Ctx(f"{SEED}:{family}:{split}:{tier}"), [], Counter(), 0
            while len(made) < n:
                attempts += 1
                if attempts > 80 * n + 200: raise RuntimeError(f"{family}/{split}/{tier}: {len(made)}/{n} after {attempts} draws ({dict(stats)})")
                k, t = ks[len(made) % len(ks)], templates[len(made) % len(templates)]
                try: item = gen(ctx, t, k, tier == "unknowable")
                except ValueError: item = None
                if item is None: stats["draw_rejected"] += 1; continue
                rec = generated_record(family, tier, split, t, k, item, checker)
                if rec is None: stats["context_rejected"] += 1; continue
                if rec["_meta"]["state_digest"] in seen: stats["duplicate_state"] += 1; continue
                hits = screener.check(rec)
                if hits: stats["overlap_reported"] += 1
                if blocking(rec, hits): stats["overlap_dropped"] += 1; continue
                rec["_meta"]["id"] = rec["_meta"]["group_id"] = f"lev-eval-v1/{split}/{tier}/{family}/{len(made):05d}"
                seen.add(rec["_meta"]["state_digest"]); made.append(rec)
            report[f"{tier}/{family}"] = dict(stats) | {"attempts": attempts, "records": len(made)}
            out += made
    return out


def revisions():
    pool = read_manifest(ROOT / "evals/public-pool-v6")["dataset_revisions"]
    return read_manifest(ROOT / "evals/v7/decision-v7")["dataset_revisions"] | {r: pool[r] for r in ADDED_REPOS}


def public(parts, sizes, seen, screener, checker, report):
    """Fresh rows of public training sources: one draw per source large enough for both partitions, screened, then dealt
    test first. Their tier is id; decision-v7's sources are in S31, the added ones are not."""
    for group, names in (("public", DECISION_V7), ("added", ADDED)):
        need = sum(sizes[s][group] for s in ("test", "development"))
        rows = build_public(4 * need + 40, split="test", seed=f"{SEED}:public", only=names, revisions=revisions(), sources=ALL_SOURCES, repos=ALL_REPOS)
        by_source = defaultdict(list)
        for r in rows: by_source[r["_meta"]["source"]].append(r)
        for name in names:
            stats, pool = Counter(), by_source[name]
            for split in ("test", "development"):
                taken = 0
                while taken < sizes[split][group]:
                    if not pool: raise RuntimeError(f"{name}: ran out of rows for {split} ({dict(stats)})")
                    r = pool.pop(0); d = state_digest(r["state"])
                    if d in seen: stats["duplicate_state"] += 1; continue
                    if checker and not checker.admit(r): stats["context_rejected"] += 1; continue
                    hits = screener.check(r)
                    if blocking(r, hits): stats["overlap_dropped"] += 1; continue
                    m = r["_meta"]
                    r["_meta"] = {**m, "id": f"lev-eval-v1/{split}/id/{name}/{m['row']}", "group_id": f"lev-eval-v1/{split}/id/{name}/{m['row']}",
                                  "tier": "id", "split": split, "difficulty": None, "unanswerable": False, "seen_s31": group == "public",
                                  "origin_id": m["id"], "state_digest": d} | ({"state_tokens": checker.state_tokens(r["state"])} if checker else {})
                    seen.add(d); parts[split].append(r); taken += 1
            report[name] = dict(stats)


def audited(parts, sizes, seen, screener, checker, report):
    """The audited sources' held-out rows: trainable sources -> id, eval-only -> far. Questions per partition and source:
    sizes[split]["audited_id" | "far"]; records are drawn group-disjointly (a third of the groups to development) and,
    where the source has a difficulty k, round-robin over k so every k is equally represented while it lasts. Several
    questions on one passage (AR-LSAT, QuAIL, ...) share a state within their group; a state is a duplicate only when an
    earlier record of another group (or a generated / public record) holds it."""
    owner = {}
    for name, spec in AUDITED.items():
        tier = "id" if spec["role"] == "trainable" else "far"
        rows = load_audited(name, "eval")
        stats = Counter({"available": len(rows)})
        for split in ("test", "development"):
            mine = [r for r in rows if (zlib.crc32(r["_meta"]["group_id"].encode()) % 3 == 0) == (split == "development")]
            random.Random(f"{SEED}:{name}:{split}").shuffle(mine)
            strata = defaultdict(list)
            for r in mine: strata[(r["_meta"].get("difficulty") or {}).get("k")].append(r)
            order = [k for k in sorted(strata, key=lambda x: (x is None, x))]
            want, taken = sizes[split]["audited_id" if tier == "id" else "far"], 0
            while taken < want and any(strata[k] for k in order):
                for k in order:
                    if taken >= want or not strata[k]: continue
                    r = strata[k].pop(0)
                    q = len(r["questions"])
                    d = state_digest(r["state"])
                    if d in seen and owner.get(d) != r["_meta"]["group_id"]: stats["duplicate_state"] += 1; continue
                    if checker and not checker.admit(r): stats["context_rejected"] += 1; continue
                    if blocking(r, screener.check(r)): stats["overlap_dropped"] += 1; continue
                    owner[d] = r["_meta"]["group_id"]; m = r["_meta"]
                    r["_meta"] = {**m, "origin_id": m["id"], "id": f"lev-eval-v1/{split}/{tier}/{m['id']}", "group_id": f"lev-eval-v1/{m['group_id']}",
                                  "tier": tier, "split": split, "unanswerable": False, "seen_s31": False, "state_digest": d,
                                  "partition_source_split": m.get("split")} | ({"state_tokens": checker.state_tokens(r["state"])} if checker else {})
                    seen.add(d); parts[split].append(r); taken += q
            stats[f"{split}_questions"] = taken
            if taken < want: stats[f"{split}_short"] = want - taken
        report[name] = dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", choices=list(SIZES), default="full")
    ap.add_argument("--no_tokenizer", action="store_true", help="skip context admission (smoke tests only)")
    ap.add_argument("--jevbench", default=str(ROOT / "vendor/jevbench/datasets/public"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit(f"{out} exists; a build never overwrites")
    sizes = SIZES[a.sizes]
    checker = None if a.no_tokenizer else Checker()
    screener = references(S31, a.jevbench, exclude=[ROOT / "evals/levbench-v1"])
    seen, parts, report = set(), {"test": [], "development": []}, {"test": {}, "development": {}}
    for split in ("test", "development"):
        parts[split] += generated(split, sizes[split], seen, screener, checker, report[split])
    report["public"], report["audited"] = {}, {}
    public(parts, sizes, seen, screener, checker, report["public"])
    audited(parts, sizes, seen, screener, checker, report["audited"])
    out.mkdir(parents=True)
    files = {}
    for split, recs in parts.items():
        random.Random(f"{SEED}:{split}:order").shuffle(recs)
        write_jsonl(out / f"{split}.jsonl", recs)
        files[f"{split}.jsonl"] = {"sha256": digest(out / f"{split}.jsonl"), "records": len(recs), "questions": sum(len(r["questions"]) for r in recs)}
    tiers = {split: dict(Counter(r["_meta"]["tier"] for r in recs)) for split, recs in parts.items()}
    write_json(out / "manifest.json", {
        "version": 1, "name": "lev-eval-v1", "seed": SEED, "sizes": a.sizes, "locked": ["test"], "files": files, "tiers": tiers,
        "families": {f: {"k_range": K_RANGE[f], "k_train": K_TRAIN[f], "unknowable_min_k": UNKNOWABLE.get(f)} for f in FAMILIES},
        "templates": {"id": TEMPLATES["id"], "near_deep_unknowable": {"development": TEMPLATES["development"], "test": TEMPLATES["test"]}},
        "public_sources": {"s31": DECISION_V7, "added": ADDED}, "dataset_revisions": revisions(),
        "trainable_sources": [f"lev_gen_{f}" for f in FAMILIES] + DECISION_V7 + ADDED + [f"lev_{n}" for n, x in AUDITED.items() if x["role"] == "trainable"],
        "eval_only_sources": [f"lev_{n}" for n, x in AUDITED.items() if x["role"] == "eval_only"],
        "audited_sources": {f"lev_{n}": {k: x[k] for k in ("repo", "revision", "role", "licence", "local_only")} | {"report_only": bool(x.get("report_only"))} for n, x in AUDITED.items()},
        "local_only_sources": [f"lev_{n}" for n, x in AUDITED.items() if x["local_only"]],
        "source_audit": "docs/DATA.md",
        "tokenizer": None if a.no_tokenizer else list(ADMISSION_TOKENIZER), "screen": screener.describe() | {"rule": "scripts/screen_lev.py blocking()"},
        "build_report": report, "code_sha256": {c: digest(ROOT / c) for c in CODE},
        "note": "Tiers are relative to train-v2; _meta.seen_s31 says whether the source is in the corpus of training run. Generated "
                "sources are eval-only at the held-out templates; train-v2 draws its own items at templates 0-3 with k <= k_train."})
    print(json.dumps({"files": files, "tiers": tiers}, indent=1))


if __name__ == "__main__":
    main()
