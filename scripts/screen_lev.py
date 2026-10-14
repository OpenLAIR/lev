"""Overlap screen of the Lev suites (docs/DATA.md, rule 5): one record is dropped when it overlaps any
reference collection. Counts and ids only; no reference text is stored.

    python scripts/screen_lev.py --records cand.jsonl --train data/local/kev08b-release-mix.jsonl \
      --train evals/v7/decision-v7:train --jevbench vendor/jevbench/datasets/public --out overlap.json

Builders import Screener and call it on each candidate. References (each item = its state plus its question text):
  train     what the model trains on (JSONL files or suite:split), given by the caller
  kev_eval  every calibration, development and test partition of every suite under evals/ (except --exclude dirs), read
            for screening only (test included, never scored here); unreadable ones (private mirrors) are listed
  jevbench  JevBench's public items
  extra     further JSONL files (evals/external/ekzhang-mmlupro-v1/records.jsonl by default)
A record overlaps a collection on any of: its normalised state equals a reference state (kev.suite.state_digest);
its state + question text + option names equal a reference item's; some reference item has >= CONTAINMENT of its word
8-grams inside the record (scripts/screen_longdoc_v1.py's rule: items under MIN_GRAMS ignored, grams shared by more than
COMMON items of a collection ignored as boilerplate).
Drop rule (blocking): a public-data record is dropped on any hit; a generated record (GENERATED sources: hard-v1 style
families, composition, contrastive, night2) only on an exact hit. Generated families share scenario sentences across
templates by construction (hard-v1 changes the wrapper and the numbers), so their containment hits are reported, not
dropped; their novelty comes from held-out templates, difficulty k and fresh numbers (docs/DATA.md, step 1).
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from kev.suite import digest, load_split, read_jsonl, read_manifest, state_digest, text_digest, write_json  # noqa: E402
from screen_longdoc_v1 import CONTAINMENT, MIN_GRAMS, Collection, grams, item_text, words  # noqa: E402

EKZHANG = ROOT / "evals/external/ekzhang-mmlupro-v1/records.jsonl"
GENERATED_PREFIXES = ("hard_", "lev_gen_", "night2_")
GENERATED = {"compositional", "composition_holdout", "legacy_policy", "legacy_holdout", "contrastive", "unknowable", "unknowable_control", "buried"}


def generated(r):
    src = r.get("_meta", {}).get("source", "")
    return src in GENERATED or src.startswith(GENERATED_PREFIXES)


def blocking(r, hits):
    """Whether these hits drop the record (the drop rule above)."""
    if not generated(r): return bool(hits)
    return any(w != "containment" for why in hits.values() for w in why)


def item_key(r):
    """state + every question's instructions and option names, normalised: two records that ask the same thing."""
    state, instructions = item_text(r)
    options = []
    for q in (r.get("questions") or {}).values() if isinstance(r.get("questions"), dict) else []:
        c = q.get("criteria")
        options += sorted(c) if isinstance(c, dict) else [str(x) for x in (c or [])]
    text = lambda x: x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)
    return text_digest(" ".join([text(state)] + [text(x) for x in instructions] + options))


class Reference(Collection):
    """A screen_longdoc_v1 Collection that also keeps state_digest and item_key sets."""

    def __init__(self, name):
        super().__init__(name); self.digests, self.items = set(), set()

    def add_record(self, r):
        state, instructions = item_text(r)
        self.add(state, instructions)
        if state: self.digests.add(state_digest(state))
        self.items.add(item_key(r))


def read_records(spec):
    """A JSONL path or suite:split (load_split, test allowed: screening only)."""
    if Path(spec).is_file(): return read_jsonl(spec), {"file": str(spec), "sha256": digest(spec)}
    suite, split = spec.rsplit(":", 1)
    return load_split(suite, split, allow_test=True), {"suite": suite, "split": split}


def collect(name, specs):
    c = Reference(name)
    for spec in specs:
        rows, info = read_records(spec); c.files[str(spec)] = info | {"records": len(rows)}
        for r in rows: c.add_record(r)
    return c


def kev_eval(exclude=()):
    c = Reference("kev_eval")
    skip = {Path(e).resolve() for e in exclude}
    for manifest in sorted((ROOT / "evals").rglob("manifest.json")):
        d = manifest.parent
        if any(d.resolve() == s or s in d.resolve().parents for s in skip): continue
        try: files = read_manifest(d).get("files", {})
        except Exception as e: c.files[str(d.relative_to(ROOT))] = f"unreadable: {type(e).__name__}"; continue
        for split in ("calibration", "development", "test"):
            if f"{split}.jsonl" not in files or not files[f"{split}.jsonl"].get("records", 1): continue
            key = f"{d.relative_to(ROOT)}/{split}.jsonl"
            try: rows = load_split(d, split, allow_test=True)
            except Exception as e: c.files[key] = f"unreadable: {type(e).__name__}"; continue
            c.files[key] = {"records": len(rows), "sha256": files[f"{split}.jsonl"]["sha256"]}
            for r in rows: c.add_record(r)
    return c


def jevbench(directory):
    c = Reference("jevbench")
    for path in sorted(Path(directory).glob("*.jsonl")):
        rows = read_jsonl(path); c.files[path.name] = {"items": len(rows), "sha256": digest(path)}
        for r in rows: c.add_record(r)
    return c


class Screener:
    def __init__(self, collections):
        self.collections = [c for c in collections if c.sets or c.digests]
        for c in self.collections: c.index()

    def check(self, r):
        """{collection: [reasons]} for every collection the record overlaps; empty when clean."""
        state, instructions = item_text(r)
        mine = grams(words(state) + [w for x in instructions for w in words(x)])
        d, k, hits = state_digest(state) if state else None, item_key(r), {}
        for c in self.collections:
            why = [w for w, hit in (("exact_state", d in c.digests), ("exact_item", k in c.items),
                                    ("containment", len(mine) >= 1 and c.screen(mine) >= CONTAINMENT)) if hit]
            if why: hits[c.name] = why
        return hits

    def describe(self):
        return {"containment_threshold": CONTAINMENT, "min_grams": MIN_GRAMS, "n": 8, "normalisation": "casefold, \\w+ tokens; state_digest for exact states",
                "references": {c.name: {"items": len(c.sets), "states": len(c.digests), "files": c.files} for c in self.collections}}


def references(train=(), jevbench_dir=None, extra=(EKZHANG,), exclude=()):
    cs = [collect("train", train)] if train else []
    cs.append(kev_eval(exclude))
    if jevbench_dir: cs.append(jevbench(jevbench_dir))
    if extra: cs.append(collect("extra", [str(e) for e in extra]))
    return Screener(cs)


def report(screener, records, key=lambda r: r["_meta"].get("source", "?")):
    """Counts per key (overlapping = any hit, dropped = blocking hits) and the dropped ids with their reasons (ids only)."""
    counts, dropped = defaultdict(Counter), {}
    for r in records:
        hits = screener.check(r); g = key(r)
        counts[g]["records"] += 1
        if hits:
            counts[g]["overlapping"] += 1
            if blocking(r, hits): counts[g]["dropped"] += 1; dropped[r["_meta"]["id"]] = hits
            for c, why in hits.items():
                for w in why: counts[g][f"{c}:{w}"] += 1
    return {"counts": {g: dict(v) for g, v in sorted(counts.items())}, "dropped_ids": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="append", required=True, help="candidate JSONL or suite:split, repeatable")
    ap.add_argument("--train", action="append", default=[], help="training JSONL or suite:split, repeatable")
    ap.add_argument("--jevbench", default=str(ROOT / "vendor/jevbench/datasets/public"))
    ap.add_argument("--exclude", action="append", default=[], help="suite directories left out of kev_eval (the suite being screened)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = references(a.train, a.jevbench, exclude=a.exclude)
    records = [r for spec in a.records for r in read_records(spec)[0]]
    out = s.describe() | {"records": a.records} | report(s, records)
    write_json(a.out, out)
    print({g: (v.get("dropped", 0), v.get("overlapping", 0), v["records"]) for g, v in out["counts"].items()})


if __name__ == "__main__":
    main()
