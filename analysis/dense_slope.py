"""study (docs/REPRODUCING.md): data-scale slopes of a looped model against a dense control
trained on the same S31 / S60 / S120 data. Per question (answerable, non-report-only rows, on the questions both sides
read: their tokenizers skip different overlong records), correctness is the argmax at the deepest depth (looped) or of the
single readout (dense); the temperature does not move an argmax. slope = accuracy at S120 - accuracy at S31; the paired 95%
CI of (looped slope - dense slope) and of (looped - dense) at each scale resamples record groups (2,000 draws, seed 0).

    python analysis/dense_slope.py --pair "Ouro-2.6B vs Qwen3.5-2B" --loop s31=<dir> --loop s60=<dir> --loop s120=<dir> \
      --dense s31=<dir> --dense s60=<dir> --dense s120=<dir> [--pair ... again] --out <dir>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_depth import bootstrap, load_rows  # noqa: E402

PANELS = ("eval-v1-development", "breadth-v1-development")
SCALES = ("s31", "s60", "s120")


def correct(d, panel):
    """{(id, question): (group, right at the deepest depth)} for the answerable, non-report-only rows."""
    rows = load_rows(Path(d) / "rows" / f"{panel}.json")[1]
    return {(r["id"], r["question"] or ""): (r["group"], float(int(np.argmax(r["logits"][-1])) == r["label"]))
            for r in rows if not r["unanswerable"] and not r["report_only"]}


def pair_report(loop, dense):
    out = {}
    for p in PANELS:
        L = {s: correct(loop[s], p) for s in SCALES}; D = {s: correct(dense[s], p) for s in SCALES}
        keys = sorted(set.intersection(*(set(x) for x in [*L.values(), *D.values()])))
        g = [L["s31"][k][0] for k in keys]
        v = lambda m, s: np.array([m[s][k][1] for k in keys])
        r = {"questions": len(keys), "loop": {s: float(v(L, s).mean()) for s in SCALES}, "dense": {s: float(v(D, s).mean()) for s in SCALES}}
        dl, dd = v(L, "s120") - v(L, "s31"), v(D, "s120") - v(D, "s31")
        r["loop_slope"] = {"delta": float(dl.mean()), "ci95": bootstrap(g, dl)}
        r["dense_slope"] = {"delta": float(dd.mean()), "ci95": bootstrap(g, dd)}
        r["slope_difference"] = {"delta": float((dl - dd).mean()), "ci95": bootstrap(g, dl - dd)}
        r["loop_minus_dense"] = {s: {"delta": float((v(L, s) - v(D, s)).mean()), "ci95": bootstrap(g, v(L, s) - v(D, s))} for s in SCALES}
        out[p] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--loop", action="append", required=True, help="SCALE=dir, three per --pair, in order")
    ap.add_argument("--dense", action="append", required=True, help="SCALE=dir, three per --pair, in order")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pct = lambda x: f"{x['delta'] * 100:+.1f} [{x['ci95'][0] * 100:+.1f}, {x['ci95'][1] * 100:+.1f}]"
    report, lines = {}, []
    for i, name in enumerate(a.pair):
        loop = dict(x.split("=", 1) for x in a.loop[3 * i:3 * i + 3]); dense = dict(x.split("=", 1) for x in a.dense[3 * i:3 * i + 3])
        report[name] = r = pair_report(loop, dense)
        lines += [f"### {name}", "", "| panel | n | looped S31 / S60 / S120 | dense S31 / S60 / S120 | looped slope | dense slope | slope difference | looped − dense at S31 / S60 / S120 |",
                  "|---|---|---|---|---|---|---|---|"]
        for p, x in r.items():
            lines.append(f"| {p} | {x['questions']} | " + " / ".join(f"{x['loop'][s]:.3f}" for s in SCALES) + " | "
                         + " / ".join(f"{x['dense'][s]:.3f}" for s in SCALES) + f" | {pct(x['loop_slope'])} | {pct(x['dense_slope'])} | "
                         + f"{pct(x['slope_difference'])} | " + " / ".join(pct(x["loop_minus_dense"][s]) for s in SCALES) + " |")
        lines.append("")
    (out / "slopes.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "slopes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
