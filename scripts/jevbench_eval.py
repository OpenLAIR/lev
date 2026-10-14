"""JevBench public items, scored in process with JevBench's own code, at every depth of a looped model in one pass.

Each public item becomes the request JevBench's TypeSafe adapter would send (jevbench.adapters.base.build_question),
goes through Kev's serving conversion (kev.api.to_record / to_answers, the /v1/systemone body), is mapped to exact-label
probabilities the way the adapter maps a native answer, and is scored by jevbench.scoring.score_task. A natively looped
checkpoint (Ouro backbone) is read out at every step from one forward pass; reading step k equals computing only k steps
(tests/test_ouro.py), so one run gives the whole accuracy-vs-depth curve. Optional latency pass: per depth, one request
at a time, the model computing only that many steps.

Summary per depth: tier accuracy, JevBench v1.3 chance-corrected Intelligence over the public tiers (easy, standard,
hard; the judge tier has no public items, so it is renormalised away exactly as composite_v13.intelligence does), hard
tier ECE and the Calibration axis (ECE part only: public items carry no gold distributions), latency p50 / p95 and the
Speed axis under JevBench's self-hosted adjustment (x2 + 0.15 s).

    python scripts/jevbench_eval.py --run <checkpoint> --out <dir> [--latency] [--jevbench ../refs/jevbench]
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.api import SystemOneRequest, to_answers, to_record   # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions   # noqa: E402
from kev.model import admit   # noqa: E402

TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}   # public file -> JevBench tier name


def jevbench_modules(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from jevbench import composite_v13, metrics, scoring, tasks
    from jevbench.adapters.base import build_question
    return composite_v13, metrics, scoring, tasks, build_question


def to_jevbench_probs(answer, task):
    """The exact-label distribution JevBench's typesafe adapter builds from a native answer."""
    t = task.question["type"]
    if t == "noul": return {"yes": answer["noul"], "no": 1.0 - answer["noul"]}
    return dict(answer["probabilities"])


def summarize(rows, comp, metrics):
    by_tier = {}
    for tier in ("easy", "standard", "hard"):
        rs = [r for r in rows if r["tier"] == tier]
        if rs: by_tier[tier] = sum(r["correct"] for r in rs) / len(rs)
    hard = [(r["confidence"], r["correct"]) for r in rows if r["tier"] == "hard" and r["confidence"] is not None]
    ece = metrics.ece_top_label(hard)["ece"] if hard else None
    n = len(rows)
    return {"n": n, "accuracy_all": sum(r["correct"] for r in rows) / n, "accuracy": by_tier,
            "intelligence_public": comp.intelligence(by_tier), "hard_ece": ece, "calibration_public": comp.calibration(ece)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jevbench", default=str(Path(__file__).resolve().parents[1] / "vendor/jevbench"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--latency", action="store_true", help="also time one request at a time per depth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rotations", type=int, default=1, help="choice items: average the distribution over this many cyclic rotations of the option order (test-time augmentation; costs that many passes)")
    ap.add_argument("--temperatures", default="", help="JSON from scripts/fit_depth_temperatures.py: divide each depth's logits by its fitted temperature")
    a = ap.parse_args()
    comp, metrics, scoring, tasks_mod, build_question = jevbench_modules(a.jevbench)
    tasks = []
    for f, tier in TIERS.items():
        tasks += [(tier, t) for t in tasks_mod.load_jsonl(str(Path(a.jevbench) / "datasets/public" / f"{f}.jsonl"))]
    if a.limit: tasks = tasks[: a.limit]
    if a.device == "cuda":   # the exact path, as kev.predictors.LocalPredictor scores
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    ck = Checkpoint(a.run)
    tok, model = ck.load(a.device, LoadOptions(dtype=torch.bfloat16 if a.dtype == "bf16" else None))
    native = getattr(model, "native_loop", False)
    depths = list(range(1, model.depth + 1)) if native else [None]
    temps = json.load(open(a.temperatures, encoding="utf-8"))["temperatures"] if a.temperatures else None
    if temps is not None:
        if len(temps) != len(depths): raise SystemExit(f"{len(temps)} temperatures for {len(depths)} depths")
        model.head.temperature = 1.0   # the per-depth temperatures replace the checkpoint's single one
    encoded = []
    for tier, t in tasks:
        req = SystemOneRequest(state=t.state, questions={"decision": build_question(t)})
        rec, meta = to_record(req)
        encoded.append((tier, t, admit(model, tok, rec), meta))
    rows = {k: [] for k in depths}

    def depth_probs(enc, meta, t):
        """JevBench exact-label distributions of one encoded request, one per depth."""
        if temps is None:
            steps = model.probs_steps(enc) if native else [model.probs(enc)]
        else:
            with torch.no_grad():
                raw = model.forward_steps_batch([enc]) if native else [model.forward_batch([enc])]
            steps = [[torch.softmax(z.float() / tk, -1).cpu() for z in step[0]] for step, tk in zip(raw, temps)]
        return [to_jevbench_probs(to_answers([ps[0].tolist()], meta)["decision"], t) for ps in steps]

    for tier, t, enc, meta in encoded:
        per_depth = depth_probs(enc, meta, t)
        if a.rotations > 1 and t.question["type"] == "choice":
            q = build_question(t); keys = list(q["criteria"])
            for r in range(1, min(a.rotations, len(keys))):
                order = keys[r:] + keys[:r]
                rec_r, meta_r = to_record(SystemOneRequest(state=t.state, questions={"decision": {**q, "criteria": {k: q["criteria"][k] for k in order}}}))
                for acc_p, p in zip(per_depth, depth_probs(admit(model, tok, rec_r), meta_r, t)):
                    for lab in acc_p: acc_p[lab] += p[lab]
            n = min(a.rotations, len(keys))
            per_depth = [{lab: v / n for lab, v in p.items()} for p in per_depth]
        for k, probs in zip(depths, per_depth):
            s = scoring.score_task(probs, t)
            conf = scoring.top_label_confidence(s["probs"]) if s["valid"] else None
            rows[k].append({"id": t.id, "tier": tier, "correct": bool(s["correct"]), "valid": s["valid"], "predicted": s.get("predicted"),
                            "confidence": conf, "probs": s.get("probs")})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"run": a.run, "base": ck.meta.base, "temperatures": temps, "rotations": a.rotations, "native_loop": native, "dtype": a.dtype, "device": a.device,
              "gpu": torch.cuda.get_device_name() if a.device == "cuda" else None, "depths": {}}
    for k in depths:
        report["depths"][str(k or 1)] = summarize(rows[k], comp, metrics)
        (out / f"rows-depth{k or 1}.json").write_text(json.dumps(rows[k]), encoding="utf-8")
    if a.latency:
        model_dtype = next(model.parameters()).dtype
        for k in depths:
            if native: model.set_depth(k)
            for _, _, enc, _ in encoded[:5]: model.probs(enc)   # warm-up
            times = []
            for _, _, enc, _ in encoded:
                if a.device == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter(); model.probs(enc)
                if a.device == "cuda": torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            times.sort()
            p50, p95 = statistics.median(times), times[min(len(times) - 1, int(0.95 * len(times)))]
            report["depths"][str(k or 1)]["latency"] = {"p50_s": p50, "p95_s": p95, "mean_s": sum(times) / len(times), "dtype": str(model_dtype),
                                                        "speed_axis_selfhosted": comp.speed(p50, p95, "gpu")}
        if native: model.set_depth(depths[-1])
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, s in report["depths"].items():
        lat = s.get("latency", {})
        print(f"depth {k}: all {s['accuracy_all']:.3f} | " + " ".join(f"{t} {v:.3f}" for t, v in s["accuracy"].items())
              + f" | intelligence {s['intelligence_public']:.1f} | hard ECE {s['hard_ece']:.3f}"
              + (f" | p50 {lat['p50_s']*1000:.0f} ms p95 {lat['p95_s']*1000:.0f} ms speed {lat['speed_axis_selfhosted']:.1f}" if lat else ""))


if __name__ == "__main__":
    main()
