"""Score Jev on a frozen suite through OpenRouter's decisions endpoint,
as kev.jev does through the Vercel AI Gateway: the same TypeSafe request body (kev.data.api_request), the same rows and
report (kev.benchmark.evaluate_records), pinned to one Jev version. Every call's cost is OpenRouter's own `usage.cost`,
and the read stops before a call could pass --budget.

    python scripts/jev_openrouter.py --suite evals/v7/decision-v7 --out runs/jev-or-decision-v7 --budget 0.2

The key is read only from the OPENROUTER_API_KEY environment variable.
"""
import argparse, json, os, sys, threading, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.benchmark import evaluate_records   # noqa: E402
from kev.data import api_request   # noqa: E402
from kev.predictors import REFUSAL_STATUSES, JevRefused, http_detail   # noqa: E402
from kev.suite import digest, load_split, read_manifest, write_json   # noqa: E402

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13-20260917"


def openrouter_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return key


class OpenRouterJev:
    """One request per record; 408, 429, 5xx and connection errors are retried with backoff 1, 2, 4, ... s (at most 30);
    400 / 413 / 422 count as a refused record (JevRefused); any other client error stops the read."""

    def __init__(self, key, budget, attempts=6, concurrency=4, timeout=120):
        self.key, self.budget, self.attempts, self.concurrency, self.timeout = key, budget, attempts, concurrency, timeout
        self.lock = threading.Lock()
        self.calls = self.retries = self.input_tokens = self.output_tokens = 0
        self.cost, self.refusals, self.served_model = 0.0, {}, None
        self.started_at = datetime.now(timezone.utc).isoformat()

    def __call__(self, record):
        payload = json.dumps({**api_request(record), "model": MODEL}).encode()
        for attempt in range(self.attempts):
            with self.lock:
                if self.cost + 0.001 > self.budget:   # a record costs about $0.00002; the margin covers calls in flight
                    raise RuntimeError("Jev evaluation reached the cost cap")
                self.calls += 1
            req = urllib.request.Request(URL, data=payload, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                break
            except urllib.error.HTTPError as error:
                if error.code in REFUSAL_STATUSES:
                    with self.lock:
                        self.refusals[str(error.code)] = self.refusals.get(str(error.code), 0) + 1
                    raise JevRefused(f"Jev refused the request (HTTP {error.code}): {http_detail(error)}") from None
                if error.code not in (408, 429) and error.code < 500:
                    raise RuntimeError(f"OpenRouter answered HTTP {error.code}: {http_detail(error)}") from None
                last = error
            except Exception as error:   # connection errors and timeouts
                last = error
            if attempt == self.attempts - 1:
                raise RuntimeError(f"OpenRouter failed after {self.attempts} attempts: {last}")
            with self.lock:
                self.retries += 1
            time.sleep(min(2 ** attempt, 30))
        latency = 1000 * (time.perf_counter() - start)
        usage = body.get("usage") or {}
        if usage.get("cost") is None:
            raise RuntimeError("OpenRouter returned no usage cost; cannot account for spend")
        with self.lock:
            self.cost += float(usage["cost"])
            self.input_tokens += usage.get("input_tokens") or 0
            self.output_tokens += usage.get("output_tokens") or 0
            self.served_model = body.get("model", self.served_model)
        probs = {}
        for qid, q in record["questions"].items():
            a = body["answers"][qid]
            probs[qid] = {"true": float(a["noul"]), "false": 1 - float(a["noul"])} if q["type"] == "noul" else \
                {str(k): float(v) for k, v in a["probabilities"].items()}
        return {"probabilities": probs, "latency_ms": latency, "input_tokens": usage.get("input_tokens")}

    def accounting(self):
        return {"route": URL, "model": MODEL, "served_model": self.served_model, "started_at": self.started_at,
                "calls": self.calls, "retries": self.retries, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.cost, 6), "refusals": dict(sorted(self.refusals.items())),
                "concurrency": self.concurrency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", default="development")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=0.2, help="USD; the read stops before passing it")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    if not 0 < a.budget <= 5:
        ap.error("budget must be in (0, 5]")
    if Path(a.out).exists():
        ap.error("output directory already exists")
    records = load_split(a.suite, a.split)
    for r in records:   # the lev suites carry neither field; the defaults scripts/depth_rows.py uses for them
        r["_meta"].setdefault("variant", "clean")
        r["_meta"].setdefault("group_id", r["_meta"]["id"])
    predictor = OpenRouterJev(openrouter_key(), a.budget, concurrency=a.concurrency)
    try:
        heldout = read_manifest(a.suite).get("holdout_sources") or ()   # the lev suites record none
        report, _ = evaluate_records(records, predictor, a.out, heldout_sources=tuple(heldout), skip_overlong=True)
        report.update(suite_sha256=digest(Path(a.suite) / "manifest.json"), split=a.split, provider=predictor.accounting())
        write_json(Path(a.out) / "report.json", report)
        print(json.dumps({"clean": report["clean"], "coverage": report["coverage"], "provider": report["provider"]}, indent=1))
    finally:
        if Path(a.out).exists():
            write_json(Path(a.out) / "usage.json", predictor.accounting())


if __name__ == "__main__":
    main()
