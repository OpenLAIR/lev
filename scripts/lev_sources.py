"""Converters from the 14 public datasets approved in docs/DATA.md ("决定", 2026-10-10) to Kev labelled
requests (the shape kev.data.load_records / kev.data.materialize take):

    {"state": str | dict, "questions": {qid: {"type": "choice" | "noul", "instructions", "criteria", "label", "src"}},
     "_meta": {"source", "id", "group_id", "repo", "revision", "split", "row", "difficulty", "licence", "local_only", ...}}

    from scripts.lev_sources import SOURCES, load
    records = load("mathqa", "train", n=1000, seed=0)     # load.stats says how many rows were read, kept and why others were not

Roles (manifest `trainable_sources` / `eval_only_sources`): six trainable (CondaQA, Social IQa, CosmosQA, MathQA, QASC,
PIQA), eight eval-only far sources (BoardgameQA, StepGame, CLadder, StrategyQA, LogiQA 2.0, AR-LSAT, QuAIL, TimeDial);
LogiQA 2.0, AR-LSAT, QuAIL and TimeDial are local only (NC / share-alike / exam copyright), never published.
Every file is read at a pinned 40-character commit (the audit's commit; `refs/convert/parquet` commits for script-only
repos); nothing reads `main`. Labels come from the datasets only, except that MathQA keeps a row only when its
`annotated_formula`, executed by `execute_formula`, lands on the labelled option (MathQA's operation set below).

Conventions follow kev/data.py: choice keys are short slugs or letters with the option text as the criteria value, the
label is the key; options keep the dataset's order (TimeDial, whose correct answers are listed first, is shuffled with a
per-row seed). The datasets' own "don't know" / "not enough information" / "unknown" / "None of the above" options are
ordinary options scored against the gold label (audit, 决定). Row ids are the dataset's own id where it has one, else
the row's index in its split file; `group_id` is the unit that must stay inside one partition (a shared passage, game,
causal model or MathQA template).
"""
import json
import math
import random
import re
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from huggingface_hub import constants as hf_constants, hf_hub_download  # noqa: E402

from kev.data import materialize  # noqa: E402
from kev.suite import text_digest  # noqa: E402

TRAINABLE = ("condaqa", "social_iqa", "cosmosqa", "mathqa", "qasc", "piqa")
EVAL_ONLY = ("boardgameqa", "stepgame", "cladder", "strategyqa", "logiqa2", "ar_lsat", "quail", "timedial")

# repo + revision: the audit's commit resolved to the full sha (HfApi().list_repo_commits). "files" maps an HF split to
# the file(s) read for it. StepGame reads the corrected benchmark of Li et al. (AAAI 2024, arXiv 2401.03991) from its
# GitHub repository: the clean test files with every example that uses one of the wrong templates removed.
SOURCES = {
    "condaqa": {"repo": "lasha-nlp/CONDAQA", "revision": "3c9caa2f2f6960711e7f4d2e800581def2b6c183", "role": "trainable",
                "licence": "Apache-2.0", "local_only": False, "splits": {"train": "train", "eval": "dev"},
                "files": {"train": ["train.json"], "dev": ["dev.json"], "test": ["test.json"]}},
    "social_iqa": {"repo": "allenai/social_i_qa", "revision": "537a2ec8ec565adc0b70b70752893e59e024df26", "branch": "refs/convert/parquet",
                   "role": "trainable", "licence": "CC-BY-4.0", "local_only": False, "splits": {"train": "train", "eval": "validation"},
                   "files": {"train": ["default/train/0000.parquet"], "validation": ["default/validation/0000.parquet"]}},
    "cosmosqa": {"repo": "allenai/cosmos_qa", "revision": "ed50fe83db62b355759053b1cd5eef344ee371e0", "branch": "refs/convert/parquet",
                 "role": "trainable", "licence": "CC-BY-4.0", "local_only": False, "splits": {"train": "train", "eval": "validation"},
                 "files": {"train": ["default/train/0000.parquet"], "validation": ["default/validation/0000.parquet"]}},
    "mathqa": {"repo": "allenai/math_qa", "revision": "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6", "branch": "refs/convert/parquet",
               "role": "trainable", "licence": "Apache-2.0", "local_only": False, "splits": {"train": "train", "eval": ("validation", "test")},
               "files": {"train": ["default/train/0000.parquet"], "validation": ["default/validation/0000.parquet"],
                         "test": ["default/test/0000.parquet"]}},
    "qasc": {"repo": "allenai/qasc", "revision": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "role": "trainable", "licence": "CC-BY-4.0",
             "local_only": False, "splits": {"train": "train", "eval": "validation"},
             "files": {"train": ["data/train-00000-of-00001.parquet"], "validation": ["data/validation-00000-of-00001.parquet"]}},
    "piqa": {"repo": "ybisk/piqa", "revision": "142c51238b3ca2bc61e9a075913871b8b600e8e1", "branch": "refs/convert/parquet",
             "role": "trainable", "licence": "AFL-3.0", "local_only": False, "splits": {"train": "train", "eval": "validation"},
             "files": {"train": ["plain_text/train/0000.parquet"], "validation": ["plain_text/validation/0000.parquet"]}},
    "boardgameqa": {"repo": "tasksource/Boardgame-QA", "revision": "78e38c3c8df3b4f6de7ae8bd1fc6a8bd1f31be56", "role": "eval_only",
                    "licence": "CC-BY-4.0", "local_only": False, "splits": {"train": None, "eval": "test"},
                    "files": {"test": ["data/test-00000-of-00001-be86a421cd54a34b.parquet"]}},
    "stepgame": {"repo": "Fangjun-Li/SpatialLM-StepGame", "host": "github", "revision": "1114cb58382e1cea8c66c8bf98b5749fb32d7575",
                 "role": "eval_only", "licence": "MIT (StepGame, ShiZhengyan/StepGame); the corrected-subset repository has no licence file",
                 "local_only": False, "splits": {"train": None, "eval": "test"}, "version": "correct_clean (Li et al., AAAI 2024)",
                 "files": {"test": [f"data/correct_clean/qa{k}_test.txt" for k in range(1, 11)]}},
    "cladder": {"repo": "causal-nlp/CLadder", "revision": "18d684e444ed07cc57ae0f3a91ee983a79b8119a", "role": "eval_only", "licence": "MIT",
                "local_only": False, "splits": {"train": None, "eval": "full_v1.5_default"},
                "files": {"full_v1.5_default": ["data/full_v1.5_default.csv"]}},
    "strategyqa": {"repo": "voidful/StrategyQA", "revision": "2279eaf9f2580aef77ed6fa0efd7846c381ab5a0", "role": "eval_only", "licence": "MIT",
                   "local_only": False, "report_only": True, "splits": {"train": None, "eval": "train"},
                   "files": {"train": ["strategyqa_train.json"]}},
    "logiqa2": {"repo": "datatune/LogiQA2.0", "revision": "edbf6440d36a03f715c0c9d479f421f2ece87d6a", "role": "eval_only",
                "licence": "CC-BY-NC-SA-4.0 (upstream csitfun/LogiQA2.0; the HF card's mit conflicts)", "local_only": True,
                "splits": {"train": None, "eval": "test"}, "files": {"test": ["MRC/test.txt"]}},
    "ar_lsat": {"repo": "tasksource/lsat-ar", "revision": "0adf424eef17df6a3506ab3d94ed194a80dfd837", "role": "eval_only",
                "licence": "MIT (zhongwanjun/AR-LSAT); questions copyright LSAC", "local_only": True,
                "splits": {"train": None, "eval": ("train", "validation", "test")},
                "files": {"train": ["data/train-00000-of-00001-123a2341f5f908d9.parquet"],
                          "validation": ["data/validation-00000-of-00001-ca642bf6efcbaaf0.parquet"],
                          "test": ["data/test-00000-of-00001-7c743adafc79bc47.parquet"]}},
    "quail": {"repo": "textmachinelab/quail", "revision": "2bd9d7f90a532fe1a910b70972cef5fda341c8fe", "role": "eval_only",
              "licence": "CC-BY-NC-SA-4.0", "local_only": True, "splits": {"train": None, "eval": ("validation", "challenge")},
              "files": {"validation": ["quail/validation-00000-of-00001.parquet"], "challenge": ["quail/challenge-00000-of-00001.parquet"]}},
    "timedial": {"repo": "google-research-datasets/time_dial", "revision": "6c65411432c135ac7cca3669db8da476c858b141",
                 "branch": "refs/convert/parquet", "role": "eval_only", "licence": "CC-BY-NC-SA-4.0", "local_only": True,
                 "splits": {"train": None, "eval": "test"}, "files": {"test": ["default/test/0000.parquet"]}},
}
assert set(SOURCES) == set(TRAINABLE) | set(EVAL_ONLY)
assert all(len(s["revision"]) == 40 and (s["role"] == "trainable") == (k in TRAINABLE) for k, s in SOURCES.items())


class Skip(Exception):
    """A row that does not convert; the message is the reason counted in load.stats."""


# ---------------------------------------------------------------------------------------------------------------- files
def _github_file(repo, revision, path):
    dest = Path(hf_constants.HF_HOME) / "lev-github" / repo / revision / path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{revision}/{path}", timeout=120) as r: data = r.read()
        tmp = dest.with_suffix(dest.suffix + ".part"); tmp.write_bytes(data); tmp.replace(dest)
    return dest


def _file(name, path):
    s = SOURCES[name]
    if s.get("host") == "github": return _github_file(s["repo"], s["revision"], path)
    return Path(hf_hub_download(s["repo"], path, repo_type="dataset", revision=s["revision"]))


def _plain(v):
    """numpy arrays / scalars from parquet -> python lists / scalars."""
    if hasattr(v, "tolist"): v = v.tolist()
    if isinstance(v, dict): return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list): return [_plain(x) for x in v]
    return v


def _read_file(name, path):
    p = _file(name, path)
    if path.endswith(".parquet") or path.endswith(".csv"):
        import pandas as pd
        df = pd.read_parquet(p) if path.endswith(".parquet") else pd.read_csv(p, dtype=str, keep_default_na=False)
        return [{k: _plain(v) for k, v in r.items()} for r in df.to_dict("records")]
    if path.endswith(".txt") and SOURCES[name].get("host") == "github":
        return _babi(p.read_text(encoding="utf-8"), int(re.search(r"qa(\d+)_", path).group(1)))
    text = p.read_text(encoding="utf-8")
    if text.lstrip().startswith("["): return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@lru_cache(maxsize=None)
def _rows(name, split):
    rows = []
    for path in SOURCES[name]["files"][split]: rows.extend(_read_file(name, path))
    return tuple(rows)


def rows(name, split):
    """The raw rows of one HF split (list of dicts, in file order)."""
    return list(_rows(name, split))


def _babi(text, k):
    """StepGame's bAbI format: numbered story lines, then '<n> question\\tanswer\\t1' closes an example."""
    out, story = [], []
    for line in text.splitlines():
        if not line.strip(): continue
        _, body = line.split(" ", 1)
        if "\t" in body:
            q, a, _ = body.split("\t")
            out.append({"story": list(story), "question": q.strip(), "label": a.strip(), "k": k, "index": len(out)})
            story = []
        else:
            story.append(body.strip())
    return out


# ------------------------------------------------------------------------------------------------------------- helpers
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _meta(name, split, row, row_id, group, difficulty=None, **extra):
    s = SOURCES[name]
    src = f"lev_{name}"
    m = {"source": src, "variant": "clean", "id": f"{src}/{split}/{row_id}", "group_id": f"{src}/{group}", "repo": s["repo"],
         "revision": s["revision"], "split": split, "row": row, "difficulty": difficulty, "licence": s["licence"], "local_only": s["local_only"]}
    if s.get("host"): m["host"] = s["host"]
    if s.get("report_only"): m["report_only"] = True
    m.update(extra)
    return m


def _choice(name, state, instructions, options, label_index, meta, qid="answer"):
    """Lettered options in dataset order -> one choice question."""
    if not 2 <= len(options) <= len(LETTERS): raise Skip("option_count")
    texts = [str(o).strip() for o in options]
    if any(not t for t in texts): raise Skip("empty_option")
    if len(set(t.casefold() for t in texts)) != len(texts): raise Skip("duplicate_options")
    if not 0 <= label_index < len(texts): raise Skip("label_out_of_range")
    keys = list(LETTERS[:len(texts)])
    return {"state": state, "questions": {qid: {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, texts)),
                                                "label": keys[label_index], "src": f"lev_{name}"}}, "_meta": meta}


def _text(v, reason="empty_text"):
    if not isinstance(v, str) or not v.strip(): raise Skip(reason)
    return v.strip()


def _int(v, reason="bad_label"):
    try: return int(str(v).strip())
    except ValueError: raise Skip(reason) from None


# ---------------------------------------------------------------------------------------------------------- trainable
CONDAQA = {"YES": "yes", "NO": "no", "DON'T KNOW": "dont_know"}
CONDAQA_CRITERIA = {"yes": None, "no": None, "dont_know": "The passage does not settle it"}


def convert_condaqa(row, split, idx):
    label = CONDAQA.get(str(row.get("label", "")).strip())
    if label is None: raise Skip("span_answer_not_yes_no_dont_know")
    meta = _meta("condaqa", split, idx, row["SampleID"], f"passage/{row['PassageID']}",
                 edit_id=row.get("PassageEditID"), question_id=row.get("QuestionID"), cue=row.get("original cue"))
    return [{"state": _text(row["sentence1"]), "questions": {"answer": {"type": "choice", "instructions": _text(row["sentence2"]),
             "criteria": dict(CONDAQA_CRITERIA), "label": label, "src": "lev_condaqa"}}, "_meta": meta}]


def convert_social_iqa(row, split, idx):
    context = _text(row["context"])
    meta = _meta("social_iqa", split, idx, idx, f"context/{text_digest(context)[:16]}")
    return [_choice("social_iqa", context, _text(row["question"]), [row["answerA"], row["answerB"], row["answerC"]],
                    _int(row["label"]) - 1, meta)]


def convert_cosmosqa(row, split, idx):
    context = _text(row["context"])
    meta = _meta("cosmosqa", split, idx, row["id"], f"context/{text_digest(context)[:16]}")
    return [_choice("cosmosqa", context, _text(row["question"]), [row[f"answer{i}"] for i in range(4)], _int(row["label"]), meta)]


def convert_qasc(row, split, idx):
    ch = row["choices"]
    labels, texts = [str(x).strip() for x in ch["label"]], list(ch["text"])
    key = str(row.get("answerKey", "")).strip()
    if key not in labels: raise Skip("no_answer_key")
    meta = _meta("qasc", split, idx, row["id"], f"fact/{text_digest(str(row.get('combinedfact') or row['id']))[:16]}")
    return [_choice("qasc", {"question": _text(row["question"])}, "Which option correctly answers the question?", texts, labels.index(key), meta)]


def convert_piqa(row, split, idx):
    goal = _text(row["goal"])
    meta = _meta("piqa", split, idx, idx, f"goal/{text_digest(goal)[:16]}")
    return [_choice("piqa", {"goal": goal}, "Which solution achieves the goal?", [row["sol1"], row["sol2"]], _int(row["label"]), meta)]


# ---- MathQA: the operation set of MathQA (Amini et al. 2019, operation_list.txt), as it appears in annotated_formula.
def _int_arg(x):
    if not float(x).is_integer(): raise ValueError("non-integer argument")
    return int(x)


def _heron(a, b, c):
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


OPS = {
    "add": lambda a, b: a + b, "subtract": lambda a, b: a - b, "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
    "power": lambda a, b: a ** b, "sqrt": math.sqrt, "inverse": lambda a: 1 / a, "negate": lambda a: -a,
    "factorial": lambda a: math.factorial(_int_arg(a)), "floor": math.floor, "reminder": lambda a, b: a % b,
    "gcd": lambda a, b: math.gcd(_int_arg(a), _int_arg(b)), "lcm": lambda a, b: math.lcm(_int_arg(a), _int_arg(b)),
    "choose": lambda n, k: math.comb(_int_arg(n), _int_arg(k)), "permutation": lambda n, k: math.perm(_int_arg(n), _int_arg(k)),
    "log": math.log, "max": max, "min": min,
    "speed": lambda d, t: d / t, "stream_speed": lambda down, up: (down - up) / 2, "speed_in_still_water": lambda down, up: (down + up) / 2,
    "negate_prob": lambda p: 1 - p,
    "original_price_before_gain": lambda gain, price: price * 100 / (100 + gain),
    "original_price_before_loss": lambda loss, price: price * 100 / (100 - loss),
    "p_after_gain": lambda gain, price: price * (100 + gain) / 100, "p_after_loss": lambda loss, price: price * (100 - loss) / 100,
    "rectangle_area": lambda a, b: a * b, "rectangle_perimeter": lambda a, b: 2 * (a + b),
    "square_area": lambda a: a * a, "square_perimeter": lambda a: 4 * a,
    "square_edge_by_perimeter": lambda p: p / 4, "square_edge_by_area": math.sqrt,
    "rhombus_area": lambda d1, d2: d1 * d2 / 2, "rhombus_perimeter": lambda a: 4 * a,
    "quadrilateral_area": lambda d, h1, h2: d * (h1 + h2) / 2, "trapezium_area": lambda a, b, h: (a + b) * h / 2,
    "triangle_area": lambda b, h: b * h / 2, "triangle_perimeter": lambda a, b, c: a + b + c, "triangle_area_three_edges": _heron,
    "diagonal": lambda a, b: math.sqrt(a * a + b * b),
    "circle_area": lambda r: math.pi * r * r, "circumface": lambda r: 2 * math.pi * r,
    "volume_cube": lambda a: a ** 3, "surface_cube": lambda a: 6 * a * a, "cube_edge_by_volume": lambda v: v ** (1 / 3),
    "volume_rectangular_prism": lambda a, b, c: a * b * c, "surface_rectangular_prism": lambda a, b, c: 2 * (a * b + b * c + a * c),
    "volume_cylinder": lambda r, h: math.pi * r * r * h, "surface_cylinder": lambda r, h: 2 * math.pi * r * (r + h),
    "volume_sphere": lambda r: 4 / 3 * math.pi * r ** 3, "surface_sphere": lambda r: 4 * math.pi * r * r,
    "volume_cone": lambda r, h: math.pi * r * r * h / 3,
    "sine": lambda x: math.sin(math.radians(x)), "cosine": lambda x: math.cos(math.radians(x)), "tangent": lambda x: math.tan(math.radians(x)),
}
CONSTS = {"const_pi": math.pi, "const_deg_to_rad": math.pi / 180}
_TOKEN = re.compile(r"\s*(?:(?P<name>const_[0-9_]+(?:\.0+)?(?![\w.])|[a-z_][a-z0-9_]*)|(?P<num>-?\d+(?:\.\d+)?(?:e-?\d+)?)|(?P<punct>[(),]))", re.I)


def _const(name):
    if name in CONSTS: return CONSTS[name]
    m = re.fullmatch(r"const_(\d+)(?:_(\d+))?(?:\.0+)?", name)   # const_2.0 occurs as a spelling of const_2
    if not m: raise ValueError(f"unknown constant {name}")
    return float(f"{m.group(1)}.{m.group(2)}") if m.group(2) else float(m.group(1))


def execute_formula(formula):
    """(value, number of operations) of a MathQA annotated_formula such as 'divide(multiply(36, const_100), 3)'.
    Raises ValueError on an unknown operation or constant, a malformed formula or a math error."""
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", formula.strip())   # thousands separators inside numbers: 30,000 (arguments are ', '-separated)
    toks, pos = [], 0
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos: raise ValueError(f"cannot parse {s[pos:pos + 20]!r}")
        toks.append((m.lastgroup, m.group(m.lastgroup))); pos = m.end()
        while pos < len(s) and s[pos].isspace(): pos += 1
    i, ops = 0, 0

    def expr():
        nonlocal i, ops
        if i >= len(toks): raise ValueError("unexpected end")
        kind, v = toks[i]; i += 1
        if kind == "num": return float(v)
        if kind != "name": raise ValueError(f"unexpected {v!r}")
        if v.startswith("const_") and (i >= len(toks) or toks[i][1] != "("): return _const(v)
        if v not in OPS: raise ValueError(f"unknown operation {v}")
        if i >= len(toks) or toks[i][1] != "(": raise ValueError(f"{v} without arguments")
        i += 1; args = []
        while True:
            args.append(expr())
            if i >= len(toks): raise ValueError("unclosed call")
            sep = toks[i][1]; i += 1
            if sep == ")": break
            if sep != ",": raise ValueError(f"unexpected {sep!r}")
        ops += 1
        try: out = OPS[v](*args)
        except (ArithmeticError, TypeError) as e: raise ValueError(f"{v}{tuple(args)}: {e}") from None
        if isinstance(out, complex) or (isinstance(out, int) and abs(out) > 1e300) or not math.isfinite(out):
            raise ValueError(f"{v}{tuple(args)}: not a finite real")
        return float(out)

    value = expr()
    if i != len(toks): raise ValueError("trailing tokens")
    return value, ops


_OPTION_SPLIT = re.compile(r"(?:^|\s*,\s*)\[?'?([a-e])\s*\)\s*")
_NUMBER = re.compile(r"(?<![\w.])(-\s?)?(\d+(?:\.\d+)?|\.\d+)(?:\s+(\d+)\s*/\s*(\d+)|\s*/\s*(\d+(?:\.\d+)?))?")


def parse_options(options):
    """MathQA's options string 'a ) rs . 400 , b ) 300 , ...' -> {'a': 'rs . 400', 'b': '300', ...}; None if malformed."""
    s = options.strip().strip("[]")
    parts = _OPTION_SPLIT.split(s)
    if parts[0].strip(): return None
    keys, texts = parts[1::2], [t.strip().strip("',") for t in parts[2::2]]
    if keys != list("abcde")[:len(keys)] or len(keys) < 2: return None
    return dict(zip(keys, texts))


def option_value(text):
    """The number an option states ('rs . 400' -> 400, '2 1 / 2' -> 2.5, '3 / 5' -> 0.6, '3 : 4' -> 0.75, '1 , 000' -> 1000, '- 5' -> -5);
    None when it states none, or more than one, or something only a formula could (√, ^, π)."""
    t = re.sub(r"(?<=\d)\s*,\s*(?=\d{3}\b)", "", text)
    if re.search(r"[√^π]|sqrt|\bpi\b", t): return None
    ratio = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", t)
    if ratio: return float(ratio.group(1)) / float(ratio.group(2)) if float(ratio.group(2)) else None
    found = list(_NUMBER.finditer(t))
    if len(found) != 1: return None
    m = found[0]
    v = float(m.group(2))
    if m.group(3): v += float(m.group(3)) / float(m.group(4))
    elif m.group(5):
        if float(m.group(5)) == 0: return None
        v /= float(m.group(5))
    return -v if m.group(1) else v


MATHQA_REL_TOL = 0.01


def mathqa_check(row):
    """(n_ops, options) when the annotated formula lands on the labelled option: its value within MATHQA_REL_TOL
    (relative; absolute 0.01 near zero) of the labelled option's, and no option of another value is closer.
    Raises Skip with the reason otherwise."""
    options = parse_options(str(row.get("options", "")))
    if options is None: raise Skip("options_unparsable")
    label = str(row.get("correct", "")).strip()
    if label not in options: raise Skip("label_not_in_options")
    want = option_value(options[label])
    if want is None: raise Skip("label_value_unparsable")
    try: got, n_ops = execute_formula(str(row.get("annotated_formula", "")))
    except ValueError: raise Skip("formula_error") from None
    if n_ops == 0: raise Skip("formula_no_operation")
    if abs(got - want) > max(MATHQA_REL_TOL * abs(want), 0.01): raise Skip("formula_misses_label")
    for k, text in options.items():
        v = option_value(text)
        if k != label and v is not None and v != want and abs(got - v) < abs(got - want): raise Skip("formula_closer_to_other_option")
    return n_ops, options


def mathqa_template(row):
    """MathQA's template key: the linear formula with the numbers abstracted (n0, n1, ... as the dataset writes them)."""
    return re.sub(r"\s+", "", str(row.get("linear_formula", ""))).strip("|").casefold()


def convert_mathqa(row, split, idx):
    n_ops, options = mathqa_check(row)
    template = mathqa_template(row)
    if not template: raise Skip("no_linear_formula")
    meta = _meta("mathqa", split, idx, idx, f"template/{text_digest(template)[:16]}", {"kind": "mathqa_ops", "k": n_ops},
                 template=template, category=row.get("category"))
    keys = list(options)
    return [_choice("mathqa", {"problem": _text(row["Problem"])}, "Which option answers the problem?", [options[k] for k in keys],
                    keys.index(str(row["correct"]).strip()), meta)]


# --------------------------------------------------------------------------------------------------------- eval-only
BOARDGAME_CRITERIA = {"proved": "The statement follows from the game state, the rules and the preferences",
                      "disproved": "The statement's negation follows from the game state, the rules and the preferences",
                      "unknown": "Neither the statement nor its negation follows"}
_BOARDGAME_Q = re.compile(r"\s*(Based on the game state and the rules and preferences, .*\?)\s*$", re.S)


def convert_boardgameqa(row, split, idx):
    example = _text(row["example"])
    m = _BOARDGAME_Q.search(example)
    if not m: raise Skip("no_question_sentence")
    label = str(row.get("label", "")).strip()
    if label not in BOARDGAME_CRITERIA: raise Skip("bad_label")
    config = str(row.get("config", ""))
    d = re.search(r"depth(\d+)", config)
    if not d: raise Skip("no_depth")
    state = example[:m.start()].strip()
    meta = _meta("boardgameqa", split, idx, idx, f"theory/{text_digest(state)[:16]}", {"kind": "boardgameqa_depth", "k": int(d.group(1))},
                 config=config)
    return [{"state": state, "questions": {"answer": {"type": "choice", "instructions": m.group(1).strip(), "criteria": dict(BOARDGAME_CRITERIA),
             "label": label, "src": "lev_boardgameqa"}}, "_meta": meta}]


STEPGAME_CRITERIA = {"left": "to the left", "right": "to the right", "above": "directly above", "below": "directly below",
                     "upper_left": "above and to the left", "upper_right": "above and to the right",
                     "lower_left": "below and to the left", "lower_right": "below and to the right", "overlap": "at the same position"}


def convert_stepgame(row, split, idx):
    label = str(row.get("label", "")).strip().replace("-", "_")
    if label not in STEPGAME_CRITERIA: raise Skip("bad_label")
    story = [s for s in row["story"] if s.strip()]
    if not story: raise Skip("empty_story")
    meta = _meta("stepgame", split, idx, f"qa{row['k']}/{row['index']}", f"qa{row['k']}/{row['index']}",
                 {"kind": "stepgame_hops", "k": int(row["k"])}, version=SOURCES["stepgame"]["version"])
    return [{"state": "\n".join(story), "questions": {"relation": {"type": "choice", "instructions": _text(row["question"]),
             "criteria": dict(STEPGAME_CRITERIA), "label": label, "src": "lev_stepgame"}}, "_meta": meta}]


_LAST_QUESTION = re.compile(r"^(.*[.!%\"')])\s+([^.!?]*\?)\s*$", re.S)


def convert_cladder(row, split, idx):
    prompt = _text(row["prompt"])
    m = _LAST_QUESTION.match(prompt)
    if not m: raise Skip("no_question_sentence")
    label = str(row.get("label", "")).strip().lower()
    if label not in ("yes", "no"): raise Skip("bad_label")
    rung = _int(row.get("rung"), "no_rung")
    background = m.group(1).strip()
    meta = _meta("cladder", split, idx, row["id"], f"model/{text_digest(background)[:16]}", {"kind": "cladder_rung", "k": rung},
                 query_type=row.get("query_type"), graph_id=row.get("graph_id"), story_id=row.get("story_id"))
    return [{"state": background, "questions": {"answer": {"type": "noul", "instructions": m.group(2).strip(), "label": label == "yes",
             "src": "lev_cladder"}}, "_meta": meta}]


def convert_strategyqa(row, split, idx):
    if not isinstance(row.get("answer"), bool): raise Skip("no_answer")
    steps = row.get("decomposition") or []
    if not steps: raise Skip("no_decomposition")
    meta = _meta("strategyqa", split, idx, row["qid"], f"term/{text_digest(str(row.get('term') or row['qid']))[:16]}",
                 {"kind": "strategyqa_steps", "k": len(steps)}, term=row.get("term"))
    return [{"state": {"question": _text(row["question"])}, "questions": {"answer": {"type": "noul", "instructions": "Is the answer to this question yes?",
             "label": row["answer"], "src": "lev_strategyqa"}}, "_meta": meta}]


def convert_logiqa2(row, split, idx):
    text = _text(row["text"])
    types = sorted(k for k, v in (row.get("type") or {}).items() if v)
    meta = _meta("logiqa2", split, idx, row["id"], f"passage/{text_digest(text)[:16]}", reasoning_types=types)
    return [_choice("logiqa2", text, _text(row["question"]), row["options"], _int(row["answer"]), meta)]


def convert_ar_lsat(row, split, idx):
    context = _text(row["context"])
    meta = _meta("ar_lsat", split, idx, row["id_string"], f"game/{text_digest(context)[:16]}")
    return [_choice("ar_lsat", context, _text(row["question"]), row["answers"], _int(row["label"]), meta)]


def convert_quail(row, split, idx):
    meta = _meta("quail", split, idx, row["id"], f"context/{row['context_id']}", question_type=row.get("question_type"),
                 domain=row.get("domain"))
    return [_choice("quail", _text(row["context"]), _text(row["question"]), row["answers"], _int(row["correct_answer_id"]), meta)]


def convert_timedial(row, split, idx):
    """One item -> two 3-way choices (each correct answer with both wrong ones), sharing a group; options shuffled per
    record with a seed from the row id so the correct answer is not always first."""
    turns = [t.strip() for t in row["conversation"] if t and t.strip()]
    if not any("<MASK>" in t for t in turns): raise Skip("no_mask")
    wrong = [_text(row["incorrect1"]), _text(row["incorrect2"])]
    out = []
    for j in (1, 2):
        right = _text(row[f"correct{j}"])
        options = [right] + wrong
        order = list(range(3)); random.Random(f"timedial:{row['id']}:{j}").shuffle(order)
        meta = _meta("timedial", split, idx, f"{row['id']}/c{j}", f"dialog/{row['id']}",
                     incorrect_rules=[row.get("incorrect1_rule"), row.get("incorrect2_rule")])
        out.append(_choice("timedial", {"conversation": turns}, "Which option best fills the <MASK> in the conversation?",
                           [options[o] for o in order], order.index(0), meta))
    return out


CONVERTERS = {name: globals()[f"convert_{name}"] for name in SOURCES}


# ------------------------------------------------------------------------------------------------------------------ load
STATS = {}


def convert_split(name, split, exclude_groups=frozenset()):
    """Every row of one split -> (per-row lists of records, Counter of skip reasons, rows read). A row whose group_id is in
    exclude_groups is skipped as group_in_train."""
    out, skipped, n = [], Counter(), 0
    for idx, row in enumerate(_rows(name, split)):
        n += 1
        if name == "mathqa" and f"lev_mathqa/template/{text_digest(mathqa_template(row))[:16]}" in exclude_groups:
            skipped["group_in_train"] += 1; continue   # before the formula filter, so the reason does not depend on it
        try:
            reqs = CONVERTERS[name](dict(row), split, idx)
            for r in reqs: materialize(r)
        except Skip as e:
            skipped[str(e)] += 1; continue
        except Exception as e:  # noqa: BLE001 - a record the serving path refuses is a skipped row, with its reason
            skipped[f"materialize:{type(e).__name__}"] += 1; continue
        if any(r["_meta"]["group_id"] in exclude_groups for r in reqs):
            skipped["group_in_train"] += 1; continue
        out.append(reqs)
    return out, skipped, n


@lru_cache(maxsize=None)
def train_groups(name):
    """group_ids of a trainable source's training split: those of its converted rows, and for MathQA the template of
    every training row, converted or not (a template must not cross train and eval even if its train rows failed)."""
    groups = {r["_meta"]["group_id"] for reqs in convert_split(name, SOURCES[name]["splits"]["train"])[0] for r in reqs}
    if name == "mathqa": groups |= {f"lev_mathqa/template/{text_digest(mathqa_template(r))[:16]}" for r in _rows("mathqa", "train")}
    return frozenset(groups)


def load(name, split_role, n=None, seed=0):
    """Labelled requests of one source: split_role "train" (the HF training split) or "eval" (held-out items: the HF split(s)
    in SOURCES[name]["splits"]["eval"]). With n, n rows are drawn with random.Random(f"{seed}:{name}:{split_role}") from
    the rows that converted (a TimeDial row gives two records), kept in file order. A trainable source's eval rows exclude
    every group of its train split (train_groups: a shared passage, goal or context; MathQA's template, the normalised
    linear_formula), so no group crosses train and eval.
    load.stats (and STATS[(name, split_role)]) records rows available, converted, returned and the skip reasons."""
    if name not in SOURCES: raise KeyError(f"unknown source {name!r}; known: {sorted(SOURCES)}")
    if split_role not in ("train", "eval"): raise ValueError("split_role is 'train' or 'eval'")
    splits = SOURCES[name]["splits"][split_role]
    if splits is None: raise ValueError(f"{name} is eval-only: it has no training split")
    splits = (splits,) if isinstance(splits, str) else tuple(splits)
    exclude = train_groups(name) if split_role == "eval" and SOURCES[name]["splits"]["train"] else frozenset()
    per_row, skipped, available = [], Counter(), 0
    for split in splits:
        got, sk, k = convert_split(name, split, exclude)
        per_row += got; skipped += sk; available += k
    picked = per_row
    if n is not None:
        idx = sorted(random.Random(f"{seed}:{name}:{split_role}").sample(range(len(per_row)), min(n, len(per_row))))
        picked = [per_row[i] for i in idx]
    records = [r for reqs in picked for r in reqs]
    stats = {"source": name, "role": split_role, "splits": list(splits), "available": available, "converted": len(per_row),
             "skipped": dict(skipped), "rows_returned": len(picked), "records_returned": len(records)}
    STATS[(name, split_role)] = load.stats = stats
    return records


load.stats = None


def summary(records):
    """Label and difficulty counts of a list of records (for reports)."""
    labels, diff = Counter(), Counter()
    for r in records:
        for q in r["questions"].values(): labels[str(q["label"])] += 1
        d = r["_meta"].get("difficulty")
        if d: diff[f"{d['kind']}={d['k']}"] += 1
    return {"labels": dict(sorted(labels.items())), "difficulty": dict(sorted(diff.items()))}


if __name__ == "__main__":
    import argparse
    from kev.api import render
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", nargs="*", default=list(SOURCES)); ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in a.sources:
        for role in ("train", "eval"):
            if SOURCES[name]["splits"][role] is None: continue
            recs = load(name, role, n=a.n, seed=a.seed)
            print(json.dumps({**load.stats, **summary(recs)}, ensure_ascii=False))
            if recs: print("   example:", repr(render(recs[0]["state"])[:300]))
