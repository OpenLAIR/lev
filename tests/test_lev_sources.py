"""scripts/lev_sources.py: the MathQA formula executor and option parser, and every converter on one hand-written row
in the dataset's real schema (offline). LEV_NETWORK=1 also reads every source at its pinned revision."""
import math
import os

import pytest

from kev.data import materialize
from scripts.lev_sources import (CONVERTERS, EVAL_ONLY, SOURCES, TRAINABLE, Skip, execute_formula, load, mathqa_check,
                                     mathqa_template, option_value, parse_options)


# ------------------------------------------------------------------------------------------------------------- MathQA
@pytest.mark.parametrize("formula, value, ops", [
    ("add(1, 2)", 3, 1),
    ("divide(multiply(const_100, divide(multiply(36, const_100), multiply(3, 10))), multiply(3, 10))", 400, 6),
    ("power(5, 4)", 625, 1),
    ("subtract(divide(multiply(add(const_100, 31.1), const_100), subtract(const_100, 5)), const_100)", 100 * 131.1 / 95 - 100, 5),
    ("circle_area(const_2)", 4 * math.pi, 1),
    ("multiply(const_2.0, divide(30,000, 3))", 20_000, 2),          # const_2.0 and a thousands separator, as the data spells them
    ("choose(5, 2)", 10, 1), ("factorial(4)", 24, 1), ("reminder(17, 5)", 2, 1), ("lcm(4, 6)", 12, 1),
    ("multiply(const_0_2778, 36)", 0.2778 * 36, 1), ("volume_cylinder(1, 2)", 2 * math.pi, 1),
    ("triangle_area_three_edges(3, 4, 5)", 6, 1), ("original_price_before_gain(25, 500)", 400, 1),
    ("speed(150, 10)", 15, 1), ("negate_prob(0.25)", 0.75, 1), ("inverse(4)", 0.25, 1), ("floor(divide(7, 2))", 3, 2),
])
def test_execute_formula(formula, value, ops):
    got, n = execute_formula(formula)
    assert got == pytest.approx(value) and n == ops


@pytest.mark.parametrize("formula", ["sqrt(negate(4))", "divide(1, subtract(2, 2))", "frobnicate(1)", "add(1, 2", "add(1, 2, 3)",
                                     "const_zz", "factorial(2.5)"])
def test_execute_formula_rejects(formula):
    with pytest.raises(ValueError):
        execute_formula(formula)


def test_parse_options_and_values():
    opts = parse_options("a ) rs . 400 , b ) 3 / 5 , c ) 2 1 / 2 , d ) 1 , 000 , e ) none of these")
    assert opts == {"a": "rs . 400", "b": "3 / 5", "c": "2 1 / 2", "d": "1 , 000", "e": "none of these"}
    assert [option_value(opts[k]) for k in "abcde"] == [400, 0.6, 2.5, 1000, None]
    assert option_value("15.5 days") == 15.5 and option_value("- 55") == -55 and option_value("3 : 4") == 0.75
    assert option_value("5 % ") == 5 and option_value("$ 3.36") == 3.36
    assert option_value("4 √ 3") is None and option_value("5 hr 30 min") is None and option_value("2 π") is None
    assert parse_options("rs . 400") is None


def mathqa_row(**kw):
    row = {"Problem": "a multiple choice test consists of 4 questions , and each question has 5 answer choices . in how many r ways "
                      "can the test be completed if every question is unanswered ?",
           "Rationale": '"5 choices for each of the 4 questions ... answer : c ."', "options": "a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024",
           "correct": "c", "annotated_formula": "power(5, 4)", "linear_formula": "power(n1,n0)|", "category": "general"}
    return {**row, **kw}


def test_mathqa_filter():
    assert mathqa_check(mathqa_row())[0] == 1
    assert mathqa_template(mathqa_row()) == "power(n1,n0)"
    with pytest.raises(Skip, match="formula_misses_label"): mathqa_check(mathqa_row(correct="b"))
    with pytest.raises(Skip, match="formula_closer_to_other_option"):   # within 1 % of the label, but another option is closer
        mathqa_check(mathqa_row(options="a ) 100 , b ) 100.5 , c ) 99", correct="a", annotated_formula="add(100, 0.4)"))
    with pytest.raises(Skip, match="formula_error"): mathqa_check(mathqa_row(annotated_formula="power(5, 4"))
    with pytest.raises(Skip, match="label_value_unparsable"): mathqa_check(mathqa_row(options="a ) 24 , b ) 120 , c ) none", correct="c"))


# ------------------------------------------------------------------------------------------------------- converters
ROWS = {  # one row per source in the schema read from the pinned files (values shortened)
    "condaqa": ("train", {"QuestionID": "q10", "original cue": "rarely", "PassageEditID": 0, "original passage": "...", "SampleID": 5294,
                          "label": "DON'T KNOW", "original sentence": "...", "PassageID": 444,
                          "sentence1": "Generally, drug possession is an arrestable offense, although first-time offenders rarely serve jail time.",
                          "sentence2": "If a drug addict is caught with marijuana, is there a chance he will be jailed?"}, "dont_know"),
    "social_iqa": ("train", {"context": "Cameron decided to have a barbecue and gathered her friends together.", "question": "How would Others feel as a result?",
                             "answerA": "like attending", "answerB": "like staying home", "answerC": "a good friend to have", "label": "1"}, "a"),
    "cosmosqa": ("validation", {"id": "3BFF##q1_a1", "context": "Do i need to go for a legal divorce ? I wanted to marry a woman ...",
                                "question": "Why is this person asking about divorce ?", "answer0": "If he gets married in the church he wo nt have to get a divorce .",
                                "answer1": "He wants to get married to a different person .", "answer2": "He wants to know if he can divorce her ?",
                                "answer3": "None of the above choices .", "label": "1"}, "b"),
    "mathqa": ("train", mathqa_row(), "c"),
    "qasc": ("train", {"id": "3E7TUJ2EGCLQNOV1WEAJ2NN9ROPD9K", "question": "What type of water formation is formed by clouds?",
                       "choices": {"text": ["pearls", "streams", "shells", "diamonds", "rain", "beads", "cooled", "liquid"],
                                   "label": ["A", "B", "C", "D", "E", "F", "G", "H"]}, "answerKey": "F",
                       "fact1": "beads of water are formed by water vapor condensing", "fact2": "Clouds are made of water vapor.",
                       "combinedfact": "Beads of water can be formed by clouds.", "formatted_question": "..."}, "f"),
    "piqa": ("validation", {"goal": "How do I ready a guinea pig cage for it's new occupants?", "sol1": "Provide bedding made of ripped paper strips.",
                            "sol2": "Provide bedding made of ripped jeans material.", "label": "0"}, "a"),
    "boardgameqa": ("test", {"proof": "...", "label": "unknown", "rules": "...", "preferences": "...", "theory": "...", "goal": "(pigeon, bring, woodpecker)",
                             "facts": "...", "config": "Main-depth3",
                             "example": "A few players are playing a boardgame. The current state of the game is as follows. The coyote swears to the pigeon. "
                                        "And the rules of the game are as follows. Rule1: If the coyote swears to the pigeon, then the pigeon shouts at the peafowl. "
                                        "Based on the game state and the rules and preferences, does the pigeon bring an oil tank for the woodpecker?"}, "unknown"),
    "stepgame": ("test", {"story": ["O is to the bottom left of Z.", "If Y is the center of a clock face, J is located between 4 and 5.",
                                    "J and Z are side by side with J on the top and Z at the bottom."],
                          "question": "What is the relation of the agent Y to the agent O?", "label": "upper-left", "k": 3, "index": 0}, "upper_left"),
    "cladder": ("full_v1.5_default", {"id": "4", "label": "no", "reasoning": "...", "rung": "3", "query_type": "nde", "graph_id": "mediation",
                                      "story_id": "alarm", "question_property": "easy", "formal_form": "...",
                                      "prompt": "Imagine a self-contained, hypothetical world with only the following conditions: Husband has a direct effect on "
                                                "wife and alarm clock. For husbands that set the alarm, the probability of alarm set by wife is 24%. If we "
                                                "disregard the mediation effect through wife, would husband positively affect alarm clock?"}, False),
    "strategyqa": ("train", {"qid": "b8677742616fef051f00", "term": "Genghis Khan", "description": "...",
                             "question": "Are more people today related to Genghis Khan than Julius Caesar?", "answer": True,
                             "facts": ["Julius Caesar had three children."], "evidence": [],
                             "decomposition": ["How many kids did Julius Caesar have?", "How many kids did Genghis Khan have?", "Is #2 greater than #1?"]}, True),
    "logiqa2": ("test", {"id": 7261, "answer": 3, "text": "All Anxi people are vegetarians, while all Zhenyuan people are ascetics. Guo Shu is an ascetic.",
                         "question": "Which of the followings can be infered", "options": ["Guo Shu is from Zhenyuan", "Guo Shu is not from Zhenyuan",
                         "Guo Shu is from Anxi", "Guo Shu is not from Anxi"], "type": {"Categorical Reasoning": True, "Conjunctive Reasoning": False}}, "d"),
    "ar_lsat": ("test", {"context": "Of the eight students, exactly six will give individual oral reports ...", "id_string": "200010_3-G_1_1",
                         "answers": ["Mon: Helen", "Mon: Irving", "Mon: Lenore", "Mon: Nina", "Mon: Olivia"], "label": 2,
                         "question": "Which one of the following could be the schedule of the students' reports?"}, "c"),
    "quail": ("validation", {"id": "f141_0", "context_id": "f141", "question_id": "0", "domain": "fiction", "metadata": "{}",
                             "context": "Candy watched the bearded man drive his silver BMW into the convenience store parking lot.",
                             "question": "How long was Candy trying to seduce Larry?", "question_type": "Event_duration",
                             "answers": ["about 10 minutes", "about 2 hours", "not enough information", "All day"], "correct_answer_id": "2"}, "c"),
}


def check(record, label):
    rec = materialize(record)
    (q,), (src_q,) = rec["questions"], record["questions"].values()
    want = q["keys"].index(label) if src_q["type"] == "choice" else int(label)
    assert src_q["label"] == label and q["label"] == want
    m = record["_meta"]
    assert m["source"] == src_q["src"] and m["id"].startswith(m["source"] + "/") and m["group_id"].startswith(m["source"] + "/")
    s = SOURCES[m["source"].removeprefix("lev_")]
    assert m["repo"] == s["repo"] and m["revision"] == s["revision"] and len(m["revision"]) == 40
    assert m["licence"] == s["licence"] and m["local_only"] == s["local_only"] and isinstance(m["row"], int)
    return m


@pytest.mark.parametrize("name", sorted(ROWS))
def test_converter(name):
    split, row, label = ROWS[name]
    recs = CONVERTERS[name](row, split, 7)
    assert len(recs) == 1
    m = check(recs[0], label)
    assert m["split"] == split and m["row"] == 7


def test_difficulty_and_flags():
    def meta(name): split, row, _ = ROWS[name]; return CONVERTERS[name](row, split, 0)[0]["_meta"]
    assert meta("mathqa")["difficulty"] == {"kind": "mathqa_ops", "k": 1}
    assert meta("boardgameqa")["difficulty"] == {"kind": "boardgameqa_depth", "k": 3}
    assert meta("stepgame")["difficulty"] == {"kind": "stepgame_hops", "k": 3}
    assert meta("cladder")["difficulty"] == {"kind": "cladder_rung", "k": 3}
    assert meta("strategyqa")["difficulty"] == {"kind": "strategyqa_steps", "k": 3} and meta("strategyqa")["report_only"] is True
    assert meta("condaqa")["difficulty"] is None and meta("condaqa")["group_id"] == "lev_condaqa/passage/444"
    assert all(meta(n)["local_only"] for n in ("logiqa2", "ar_lsat", "quail"))
    cl = CONVERTERS["cladder"](ROWS["cladder"][1], "full_v1.5_default", 0)[0]
    assert cl["questions"]["answer"]["instructions"].startswith("If we disregard") and cl["state"].endswith("24%.")
    bg = CONVERTERS["boardgameqa"](ROWS["boardgameqa"][1], "test", 0)[0]
    assert "Based on" not in bg["state"] and bg["questions"]["answer"]["instructions"].startswith("Based on the game state")
    sq = CONVERTERS["strategyqa"](ROWS["strategyqa"][1], "train", 0)[0]
    assert sq["state"] == {"question": "Are more people today related to Genghis Khan than Julius Caesar?"}   # no facts


def test_timedial_two_records_one_group():
    row = {"id": 4, "conversation": ["A: What about your company's QC management ?",
                                     "B: We established strict QC standards and have followed them for <MASK> with good results ."],
           "correct1": "about five years", "correct2": "more than one and half year ", "incorrect1": "about 36 days ", "incorrect1_rule": "Rule 2",
           "incorrect2": "about 14 hours ", "incorrect2_rule": "Rule 3"}
    a, b = CONVERTERS["timedial"](row, "test", 3)
    for rec, right in ((a, "about five years"), (b, "more than one and half year")):
        q = rec["questions"]["answer"]
        assert len(q["criteria"]) == 3 and q["criteria"][q["label"]] == right
        assert set(q["criteria"].values()) - {right} == {"about 36 days", "about 14 hours"}
        check(rec, q["label"])
    assert a["_meta"]["group_id"] == b["_meta"]["group_id"] and a["_meta"]["id"] != b["_meta"]["id"]
    assert CONVERTERS["timedial"](row, "test", 3) == [a, b]   # the shuffle is seeded by the row


def test_skips():
    with pytest.raises(Skip, match="span_answer"):
        CONVERTERS["condaqa"]({**ROWS["condaqa"][1], "label": "Kindergarten"}, "dev", 0)
    with pytest.raises(Skip, match="duplicate_options"):
        CONVERTERS["social_iqa"]({**ROWS["social_iqa"][1], "answerC": "Like attending"}, "train", 0)


def test_registry():
    assert set(CONVERTERS) == set(SOURCES) and len(TRAINABLE) == 6 and len(EVAL_ONLY) == 8
    for name in EVAL_ONLY:
        assert SOURCES[name]["splits"]["train"] is None
        with pytest.raises(ValueError, match="eval-only"): load(name, "train")
    assert {n for n, s in SOURCES.items() if s["local_only"]} == {"logiqa2", "ar_lsat", "quail", "timedial"}


# ---------------------------------------------------------------------------------------------------------- network
network = pytest.mark.skipif(not os.environ.get("LEV_NETWORK"), reason="set LEV_NETWORK=1 to read the pinned datasets")


@network
@pytest.mark.parametrize("name", sorted(SOURCES))
def test_load_pinned(name):
    for role in ("train", "eval"):
        if SOURCES[name]["splits"][role] is None: continue
        a, b = load(name, role, n=20, seed=1), load(name, role, n=20, seed=1)
        assert a == b and load.stats["rows_returned"] == 20 and load.stats["converted"] > 300
        for r in a: materialize(r)
    if SOURCES[name]["splits"]["train"]:   # no group (MathQA: template) crosses train and eval
        assert not {r["_meta"]["group_id"] for r in load(name, "train")} & {r["_meta"]["group_id"] for r in load(name, "eval")}
    if name == "mathqa":
        assert not {r["_meta"]["template"] for r in load("mathqa", "train")} & {r["_meta"]["template"] for r in load("mathqa", "eval")}
