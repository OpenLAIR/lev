"""Synthetic checks of the Lev analysis helpers (analysis/stopper_ablations.py, analysis/cascade.py); no weights."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from cascade import escalate_by_record, mix_at, record_mean   # noqa: E402
from stopper_ablations import frontier_at, patience_stop   # noqa: E402


def test_patience_stops_once_the_answer_holds():
    ans = np.array([[0, 0, 0, 0],    # unchanged from depth 1: patience 1 stops at depth 2 (index 1)
                    [0, 1, 1, 1],    # changes once: stops at depth 3
                    [0, 1, 2, 3]])   # never holds: last depth
    assert patience_stop(ans, 1).tolist() == [1, 2, 3]
    assert patience_stop(ans, 2).tolist() == [2, 3, 3]
    conf = np.array([[0.4, 0.4, 0.95, 0.9], [0.9] * 4, [0.9] * 4])
    assert patience_stop(ans, 1, conf, 0.9).tolist() == [2, 2, 3]


def test_record_level_escalation_and_cost():
    ids = ["a", "a", "b"]
    assert escalate_by_record(ids, [False, True, False]).tolist() == [True, True, False]
    assert record_mean(ids, [1.0, 3.0, 2.0]) == 2.5   # record a costs its dearest row


def test_cascade_mixture_hits_the_cost_exactly():
    pts = [{"cost": 1.0, "accuracy": 0.5}, {"cost": 3.0, "accuracy": 0.9}]
    lo, hi, lam = mix_at(pts, 2.0)
    assert (lo["cost"], hi["cost"], lam) == (1.0, 3.0, 0.5)
    assert mix_at(pts, 0.5) is None and mix_at(pts, 3.5) is None


def test_frontier_mixes_neighbouring_depths():
    right = np.array([[0, 1, 1, 1], [0, 0, 1, 1]], bool)   # depth accuracies .0 / .5 / 1 / 1
    assert frontier_at(right, 0.5) == 0.5                   # depth 2
    assert abs(frontier_at(right, 0.625) - 0.75) < 1e-12    # half depth 2, half depth 3


def test_upper_hull_drops_dominated_and_concave_points():
    from cascade import upper_hull
    pts = [{"cost": 1, "accuracy": 0.5}, {"cost": 2, "accuracy": 0.55}, {"cost": 3, "accuracy": 0.9}, {"cost": 4, "accuracy": 0.8}]
    assert [q["cost"] for q in upper_hull(pts)] == [1, 3]   # (2, .55) lies under the chord, (4, .8) is dearer and worse
