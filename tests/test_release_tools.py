import copy
import json
import numpy as np
import pytest
from analysis import adaptive_depth as A
from scripts.export_results import project
from scripts.stopper_config import fit_config, apply_config, scores

MODEL = 'example/Lev@revision1'


def document():
    return {'split': 'development', 'depths': 2, 'rows': [
        {'id': 'dev/r1', 'group': 'g1', 'question': 'q1', 'source': 'toy', 'label': 0,
         'tokens': 17, 'logits': [[2.0, 0.0], [0.0, 2.0]], 'tier': 'deep',
         'difficulty': {'kind': 'chain', 'k': 3}, 'unanswerable': False,
         'state': 'This text must not be exported', 'criteria': {'A': 'Private text'}}]}


def test_export_preserves_analysis_fields_and_omits_text():
    d = document(); out = project(d, 'toy-development', MODEL)
    r = out['rows'][0]
    assert (r['group'], r['question'], r['tokens'], r['label'], r['difficulty']) == ('g1', 'q1', 17, 0, {'kind': 'chain', 'k': 3})
    assert 'state' not in r and 'criteria' not in r
    assert json.loads(json.dumps(out))['rows'][0]['logits'] == d['rows'][0]['logits']


@pytest.mark.parametrize('field', ['source', 'task', 'question', 'group'])
def test_export_rejects_nested_text(field):
    d = document(); d['rows'][0][field] = {'text': 'hidden prompt'}
    with pytest.raises(ValueError): project(d, 'toy-development', MODEL)


def test_export_rejects_nested_difficulty_and_locked_split():
    d = document(); d['rows'][0]['difficulty']['kind'] = ['hidden prompt']
    with pytest.raises(ValueError): project(d, 'toy-development', MODEL)
    d = document(); d['split'] = 'test'
    with pytest.raises(ValueError): project(d, 'toy-development', MODEL)


def make_rows(seed):
    rng = np.random.default_rng(seed)
    return [{'label': int(rng.integers(2)), 'logits': [rng.normal(size=2) for _ in range(3)]} for _ in range(60)]


def test_stopper_json_roundtrip_matches_sklearn_and_ignores_adaptation_labels():
    cal, adapt = make_rows(3), make_rows(4)
    config = fit_config(cal, adapt, MODEL)
    restored = json.loads(json.dumps(config))
    changed = copy.deepcopy(adapt)
    for r in changed: r['label'] = 99
    assert fit_config(cal, changed, MODEL) == config
    Fc, Gc = A.features(cal, config['temperatures']); Fa, Ga = A.features(adapt, config['temperatures'])
    clfs = A.fit_stopper(np.concatenate([Fc, Fa]), np.concatenate([Gc, Ga]))
    np.testing.assert_allclose(scores(restored, Fa), A.stopper_scores(clfs, Fa), atol=1e-12)
    expected = A.stop_depth(A.stopper_scores(clfs, Fa), config['threshold'], True)
    np.testing.assert_array_equal(apply_config(restored, adapt, MODEL), expected)
    with pytest.raises(ValueError, match='identity'): apply_config(restored, adapt, 'different/model')


def test_stopper_constant_agreement_target():
    rows = [{'label': 0, 'logits': [np.array([3., 0.]), np.array([3., 0.])]} for _ in range(8)]
    config = fit_config(rows, rows, MODEL)
    assert config['classifiers'] == [{'constant': 1.0}]
    assert apply_config(config, rows, MODEL).tolist() == [0] * 8
