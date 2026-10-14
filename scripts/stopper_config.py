"""Fit and apply a portable JSON agreement stopper to already-computed depth logits.

This is a workload calibration utility, not online model continuation and not a
serialized copy of the paper's per-panel, two-fold cross-fitted models.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import adaptive_depth as A
from scripts.export_results import is_test, IDENTIFIER


def read_rows(path, labelled=False):
    if is_test(path):
        raise ValueError('Use calibration/development inputs, not locked test')
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    if is_test(d.get('split', '')) or is_test(d.get('panel', '')):
        raise ValueError('Locked test input is not supported')
    rows = []
    for r in d['rows']:
        zs = [np.asarray(z, float) for z in r['logits']]
        if len(zs) != d['depths'] or any(len(z) < 2 or not np.isfinite(z).all() for z in zs):
            raise ValueError('Invalid logits')
        out = {'logits': zs}
        if labelled:
            if not isinstance(r.get('label'), int) or not 0 <= r['label'] < len(zs[0]):
                raise ValueError('Temperature calibration requires valid labels')
            out['label'] = r['label']
        rows.append(out)
    if not rows:
        raise ValueError('Input rows are empty')
    return d['depths'], rows


def fit_config(calibration, adaptation, model, min_agree=0.98):
    if not 0 < min_agree <= 1:
        raise ValueError('min_agree must be in (0, 1]')
    D = len(calibration[0]['logits'])
    if D < 2 or any(len(r['logits']) != D for r in calibration + adaptation):
        raise ValueError('At least two consistent depths are required')
    temps = [A.fit_temperature(calibration, k) for k in range(D)]
    Fc, Gc = A.features(calibration, temps)
    Fa, Ga = A.features(adaptation, temps)
    F, G = np.concatenate([Fc, Fa]), np.concatenate([Gc, Ga])
    from sklearn.linear_model import LogisticRegression
    models = []
    for k in range(D - 1):
        labels = np.unique(G[:, k])
        if len(labels) == 1:
            models.append({'constant': float(labels[0])})
        else:
            clf = LogisticRegression(max_iter=3000).fit(F[:, k], G[:, k])
            models.append({'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0])})
    result = {'schema_version': 1, 'method': 'agreement-workload-calibration', 'model': model,
              'features': A.FEATURES, 'depths': D, 'temperatures': temps,
              'classifiers': models, 'min_agree': min_agree,
              'fit_scope': 'calibration plus explicitly provided unlabeled workload; threshold on workload',
              'depth_indexing': 'zero-based'}
    result['threshold'] = A.stopper_tau(scores(result, Fa), Ga, min_agree)
    return result


def validate(config):
    if config.get('schema_version') != 1 or config.get('method') != 'agreement-workload-calibration':
        raise ValueError('Unsupported stopper schema or method')
    if config.get('features') != A.FEATURES:
        raise ValueError('Feature order differs')
    if not isinstance(config.get('model'), str) or not IDENTIFIER.fullmatch(config['model']):
        raise ValueError('A pinned public model identity is required')
    D = config.get('depths')
    if not isinstance(D, int) or D < 2 or len(config.get('temperatures', [])) != D or len(config.get('classifiers', [])) != D - 1:
        raise ValueError('Invalid depth dimensions')
    if any(not isinstance(t, (int, float)) or not np.isfinite(t) or t <= 0 for t in config['temperatures']):
        raise ValueError('Temperatures must be positive and finite')
    if 'threshold' in config and not 0 <= config['threshold'] <= 1.01:
        raise ValueError('Invalid threshold')
    for c in config['classifiers']:
        if 'constant' in c:
            if c['constant'] not in (0.0, 1.0): raise ValueError('Invalid constant classifier')
        elif len(c.get('coef', [])) != len(A.FEATURES) or not np.isfinite(c['coef'] + [c.get('intercept', float('nan'))]).all():
            raise ValueError('Invalid classifier parameters')


def scores(config, F):
    validate(config)
    out = []
    for k, c in enumerate(config['classifiers']):
        if 'constant' in c:
            out.append(np.full(F.shape[0], c['constant']))
        else:
            z = np.clip(F[:, k] @ np.asarray(c['coef']) + c['intercept'], -700, 700)
            out.append(1 / (1 + np.exp(-z)))
    return np.stack(out, axis=1)


def apply_config(config, rows, model):
    validate(config)
    if model != config['model']:
        raise ValueError('Stopper/model identity mismatch')
    if any(len(r['logits']) != config['depths'] for r in rows):
        raise ValueError('Input/config depths differ')
    F, _ = A.features(rows, config['temperatures'])
    return A.stop_depth(scores(config, F), config['threshold'], scores_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    fit = sub.add_parser('fit'); fit.add_argument('--calibration', required=True); fit.add_argument('--adaptation', required=True)
    fit.add_argument('--model', required=True); fit.add_argument('--min-agree', type=float, default=0.98); fit.add_argument('--out', required=True)
    apply = sub.add_parser('apply'); apply.add_argument('--model', required=True); apply.add_argument('--config', required=True); apply.add_argument('--rows', required=True); apply.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out)
    if out.exists(): ap.error('Output already exists')
    if a.action == 'fit':
        Dc, cal = read_rows(a.calibration, labelled=True); Da, adapt = read_rows(a.adaptation)
        if Dc != Da: ap.error('Calibration/adaptation depths differ')
        result = fit_config(cal, adapt, a.model, a.min_agree)
        result['input_sha256'] = {k: hashlib.sha256(Path(getattr(a, k)).read_bytes()).hexdigest() for k in ('calibration', 'adaptation')}
    else:
        config = json.loads(Path(a.config).read_text(encoding='utf-8')); _, rows = read_rows(a.rows)
        result = {'depth_indexing': 'zero-based', 'stops': apply_config(config, rows, a.model).tolist(), 'mode': 'offline'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
