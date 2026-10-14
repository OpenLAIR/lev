"""Export text-free depth rows, preserving metadata needed by offline analyses.

Locked test export is deliberately unsupported in this staging tool. No input
is changed. Use only development/calibration records with cleared provenance.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

IDENTIFIER = re.compile(r"[A-Za-z0-9_.:/@+>=-]{1,300}\Z")
ROW_FIELDS = ('id', 'group', 'question', 'source', 'task', 'type', 'variant',
              'label', 'correct', 'state_tokens', 'tokens', 'logits',
              'tier', 'unanswerable', 'report_only')


def is_test(value):
    return bool(re.search(r'(?:^|[^a-z])test(?:$|[^a-z])', str(value).lower()))


def project(document, panel, model):
    if is_test(panel) or is_test(document.get('split', '')):
        raise ValueError('Locked test export is not supported by this staging tool')
    if not IDENTIFIER.fullmatch(panel) or not IDENTIFIER.fullmatch(model):
        raise ValueError('Use public identifiers for panel and model')
    depth = document['depths']
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('Invalid number of depths')
    result, seen = [], set()
    for original in document['rows']:
        row = {k: original[k] for k in ROW_FIELDS if k in original}
        for key in ('id', 'group', 'question', 'tokens', 'logits'):
            if key not in row:
                raise ValueError(f'Missing required row field: {key}')
        for k in ('id', 'group', 'question', 'source', 'task', 'type', 'variant', 'tier'):
            if k in row and row[k] is not None and (not isinstance(row[k], str) or not IDENTIFIER.fullmatch(row[k])):
                raise ValueError(f'{k} must be an identifier, not free text or nested data')
        for k in ('unanswerable', 'report_only'):
            if k in row and type(row[k]) is not bool:
                raise ValueError(f'{k} must be boolean')
        if 'state_tokens' in row and (type(row['state_tokens']) is not int or row['state_tokens'] < 0):
            raise ValueError('state_tokens must be nonnegative integer')
        if is_test(row['id']):
            raise ValueError('A test identifier is present')
        key = (row['id'], row['question'])
        if key in seen:
            raise ValueError('Duplicate question identity')
        seen.add(key)
        zs = row['logits']
        if len(zs) != depth or not zs or len(zs[0]) < 2 or any(len(z) != len(zs[0]) for z in zs):
            raise ValueError('Logit dimensions differ across depths')
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for z in zs for x in z):
            raise ValueError('Logits must be finite numbers')
        if not isinstance(row['tokens'], int) or row['tokens'] <= 0:
            raise ValueError('tokens must be positive')
        if 'label' not in row and 'correct' not in row:
            raise ValueError('Retain label or per-depth correctness')
        if 'label' in row and row['label'] is not None and not (isinstance(row['label'], int) and 0 <= row['label'] < len(zs[0])):
            raise ValueError('Label outside option range')
        if 'correct' in row and (len(row['correct']) != depth or any(type(x) is not bool for x in row['correct'])):
            raise ValueError('correct must contain one boolean per depth')
        difficulty = original.get('difficulty')
        if difficulty is not None:
            if not isinstance(difficulty, dict):
                raise ValueError('difficulty must be a mapping')
            row['difficulty'] = {k: difficulty[k] for k in ('kind', 'k') if k in difficulty}
            if 'k' in row['difficulty'] and type(row['difficulty']['k']) is not int:
                raise ValueError('difficulty.k must be integer')
            if 'kind' in row['difficulty'] and not isinstance(row['difficulty']['kind'], str):
                raise ValueError('difficulty.kind must be an identifier')
            if any(isinstance(v, str) and not IDENTIFIER.fullmatch(v) for v in row['difficulty'].values()):
                raise ValueError('Difficulty metadata must not contain free text')
        result.append(row)
    return {'schema_version': 1, 'model': model, 'panel': panel,
            'split': document.get('split'), 'depths': depth, 'rows': result,
            'omitted_fields': ['state', 'instructions', 'criteria', 'option_text', 'target', 'private_paths', 'skipped_reasons']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--panel', required=True)
    ap.add_argument('--model', required=True)
    a = ap.parse_args()
    if is_test(a.input) or is_test(a.panel):
        ap.error('Locked test export is not supported; input was not opened')
    destination = Path(a.output)
    if destination.exists():
        ap.error('Output already exists')
    raw = Path(a.input).read_bytes()
    result = project(json.loads(raw.decode('utf-8')), a.panel, a.model)
    result['source_sha256'] = hashlib.sha256(raw).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(result["rows"])} questions')


if __name__ == '__main__':
    main()
