"""Score an original toy request at one chosen depth or all native depths."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--all-depths', action='store_true')
    ap.add_argument('--temperatures')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    tok, model = Checkpoint(a.checkpoint).load(a.device, LoadOptions(temperature=1.0))
    if not model.native_loop: ap.error('This example requires a native recurrent checkpoint')
    D = model.depth
    temperatures = [1.0] * D
    if a.temperatures:
        temperatures = json.loads(Path(a.temperatures).read_text(encoding='utf-8'))['temperatures']
        if len(temperatures) != D or any(t <= 0 for t in temperatures):
            ap.error('One positive temperature per native depth is required')
    request = SystemOneRequest(state='The blue crate weighs 8 kilograms. The red crate weighs 3 kilograms.',
        questions={'heavier': {'type': 'choice', 'instructions': 'Which crate is heavier?',
                               'criteria': {'blue': 'The blue crate', 'red': 'The red crate'}}})
    record, meta = to_record(request)
    encoded = model.encode(tok, record)
    with torch.no_grad():
        if a.all_depths:
            depths = list(range(1, D + 1)); outputs = model.forward_steps_batch([encoded])
        else:
            model.set_depth(a.depth); depths = [a.depth]; outputs = [model.forward_batch([encoded])]
    rows = []
    for depth, output in zip(depths, outputs):
        probs = (output[0][0].float() / temperatures[depth - 1]).softmax(-1).cpu().tolist()
        rows.append({'depth': depth, 'probabilities': dict(zip(meta[0]['keys'], probs))})
    print(json.dumps({'calibration': 'per-depth' if a.temperatures else 'uncalibrated',
                      'execution': 'full-depth readouts' if a.all_depths else 'fixed-depth', 'outputs': rows}, indent=2))


if __name__ == '__main__': main()
