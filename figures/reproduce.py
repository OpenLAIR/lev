"""Redraw seven aggregate-input figures into an external destination."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
COMMANDS = [('make_ablation.py',), ('make_training_v2.py',), ('make_appendix.py',), ('make_cascade_v2.py', '--column')]
EXPECTED = {'fig_ablation.pdf', 'fig_training_col_v2.pdf', 'fig_paired.pdf', 'fig_unlabelled.pdf',
            'fig_difficulty.pdf', 'fig_pareto_v2.pdf', 'fig_cascade_col.pdf'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    out = a.output_dir.resolve()
    if out.is_relative_to(HERE.parent): ap.error('Figure output must be outside the source artifact')
    if out.exists(): ap.error('Refusing an existing output destination')
    with tempfile.TemporaryDirectory(prefix='lev-figures-') as tmp:
        work = Path(tmp)
        shutil.copytree(HERE, work/'figures')
        shutil.copytree(HERE.parent/'results', work/'results')
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'MPLCONFIGDIR':str(work/'plot-config')}
        for command in COMMANDS:
            subprocess.run([sys.executable, str(work/'figures'/command[0]), *command[1:]], cwd=work, env=env, check=True)
        generated = {p.name for p in (work/'figures').glob('*.pdf')}
        if generated != EXPECTED: raise ValueError('Unexpected figure output inventory')
        out.mkdir(parents=True)
        for name in sorted(EXPECTED): shutil.copyfile(work/'figures'/name, out/name)
    print('Reproduced seven figures from packaged aggregates; no weights or evaluation rows loaded.')


if __name__ == '__main__': main()
