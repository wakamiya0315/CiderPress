"""One-off CPU fixture generator; supply tuner.py in a complete old git archive.

Run on TSUBAME cpu_4 with OMP/MKL/OpenBLAS threads=1, never during pytest.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import scipy

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('old_tuner', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    # Invoke this file directly. All CiderPress imports, including records.py,
    # then come from the 77045b4 archive rather than the current package.
    archive = args.old_tuner.resolve().parents[2]
    sys.path.insert(0, str(archive))
    import ciderpress.ft.tuner as old
    assert Path(old.__file__).resolve() == args.old_tuner.resolve()
    spec = importlib.util.spec_from_file_location('golden_recipe',
                                                 Path(__file__).with_name('test_legacy_golden.py'))
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    arrays = {}
    for component in ('x', 'c', 'xc'):
        arrays.update(recipe.golden_fit(old.FineTuner, component))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = dict(tuner_commit='77045b4ec9',
                    tuner_sha256=hashlib.sha256(args.old_tuner.read_bytes()).hexdigest(),
                    fixture_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                    numpy=np.__version__, scipy=scipy.__version__,
                    threads=1, location='TSUBAME cpu_4',
                    ciderpress_source='complete git archive of 77045b4ec9; shared native libraries',
                    ft_source_sha256={str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((archive/'ciderpress/ft').glob('*.py'))},
                    inputs='synthetic(seed=119,nspin=2,nsystems=3); n_control=8; '
                           'max_candidates=100; default constraints/prior/cutoff/jitter; x,c,xc')
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
