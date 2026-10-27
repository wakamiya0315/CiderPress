"""Frozen outputs generated once with the actual 77045b4ec9 tuner on TSUBAME.

The fixture records the dependency versions and source hash. Bitwise checks
use one BLAS thread, as recorded in its manifest; they are not tolerance tests.
"""
from pathlib import Path

import numpy as np
import pytest

from ciderpress.ft import FineTuner
from ciderpress.ft.tests.test_ft import synthetic, add_known_labels


def golden_fit(tuner_type, component):
    base, data = synthetic(nspin=2, nsystems=3)
    ft = tuner_type(base, n_control=8, max_candidates=100, components=component)
    add_known_labels(data, ft)
    ft.fit(data)
    values = {}
    for i, kernel in enumerate(ft.engine.kernels):
        values[f'{component}_controls_{i}'] = kernel.X1ctrl
        values[f'{component}_alpha_{i}'] = kernel.alpha
    predictions = ft.predict(data)
    values[f'{component}_predictions'] = np.array([
        [predictions[k][name] for name in ('base', 'correction', 'fine_tuned')]
        for k in sorted(predictions)])
    return values


@pytest.mark.parametrize('component', ['x', 'c', 'xc'])
def test_default_matches_77045b4_bitwise(component):
    with np.load(Path(__file__).with_name('data') / 'ft_77045b4.npz') as fixture:
        for name, actual in golden_fit(FineTuner, component).items():
            expected = fixture[name]
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes(), name
