"""Opt-in candidate masks use the same density units and edges as inference."""
from copy import deepcopy

import numpy as np
import pytest

from ciderpress.ft import FineTuner
from ciderpress.ft.records import covariance, integrate
from ciderpress.ft.tests.test_ft import synthetic, unconstrained, add_known_labels
from ciderpress.dft.xc_evaluator import RBFEvaluator
from ciderpress.dft.xc_evaluator2 import MappedXC2


@pytest.mark.parametrize('grouped', [False, True])
def test_empty_spin_descriptors_do_not_select_controls(grouped):
    base, data = synthetic(nspin=2, nsystems=2)
    records = deepcopy(list(data.records.values()))
    for r in records:
        r['rho_data'][1] = 0
    changed = deepcopy(records)
    rng = np.random.default_rng(7)
    for r in changed:
        r['desc'][1] = rng.normal(size=r['desc'][1].shape) * 1e5
    kwargs = (dict(candidate_groups={'0': 'p', '1': 'p'}, group_fractions={'p': 1.})
              if grouped else {})
    ft = FineTuner(base, n_control=10, max_candidates=46, constraints=unconstrained(),
                   candidate_mask='inference', **kwargs)
    a, b = ft._kernels()[0], ft._kernels()[0]
    info = ft._select_controls(a, records, ['0', '1'])
    assert info == ft._select_controls(b, changed, ['0', '1'])
    assert a.X1ctrl.tobytes() == b.X1ctrl.tobytes()
    assert all(r['valid_candidate_rows'] == 22 for r in info['records'])


@pytest.mark.parametrize('component', ['x', 'c'])
@pytest.mark.parametrize('nspin', [1, 2])
def test_cutoff_edges_and_density_units(component, nspin):
    base, data = synthetic(nspin=nspin, nsystems=1)
    record = deepcopy(data.records['0'])
    cut = .1
    edge = np.array([np.nextafter(cut, 0), cut, np.nextafter(cut, 1)])
    record['rho_data'][:, 0] = 0
    # SEP cutoff is spin-scaled; NPOL cutoff is the physical total.
    record['rho_data'][0, 0, :3] = edge * (nspin if component == 'c' else 1)
    ft = FineTuner(base, components=component, n_control=2, max_candidates=20,
                   rhocut=cut, candidate_mask='inference')
    kernel = ft._kernels()[0]
    np.testing.assert_array_equal(ft._candidate_rows(kernel, record), [1, 2])
    info = ft._select_controls(kernel, [record])
    assert info['candidate_rows'] == 2
    # The review's two-spin NPOL 0.075 physical-density reproducer.
    if component == 'c' and nspin == 2:
        record['rho_data'][:, 0] = .075
        with pytest.raises(ValueError, match='density cutoff'):
            ft._select_controls(kernel, [record])
    # With rhocut=0 inference disables its mask, even for negative PAW rho.
    ft.rhocut = 0
    record['rho_data'][:, 0] = -1
    assert len(ft._candidate_rows(kernel, record)) == (nspin if component == 'x' else 1)*23


@pytest.mark.parametrize('component', ['x', 'c'])
def test_mixed_spin_budgets_signed_weights_and_inference(component):
    base, one = synthetic(nspin=1, nsystems=2)
    _, two = synthetic(nspin=2, nsystems=2)
    records = [one.records['0'], two.records['0'], one.records['1'], two.records['1']]
    ft = FineTuner(base, n_control=6, max_candidates=32, components=component,
                   candidate_mask='inference', candidate_groups=dict(a='p', b='p', c='m', d='m'),
                   group_fractions=dict(p=.5, m=.5))
    k = ft._kernels()[0]
    info = ft._select_controls(k, records, list('abcd'))
    assert [r['candidate_rows'] for r in info['records']] == [8]*4
    altered = deepcopy(records)
    for r in altered:
        r['wt'] *= -13
    k2 = ft._kernels()[0]
    assert info == ft._select_controls(k2, altered, list('abcd'))
    assert k.X1ctrl.tobytes() == k2.X1ctrl.tobytes()
    k.alpha = np.linspace(-.02, .03, k.Nctrl)
    mapped = MappedXC2([k.map(lambda x: RBFEvaluator(x.kernel, x.X1ctrl, x.alpha))], base.settings)
    for r in records:
        assert np.any(r['wt'] < 0)
        np.testing.assert_allclose(
            covariance(k, r, base.settings, 7, ft.rhocut) @ k.alpha,
            integrate(mapped, r, rhocut=ft.rhocut, core=False), rtol=1e-12, atol=1e-14)


def test_mask_saved_and_reloaded(tmp_path):
    base, data = synthetic(nspin=2, nsystems=3)
    ft = FineTuner(base, n_control=8, max_candidates=100, constraints=unconstrained(),
                   candidate_mask='inference')
    add_known_labels(data, ft)
    ft.fit(data)
    loaded = FineTuner.load(ft.save(tmp_path / 'masked.yaml'))
    assert loaded.candidate_mask == 'inference'
    assert loaded.predict(data) == ft.predict(data)
    loaded.fit(data)
    assert loaded.predict(data) == ft.predict(data)
    with pytest.raises(ValueError, match='candidate_mask'):
        FineTuner(base, candidate_mask='physical')
