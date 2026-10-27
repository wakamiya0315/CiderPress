"""Real PASDW table reuse, settings keying and private mutable ownership."""
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest
from gpaw.setup import create_setup

from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.gpaw.atom_descriptor_utils import PASDWCiderFeatureKernel
from ciderpress.gpaw.atom_utils import FastPASDWCiderKernel
from ciderpress.gpaw.new_backend import CiderXC
from ciderpress.gpaw.new_export import ExportPAWCache


def kernel(setup, *, nspin=1, feature=False, lambd=1.8):
    plan = NLDFSplinePlan(CiderXC().mlfunc.settings.nldf_settings,
                         nspin, .01, lambd, 18, coef_order='gq')
    cls = PASDWCiderFeatureKernel if feature else FastPASDWCiderKernel
    k = cls(None, plan, None, False)
    private = copy(setup)
    private.cider_contribs = None
    k.initialize(SimpleNamespace(nt_sg=np.zeros((nspin, 1))), None, None,
                 [private, private])
    return k


def test_private_owners_and_cross_spin_reuse():
    cache = ExportPAWCache()
    source = create_setup('H', xc='PBE')
    original = vars(source).copy()
    first = kernel(source, feature=True)
    cache.initialize(first, ('cpu', 'float64'))
    second = kernel(create_setup('H', xc='PBE'), nspin=2)
    cache.initialize(second, ('cpu', 'float64'))
    assert cache.info() == dict(size=1, maxsize=16, misses=1, hits=1)
    a, b = first.setups[0], second.setups[0]
    assert a.cider_contribs is not b.cider_contribs
    assert a.cider_contribs.plan is not b.cider_contribs.plan
    assert a.cider_contribs.nspin == 1 and b.cider_contribs.nspin == 2
    assert a.ps_setup is not b.ps_setup
    assert a.ps_setup.pcol is not b.ps_setup.pcol
    assert a.ps_setup._radial_transforms is not b.ps_setup._radial_transforms
    nm = a.ps_setup._grid_nlm
    am, bm = a.ps_setup._radial_transforms[nm], b.ps_setup._radial_transforms[nm]
    assert am is not bm
    assert am.matrices[0] is bm.matrices[0]
    assert a.ps_setup.pfuncs_ng is b.ps_setup.pfuncs_ng
    assert not a.ps_setup.pfuncs_ng.flags.writeable
    a.cider_contribs.set_D_sp(np.ones((1, 1)), a.xc_correction)
    a.cider_contribs.plan._cached_p_i_qg[0].append('scratch')
    a.ps_setup._pasdw_device_cache = {'private': 1}
    assert not hasattr(b.cider_contribs, '_D_sp')
    assert b.cider_contribs.plan._cached_p_i_qg == {0: [], 1: []}
    assert not hasattr(b.ps_setup, '_pasdw_device_cache')
    assert vars(source).keys() == original.keys()
    assert all(vars(source)[k] is v for k, v in original.items())
    cache.clear()
    assert cache.info() == dict(size=0, maxsize=16, misses=0, hits=0)
    # An engine that already holds tables survives clearing the cache.
    assert a.ps_setup.pfuncs_ng is b.ps_setup.pfuncs_ng


def test_different_settings_build_distinct_tables():
    cache = ExportPAWCache(maxsize=1)
    setup = create_setup('H', xc='PBE')
    a, b = kernel(setup), kernel(setup, lambd=1.9)
    cache.initialize(a, ('cpu', 'float64'))
    cache.initialize(b, ('cpu', 'float64'))
    assert cache.info() == dict(size=1, maxsize=1, misses=2, hits=0)
    assert a.setups[0].ps_setup.pfuncs_ng is not b.setups[0].ps_setup.pfuncs_ng


def test_cached_plan_does_not_alias_caller_settings():
    cache = ExportPAWCache()
    setup = create_setup('H', xc='PBE')
    first = kernel(setup)
    expected = first.plan.nldf_settings.theta_params[0]
    cache.initialize(first, ('cpu', 'float64'))
    first.plan.nldf_settings.theta_params[0] *= 2
    first.setups[0].cider_contribs.plan.nldf_settings.theta_params[0] *= 3
    # A separate caller with the original settings hits the old entry. Neither
    # edits to the source settings nor to a private radial plan may poison it.
    original = kernel(create_setup('H', xc='PBE'))
    cache.initialize(original, ('cpu', 'float64'))
    assert cache.info()['hits'] == 1
    assert original.setups[0].cider_contribs.plan.nldf_settings.theta_params[0] == expected


@pytest.mark.parametrize('change', ['setup', 'nldf', 'smooth', 'pasdw',
                                   'device', 'dtype', 'model', 'cutoff'])
def test_cache_key_dependencies(change):
    setup = create_setup('H', xc='PBE')
    k = kernel(setup)
    signature = dict(model='model-a', nldf={'qmax': 300.},
                     pasdw={'ovlp_fit': True, 'store_funcs': False},
                     device=None, dtype='<f8')
    before = ExportPAWCache._key(setup, k, signature)
    if change == 'setup':
        setup.xc_correction.nc_g = setup.xc_correction.nc_g.copy()
        setup.xc_correction.nc_g[0] += .1
    elif change == 'nldf':
        signature['nldf']['qmax'] = 301.
    elif change == 'smooth':
        k.plan.smooth_floor_width = .01
    elif change == 'pasdw':
        signature['pasdw']['store_funcs'] = True
    elif change == 'device':
        signature['device'] = 0
    elif change == 'dtype':
        signature['dtype'] = '<f4'
    elif change == 'model':
        signature['model'] = 'model-b'
    elif change == 'cutoff':
        k.plan.rhocut *= 2
    assert ExportPAWCache._key(setup, k, signature) != before


def test_invalid_capacity():
    with pytest.raises(ValueError, match='positive integer'):
        ExportPAWCache(0)


@pytest.mark.parametrize('gpu', [False, True])
def test_full_export_cold_warm_uncached_bitwise(gpu, tmp_path):
    """Real PAW/FFT fixture: compare every exported field and both replays."""
    from ase import Atoms
    from gpaw import GPAW
    from ciderpress import gpu as device
    from ciderpress.gpaw.new_export import (DEFAULT_PAW_CACHE, export_training_data,
                                           integrate_xc)
    if gpu and device._real_cupy() is None:
        pytest.skip('requires real CUDA CuPy')
    atoms = Atoms('H2', positions=[[2, 2, 1.6], [2, 2, 2.4]],
                  cell=[4]*3, pbc=True)
    param = CiderXC()
    calc = GPAW(legacy_gpaw=False, mode=dict(name='pw', ecut=100), xc=param,
                h=.3, nbands=2, symmetry='off', txt=None,
                parallel=dict(gpu=gpu, domain=1, band=1, distribute_density=False))
    # A prescribed, finite n/tau/D fixture exercises the complete exporter
    # without conflating this cache identity test with SCF convergence.
    calc.create_new_calculation(atoms)
    dft = calc.dft
    for wfs in dft.ibzwfs:
        wfs.psit_nX.data[:] = 0
        wfs.psit_nX.data[0] = wfs.psit_nX.xp.asarray(np.exp(-.7*wfs.psit_nX.desc.ekin_G))
        wfs._occ_n = np.array([1., 0.])
        wfs._eig_n = np.zeros(2)
    dft.density.update_ked(dft.ibzwfs)
    dft.density.taut_sR.data[:] += .05
    dft.potential, dft.energies, _ = dft.pot_calc.calculate(dft.density, dft.ibzwfs)
    dft.energies.set(band=0., entropy=0., extrapolation=0.)
    dft.calculate_energy()
    DEFAULT_PAW_CACHE.clear()
    try:
        cold = export_training_data(calc, gpu=gpu)
        first = DEFAULT_PAW_CACHE.info().copy()
        warm = export_training_data(calc, gpu=gpu)
        second = DEFAULT_PAW_CACHE.info().copy()
        uncached = export_training_data(calc, gpu=gpu, paw_cache=None)
        assert first['misses'] > 0 and second['hits'] > first['hits']
        assert second['misses'] == first['misses']
        assert DEFAULT_PAW_CACHE.info() == second
        assert cold.keys() == warm.keys() == uncached.keys()
        for key in cold:
            a, b, c = (r[key] for r in (cold, warm, uncached))
            if isinstance(a, np.ndarray):
                assert a.dtype == b.dtype == c.dtype
                assert a.shape == b.shape == c.shape
                assert a.tobytes() == b.tobytes() == c.tobytes(), key
            elif isinstance(a, (float, np.floating)):
                assert np.float64(a).tobytes() == np.float64(b).tobytes() == np.float64(c).tobytes(), key
            else:
                assert a == b == c, key
        replay = [np.float64(integrate_xc(r, param, gpu=gpu)).tobytes()
                  for r in (cold, warm, uncached)]
        assert replay[0] == replay[1] == replay[2]
    finally:
        DEFAULT_PAW_CACHE.clear()
