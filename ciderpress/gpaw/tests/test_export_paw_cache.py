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
