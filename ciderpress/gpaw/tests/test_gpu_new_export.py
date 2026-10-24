"""FP64 export quadrature versus Classic, without an SCF or checkpoint."""
from types import SimpleNamespace

import numpy as np
import pytest

from ciderpress import gpu
from ciderpress.dft.grids_indexer import AtomicGridsIndexer
from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.gpaw.atom_descriptor_utils import (
    PASDWCiderFeatureKernel, PAWCiderFeatContribs, calculate_paw_sl_features,
)
from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional, CiderXC


def test_gpu_option_requires_cuda_functional():
    from ciderpress.gpaw.new_export import export_features

    functional = object.__new__(CiderNewMGGAFunctional)
    functional.setups = []
    functional._device = None
    with pytest.raises(ValueError, match='CUDA functional'):
        export_features(functional, None, None, None, gpu=True)


def make_engine(nspin):
    from gpaw.sphere.lebedev import Y_nL, weight_n

    rng = np.random.default_rng(2719)
    nr, nq, npack, nlm = 8, 3, 6, Y_nL.shape[1]
    setups = []
    for _ in range(2):
        r = np.linspace(0., 2., nr)
        rgd = SimpleNamespace(r_g=r, dr_g=np.full(nr, .1), dv_g=.1*r*r)
        rgd.empty = lambda n, nr=nr: np.empty((n, nr))
        x = SimpleNamespace(rgd=rgd, B_pqL=.01*rng.normal(size=(npack, nq, 4)),
                            nc_corehole_g=None, dc_corehole_g=None)
        for name in ('n_qg', 'nt_qg', 'd_qg', 'dt_qg'):
            setattr(x, name, .01*rng.normal(size=(nq, nr)))
        for name in ('nc_g', 'nct_g', 'tauc_g', 'tauct_g'):
            setattr(x, name, rng.uniform(.4, .6, nr))
        for name in ('dc_g', 'dct_g'):
            setattr(x, name, .01*rng.normal(size=nr))
        for name in ('tau_pg', 'taut_pg'):
            setattr(x, name, .01*rng.normal(size=(npack, nr*len(weight_n))))
        c = PAWCiderFeatContribs.__new__(PAWCiderFeatContribs)
        c.xcc = x
        c.plan = NLDFSplinePlan(CiderXC().mlfunc.settings.nldf_settings, nspin,
                               .01, 1.8, 18, coef_order='gq')
        c.grids_indexer = AtomicGridsIndexer.make_single_atom_indexer(Y_nL, r)
        c.w_g = (rgd.dv_g[:, None]*weight_n).ravel()
        c.grids_indexer.set_weights(c.w_g)
        p = SimpleNamespace(ni=6, slrgd=rgd, alphas=c.plan.alphas,
                            nlist_i=np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
                            jlist_i=np.array([1, 0, 1, 0, 1, 0], dtype=np.int32),
                            lmlist_i=np.array([0, 1, 2, 3, 0, 1], dtype=np.int32),
                            pfuncs_ng=rng.normal(size=(2, nr)),
                            ffuncs_jg=.01*rng.normal(size=(2, nr)))
        setups.append(SimpleNamespace(cider_contribs=c, xc_correction=x, pa_setup=p))
    setups.append(setups[0])
    D = {a: rng.uniform(.01, .1, (nspin, npack)) for a in (2, 1, 0)}
    paw = PASDWCiderFeatureKernel.__new__(PASDWCiderFeatureKernel)
    paw.fr_asgLq = {a: rng.normal(size=(nspin, nr, nlm, 18)) for a in D}
    paw.df_asgLq = {a: .1*rng.normal(size=(nspin, nr, nlm, 18)) for a in D}
    coef = {a: rng.normal(size=(nspin, 6, 16)) for a in D}
    return SimpleNamespace(setups=setups, D_asp=D, paw_kernel=paw, c_asiq=coef)


@pytest.mark.parametrize('nspin', [1, 2])
def test_paw_features(nspin):
    cp = gpu._real_cupy()
    if cp is None:
        pytest.skip('requires real CUDA CuPy')
    from ciderpress.gpaw._export_gpu import paw_features

    engine = make_engine(nspin)
    paw = engine.paw_kernel
    with gpu.policy(None):
        ref_nl = paw.calculate_paw_cider_features_p2(
            engine.setups, engine.D_asp, engine.c_asiq, None, None)
        ref_raw = calculate_paw_sl_features(engine.setups, engine.D_asp, True)
    original = {name: {a: x.copy() for a, x in getattr(paw, name).items()}
                for name in ('fr_asgLq', 'df_asgLq')}
    with gpu.policy(cp.cuda.Device().id):
        # Both host coefficients and resident forward caches are supported.
        for resident in (False, True):
            if resident:
                for name in original:
                    setattr(paw, name, {a: cp.asarray(x) for a, x in original[name].items()})
            got = paw_features(engine)
            for result, expected in zip(got, (*ref_nl, *ref_raw)):
                assert set(result) == set(engine.D_asp)
                for a in expected:
                    assert result[a].dtype == np.float64
                    np.testing.assert_allclose(result[a], expected[a], rtol=1e-12, atol=1e-13)
            for result, expected in zip(got[2:], ref_raw):
                for a in expected:
                    np.testing.assert_array_equal(result[a].view(np.uint64), expected[a].view(np.uint64))
            for name in original:
                for a, x in getattr(paw, name).items():
                    np.testing.assert_array_equal(cp.asnumpy(x), original[name][a])


def test_empty_paw():
    if gpu._real_cupy() is None:
        pytest.skip('requires real CUDA CuPy')
    from ciderpress.gpaw._export_gpu import paw_features

    engine = SimpleNamespace(setups=[], D_asp={}, paw_kernel=SimpleNamespace())
    assert paw_features(engine) == ({}, {}, {}, {})


@pytest.mark.parametrize('nspin', [1, 2])
def test_relative_guard_reference_cpu(nspin, monkeypatch):
    """The host guard must reproduce Classic, including spin BLAS ordering."""
    import ciderpress.gpaw._export_gpu as module

    # Only the device-to-host copy is replaced; this tests the host reference
    # independently of CUDA. Actual device quadrature is checked above.
    monkeypatch.setattr(module, 'cupy', lambda: SimpleNamespace(asnumpy=np.asarray))
    engine = make_engine(nspin)
    with gpu.policy(None):
        nl = engine.paw_kernel.calculate_paw_cider_features_p2(
            engine.setups, engine.D_asp, engine.c_asiq, None, None)
        raw = calculate_paw_sl_features(engine.setups, engine.D_asp, True)
    for expected, rho, ae in zip(nl, raw, (True, False)):
        for a, values in expected.items():
            fake = np.ones_like(values)
            nn = engine.setups[a].cider_contribs.grids_indexer.ylm.shape[0]
            fake[..., ::nn] = 0  # select every radial shell
            got = module._repair_small_features(engine, a, fake, rho[a], ae)
            np.testing.assert_array_equal(got.view(np.uint64), values.view(np.uint64))


@pytest.mark.parametrize('nspin', [1, 2])
def test_saved_xc(nspin, monkeypatch):
    if gpu._real_cupy() is None:
        pytest.skip('requires real CUDA CuPy')
    from ciderpress.gpaw.new_export import integrate_xc, xc_energy_density
    from ciderpress.gpaw.tests.test_new_export import record

    rec, param = record(nspin)
    with gpu.policy(None):
        expected = xc_energy_density(rec, param)
    monkeypatch.setenv('CIDER_GPU_KERNELS', 'all')
    for blocksize in (1, 7, 100):
        np.testing.assert_allclose(xc_energy_density(rec, param, blocksize=blocksize, gpu=True),
                                   expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(integrate_xc(rec, param, blocksize=blocksize, gpu=True),
                                   expected @ rec['wt']+rec['exc_core'], rtol=0, atol=1e-12)
    monkeypatch.setenv('CIDER_GPU_KERNELS', 'none')
    with pytest.raises(ValueError, match='requires semilocal'):
        integrate_xc(rec, param, gpu=True)
