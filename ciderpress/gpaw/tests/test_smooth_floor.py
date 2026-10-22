"""Value/derivative, clone, serialization and fail-closed variant contracts."""
import numpy as np
import pytest

from ciderpress.dft.exponent_floor import smooth_floor_q
from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.gpaw.new_backend import CiderXC, restore_cider_xc


def plan(width=.1):
    param = CiderXC()
    return NLDFSplinePlan(param.mlfunc.settings.nldf_settings, 1,
                          .00390625, 1.8, 21, smooth_floor_width=width)


@pytest.mark.parametrize('width', [.01, .1, 1.0])
def test_map_derivative_and_tail(width):
    a0 = .00390625
    a = a0*np.array([-1000., -1., .9, 1., 1.1, 100.])
    q, dq = smooth_floor_q(a, a0, 1.8, width)
    assert np.isfinite(q).all() and np.isfinite(dq).all()
    assert (q >= 0).all() and (dq >= 0).all()
    delta = a0*1e-6
    fd = (smooth_floor_q(a+delta, a0, 1.8, width)[0] -
          smooth_floor_q(a-delta, a0, 1.8, width)[0])/(2*delta)
    np.testing.assert_allclose(fd, dq, rtol=1e-7, atol=1e-7)


def test_four_exponents_and_radial_clone():
    p = plan()
    radial = p.new(nalpha=30, use_smooth_expnt_cutoff=True)
    assert radial.smooth_floor_width == p.smooth_floor_width
    for obj in (p, radial):
        for i in (-1, 0, 1, 2):
            n = np.array([1.665556245187e-4])
            sig = np.zeros(1)
            a, da = obj.eval_feat_exp((n, sig, np.zeros(1)), i=i)
            tc = (obj.alpha0-a)/da[2]
            # Exact exponent-space midpoint, with a signed tau sweep.
            eps = 1e-10
            qm, dm = obj.get_interpolation_arguments((n, sig, tc-eps), i=i)
            qp, dp = obj.get_interpolation_arguments((n, sig, tc+eps), i=i)
            q, d = obj.get_interpolation_arguments((n, sig, tc), i=i)
            np.testing.assert_allclose((qp-qm)/(2*eps), d[2], rtol=1e-7)
            # Derivative difference must decrease tenfold with the step.
            _, dm2 = obj.get_interpolation_arguments((n, sig, tc-eps/10), i=i)
            _, dp2 = obj.get_interpolation_arguments((n, sig, tc+eps/10), i=i)
            np.testing.assert_allclose((dp[2]-dm[2])/10, dp2[2]-dm2[2], rtol=1e-6)


def test_zero_width_uses_official_native_map():
    import ctypes
    from ciderpress.dft.plans import libcider
    p = plan(0)
    a = p.alpha0*np.array([-1., 0., .99, 1., 1.01, 2., 1e7])
    q, dq = p.get_a2q_fast(a)
    expected = np.empty_like(a)
    de = np.empty_like(a)
    ptr = lambda x: x.ctypes.data_as(ctypes.c_void_p)
    libcider.cider_ind_etb(ptr(expected), ptr(de), ptr(a), ctypes.c_int(a.size),
                         ctypes.c_double(p.alpha0), ctypes.c_double(p.lambd))
    libcider.cider_ind_clip(ptr(expected), ptr(de), ctypes.c_int(20), ctypes.c_int(a.size))
    assert np.array_equal(q, expected) and np.array_equal(dq, de)


def test_negative_tail_keeps_q_derivative_without_lower_clip():
    p = plan(.1)
    # a_eff would round to alpha0 here; log1p keeps the analytic tail.
    a = np.array([p.alpha0*(1-40*.1)])
    q, dq = p.get_a2q_fast(a)
    assert q[0] > 0 and dq[0] > 0
    assert q[0] < 1e-16


def test_serialization_and_invalid_width():
    for width in (0., .1):
        xc = CiderXC(smooth_floor_width=width)
        restored = restore_cider_xc(xc.todict(), expected=xc)
        assert restored.todict() == xc.todict()
    assert 'smooth_floor_width' not in CiderXC().todict()['options']
    for width in (-1., np.inf, np.nan):
        with pytest.raises(ValueError, match='smooth_floor_width'):
            CiderXC(smooth_floor_width=width)


def test_classic_option_serialization():
    from ciderpress.gpaw.calculator import get_cider_functional
    xc = get_cider_functional('CIDER26XCSURFSCI', xkernel=None, ckernel=None,
                              smooth_floor_width=.1)
    assert xc.todict()['xc_params']['smooth_floor_width'] == .1


@pytest.mark.parametrize('cached_width,width', [(0., 0.), (.1, .1), (0., .1), (.1, 0.)])
def test_radial_cache_cannot_mix_variants(cached_width, width):
    from types import SimpleNamespace
    from ciderpress.gpaw.atom_utils import FastPASDWCiderKernel
    kernel = object.__new__(FastPASDWCiderKernel)
    kernel.dens = SimpleNamespace(nt_sg=np.zeros((1, 1)))
    kernel.plan = SimpleNamespace(smooth_floor_width=width)
    setup = SimpleNamespace(cider_contribs=SimpleNamespace(
        plan=SimpleNamespace(smooth_floor_width=cached_width)))
    if cached_width == width:
        kernel.initialize_more_things([setup])
    else:
        with pytest.raises(ValueError, match='fresh setups'):
            kernel.initialize_more_things([setup])


@pytest.mark.parametrize('selection', ['all', 'pasdw', 'fine_grid', 'grid_nldf', 'rbf'])
def test_gpu_selection_rejected(monkeypatch, selection):
    from ciderpress import gpu
    from ciderpress.dft.exponent_floor import check_floor_cpu
    monkeypatch.setenv('CIDER_GPU_KERNELS', selection)
    with gpu.policy(None):
        # Avoid needing CUDA on the Mac: explicit device validation happens
        # before the New functional builds any engine or dispatches a kernel.
        with pytest.raises(NotImplementedError, match='CPU-only'):
            check_floor_cpu(.1, device=0)
    token = gpu._policy.set((True, 0))
    try:
        with pytest.raises(NotImplementedError, match='CPU-only'):
            CiderXC(smooth_floor_width=.1)
    finally:
        gpu._policy.reset(token)
