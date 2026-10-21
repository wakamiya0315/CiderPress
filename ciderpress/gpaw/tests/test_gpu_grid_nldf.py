"""Fused grid interpolation versus the original C coefficient/fill calls."""
import numpy as np
import pytest

from ciderpress import gpu
from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.gpaw.new_backend import CiderXC
from ciderpress.gpaw.tests.test_gpu_nldf_fft import active_work, make_pw


@pytest.mark.parametrize('shape', [(7, 9, 11), (8, 6, 10)])
def test_grid_nldf(shape, monkeypatch):
    if gpu._real_cupy() is None:
        pytest.skip('requires real CUDA CuPy')
    from ciderpress.gpu.grid_nldf import fill, contract
    plan = NLDFSplinePlan(CiderXC().mlfunc.settings.nldf_settings, 1,
                          .01, 1.8, 18, coef_order='gq')
    rng = np.random.default_rng(193)
    a = rng.uniform(0, plan._spline_size-1-1e-10, shape)
    a.flat[:4] = [0, 1, 2-1e-14, plan._spline_size-1-1e-10]
    f = rng.normal(size=shape)
    gpu.enable()
    try:
        pw = make_pw(shape, plan.nalpha)
        monkeypatch.setenv('CIDER_GPU_KERNELS', 'nldf_fft')
        work = pw._gpu_work()
        for i in (-1, 0, 1, 2):
            p, dp = plan.get_interpolation_coefficients(a, i=i)
            p = p.reshape(*shape, plan.nalpha)
            dp = dp.reshape(p.shape)
            monkeypatch.setenv('CIDER_GPU_KERNELS', 'none')
            pw.set_work(f, p)
            ref_work = active_work(pw)
            v, dv = np.empty(shape), np.empty(shape)
            pw.fill_vj_feature_(v, p)
            pw.fill_vj_feature_(dv, dp)
            monkeypatch.setenv('CIDER_GPU_KERNELS', 'nldf_fft')
            work = pw._gpu_work()
            fill(work, plan, a, f, i)
            np.testing.assert_allclose(active_work(pw), ref_work, atol=1e-12, rtol=1e-12)
            actual = contract(work, plan, a, i)
            for x, y in zip(actual, (v, dv)):
                np.testing.assert_allclose(x, y, atol=1e-12, rtol=1e-12)
            fill(work, plan, a, f, i, add=True)
            np.testing.assert_allclose(active_work(pw), 2*ref_work, atol=1e-12, rtol=1e-12)
            # Mutable host spline tables invalidate the upload cache.
            table = plan._local_alpha_transform[i]
            original = table.copy()
            table *= 1.01
            fill(work, plan, a, f, i)
            np.testing.assert_allclose(active_work(pw), 1.01*ref_work, atol=1e-12, rtol=1e-12)
            table[:] = original
    finally:
        gpu.disable()
