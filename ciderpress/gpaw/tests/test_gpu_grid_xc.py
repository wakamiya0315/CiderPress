"""Resident grid composition keeps all CIDER values and derivatives."""
import numpy as np
import pytest

from ciderpress import gpu
from ciderpress.gpaw.new_backend import CiderXC, _GridHybridKernel
from ciderpress.gpaw.cider_kernel import CiderMGGAHybridKernel


@pytest.mark.parametrize('nspin', [1, 2])
def test_grid_xc_cpu_composition(nspin, monkeypatch):
    """Check composition/order separately from CUDA/libm kernel arithmetic."""
    from types import SimpleNamespace
    from ciderpress.gpu import grid_xc
    monkeypatch.setattr(grid_xc, 'cupy', lambda: SimpleNamespace(
        asarray=np.asarray, asnumpy=np.asarray, empty=np.empty, float64=np.float64))
    monkeypatch.setattr(grid_xc, 'evaluate_device',
                        lambda model, x, rho, cut: model(x, rho, rhocut=cut))
    kernel = CiderMGGAHybridKernel(CiderXC().mlfunc, .37, None, None)
    rng = np.random.default_rng(193)
    n = 31
    rho = rng.uniform(.1, 2., (nspin, n))
    sigma = rng.uniform(.1, .2, (2*nspin-1, n))
    tau = rng.uniform(.5, 2, rho.shape)
    feat = rng.uniform(.1, 2, (nspin, 3, n))
    values = [rng.normal(size=shape) for shape in
              ((n,), rho.shape, sigma.shape, rho.shape, feat.shape)]
    result = []
    with gpu.policy(None):
        for function in (kernel.call_xc_kernel,
                         lambda *a, **kw: grid_xc.evaluate(kernel, *a, **kw)):
            e, v, ds, dt, df = [a.copy() for a in values]
            function(e, rho, sigma, feat, v, ds, df,
                     tau_sg=tau, dedtau_sg=dt)
            result.append((e, v, ds, dt, df))
    for a, b in zip(result[1], result[0]):
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('nspin', [1, 2])
@pytest.mark.parametrize('xmix', [1., .37])
def test_grid_xc(nspin, xmix, monkeypatch):
    if gpu._real_cupy() is None:
        pytest.skip('requires real CUDA CuPy')
    model = CiderXC().mlfunc
    rng = np.random.default_rng(113)
    n = 129
    rho = 10**rng.uniform(-9, 1, (nspin, n))
    sigma = rng.uniform(.01, .2, (2*nspin-1, n)) * rho[0]**(8./3)
    if nspin == 2:
        sigma[1] = .2*np.sqrt(sigma[0]*sigma[2])
    tau = rng.uniform(1, 4, rho.shape) * rho**(5./3)
    feat = rng.uniform(.01, 2, (nspin, 3, n))
    initial = [rng.normal(size=a) for a in ((n,), rho.shape, sigma.shape, rho.shape)]
    kernels = [CiderMGGAHybridKernel, _GridHybridKernel]
    result = []
    gpu.enable()
    try:
        for cls, selection in zip(kernels, ('none', 'all')):
            monkeypatch.setenv('CIDER_GPU_KERNELS', selection)
            kernel = cls(model, xmix, None if xmix == 1 else 'GGA_X_PBE', None)
            e, v, ds, dt = [a.copy() for a in initial]
            df = kernel.calculate(e, rho.copy(), v, sigma.copy(), ds,
                                  tau.copy(), dt, feat.copy())
            result.append((e, v, ds, dt, df))
        for a, b in zip(result[1], result[0]):
            np.testing.assert_allclose(a, b, atol=2e-13*max(1., abs(b).max()), rtol=1e-12)
            assert type(a) is np.ndarray
    finally:
        gpu.disable()
