"""CUDA saturation against the independent 80-digit reference and host plans."""
import numpy as np
import pytest

from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.dft.settings import NLDFSettingsVJ
from ciderpress.dft.tests.test_exponent_saturation import reference, saturate
from ciderpress.gpu import cupy
from ciderpress.gpu.pasdw_exponents import _SOURCE, arguments


@pytest.fixture
def cp():
    module = pytest.importorskip('cupy')
    try:
        if module.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('real CUDA device required')
    except module.cuda.runtime.CUDARuntimeError:
        pytest.skip('real CUDA device required')
    return module


@pytest.mark.parametrize('cutoff', [1., 1.2e6])
def test_cuda_saturation_reference(cp, cutoff):
    # Compile the same device function used by the production argument kernel.
    source = _SOURCE + r'''
extern "C" __global__ void check(double* a, double* d, double cut, int n) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) a[i] = saturate_exponent(a[i], cut, d[i]);
}
'''
    kernel = cp.RawKernel(source, 'check', options=('--fmad=false',))
    values = cutoff*np.array([-1e-12, -1e-20, 0., 1e-20, 1e-16, 1e-12,
                              1e-8, 1e-3, .1, .9, 1., 1.1])
    a, d = cp.asarray(values), cp.empty(values.shape)
    kernel((1,), (32,), (a, d, np.float64(cutoff), np.int32(len(values))))
    expected = np.array([reference(v, cutoff) for v in values]).T
    actual = np.array([cp.asnumpy(a), cp.asnumpy(d)])
    np.testing.assert_allclose(actual, expected, atol=0., rtol=5e-14)
    host = saturate(values, cutoff)
    np.testing.assert_allclose(actual[0], host[0], atol=0., rtol=1e-12)
    np.testing.assert_allclose(actual[1], host[1][0], atol=0., rtol=1e-12)


@pytest.mark.parametrize('nspin', [1, 2])
@pytest.mark.parametrize('level', ['MGGA', 'GGA'])
@pytest.mark.parametrize('grad', [0., .2])
def test_cuda_plan_arguments(cp, nspin, level, grad):
    params = [1., grad, .03125] if level == 'MGGA' else [1., grad]
    settings = NLDFSettingsVJ(level, params, 'one', ['se'], [params])
    plan = NLDFSplinePlan(settings, nspin, 1e-8, 1.8, 60, rhocut=1e-8)
    rho = np.zeros((5 if level == 'MGGA' else 4, 103))
    rho[0] = np.r_[0., .25e-8, .5e-8, .75e-8, 1e-8,
                   np.geomspace(1e-8, 100., 98)] / nspin
    rng = np.random.default_rng(514)
    rho[1:4] = rng.normal(size=(3, 103))*rho[0]*.2
    if level == 'MGGA':
        rho[4] = .5*(nspin*rho[0])**(5/3)/nspin
    for index in [-1, 0]:
        expected = plan.get_interpolation_arguments(plan.get_rho_tuple(rho), i=index)
        actual = arguments(plan, rho, index)
        for a, b in zip((actual[0], *actual[1]), (expected[0], *expected[1])):
            # Log interpolation can amplify a last-bit exponent error.
            np.testing.assert_allclose(cp.asnumpy(a), b, atol=1e-12, rtol=1e-12)


def test_cuda_small_exponent_chain_rule(cp):
    settings = NLDFSettingsVJ('MGGA', [1., .2, .03125], 'one',
                              ['se'], [[1., .2, .03125]])
    plan = NLDFSplinePlan(settings, 2, 1e-8, 1.8, 60)
    rho = np.array([[.1, 1., 10.], [.02, .1, .3], [.01, .2, .1],
                    [.03, .1, .2], [.1, .4, 2.]])
    _, derivative = arguments(plan, rho, -1)
    for index, dindex in [(0, 0), (4, 2)]:
        for step in [2e-5, 1e-5, 5e-6]:
            plus, minus = rho.copy(), rho.copy()
            plus[index] += step
            minus[index] -= step
            numeric = (arguments(plan, plus, -1)[0] -
                       arguments(plan, minus, -1)[0])/(2*step)
            np.testing.assert_allclose(cp.asnumpy(numeric),
                                       cp.asnumpy(derivative[dindex]),
                                       atol=2e-8, rtol=2e-7)
