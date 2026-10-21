"""Same-coefficient stencil-2 GPU gradient and correction, including edges."""
import numpy as np
import pytest

from ciderpress import gpu
from ciderpress.gpaw.cider_fft import get_rho_sxg, add_gradient_correction
from gpaw.fd_operators import Gradient
from gpaw.old.grid_descriptor import GridDescriptor


@pytest.mark.parametrize('pbc', [(1, 1, 1), (1, 1, 0), (0, 0, 0)])
@pytest.mark.parametrize('skew', [False, True])
def test_fine_grid(pbc, skew):
    if gpu._real_cupy() is None:
        pytest.skip('requires real CUDA CuPy')
    from ciderpress.gpu.fine_grid import get_rho, add_correction
    cell = np.diag([6., 7., 8.])
    if skew:
        cell[1, 0] = 1.5
    gd = GridDescriptor([12, 14, 16], cell, pbc_c=pbc)
    gradients = [Gradient(gd, v, n=2).apply for v in range(3)]
    rng = np.random.default_rng(193)
    n, tau = rng.normal(size=(2, 1, *gd.n_c))
    ref = get_rho_sxg(n, gradients, tau)
    gpu.enable()
    try:
        actual = get_rho(n, gradients, tau)
        np.testing.assert_allclose(actual, ref, atol=1e-12, rtol=1e-12)
        dedrho = rng.normal(size=ref.shape)
        expected = dedrho[:, 0].copy()
        actual = expected.copy()
        add_gradient_correction(gradients, dedrho, expected)
        add_correction(gradients, dedrho, actual)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    finally:
        gpu.disable()
