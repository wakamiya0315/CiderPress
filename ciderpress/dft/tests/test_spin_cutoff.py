"""Representation and spin-exchange invariance of evaluator density gates."""
from types import SimpleNamespace

import numpy as np

from ciderpress.dft.xc_evaluator2 import MappedDFTKernel2


def linear_separate_kernel():
    # Isolate gating from trained coefficients and the baseline library.
    kernel = object.__new__(MappedDFTKernel2)
    kernel.mode = "SEP"
    kernel.feature_list = SimpleNamespace(nfeat=1)

    def evaluate(X, f, df):
        f += X[:, 0]
        df[:, 0] += 1.

    def baseline(f, df, rho, vrho):
        f /= len(rho[0])
        df /= len(rho[0])

    kernel.fevals = [evaluate]
    kernel.get_descriptors = lambda X, **kwargs: X.transpose(0, 2, 1).reshape(-1, 1)
    kernel.apply_libxc_baseline_ = baseline
    kernel.apply_descriptor_grad = lambda X, df, **kwargs: df.transpose(0, 2, 1)
    return kernel


def evaluate(rho):
    kernel = linear_separate_kernel()
    energy, derivative = kernel(len(rho) * rho[:, None], (rho,), (np.zeros_like(rho),), rhocut=1e-8)
    return energy, len(rho) * derivative[:, 0]


def test_equivalent_unpolarized_representations_at_cutoff():
    total = np.array([.75e-8, 1.5e-8, 3e-8])
    e1, v1 = evaluate(total[None])
    e2, v2 = evaluate(np.tile(total / 2, (2, 1)))
    np.testing.assert_allclose(e1, [0., 1.5e-8, 3e-8], atol=0, rtol=1e-15)
    np.testing.assert_array_equal(e2, e1)
    np.testing.assert_array_equal(v2, np.tile(v1, (2, 1)))


def test_one_inactive_spin_and_spin_exchange():
    rho = np.array([[.1e-8, 2e-8], [1.5e-8, .1e-8]])
    energy, derivative = evaluate(rho)
    swapped_energy, swapped_derivative = evaluate(rho[::-1].copy())
    np.testing.assert_array_equal(energy, [1.5e-8, 2e-8])
    np.testing.assert_array_equal(derivative, [[0., 1.], [1., 0.]])
    np.testing.assert_array_equal(energy, swapped_energy)
    np.testing.assert_array_equal(derivative, swapped_derivative[::-1])
