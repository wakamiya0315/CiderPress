"""Representation and spin-exchange invariance of evaluator density gates."""
from types import SimpleNamespace

import numpy as np
import pytest

from ciderpress.dft.xc_evaluator2 import MappedDFTKernel2
from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.dft.settings import NLDFSettingsVJ


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


@pytest.mark.parametrize('nspin', [1, 2])
def test_plan_copy_preserves_density_gate(nspin):
    settings = NLDFSettingsVJ('MGGA', [1., 0., .03125], 'one',
                              ['se'], [[1., 0., .03125]])
    plan = NLDFSplinePlan(settings, nspin, .01, 1.8, 12, rhocut=1e-8)
    copied = plan.new()
    twice = copied.new()
    assert copied.rhocut == twice.rhocut == plan.rhocut
    rho = np.array([.25e-8, .75e-8, 1.5e-8]) / nspin
    inputs = (rho, np.zeros_like(rho), rho ** (5 / 3))
    function, derivatives = plan.get_function_to_convolve(inputs)
    for candidate in (copied, twice):
        actual, gradient = candidate.get_function_to_convolve(inputs)
        np.testing.assert_array_equal(actual, function)
        np.testing.assert_array_equal(gradient, derivatives)
    switched = plan.new(nspin=3-nspin)
    assert switched.rhocut * switched.nspin == plan.rhocut * plan.nspin
