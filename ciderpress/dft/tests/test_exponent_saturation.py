"""Accuracy of exponent saturation near zero and its analytic derivative."""
import ctypes
from decimal import Decimal, localcontext

import numpy as np
import pytest

from ciderpress.dft.plans import libcider


def saturate(values, cutoff, derivatives=None):
    values = np.array(values, dtype=float)
    derivatives = (np.ones((1, len(values))) if derivatives is None
                   else np.array(derivatives, dtype=float))
    pointers = (ctypes.c_void_p * len(derivatives))(
        *[row.ctypes.data for row in derivatives])
    libcider.smooth_cider_exponents(
        values.ctypes.data_as(ctypes.c_void_p), pointers,
        ctypes.c_double(cutoff), ctypes.c_int(values.size),
        ctypes.c_int(len(derivatives)))
    return values, derivatives


def reference(value, cutoff):
    with localcontext() as context:
        context.prec = 80
        x = Decimal.from_float(float(value)) / Decimal.from_float(float(cutoff))
        polynomial = sum(x**k / k for k in range(1, 13))
        exponential = (-polynomial).exp()
        return (float(Decimal.from_float(float(cutoff)) * (1 - exponential)),
                float(exponential * (1 + sum(x**k for k in range(1, 12)))))


@pytest.mark.parametrize('cutoff', [1., 1.2e6])
def test_saturation_high_precision_reference(cutoff):
    ratios = np.array([-1e-12, -1e-20, 0., 1e-20, 1e-16, 1e-12,
                       1e-8, 1e-3, .1, .9, 1., 1.1])
    values = ratios * cutoff
    expected = np.array([reference(value, cutoff) for value in values])
    actual, derivative = saturate(values, cutoff)
    np.testing.assert_allclose(actual, expected[:, 0], atol=0., rtol=5e-14)
    np.testing.assert_allclose(derivative[0], expected[:, 1], atol=0., rtol=5e-14)


def test_small_exponent_derivative_and_chain_rule():
    value, cutoff = 1e-4, 1.2e6
    slopes = np.array([[1.], [-3.], [.25]])
    _, derivative = saturate([value], cutoff, slopes)
    for step in [1e-7, 5e-8, 2.5e-8]:
        plus, _ = saturate([value + step], cutoff)
        minus, _ = saturate([value - step], cutoff)
        numeric = (plus - minus) / (2 * step)
        np.testing.assert_allclose(derivative, slopes * numeric,
                                   atol=2e-12, rtol=2e-12)
