"""Compact interpolation remains differentiable when a point crosses its sphere."""
import numpy as np
import pytest
from gpaw.xc.vdw import spline

from ciderpress.dft import pwutil
from ciderpress.gpaw.atom_utils import _terminate_spline_at_cutoff


def evaluate(coefficients, coordinates, derivative=False):
    coordinates = np.asarray(coordinates)
    indices = np.minimum(np.floor(coordinates).astype(np.int32),
                         coefficients.shape[1] - 1)
    fractions = np.ascontiguousarray(coordinates - indices)
    kernel = (pwutil.eval_cubic_spline_deriv if derivative
              else pwutil.eval_cubic_spline)
    return kernel(coefficients, np.ascontiguousarray(indices), fractions)


@pytest.mark.parametrize('cutoff', [2., 2.2, 2.9, 5., np.nextafter(5., np.inf)])
def test_compact_boundary_and_native_derivative(cutoff):
    x = np.arange(7, dtype=float)
    values = np.maximum(1 - (x / cutoff)**2, 0.)**2
    coefficients = np.stack([spline(x, values), spline(x, 2 * values)])
    original = coefficients.copy()
    _terminate_spline_at_cutoff(coefficients, cutoff)

    # Use the compiled interpolation and derivative kernels independently
    # of the construction of the terminal polynomial.
    left_knot = np.ceil(cutoff - 1e-12) - 1
    prefix = np.linspace(.01, left_knot, 11)
    np.testing.assert_array_equal(evaluate(coefficients, prefix),
                                  evaluate(original, prefix))
    at_cut = np.array([np.nextafter(cutoff, 0.), cutoff])
    np.testing.assert_allclose(evaluate(coefficients, at_cut), 0., atol=2e-14)
    np.testing.assert_allclose(evaluate(coefficients, at_cut, True), 0., atol=2e-13)
    middle = .5 * (left_knot + cutoff)
    delta = 1e-6
    difference = (evaluate(coefficients, [middle + delta])
                  - evaluate(coefficients, [middle - delta])) / (2 * delta)
    np.testing.assert_allclose(evaluate(coefficients, [middle], True),
                               difference, atol=2e-9, rtol=2e-8)

    # The production sphere includes the point only on its inner side.
    # A disappearing point must carry neither a value jump nor a finite
    # limiting force.  Halving its distance shows the quadratic vanishing.
    inner = evaluate(coefficients, cutoff - np.array([1e-4, 5e-5, 2.5e-5]))
    np.testing.assert_allclose(inner[:, 1] / inner[:, 0], .25, rtol=.01)
    np.testing.assert_allclose(inner[:, 2] / inner[:, 1], .25, rtol=.01)


@pytest.mark.parametrize('cutoff', [0., -1., 7.])
def test_invalid_cutoff_is_rejected(cutoff):
    with pytest.raises(ValueError, match='cutoff'):
        _terminate_spline_at_cutoff(np.zeros((2, 7, 4)), cutoff)
