"""Exact-dot and transpose checks for the paired PASDW density projection."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from ciderpress.gpu import _real_cupy
from ciderpress.gpu.pasdw import density_projection


@pytest.fixture
def cp():
    module = _real_cupy()
    if module is None:
        pytest.skip('requires real CUDA/CuPy')
    return module


def exact_dot(values, table):
    output = np.empty((*values.shape[:-1], table.shape[1]))
    with localcontext() as context:
        context.prec = 80
        for index in np.ndindex(output.shape):
            row, col = index[:-1], index[-1]
            output[index] = float(sum((Decimal.from_float(float(values[(*row, k)]))*
                                       Decimal.from_float(float(table[k, col]))
                                       for k in range(table.shape[0])), Decimal(0)))
    return output


def test_density_projection_retains_cancelled_bits(cp):
    values = np.tile([1e16, 1., -1e16], (2, 2, 1))
    table = np.array([[1., 2.], [1., 3.], [1., 2.]])
    result = cp.asnumpy(density_projection(cp.asarray(values), cp.asarray(table)))
    np.testing.assert_array_equal(result, np.broadcast_to([1., 3.], (2, 2, 2)))


def test_density_projection_and_transpose_match_decimal(cp):
    rng = np.random.default_rng(1701)
    values = rng.normal(size=(2, 2, 11))
    table = rng.normal(size=(11, 13))
    dual = rng.normal(size=(2, 2, 13))
    device_table = cp.asarray(table)
    forward = cp.asnumpy(density_projection(cp.asarray(values), device_table))
    backward = cp.asnumpy(density_projection(cp.asarray(dual), device_table.T))
    np.testing.assert_array_equal(forward, exact_dot(values, table))
    np.testing.assert_array_equal(backward, exact_dot(dual, table.T))
    np.testing.assert_allclose(np.sum(forward*dual), np.sum(values*backward), rtol=2e-15, atol=2e-14)


def test_density_projection_empty_inner_and_batch(cp):
    result = density_projection(cp.empty((2, 0)), cp.empty((0, 3)))
    np.testing.assert_array_equal(cp.asnumpy(result), np.zeros((2, 3)))
    assert density_projection(cp.empty((0, 2, 3)), cp.empty((3, 4))).shape == (0, 2, 4)


def test_density_projection_rejects_lossy_dtype(cp):
    with pytest.raises(ValueError, match='float64'):
        density_projection(cp.ones((2, 3), dtype=cp.float32), cp.ones((3, 4)))
    with pytest.raises(ValueError, match='matching'):
        density_projection(cp.ones((2, 3)), cp.ones((2, 4)))
