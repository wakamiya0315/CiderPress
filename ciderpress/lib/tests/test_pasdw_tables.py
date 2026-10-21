"""In-tree C table ABI and invalid-dimension guards (no large allocations)."""
import ctypes
import os

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('CIDER_TEST_REBUILT_TABLES'),
                    reason='requires the p4-fixes in-tree native rebuild')
def test_pasdw_table_dimension_guards():
    from ciderpress.dft.plans import libcider
    null = ctypes.c_void_p()
    # Invalid sizes must return before dereferencing even null input pointers.
    assert libcider.tabulate_pasdw_radial(null, null, ctypes.c_int(-1), null) == -1
    for nk, nq in ((-1, 2), (2, -1), (2**31-1, 2**31-1)):
        assert libcider.tabulate_pasdw_convolution(null, null, null,
                    ctypes.c_int(nk), ctypes.c_int(nq), null) == -1
    k, alpha, norms = np.array([0., 1.]), np.array([1., 2.]), np.array([.5, .7])
    out = np.empty((2, 2, 2))
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    assert libcider.tabulate_pasdw_convolution(ptr(k), ptr(alpha), ptr(norms),
                ctypes.c_int(2), ctypes.c_int(2), ptr(out)) == 0
    expected = np.empty_like(out)
    for q2 in range(2):
        for q1 in range(2):
            exponent = .25 / (alpha[q1]+alpha[q2])
            factor = norms[q1]*norms[q2]
            factor *= (4*np.pi*exponent)**1.5
            expected[:, q2, q1] = factor*np.exp(-k*k*exponent)
    np.testing.assert_allclose(out, expected, atol=1e-15, rtol=1e-15)
