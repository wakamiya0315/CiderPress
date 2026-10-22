"""Real CUDA checks for the shared fixed radial map and both entry points."""
import numpy as np
import pytest

from ciderpress import gpu
from ciderpress.gpaw.atom_utils import PSmoothSetupV1
from ciderpress.gpaw.tests import test_paw_radial


@pytest.fixture
def device():
    cp = gpu._real_cupy()
    if cp is None:
        pytest.skip('Real CUDA device required')
    with gpu.policy(cp.cuda.Device().id):
        yield cp
    assert not gpu.is_enabled()


@pytest.mark.parametrize('channels', [0, 1, 3])
def test_fixed_pair(device, channels):
    from ciderpress.gpu.pasdw import fixed_radial
    from ciderpress.gpu.pasdw_pipeline import k2orb, orb2k

    cp = device
    mapping = test_paw_radial.RadialTransformTests().make_map()
    setup = PSmoothSetupV1.__new__(PSmoothSetupV1)
    setup._grid_nlm = mapping.nlm
    setup._radial_transforms = {mapping.nlm: mapping}
    rng = np.random.default_rng(713)
    x = rng.normal(size=(2, mapping.nk, mapping.nlm, channels))
    y = rng.normal(size=(2, mapping.loc[-1], channels))
    dx, dy = cp.asarray(x), cp.asarray(y)
    for forward, reverse in ((setup.k2orb, setup.orb2k),
                             (lambda x: k2orb(setup, x), lambda x: orb2k(setup, x))):
        fx, ay = forward(dx), reverse(dy)
        assert isinstance(fx, cp.ndarray) and isinstance(ay, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(fx), mapping.forward(x), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(cp.asnumpy(ay), mapping.adjoint(y), rtol=1e-12, atol=1e-13)
        left = cp.sum(fx * dy).item()
        right = cp.sum(dx * ay * cp.asarray(mapping.weights)[None, :, None, None]).item()
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(cp.asnumpy(dx), x)
    np.testing.assert_array_equal(cp.asnumpy(dy), y)
    for to_orb, value in ((True, x), (False, y)):
        out = fixed_radial(mapping, value, to_orb)
        assert isinstance(out, np.ndarray) and out.dtype == np.float64
    # The zero-weight sample remains available to the unweighted reverse map.
    assert not channels or np.any(mapping.adjoint(y)[:, 0])


def test_invalid_shapes(device):
    from ciderpress.gpu.pasdw import fixed_radial

    mapping = test_paw_radial.RadialTransformTests().make_map()
    for values, direction in ((np.zeros((2, 4, 9, 3)), True),
                              (np.zeros((2, 6, 3)), False),
                              (np.zeros((2, 7, 3), complex), False)):
        with pytest.raises(ValueError):
            fixed_radial(mapping, values, direction)


def test_compensated_products_and_cancellation(device):
    """FP64 GEMM/Kahan alone loses the product residual before summation."""
    from ciderpress.gpu.pasdw import fixed_radial

    cp = device
    mapping = test_paw_radial.RadialTransformTests().make_map()
    mapping.nk, mapping.nlm = 4, 1
    mapping.loc = np.array([0, 4])
    mapping.weights = np.array([1. + 2.**-27, 1., 1., 0.])
    mapping.matrices = (np.array([
        [1. - 2.**-27, -1., 1., 0.],
        [1., 1., 1., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 0.]], dtype=np.longdouble),)
    x = np.array([1., 1., 0., 1.]).reshape(1, 4, 1, 1)
    y = np.array([1e16, 1., -1e16, 0.]).reshape(1, 4, 1)
    for value, to_orb, reference in (
            (x, True, mapping.forward(x)), (y, False, mapping.adjoint(y))):
        got = fixed_radial(mapping, cp.asarray(value), to_orb)
        np.testing.assert_array_equal(cp.asnumpy(got), reference)
        np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(
            fixed_radial(mapping, cp.asarray(value), to_orb)))


def test_strided_inputs_and_empty_spins(device):
    from ciderpress.gpu.pasdw import fixed_radial

    cp = device
    mapping = test_paw_radial.RadialTransformTests().make_map()
    rng = np.random.default_rng(518)
    x = cp.asarray(rng.normal(size=(2, 5, 9, 6)))[..., ::2]
    y = cp.asarray(rng.normal(size=(2, 7, 6)))[..., ::2]
    for value, to_orb, method in ((x, True, mapping.forward),
                                   (y, False, mapping.adjoint)):
        np.testing.assert_allclose(cp.asnumpy(fixed_radial(mapping, value, to_orb)),
                                   method(cp.asnumpy(value)), rtol=1e-12, atol=1e-13)
        empty = value[:0]
        assert fixed_radial(mapping, empty, to_orb).shape == method(cp.asnumpy(empty)).shape


def test_subnormal_radial_tails(device):
    """Product residuals must survive until the final subnormal output."""
    from ciderpress.gpu.pasdw import fixed_radial

    cp = device
    mapping = test_paw_radial.RadialTransformTests().make_map()
    tiny = np.nextafter(0., 1.)
    mapping.nk, mapping.nlm = 4, 1
    mapping.loc = np.array([0, 4])
    mapping.weights = np.ones(4)
    mapping.matrices = (np.full((4, 4), tiny, dtype=np.longdouble),)
    x = np.full((1, 4, 1, 1), .5)
    y = x.reshape(1, 4, 1)
    for value, fwd, expected in ((x, True, mapping.forward(x)),
                                  (y, False, mapping.adjoint(y))):
        assert np.all(expected == 2*tiny)
        np.testing.assert_array_equal(cp.asnumpy(fixed_radial(mapping, cp.asarray(value), fwd)), expected)
