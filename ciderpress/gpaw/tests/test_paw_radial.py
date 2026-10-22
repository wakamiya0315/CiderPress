# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise multi-spin channels, empty shells, and quadrature factors."""
import unittest

import numpy as np

from ciderpress.gpaw.paw_radial import RadialTransform


class RadialTransformTests(unittest.TestCase):
    def make_map(self):
        m = RadialTransform.__new__(RadialTransform)
        m.dtype = np.longdouble
        m.nk, m.nlm = 5, 9
        m.loc = np.array([0, 2, 2, 7])  # The l=1 shell is empty.
        m.weights = np.array([0.0, 0.2, 0.5, 0.8, 1.1])
        m.matrices = [
            np.arange(10).reshape(2, 5).astype(np.longdouble) / 7,
            np.empty((0, 5), dtype=np.longdouble),
            np.arange(5).reshape(1, 5).astype(np.longdouble) / 3,
        ]
        return m

    def test_channels_and_empty_shells(self):
        m = self.make_map()
        x = np.random.default_rng(28).normal(size=(2, 5, 9, 3))
        original = x.copy()
        actual = m.forward(x)
        expected = np.zeros((2, 7, 3))
        for l, matrix in enumerate(m.matrices):
            for radial in range(len(matrix)):
                for component in range(2 * l + 1):
                    for spin in range(2):
                        for channel in range(3):
                            expected[
                                spin,
                                m.loc[l] + radial * (2 * l + 1) + component,
                                channel,
                            ] = sum(
                                matrix[radial, k]
                                * m.weights[k]
                                * x[spin, k, l * l + component, channel]
                                for k in range(5)
                            )
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(x, original)

    def test_weighted_adjoint_with_zero_weight(self):
        m = self.make_map()
        rng = np.random.default_rng(19)
        x = rng.normal(size=(2, 5, 9, 3))
        y = rng.normal(size=(2, 7, 3))
        original = y.copy()
        left = np.sum(m.forward(x) * y)
        right = np.sum(x * m.adjoint(y) * m.weights[None, :, None, None])
        np.testing.assert_allclose(left, right, atol=1e-13, rtol=1e-13)
        np.testing.assert_array_equal(y, original)
        np.testing.assert_array_equal(m.adjoint(y)[:, :, 1:4], 0.0)

    def test_zero_feature_channels(self):
        m = self.make_map()
        self.assertEqual(m.forward(np.empty((2, 5, 9, 0))).shape, (2, 7, 0))
        self.assertEqual(m.adjoint(np.empty((2, 7, 0))).shape, (2, 5, 9, 0))

    def test_float64_accumulation(self):
        m = self.make_map()
        x = np.random.default_rng(18).normal(size=(2, 5, 9, 3))
        reference = m.forward(x)
        m.dtype = np.float64
        m.matrices = [a.astype(m.dtype) for a in m.matrices]
        np.testing.assert_allclose(m.forward(x), reference, atol=1e-13, rtol=1e-13)

    def test_setup_pair_and_input_protection(self):
        from ciderpress.gpaw.atom_utils import PSmoothSetupV1

        mapping = self.make_map()
        setup = PSmoothSetupV1.__new__(PSmoothSetupV1)
        setup._grid_nlm = 9
        setup._radial_transforms = {9: mapping}
        rng = np.random.default_rng(35)
        x = rng.normal(size=(2, 5, 9, 3))
        y = rng.normal(size=(2, 7, 3))
        saved_x, saved_y = x.copy(), y.copy()
        np.testing.assert_array_equal(setup.k2orb(x), mapping.forward(x))
        np.testing.assert_array_equal(setup.orb2k(y), mapping.adjoint(y))
        np.testing.assert_array_equal(x, saved_x)
        np.testing.assert_array_equal(y, saved_y)


if __name__ == "__main__":
    unittest.main()
