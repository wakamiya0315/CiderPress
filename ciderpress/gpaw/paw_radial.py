# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable radial projection with a paired adjoint."""
import ctypes

import numpy as np
from gpaw.sphere.lebedev import Y_nL

from ciderpress.dft.grids_indexer import AtomicGridsIndexer
from ciderpress.dft.plans import libcider


class RadialTransform:
    """A fixed radial map and its quadrature-weighted adjoint.

    If R is the radial-to-orbital projection, S the overlap matrix and W
    the radial quadrature, the forward map is (S**-1 R) W.  Construct
    S**-1 R once on unit vectors and use its transpose for the reverse
    map.  Repeated overlap solves on nearly equal inputs otherwise add
    input-dependent cancellation to a mathematically linear operation.

    Each angular component uses the same radial block.  Accumulation uses
    the widest NumPy real type; returned arrays retain the native float64
    interface.  The map does not depend on an SCF reference state.
    """

    def __init__(self, collection, radial_grid, nlm, weights):
        dtype = np.longdouble
        self.dtype = dtype
        self.loc = np.array(collection.vloc_l, dtype=int, copy=True)
        self.nk = len(radial_grid.k_g)
        self.nlm = nlm
        self.weights = np.array(weights, dtype=float, copy=True)
        self.matrices = [
            np.zeros(((b - a) // (2 * l + 1), self.nk))
            for l, (a, b) in enumerate(zip(self.loc[:-1], self.loc[1:]))
        ]
        if (len(self.loc) - 1) ** 2 > nlm:
            raise ValueError("Radial map exceeds the available harmonics")
        basis = collection.atco_recip
        grid = AtomicGridsIndexer.make_single_atom_indexer(
            Y_nL[:, :nlm], radial_grid.k_g
        )
        for begin in range(0, self.nk, 16):
            end = min(self.nk, begin + 16)
            n = end - begin
            values = np.zeros((self.nk, nlm, n))
            for l in range(len(self.matrices)):
                values[np.arange(begin, end), l * l, np.arange(n)] = 1.0
            coefficients = np.zeros((basis.nao, n))
            basis.convert_rad2orb_(
                values, coefficients, grid, grid.rad_arr, rad2orb=True, offset=0
            )
            libcider.solve_atc_coefs_arr(
                collection.atco.atco_c_ptr,
                coefficients.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(n),
            )
            for l, matrix in enumerate(self.matrices):
                a, b = self.loc[l : l + 2]
                block = coefficients[a:b].reshape(-1, 2 * l + 1, n)
                if block.shape[1] > 1 and np.any(block[:, 1:]):
                    raise RuntimeError("Radial transform mixed harmonic components")
                matrix[:, begin:end] = block[:, 0]
        self.matrices = tuple(x.astype(self.dtype) for x in self.matrices)
        for array in (*self.matrices, self.loc, self.weights):
            array.flags.writeable = False

    def forward(self, values):
        if values.ndim != 4 or np.iscomplexobj(values):
            raise ValueError("Expected real spin/radial/angular/feature values")
        spins, nk, nlm, channels = values.shape
        if nk != self.nk or nlm != self.nlm:
            raise ValueError("Radial grid shape changed")
        output = np.zeros((spins, self.loc[-1], channels))
        if channels == 0:
            return output
        for l, matrix in enumerate(self.matrices):
            if not len(matrix):
                continue
            nm = 2 * l + 1
            block = np.asarray(values[:, :, l * l : (l + 1) ** 2], dtype=self.dtype)
            block = block * self.weights[None, :, None, None]
            projected = matrix @ block.transpose(1, 0, 2, 3).reshape(nk, -1)
            output[:, self.loc[l] : self.loc[l + 1]] = (
                projected.reshape(len(matrix), spins, nm, channels)
                .transpose(1, 0, 2, 3)
                .reshape(spins, len(matrix) * (2 * l + 1), channels)
            )
        return output

    def adjoint(self, coefficients):
        if coefficients.ndim != 3 or np.iscomplexobj(coefficients):
            raise ValueError("Expected real spin/orbital/feature coefficients")
        spins, orbitals, channels = coefficients.shape
        if orbitals != self.loc[-1]:
            raise ValueError("Orbital coefficient count changed")
        output = np.zeros((spins, self.nk, self.nlm, channels))
        if channels == 0:
            return output
        for l, matrix in enumerate(self.matrices):
            if not len(matrix):
                continue
            nm = 2 * l + 1
            block = np.asarray(
                coefficients[:, self.loc[l] : self.loc[l + 1]], dtype=self.dtype
            ).reshape(spins, len(matrix), nm, channels)
            projected = matrix.T @ block.transpose(1, 0, 2, 3).reshape(len(matrix), -1)
            output[:, :, l * l : (l + 1) ** 2] = projected.reshape(
                self.nk, spins, nm, channels
            ).transpose(1, 0, 2, 3)
        return output
