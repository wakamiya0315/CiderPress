"""Hermite sphere termination and discrete B/Q position adjoints."""
from types import SimpleNamespace

import numpy as np
import pytest
from gpaw.xc.vdw import spline

from ciderpress.gpaw.atom_utils import (
    FastAtomPASDWSlice, _terminate_spline_at_cutoff,
)


@pytest.mark.parametrize('cutoff', [1.0, 2.3, 4.0, 4.0 + 1e-15])
def test_termination(cutoff):
    table = np.random.default_rng(17).normal(size=(3, 6, 4))
    original = table.copy()
    _terminate_spline_at_cutoff(table, cutoff)
    end = int(np.ceil(round(cutoff, 13))) - 1
    width = round(cutoff, 13) - end
    c = table[:, end]
    np.testing.assert_array_equal(c[:, :2], original[:, end, :2])
    np.testing.assert_array_equal(table[:, :end], original[:, :end])
    np.testing.assert_allclose(c @ np.array([1, width, width**2, width**3]), 0, atol=2e-14)
    np.testing.assert_allclose(c @ np.array([0, 1, 2*width, 3*width**2]), 0, atol=2e-14)
    assert not table[:, end+1:].any()


@pytest.mark.parametrize('cutoff', [0, -1, 5.1])
def test_invalid_cutoff(cutoff):
    with pytest.raises(ValueError, match='within'):
        _terminate_spline_at_cutoff(np.zeros((6, 4)), cutoff)


def slice_factory():
    # Four angular channels, an off-knot cutoff, and a point exactly on the
    # sphere. Membership is recomputed at every displaced center.
    h, cutoff = .07, 1.43
    r = h*np.arange(32)
    tables = []
    for scale in (1., 1.3):
        values = np.exp(-scale*r*r) * (1 + np.cos(np.pi*np.minimum(r/cutoff, 1)))/2
        table = spline(np.arange(len(r), dtype=float), values)[None]
        _terminate_spline_at_cutoff(table, cutoff/h)
        tables.append(table)
    setup = SimpleNamespace(pfuncs_ntp=tables[0], ffuncs_jtp=tables[1],
        lmax=1, nlist_i=np.zeros(4, np.int32), jlist_i=np.zeros(4, np.int32),
        lmlist_i=np.arange(4, dtype=np.int32), exact_ovlp_pf=np.eye(4))
    xyz = np.stack(np.meshgrid(*[np.arange(-5, 6)*.31]*3, indexing='ij'), axis=-1).reshape(-1, 3)
    xyz += [.017, .031, .053]
    xyz = np.vstack([xyz, [cutoff, 0, 0]])

    def make(center):
        dr = xyz - center
        radius = np.linalg.norm(dr, axis=1)
        mask = radius <= cutoff
        radius, dr = radius[mask], dr[mask]
        g = radius/h
        return FastAtomPASDWSlice(np.flatnonzero(mask), np.floor(g), g % 1,
            radius, dr/radius[:, None], cutoff, setup, .31**3, h, ovlp_fit=True)
    return xyz, make


def test_B_Q_position_derivatives():
    xyz, make = slice_factory()
    center = np.zeros(3)
    sl = make(center)
    p, dp = sl.get_funcs(), sl.get_grads()
    S, dS = sl.sinv_pf, sl.get_ovlp_deriv(p, dp)
    rng = np.random.default_rng(91)
    c = rng.normal(size=(4, 2))
    y = rng.normal(size=(len(xyz), 2))
    # B c = p.T S c, Q y = S.T p y dv. Their negative R
    # derivatives have the Classic get_grads() sign, including dS.
    def maps(s):
        full = np.zeros((len(xyz), 4))
        full[s.indset] = s.get_funcs().T @ s.sinv_pf
        return full @ c, full.T @ y * s.dv
    bgrad = np.zeros((3, len(xyz), 2))
    qgrad = np.empty((3, 4, 2))
    for v in range(3):
        derivative = dp[v].T @ S + p.T @ dS[v]
        bgrad[v, sl.indset] = derivative @ c
        qgrad[v] = derivative.T @ y[sl.indset] * sl.dv
    errors = []
    for step in (2e-4, 1e-4, 5e-5, 2.5e-5):
        error = 0.
        for v in range(3):
            delta = np.eye(3)[v]*step
            plus, minus = maps(make(center+delta)), maps(make(center-delta))
            for actual, a, b in zip((bgrad[v], qgrad[v]), plus, minus):
                error = max(error, float(np.max(abs(actual + (a-b)/(2*step)))))
        errors.append(error)
    # C1 sphere termination permits entry/exit; the one-sided second
    # derivative causes O(h) convergence for the point on the boundary.
    assert errors[-1] < 2e-3, errors
    assert errors[-1] < .35*errors[0], errors
