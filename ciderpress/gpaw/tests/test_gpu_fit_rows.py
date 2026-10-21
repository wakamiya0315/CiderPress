"""Parallel host fits retain the exact independent single-RHS answers."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import cho_factor

from ciderpress.gpu.pasdw_pipeline import _fit_rows, _native_potrs, _solve_cpu, _solve_native


@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('n', [0, 1, 7, 129])
def test_native_single_rhs(lower, n):
    function = _native_potrs()
    if function is None:
        pytest.skip('SciPy Cython LAPACK ABI unavailable; serial fallback')
    rng = np.random.default_rng(171)
    if n:
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        matrix = (q*np.geomspace(1., 1e-9, n)) @ q.T
        factor = cho_factor(matrix, lower=lower)
    else:
        factor = np.empty((0, 0)), lower
    rhs = rng.normal(size=n)
    old_factor, old_rhs = factor[0].copy(), rhs.copy()
    ref = _solve_cpu(factor, rhs)
    actual = _solve_native(function, factor, rhs)
    assert actual.tobytes() == ref.tobytes()
    assert factor[0].tobytes() == old_factor.tobytes()
    assert rhs.tobytes() == old_rhs.tobytes()


def test_native_unavailable_fallback(monkeypatch):
    from ciderpress.gpu import pasdw_pipeline as m
    monkeypatch.setattr(m, '_native_potrs', lambda: None)
    test_fit_rows(7, 2, 3, False)


def test_native_rejects_changed_abi(monkeypatch):
    import ctypes
    from scipy.linalg import cython_lapack
    create = ctypes.pythonapi.PyCapsule_New
    create.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    create.restype = ctypes.py_object
    # This address must never execute: the unsupported ABI is rejected first.
    capsule = create(1, b'void (char *, long *, double *)', None)
    _native_potrs.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setitem(cython_lapack.__pyx_capi__, 'dpotrs', capsule)
            assert _native_potrs() is None
    finally:
        _native_potrs.cache_clear()


def test_failed_fit_joins_workers(monkeypatch):
    from threading import Lock
    import time
    from ciderpress.gpu import pasdw_pipeline as m
    lock = Lock()
    calls = active = 0
    def fail(function, factor, rhs):
        nonlocal calls, active
        with lock:
            current = calls
            calls += 1
            active += 1
        try:
            if current == 0:
                raise ValueError('injected fit failure')
            time.sleep(.02)
            return rhs.copy()
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(m, '_native_potrs', lambda: object())
    monkeypatch.setattr(m, '_solve_native', fail)
    p = SimpleNamespace(lmax=0, p_cl_l_ii=[(np.eye(2), True)],
                        pcol=SimpleNamespace(uloc_l=[0, 1]),
                        jcol=SimpleNamespace(uloc_l=[0, 1]))
    with pytest.raises(ValueError, match='injected fit failure'):
        m._fit_rows(p, np.ones((8, 1, 1)), np.ones((8, 1, 1)), workers=4)
    assert active == 0


@pytest.mark.parametrize('ns,nb,na,empty', [(0, 2, 3, False), (1, 2, 3, False),
                                         (7, 2, 3, False), (7, 2, 3, True)])
def test_fit_rows(ns, nb, na, empty):
    rng = np.random.default_rng(151)
    factors = []
    for n in (3*nb+2*na, 0 if empty else 2*nb+na):
        if n:
            q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            matrix = (q*np.geomspace(1., 1e-9, n)) @ q.T
            factors.append(cho_factor(matrix, lower=True))
        else:
            factors.append((np.empty((0, 0)), True))
    p = SimpleNamespace(lmax=1, p_cl_l_ii=factors,
        pcol=SimpleNamespace(uloc_l=[0, 3, 3 if empty else 9]),
        jcol=SimpleNamespace(uloc_l=[0, 2, 2 if empty else 5]))
    a = rng.normal(size=(ns, p.pcol.uloc_l[-1], nb))
    b = rng.normal(size=(ns, p.jcol.uloc_l[-1], na))
    copies = [x.copy() for x in (a, b, *(v[0] for v in factors))]
    serial = _fit_rows(p, a, b)
    for _ in range(3):
        parallel = _fit_rows(p, a, b, workers=4)
        for x, y in zip(parallel, serial):
            assert x.tobytes() == y.tobytes()
    for x, y in zip((a, b, *(v[0] for v in factors)), copies):
        assert x.tobytes() == y.tobytes()
