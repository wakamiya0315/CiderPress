"""Native snapshot boundary, ownership, and mandatory validation without CUDA."""
import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pytest

from ciderpress.dft.tests.test_gpu_lcao_tables import native_generator


def test_snapshot_capacity_and_ownership(native_generator):
    from ciderpress.gpu.lcao_tables import _accessor, _snapshot, extract_atc
    ccl = native_generator.ccl
    native_export = _accessor()
    def export(pointer, *args):
        return native_export(pointer, ccl.atco_inp._atco, ccl.atco_out._atco, *args)
    meta = np.full(9, 123, dtype=np.uintp)
    ptr = lambda a: ct.c_void_p(a.ctypes.data)
    assert export(None, ptr(meta), 9, None, 0, None, 0) == 1
    assert np.all(meta == 123)
    assert native_export(ccl._ccl, ccl.atco_out._atco, ccl.atco_inp._atco,
                         ptr(meta), 9, None, 0, None, 0) == 3
    assert np.all(meta == 123)
    assert export(ccl._ccl, ptr(meta), 8, None, 0, None, 0) == 2
    assert np.all(meta == 123)
    assert export(ccl._ccl, ptr(meta), 9, None, 0, None, 0) == 0
    rows = np.full((int(meta[5]), 11), -7, dtype=np.intc)
    data = np.full(int(meta[6]), -7., dtype=np.float64)
    for nr, nd in ((rows.size-1, data.size), (rows.size, data.size-1)):
        assert export(ccl._ccl, ptr(meta), 9, ptr(rows), nr, ptr(data), nd) == 2
        assert np.all(rows == -7) and np.all(data == -7)
    assert export(ccl._ccl, ptr(meta), 9, ptr(rows), rows.size, None, 0) == 1
    assert export(ccl._ccl, ptr(meta), 9, ptr(rows), rows.size, ptr(data), data.size) == 0
    _, raw, _ = _snapshot(ccl)
    before = raw.tobytes()
    data[:] = 0
    assert _snapshot(ccl)[1].tobytes() == before
    blocks, info = extract_atc(ccl)
    assert info["abi_version"] == 1
    assert all(b.matrix.flags.owndata and not b.matrix.flags.writeable for b in blocks)
    saved = [b.matrix.tobytes() for b in blocks]
    ccl.solve_projection_coefficients()  # Mutates the native table, not the copies.
    assert [b.matrix.tobytes() for b in blocks] == saved


def test_missing_accessor_requires_rebuild(native_generator, monkeypatch):
    from ciderpress.dft import lcao_convolutions
    from ciderpress.gpu.lcao_tables import extract_atc
    monkeypatch.setattr(lcao_convolutions, "libcider", SimpleNamespace())
    with pytest.raises(RuntimeError, match="libmcider lacks.*rebuild CiderPress"):
        extract_atc(native_generator.ccl)
    # Keep the native destructor alive after monkeypatch teardown.
    monkeypatch.undo()


def test_self_check_runs_on_every_snapshot(native_generator, monkeypatch):
    from ciderpress.gpu.lcao_tables import extract_atc
    ccl = native_generator.ccl
    multiply = ccl.multiply_atc_integrals
    calls = []
    def tracked(x, fwd=True):
        calls.append(fwd)
        return multiply(x, fwd=fwd)
    monkeypatch.setattr(ccl, "multiply_atc_integrals", tracked)
    extract_atc(ccl)
    extract_atc(ccl)
    assert calls == [True, False, True, False]
    monkeypatch.setattr(ccl, "multiply_atc_integrals", lambda x, fwd=True: multiply(x, fwd=fwd)+1e4)
    with pytest.raises(RuntimeError, match="self-check failed"):
        extract_atc(ccl)


def test_uninitialized_snapshot_rejected(native_generator):
    from ciderpress.gpu.lcao_tables import extract_atc
    native_generator.ccl._integrals_initialized = False
    with pytest.raises(ValueError, match="initialized version-j"):
        extract_atc(native_generator.ccl)
