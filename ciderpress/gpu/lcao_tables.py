"""Owned molecular tables copied through libmcider's versioned snapshot API.

Setup only: native Gaussian integrals/solves remain on the CPU. No private C
structure layout is mirrored here; all retained arrays own their storage.
"""
import ctypes as ct
from dataclasses import dataclass
from hashlib import sha256

import numpy as np


def _accessor():
    from ciderpress.dft.lcao_convolutions import libcider
    try:
        export = libcider.cider_export_atc_v1
    except AttributeError as exc:
        raise RuntimeError(
            "libmcider lacks cider_export_atc_v1; rebuild CiderPress native "
            "libraries from this checkout (python setup.py build_py, or a "
            "clean CMake build of ciderpress/lib) before GPU LCAO setup"
        ) from exc
    export.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                       ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t)
    export.restype = ct.c_int
    return export


def _snapshot(ccl):
    export = _accessor()
    meta = np.empty(9, dtype=np.uintp)
    def call(rows=None, data=None):
        rc = export(ccl._ccl, ccl.atco_inp._atco, ccl.atco_out._atco,
                    ct.c_void_p(meta.ctypes.data), meta.size,
                    None if rows is None else ct.c_void_p(rows.ctypes.data),
                    0 if rows is None else rows.size,
                    None if data is None else ct.c_void_p(data.ctypes.data),
                    0 if data is None else data.size)
        if rc:
            raise RuntimeError(f"Native ATC snapshot failed (error {rc})")
    call()
    version, natm, na, nb, pairs, count, size, ni, no = map(int, meta)
    if (version, natm, na, nb, ni, no) != (
            1, ccl.atco_inp.natm, ccl.nalpha, ccl.nbeta,
            ccl.atco_inp.nao, ccl.atco_out.nao):
        raise RuntimeError("Native ATC snapshot dimension/version mismatch")
    if pairs <= 0 or count <= 0 or size != pairs*na*nb + 2*na:
        raise RuntimeError("Native ATC snapshot size mismatch")
    expected = meta.copy()
    rows = np.empty((count, 11), dtype=np.intc)
    data = np.empty(size, dtype=np.float64)
    call(rows, data)
    if not np.array_equal(meta, expected):
        raise RuntimeError("Native ATC snapshot changed during copy")
    for value, ref in ((data[:na], ccl._alphas), (data[na:2*na], ccl._alpha_norms)):
        if not np.array_equal(value, ref):
            raise RuntimeError("Native ATC snapshot alpha mismatch")
    return rows, data[2*na:], pairs


def basis_groups(atco):
    """(atom,l,shell_start,shell_stop,orbital_start,orbital_stop)."""
    bas, loc = atco.bas, atco.ao_loc
    result = []
    for a in range(atco.natm):
        for l in range(int(bas[bas[:, 0] == a, 1].max()) + 1):
            ids = np.flatnonzero((bas[:, 0] == a) & (bas[:, 1] == l))
            if not len(ids):
                continue
            lo, hi = int(ids[0]), int(ids[-1] + 1)
            if not np.array_equal(ids, np.arange(lo, hi)):
                raise ValueError("ATC shells must be contiguous in atom/l order")
            result.append((a, l, lo, hi, int(loc[lo]), int(loc[hi])))
    return result


@dataclass(frozen=True)
class ATCBlock:
    atom: int
    l: int
    input_slice: slice
    output_slice: slice
    matrix: np.ndarray  # (out_shell * beta, in_shell * alpha)


def apply_atc_numpy(blocks, x, nao, q, forward=True):
    """Independent packed-table oracle used by the mandatory native self-check."""
    out = np.zeros((nao, q))
    for b in blocks:
        ins, outs = (b.input_slice, b.output_slice) if forward else (b.output_slice, b.input_slice)
        nm = 2 * b.l + 1
        packed = x[ins].reshape(-1, nm, q).transpose(0, 2, 1).reshape(-1, nm)
        value = (b.matrix if forward else b.matrix.T) @ packed
        out[outs] = value.reshape(-1, q, nm).transpose(0, 2, 1).reshape(-1, q)
    return out


def extract_atc(ccl):
    """Read solved tables, validate ABI/dimensions, and check both native actions.

    The caller must already have called solve_projection_coefficients. There is
    no native 'solved' flag: the self-check proves the copied table implements
    the live operator; setup provenance must establish that the solve was run.
    """
    if not ccl._integrals_initialized or ccl.is_vk or ccl.nalpha != ccl.nbeta:
        raise ValueError("Need initialized version-j convolution tables")
    rows, data, pairs = _snapshot(ccl)
    ing, outg = basis_groups(ccl.atco_inp), basis_groups(ccl.atco_out)
    if [(g[0], g[1]) for g in ing] != [(g[0], g[1]) for g in outg]:
        raise ValueError("Input/output atom/l blocks must match")
    if len(rows) != len(ing):
        raise RuntimeError("Native ATC snapshot block count mismatch")
    blocks, expected = [], 0
    for row, i, o in zip(rows, ing, outg):
        a, l = i[:2]
        n, m = i[3]-i[2], o[3]-o[2]
        if tuple(row) != (*i, *o[2:], expected):
            raise RuntimeError("Native ATC snapshot block index mismatch")
        count = n*m*ccl.nalpha*ccl.nbeta
        offset = expected*ccl.nalpha*ccl.nbeta
        raw = data[offset:offset+count].reshape(m, n, ccl.nbeta, ccl.nalpha)
        matrix = np.array(raw.transpose(0, 2, 1, 3).reshape(m*ccl.nbeta, n*ccl.nalpha),
                          copy=True, order="C")
        if not np.isfinite(matrix).all():
            raise RuntimeError("Non-finite solved ATC table")
        matrix.flags.writeable = False
        blocks.append(ATCBlock(a, l, slice(i[4], i[5]), slice(o[4], o[5]), matrix))
        expected += n*m
    if expected != pairs:
        raise RuntimeError("Native ATC snapshot shell-pair mismatch")
    rng, checks = np.random.default_rng(9282196), {}
    for forward in (True, False):
        src = ccl.atco_inp if forward else ccl.atco_out
        dst = ccl.atco_out if forward else ccl.atco_inp
        x = rng.normal(size=(src.nao, ccl.nalpha))
        ref = ccl.multiply_atc_integrals(x, fwd=forward)
        value = apply_atc_numpy(blocks, x, dst.nao, ccl.nalpha, forward)
        scale = max(1., float(np.max(np.abs(ref))))
        err = np.abs(value-ref)
        passed = bool(np.all(err <= 1e-12*scale + 1e-12*np.abs(ref)))
        checks["forward" if forward else "adjoint"] = dict(max_abs=float(err.max()), scale=scale, passed=passed)
        if not passed:
            raise RuntimeError(f"ATC native self-check failed: {checks}")
    return tuple(blocks), dict(accessor="cider_export_atc_v1", abi_version=1,
                               shell_pairs=pairs, checks=checks)


def radial_tables(atco, indexer):
    """Tabulate the native libm radial expression, then retain only local blocks."""
    from ciderpress.dft.lcao_convolutions import libcider
    table = np.empty((atco.nbas, indexer.nrad))
    rc = libcider.tabulate_pasdw_radial(atco._atco,
        indexer.rad_arr.ctypes.data_as(ct.c_void_p), ct.c_int(indexer.nrad),
        table.ctypes.data_as(ct.c_void_p))
    if rc != 0:
        raise RuntimeError("Native radial tabulation failed")
    return tuple(np.ascontiguousarray(table[lo:hi, indexer.ra_loc[a]:indexer.ra_loc[a+1]])
                 for a, l, lo, hi, u, v in basis_groups(atco))


def geometry_digest(generator):
    """Content identity for an immutable setup snapshot; recompute only at reset."""
    g, it, p = generator.grids_indexer, generator.interpolator, generator.plan
    h = sha256()
    for x in (it.atom_coords, it.all_coords, g.all_weights, g.idx_map, g.iatom_list,
              g.rad_arr, g.rad_loc, g.ylm, g.ylm_loc, p.alphas,
              generator.ccl.atco_inp.bas, generator.ccl.atco_inp.env,
              generator.ccl.atco_out.bas, generator.ccl.atco_out.env):
        x = np.ascontiguousarray(x)
        h.update(str((x.shape, x.dtype.str)).encode()); h.update(x.tobytes())
    h.update(repr((p.nspin, p.rhocut, p.nldf_settings.__dict__, it.nrad,
                   it.aparam, it.dparam, g.padding)).encode())
    return h.hexdigest()
