"""Owned molecular tables for the 9282196 version-j prototype.

Setup only: native Gaussian integrals/solves remain on the CPU. The private
ctypes ABI is deliberately pinned to the exact header; never retain C views.
This is not a replacement for a future native snapshot accessor.
"""
import ctypes as ct
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

ABI_COMMIT = "9282196500a5e2635d4bf88555765cef55bc36fb"
HEADER_SHA256 = "05d752a93e20ef595faebecaf8c5e40d551a735d51d9b4d64e4d5ba734022a80"
_I = ct.POINTER(ct.c_int)
_D = ct.POINTER(ct.c_double)


class _Collection(ct.Structure):
    _fields_ = [(n, ct.c_int) for n in
                ("natm", "tot_shl_pairs", "nalpha", "nbeta", "nfeat_i", "has_vj")]
    _fields_ += [("icontrib_ids", _I), ("feat_orders", _I), ("max_pair", ct.c_int),
                 ("atco_inp", ct.c_void_p), ("atco_out", ct.c_void_p),
                 ("pair_loc", ct.POINTER(_I)), ("alphas", _D),
                 ("alpha_norms", _D), ("ovlp_mats", _D)]


def _owned(pointer, shape):
    if not pointer:
        raise RuntimeError("NULL pointer in pinned ATC ABI")
    return np.array(np.ctypeslib.as_array(pointer, shape=shape), copy=True, order="C")


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
    header = Path(__file__).parents[1] / "lib/mod_cider/convolutions.h"
    if sha256(header.read_bytes()).hexdigest() != HEADER_SHA256:
        raise RuntimeError(f"Unknown convolution_collection ABI; expected {ABI_COMMIT}")
    if ct.sizeof(ct.c_void_p) != 8 or ct.sizeof(_Collection) != 96:
        raise RuntimeError("Pinned ATC ABI requires a 64-bit layout of size 96")
    if not ccl._integrals_initialized or ccl.is_vk:
        raise ValueError("Need initialized version-j convolution tables")
    c = ct.cast(ccl._ccl, ct.POINTER(_Collection)).contents
    if (c.natm, c.nalpha, c.nbeta, c.nfeat_i, c.has_vj) != (
            ccl.atco_inp.natm, ccl.nalpha, ccl.nbeta, 0, 1):
        raise RuntimeError("ATC ABI dimension/feature mismatch")
    if c.atco_inp != ct.cast(ccl.atco_inp._atco, ct.c_void_p).value or c.atco_out != ct.cast(ccl.atco_out._atco, ct.c_void_p).value:
        raise RuntimeError("ATC ABI basis pointer mismatch")
    for ptr, ref in ((c.alphas, ccl._alphas), (c.alpha_norms, ccl._alpha_norms)):
        if not np.array_equal(_owned(ptr, ref.shape), ref):
            raise RuntimeError("ATC ABI alpha mismatch")
    ing, outg = basis_groups(ccl.atco_inp), basis_groups(ccl.atco_out)
    if [(g[0], g[1]) for g in ing] != [(g[0], g[1]) for g in outg]:
        raise ValueError("Input/output atom/l blocks must match")
    pairs = sum((i[3]-i[2])*(o[3]-o[2]) for i, o in zip(ing, outg))
    if pairs != c.tot_shl_pairs or pairs <= 0:
        raise RuntimeError("ATC ABI shell-pair mismatch")
    blocks, expected = [], 0
    for i, o in zip(ing, outg):
        a, l = i[:2]
        n, m = i[3]-i[2], o[3]-o[2]
        if c.pair_loc[a][l] != expected:
            raise RuntimeError("ATC ABI pair offset mismatch")
        count = n*m*c.nalpha*c.nbeta
        ptr = ct.cast(ct.addressof(c.ovlp_mats.contents) + expected*c.nalpha*c.nbeta*8, _D)
        raw = _owned(ptr, (count,)).reshape(m, n, c.nbeta, c.nalpha)
        matrix = np.ascontiguousarray(raw.transpose(0, 2, 1, 3).reshape(m*c.nbeta, n*c.nalpha))
        if not np.isfinite(matrix).all():
            raise RuntimeError("Non-finite solved ATC table")
        matrix.flags.writeable = False
        blocks.append(ATCBlock(a, l, slice(i[4], i[5]), slice(o[4], o[5]), matrix))
        expected += n*m
    rng, checks = np.random.default_rng(9282196), {}
    for forward in (True, False):
        src = ccl.atco_inp if forward else ccl.atco_out
        dst = ccl.atco_out if forward else ccl.atco_inp
        x = rng.normal(size=(src.nao, c.nalpha))
        ref = ccl.multiply_atc_integrals(x, fwd=forward)
        value = apply_atc_numpy(blocks, x, dst.nao, c.nalpha, forward)
        scale = max(1., float(np.max(np.abs(ref))))
        err = np.abs(value-ref)
        passed = bool(np.all(err <= 1e-12*scale + 1e-12*np.abs(ref)))
        checks["forward" if forward else "adjoint"] = dict(max_abs=float(err.max()), scale=scale, passed=passed)
        if not passed:
            raise RuntimeError(f"Pinned ATC native self-check failed: {checks}")
    return tuple(blocks), dict(commit=ABI_COMMIT, header_sha256=HEADER_SHA256,
                               struct_size=ct.sizeof(_Collection), shell_pairs=pairs, checks=checks)


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
