"""Private device-resident linear PASDW chains; CPU fits/solves are retained."""
import numpy as np
import os
import ctypes
from concurrent.futures import ThreadPoolExecutor, wait
from functools import lru_cache
from scipy.linalg import cho_solve
from scipy.linalg.lapack import dpotrs

from ciderpress.gpu import cupy, use_gpu
from ciderpress.gpu.pasdw import (_cached, _signature, _table_revision, radial, solve, r2k,
                                  reciprocal_convolution, numeric_basis, coefficients,
                                  fixed_radial)


def _slices(p, l, m):
    nm = 2*l+1
    return (slice(p.pcol.uloc_l[l]+m, p.pcol.uloc_l[l+1]+m, nm),
            slice(p.jcol.uloc_l[l]+m, p.jcol.uloc_l[l+1]+m, nm))


def _solve_cpu(factor_and_lower, rhs):
    # SciPy explicitly handles empty RHSs before calling its LAPACK wrapper.
    # Empty angular blocks occur for H and must retain that behavior.
    if rhs.size == 0:
        return cho_solve(factor_and_lower, rhs, check_finite=False)
    factor, lower = factor_and_lower
    answer, info = dpotrs(factor, rhs, lower=lower, overwrite_b=False)
    if info:
        raise ValueError(f'illegal value in {-info}th argument of internal potrs')
    return answer


@lru_cache(maxsize=1)
def _fit_executor():
    # Four host cores per gpu_h rank. Workers never call CUDA or change the
    # process BLAS/OpenMP configuration; each solve retains one RHS.
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix='cider_fit')


@lru_cache(maxsize=1)
def _native_potrs():
    """SciPy's checked Cython LAPACK ABI, called without Python's GIL.

    Validate the complete capsule signature before constructing a C callable.
    A different SciPy ABI retains the ordinary serial f2py implementation.
    """
    try:
        from scipy.linalg import cython_lapack
        capsule = cython_lapack.__pyx_capi__.get('dpotrs')
    except (ImportError, AttributeError):
        return None
    if capsule is None or ctypes.sizeof(ctypes.c_int) != 4:
        return None
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.argtypes = [ctypes.py_object]
    get_name.restype = ctypes.c_char_p
    name = get_name(capsule)
    alias = b'__pyx_t_5scipy_6linalg_13cython_lapack_d'
    signatures = [b'void (char *, int *, int *, '+d+b' *, int *, '+d+b' *, int *, int *)'
                  for d in (alias, b'double')]
    if name not in signatures:
        return None
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, name)
    if not pointer:
        return None
    pi, pd = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
    function = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), pi, pi,
                               pd, pi, pd, pi, pi)(pointer)
    function._scipy_module = cython_lapack  # Keep the exporting module alive.
    return function


def _solve_native(function, factor_and_lower, rhs):
    factor, lower = factor_and_lower
    if (rhs.size == 0 or rhs.ndim != 1 or rhs.dtype != np.float64
            or factor.dtype != np.float64 or factor.shape != (rhs.size, rhs.size)):
        return _solve_cpu(factor_and_lower, rhs)
    if rhs.size > np.iinfo(np.int32).max:
        raise ValueError('Host fit exceeds the LAPACK integer dimension limit')
    factor = np.asfortranarray(factor)
    answer = np.array(rhs, dtype=np.float64, order='F', copy=True)
    uplo = ctypes.c_char(b'L' if lower else b'U')
    n, nrhs, lda, ldb = (ctypes.c_int(v) for v in (len(rhs), 1, len(rhs), len(rhs)))
    info = ctypes.c_int(0)
    function(ctypes.byref(uplo), ctypes.byref(n), ctypes.byref(nrhs),
             factor.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(lda),
             answer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(ldb),
             ctypes.byref(info))
    if info.value:
        raise ValueError(f'illegal value in {-info.value}th argument of internal potrs')
    return answer


def _fit_rows(p, b1, b2, workers=1):
    """Independent atom/spin rows, preserving every single-RHS solve."""
    ns, _, nb = b1.shape
    na = b2.shape[-1]
    out1, out2 = np.zeros_like(b1), np.zeros_like(b2)
    native = _native_potrs() if workers == 4 and ns >= 4 else None
    for l in range(p.lmax+1):
        pairs = [_slices(p, l, m) for m in range(2*l+1)]
        def row(s):
            for i, j in pairs:
                rhs = np.append(b1[s, i].flatten(), b2[s, j].flatten())
                answer = (_solve_native(native, p.p_cl_l_ii[l], rhs) if native is not None
                          else _solve_cpu(p.p_cl_l_ii[l], rhs))
                n1 = b1[s, i].size
                out1[s, i] = answer[:n1].reshape(-1, nb)
                out2[s, j] = answer[n1:].reshape(-1, na)
        if native is not None:
            # Keep the shared factor hot while independent RHSs run in C.
            # Join running calls even on an exception: no row may outlive
            # this evaluation or observe the next angular block's variables.
            futures = [_fit_executor().submit(row, s) for s in range(ns)]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                wait(futures)
                raise
        else:
            for s in range(ns):
                row(s)
    return out1, out2


def constant(owner, name, x):
    return _cached(owner, ('pipeline', name), _signature(x), lambda cp: cp.asarray(x))


def k2orb(p, x):
    return fixed_radial(p._get_radial_transform(x.shape[2]), x, True)


def orb2k(p, x):
    return fixed_radial(p._get_radial_transform(p._grid_nlm), x, False)


def basis(p, x, fwd):
    cp = cupy()
    ns, nq = len(x), x.shape[-1]
    if fwd:
        out = cp.zeros((ns, p.funcs_ig.shape[1], p.nlm, nq))
        numeric_basis(p, x, out, True)
    else:
        out = cp.zeros((ns, p.nu, nq))
        numeric_basis(p, out, x, False)
    return out


def fitted(p, first, second, adjoint=False):
    """Small coefficient arrays cross the CPU fit boundary once per pair.

    Preserve the original host BLAS products and single-RHS LAPACK calls
    surrounding the sensitive fit. Multiple-RHS solves are not equivalent
    at the accuracy required by these ill-conditioned coefficients.
    """
    cp = cupy()
    if adjoint:
        b1, b2 = cp.asnumpy(basis(p.pcol, second, False)), cp.asnumpy(first)
    else:
        b1 = np.stack([p.pcol.convert(row, fwd=False) for row in cp.asnumpy(k2orb(p, first))])
        b2 = np.stack([p.jcol.convert(row, fwd=False) for row in cp.asnumpy(k2orb(p, second))])
    ns, _, nb = b1.shape
    na = b2.shape[-1]
    # Retain finite-value validation, but scan each shared factor only once
    # when its contents change, and scan each coefficient batch once.
    factors = [factor for factor, lower in p.p_cl_l_ii]
    signature = _table_revision(p, 'fit_factors', *factors)
    if getattr(p, '_pasdw_finite_factors', None) != signature:
        for factor in factors:
            np.asarray_chkfinite(factor)
        p._pasdw_finite_factors = signature
    np.asarray_chkfinite(b1)
    np.asarray_chkfinite(b2)
    try:
        threads = int(os.environ.get('OMP_NUM_THREADS', '1').split(',')[0])
    except ValueError:
        threads = 1
    workers = 4 if use_gpu('pasdw_fit') and threads >= 4 else 1
    out1, out2 = _fit_rows(p, b1, b2, workers)
    if adjoint:
        return (orb2k(p, cp.asarray(np.stack([p.pcol.convert(row, fwd=True) for row in out1]))),
                orb2k(p, cp.asarray(np.stack([p.jcol.convert(row, fwd=True) for row in out2]))))
    return cp.asarray(out2), basis(p.pcol, cp.asarray(out1), True)


def df_only(p, x, adjoint=False):
    cp = cupy()
    if adjoint:
        b = cp.asnumpy(basis(p.pcol, x, False))
        return orb2k(p, cp.asarray(np.stack([p.pcol.convert(row, fwd=True) for row in b])))
    b = cp.asnumpy(k2orb(p, x))
    return basis(p.pcol, cp.asarray(np.stack([p.pcol.convert(row, fwd=False) for row in b])), True)


def linear(contrib, p, first, second, vc=None):
    """v1 forward/reverse chains with no radial-grid host intermediates."""
    from ciderpress.gpaw.fit_paw_gauss_pot import get_dv, get_dvk

    cp = cupy()
    first, second = cp.asarray(first), cp.asarray(second)
    contrib.sbt_rgd = p.sbt_rgd
    na = p.alphas.size
    dv = constant(p, 'dv', get_dv(p.slrgd))[:, None, None]
    dvk = constant(p, 'dvk', get_dvk(p.sbt_rgd))[:, None, None]
    w = constant(p, 'w_b', p.w_b)
    def conv(x, fitting=False):
        return reciprocal_convolution(contrib, x, p.sbt_rgd,
            contrib._galphas if fitting else contrib.plan.alphas,
            contrib._gnorms if fitting else contrib.plan.alpha_norms)
    def forward(x):
        return r2k(contrib, x, p.sbt_rgd, True)
    def backward(x):
        return r2k(contrib, x*dvk, p.sbt_rgd, False)
    if vc is None:
        dy = conv(forward(second))
        dy1 = dy[..., :na]*w
        yy = conv(dy1, True)
        c, df1 = fitted(p, dy1, yy)
        df2 = df_only(p, dy[..., na:])
        df = cp.concatenate((df1, df2), axis=-1)
        first[..., :na] += coefficients(p, c, None, True)*dv
        fr = backward(conv(forward(first)))
        return fr, df, cp.asnumpy(c)
    vxt = backward(conv(forward(first)))
    vdy2 = df_only(p, second[..., na:], True)
    vc_device = cp.asarray(vc) + coefficients(p, vxt[..., :na]*dv, None, False)
    vc[:] = cp.asnumpy(vc_device)
    vdy1, vyy = fitted(p, vc_device, second[..., :na], True)
    vdy1 += conv(vyy, True)
    vdy = cp.concatenate((vdy1*w, vdy2), axis=-1)
    vdx = backward(conv(vdy))
    return vxt, vdx
