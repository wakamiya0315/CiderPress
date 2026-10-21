"""Batched PASDW dense transforms behind their existing NumPy interfaces.

All radial tables are evaluated by the original C/libm expressions on the
host. Reductions preserve the C loop order (CUDA FMA contraction is disabled).
The ill-conditioned PAW basis fits and all Cholesky solves remain on the CPU.
"""

import ctypes
from functools import lru_cache
from collections import OrderedDict
from math import prod
from operator import index
from hashlib import sha256

import numpy as np

from ciderpress.gpu import cupy


_SOURCE = r'''
extern "C" __global__ void rad2orb(
    const double* x, const double* table, const int* shell, const int* lm,
    double* out, long long ns, long long nr, long long nlm, long long nu, long long nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nu * nq) return;
    long long q = t % nq, u = t / nq % nu, s = t / (nq * nu);
    double v = out[t];
    for (long long r = 0; r < nr; r++)
        v += table[shell[u] * nr + r] * x[((s * nr + r) * nlm + lm[u]) * nq + q];
    out[t] = v;
}
extern "C" __global__ void orb2rad(
    const double* x, const double* table, const int* start, const int* end,
    const int* ao, double* out, long long ns, long long nr, long long nlm, long long nu, long long nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nr * nlm * nq) return;
    long long q = t % nq, L = t / nq % nlm, r = t / (nq * nlm) % nr;
    long long s = t / (nq * nlm * nr), m = L - start[nlm + L];
    double v = out[t];
    for (long long j = start[L]; j < end[L]; j++)
        v += table[j * nr + r] * x[(s * nu + ao[j] + m) * nq + q];
    out[t] = v;
}
extern "C" __global__ void convolve(
    const double* x, const double* table, double* out,
    long long ns, long long nk, long long nlm, long long nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nk * nlm * nq) return;
    long long q2 = t % nq, k = t / (nq * nlm) % nk;
    long long base = t - q2;
    double v = 0.0;
    for (long long q1 = 0; q1 < nq; q1++)
        v += table[(k * nq + q2) * nq + q1] * x[base + q1];
    out[t] = v;
}
extern "C" __global__ void coef2rad(
    const double* c, const double* table, const int* lm, double* out,
    long long ns, long long nr, long long nlm, long long ni, long long nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nr * nlm * nq) return;
    long long q = t % nq, L = t / nq % nlm, r = t / (nq * nlm) % nr;
    long long s = t / (nq * nlm * nr);
    double v = out[t];
    for (long long i = 0; i < ni; i++)
        if (lm[i] == L) v += c[(s * ni + i) * nq + q] * table[i * nr + r];
    out[t] = v;
}
extern "C" __global__ void rad2coef(
    const double* x, const double* table, const int* lm, double* out,
    long long ns, long long nr, long long nlm, long long ni, long long nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * ni * nq) return;
    long long q = t % nq, i = t / nq % ni, s = t / (nq * ni);
    double v = 0.0;
    for (long long r = 0; r < nr; r++)
        v += table[i * nr + r] * x[((s * nr + r) * nlm + lm[i]) * nq + q];
    out[t] += v;
}
extern "C" __global__ void spline_features(
    const double* x, const double* arg, const double* table, double* p,
    double* feat, double* dfeat, long long ng, long long nq, long long nspin) {
    long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    long long cell = (long long)arg[g];
    double d = arg[g] - cell, v = 0.0, dv = 0.0;
    for (long long q = 0; q < nq; q++) {
        const double* w = table + (cell * nq + q) * 4;
        double f = w[0] + d * (w[1] + d * (w[2] + d * w[3]));
        double df = w[1] + d * (2 * w[2] + d * 3 * w[3]);
        p[g*nq+q] = f;
        v += f * x[g*nq+q];
        dv += df * x[g*nq+q];
    }
    feat[g] = v * nspin;
    dfeat[g] = dv;
}
extern "C" __global__ void theta_grid(
    const double* arg, const double* fun, const double* table,
    double* out, long long ng, long long nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ng*nq) return;
    long long g = t/nq, q = t%nq, cell = (long long)arg[g];
    double d = arg[g]-cell;
    const double* w = table + (cell*nq+q)*4;
    out[t] = (w[0] + d*(w[1] + d*(w[2] + d*w[3]))) * fun[g];
}
extern "C" __global__ void theta_adjoint(
    const double* x, const double* arg, const double* fun, const double* table,
    double* dedarg, double* dedfun, long long ng, long long nq) {
    long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    long long cell = (long long)arg[g];
    double d = arg[g]-cell, v = 0.0, dv = 0.0;
    for (long long q = 0; q < nq; q++) {
        const double* w = table + (cell*nq+q)*4;
        double p = w[0] + d*(w[1] + d*(w[2] + d*w[3]));
        double dp = w[1] + d*(2*w[2] + d*3*w[3]);
        v += p*x[g*nq+q];
        dv += dp*x[g*nq+q];
    }
    dedarg[g] = dv*fun[g];
    dedfun[g] = v;
}
'''


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _signature(*arrays):
    return tuple((a.shape, a.dtype.str, sha256(a.tobytes()).digest()) for a in arrays)


def _table_revision(owner, name, *arrays):
    """Exact byte comparisons for large FP64 tables, avoiding repeated SHA.

    A private host snapshot costs one additional setup-table copy. Every
    element is checked on reuse, including signed zero and NaN payload bits.
    Replacing an array with identical contents needs no device rebuild.
    """
    assert all(a.dtype == np.float64 for a in arrays)
    metadata = tuple((a.shape, a.dtype.str) for a in arrays)
    cache = getattr(owner, '_pasdw_host_table_cache', None)
    if cache is None:
        cache = owner._pasdw_host_table_cache = {}
    previous = cache.get(name)
    if previous is not None and previous[0] == metadata:
        if all(np.array_equal(a.view(np.uint64), b) for a, b in zip(arrays, previous[1])):
            return previous[2]
    revision = 1 if previous is None else previous[2]+1
    cache[name] = metadata, tuple(a.copy(order='C').view(np.uint64) for a in arrays), revision
    return revision


def _cached(owner, name, signature, build):
    cp = cupy()
    # Device selection and CPU table contents are part of every cache key.
    key = cp.cuda.Device().id, signature
    cache = getattr(owner, '_pasdw_device_cache', None)
    if cache is None:
        cache = owner._pasdw_device_cache = OrderedDict()
    if name not in cache or cache[name][0] != key:
        cache[name] = key, build(cp)
    cache.move_to_end(name)
    while len(cache) > 32:
        cache.popitem(last=False)
    return cache[name][1]


@lru_cache(None)
def _kernel(name):
    return cupy().RawKernel(_SOURCE, name, options=('--fmad=false',))


def _checked_size(*dimensions):
    dims = tuple(index(d) for d in dimensions)
    limit = np.iinfo(np.int64).max // 8
    if any(d < 0 or d > limit for d in dims) or prod(dims) > limit:
        raise ValueError('PASDW dimensions exceed signed 64-bit byte addressing')
    return prod(dims)


def _launch(name, size, args):
    _checked_size(size)
    for arg in args:
        if hasattr(arg, 'shape') and arg.shape:
            _checked_size(*arg.shape)
        elif isinstance(arg, np.int64):
            _checked_size(int(arg))
    if (size + 255) // 256 > 2**31 - 1:
        raise ValueError('PASDW launch exceeds CUDA grid size')
    if size:
        _kernel(name)(((size + 255) // 256,), (256,), args)


def _device_array(x):
    cp = cupy()
    return cp.ascontiguousarray(x) if isinstance(x, cp.ndarray) else cp.asarray(np.ascontiguousarray(x))


def _join_device(values, stack=True):
    """Accept legacy/replay host states and resident states from this cycle."""
    cp = cupy()
    if any(isinstance(x, cp.ndarray) for x in values):
        fn = cp.stack if stack else cp.concatenate
        return fn([cp.asarray(x) for x in values])
    fn = np.stack if stack else np.concatenate
    return cp.asarray(fn(values))


def _tables(cp, table, ls, nlm):
    nbas = len(ls)
    ao = np.r_[0, np.cumsum(2 * ls + 1)].astype(np.int32)
    shell = np.repeat(np.arange(nbas, dtype=np.int32), 2 * ls + 1)
    lm = np.concatenate([np.arange(l*l, (l+1)*(l+1), dtype=np.int32) for l in ls])
    start, end, l0 = [], [], []
    for L in range(nlm):
        l = int(np.sqrt(L))
        indices = np.flatnonzero(ls == l)
        start.append(int(indices[0]) if len(indices) else 0)
        end.append(int(indices[-1]+1) if len(indices) else 0)
        l0.append(l*l)
    return tuple(cp.asarray(v) for v in (table, shell, lm,
        np.array(start+l0, dtype=np.int32), np.array(end, dtype=np.int32), ao))


def _contract(tables, x, ns, nr, nlm, nu, nq, to_orb, initial=None):
    cp = cupy()
    assert x.dtype == np.float64
    table, shell, lm, start, end, ao = tables
    device = isinstance(x, cp.ndarray)
    dx = _device_array(x)
    shape = (ns, nu, nq) if to_orb else (ns, nr, nlm, nq)
    out = cp.zeros(shape, dtype=cp.float64) if initial is None else cp.array(initial)
    ints = tuple(np.int64(i) for i in (ns, nr, nlm, nu, nq))
    if to_orb:
        _launch('rad2orb', out.size, (dx, table, shell, lm, out, *ints))
    else:
        _launch('orb2rad', out.size, (dx, table, start, end, ao, out, *ints))
    return out if device else cp.asnumpy(out)


def radial(owner, atco, r, nlm, x, to_orb):
    """Host-tabulated Gaussian basis, fixed-order device contractions."""
    from ciderpress.dft.lcao_convolutions import ANG_OF
    from ciderpress.dft.plans import libcider

    r = np.ascontiguousarray(r)
    bas, env = atco.bas, atco.env
    ls = bas[:, ANG_OF]
    assert atco.natm == 1
    signature = _signature(bas, env, r) + (nlm,)
    def build(cp):
        nbas = len(bas)
        table = np.empty((nbas, len(r)))
        _checked_size(nbas, len(r))
        if len(r) > np.iinfo(np.int32).max:
            raise ValueError('PASDW radial dimension exceeds C int ABI')
        status = libcider.tabulate_pasdw_radial(atco.atco_c_ptr, _ptr(r),
                                      ctypes.c_int(len(r)), _ptr(table))
        if status != 0:
            raise ValueError('Invalid PASDW radial table dimensions')
        return _tables(cp, table, ls, nlm)
    # Semantic slots survive table replacement without retaining old object
    # IDs, while the distinct input/reciprocal/feature bases can coexist.
    slot = ('other', to_orb)
    for name in ('atco_inp', '_atco_recip', 'atco_feat'):
        if getattr(owner, name, None) is atco:
            slot = name
            break
    else:
        for name in ('pcol', 'jcol'):
            if getattr(getattr(owner, name, None), 'atco_recip', None) is atco:
                slot = (name, 'reciprocal')
                break
    tables = _cached(owner, ('radial', slot), signature, build)
    ns, nq, nr, nu = len(x), x.shape[-1], len(r), atco.nao
    return _contract(tables, x, ns, nr, nlm, nu, nq, to_orb)


def numeric_basis(owner, p_uq, p_rlmq, fwd):
    """The already fitted PAW radial projectors; no device fitting/solve."""
    signature = _signature(owner.funcs_ig, owner.iloc_l) + (owner.nlm,)
    def build(cp):
        ls = np.repeat(np.arange(len(owner.iloc_l)-1), np.diff(owner.iloc_l))
        return _tables(cp, owner.funcs_ig, ls, owner.nlm)
    tables = _cached(owner, 'numeric_radial', signature, build)
    single = p_uq.ndim == 2
    p = p_uq[None] if single else p_uq
    r = p_rlmq[None] if single else p_rlmq
    x, initial = (p, r) if fwd else (r, p)
    out = _contract(tables, x, len(p), owner.funcs_ig.shape[1], owner.nlm,
                    owner.nu, p_uq.shape[-1], not fwd, initial)
    if single:
        out = out[0]
    if fwd:
        p_rlmq[:] = out
    else:
        p_uq[:] = out


def coefficients(owner, x, out, to_radial, check_q=True, use_ffuncs=False, _nlm=None):
    """PASDW projectors, including additive output semantics."""
    cp = cupy()
    nr, ni = owner.slrgd.r_g.size, owner.ni
    nlm = owner.pcol.nlm if _nlm is None else _nlm
    nq = owner.alphas.size if check_q else x.shape[-1]
    ns = len(x)
    table = (owner.ffuncs_jg[owner.jlist_i] if use_ffuncs else
             owner.pfuncs_ng[owner.nlist_i])
    signature = _signature(table, owner.lmlist_i)
    tables = _cached(owner, ('coefficients', use_ffuncs), signature,
                     lambda cp: (cp.asarray(table), cp.asarray(owner.lmlist_i)))
    expected = (ns, ni, nq) if to_radial else (ns, nr, nlm, nq)
    assert x.shape == expected and x.dtype == np.float64
    shape = (ns, nr, nlm, nq) if to_radial else (ns, ni, nq)
    if out is None:
        dest = cp.zeros(shape, dtype=cp.float64)
    else:
        assert out.shape == shape and out.dtype == np.float64
        dest = cp.array(out, order='C')
    device = isinstance(x, cp.ndarray)
    dx = _device_array(x)
    _launch('coef2rad' if to_radial else 'rad2coef', dest.size,
            (dx, *tables, dest, *(np.int64(i) for i in (ns, nr, nlm, ni, nq))))
    if device:
        if out is not None:
            out[:] = dest
            return out
        return dest
    host = cp.asnumpy(dest)
    if out is not None:
        out[:] = host
        return out
    return host


def contract_gradient(owner, rgd, n_qg, dndr_qg, dedn_sg, dedgrad_svg, nlm):
    """Dense PAW potential quadrature, NumPy in/out and cached radial tables."""
    from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
    from gpaw.xc.pawcorrection import rnablaY_nLv

    cp = cupy()
    signature = _signature(n_qg, dndr_qg, rgd.dv_g, rgd.dr_g, rgd.r_g,
                           Y_nL, weight_n, rnablaY_nLv, R_nv) + (nlm,)
    def build(cp):
        return tuple(cp.asarray(x) for x in (
            n_qg * rgd.dv_g, dndr_qg * rgd.dv_g,
            weight_n[:, None] * Y_nL[:, :nlm],
            n_qg * (rgd.dr_g * rgd.r_g),
            (4*np.pi) * weight_n[:, None, None] * rnablaY_nLv[:, :nlm], R_nv))
    n, dn, y, nr, dy, directions = _cached(
        owner, ('gradient', n_qg is owner.xcc.n_qg), signature, build
    )
    ns, nn = len(dedn_sg), len(weight_n)
    rho = cp.asarray(dedn_sg).reshape(ns, -1, nn)
    grad = cp.asarray(dedgrad_svg).reshape(ns, 3, -1, nn)
    radial_grad = cp.einsum('svgn,nv->sgn', grad, directions)
    b = cp.matmul(n, rho)
    b += cp.matmul(dn, radial_grad)
    out = cp.matmul(b, y)
    b = cp.einsum('qg,svgn->svqn', nr, grad, optimize=True)
    out += cp.einsum('nLv,svqn->sqL', dy, b, optimize=True)
    return cp.asnumpy(out)


def solve(atco, x):
    """Use the existing host factor/solve, without forming an inverse."""
    from ciderpress.dft.plans import libcider

    cp = cupy()
    if isinstance(x, cp.ndarray):
        return cp.asarray(solve(atco, cp.asnumpy(x)))
    for s in range(len(x)):
        libcider.solve_atc_coefs_arr(atco.atco_c_ptr, _ptr(x[s]),
                                    ctypes.c_int(x.shape[-1]))
    return x


def r2k(owner, x, rgd, fwd):
    ai, ao = ((owner.atco_inp, owner._atco_recip) if fwd else
              (owner._atco_recip, owner.atco_feat))
    ri, ro = ((owner.grids_indexer.rad_arr, rgd.k_g) if fwd else
              (rgd.k_g, owner.grids_indexer.rad_arr))
    nlm = x.shape[-2]
    p = radial(owner, ai, ri, nlm, x, True)
    p = solve(owner.atco_inp, p)
    return radial(owner, ao, ro, nlm, p, False)


def reciprocal_convolution(owner, x, rgd, alphas, norms):
    from ciderpress.dft.plans import libcider

    k = np.ascontiguousarray(rgd.k_g)
    signature = _signature(k, alphas, norms)
    def build(cp):
        table = np.empty((len(k), len(alphas), len(alphas)))
        _checked_size(len(k), len(alphas), len(alphas))
        if max(len(k), len(alphas)) > np.iinfo(np.int32).max:
            raise ValueError('PASDW convolution dimension exceeds C int ABI')
        status = libcider.tabulate_pasdw_convolution(_ptr(k), _ptr(alphas), _ptr(norms),
            ctypes.c_int(len(k)), ctypes.c_int(len(alphas)), _ptr(table))
        if status != 0:
            raise ValueError('Invalid PASDW convolution table dimensions')
        return cp.asarray(table)
    table = _cached(owner, ('convolution', alphas is getattr(owner, '_galphas', None)),
                    signature, build)
    cp = cupy()
    device = isinstance(x, cp.ndarray)
    dx = _device_array(x)
    out = cp.empty_like(dx)
    assert x.shape[1] == len(k) and x.shape[-1] == len(alphas)
    _launch('convolve', out.size, (dx, table, out,
            *(np.int64(i) for i in x.shape)))
    return out if device else cp.asnumpy(out)


def _groups(setups, D):
    groups = {}
    for a in D:
        groups.setdefault(id(setups[a]), []).append(a)
    return groups.values()


class _BatchIndexer:
    """Fold atoms into the point axis, retaining the original BLAS transforms."""

    def __init__(self, base, count):
        self.base, self.count = base, count

    def empty_gq(self, nalpha=1):
        return np.empty((self.count*self.base.ngrids, nalpha))

    def empty_rlmq(self, nalpha=1):
        return np.empty((self.count, self.base.nrad, self.base.nlm, nalpha))

    def reduce_angc_ylm_(self, radial, grid, a2y=True, offset=0):
        chunks = grid.reshape(self.count, self.base.ngrids, -1)
        for a in range(self.count):
            self.base.reduce_angc_ylm_(radial[a], chunks[a], a2y=a2y, offset=offset)


class _BatchContribs:
    def __init__(self, base, count, theta_cache=None):
        from ciderpress.gpu import use_gpu

        self.base = base
        self.theta_cache = theta_cache
        self.grids_indexer = _BatchIndexer(base.grids_indexer, count)
        self.w_g = np.tile(base.w_g, count)
        self.plan = _DevicePlan(base.plan) if use_gpu('pasdw') else base.plan

    def __getattr__(self, name):
        return getattr(self.base, name)

    def features(self, rho, theta_index=None):
        from ciderpress.gpaw.atom_utils import _PAWCiderContribs

        if isinstance(self.plan, _DevicePlan) and self.plan._supported():
            plan = self.base.plan
            from ciderpress.gpu.pasdw_reductions import spherical

            out = []
            theta = []
            for s in range(len(rho)):
                rt = plan.get_rho_tuple(rho[s])
                arg, da = plan.get_interpolation_arguments(rt, i=-1)
                fun, df = plan.get_function_to_convolve(rt)
                theta.append((arg, da, fun, df))
                x = theta_grid(plan, arg, fun*self.w_g, _device=True)
                out.append(spherical(self.base.grids_indexer, x, True))
            if self.theta_cache is not None and theta_index is not None:
                self.theta_cache[theta_index] = theta
            return cupy().stack(out)
        return _PAWCiderContribs.get_paw_atom_contribs(self, rho)

    def energy(self, e, rho, v, f):
        from ciderpress.gpaw.atom_utils import _PAWCiderContribs

        if isinstance(self.plan, _DevicePlan) and self.plan._supported():
            from ciderpress.gpu.pasdw_reductions import energy

            return energy(self, e, rho, v, f)
        if isinstance(f, cupy().ndarray):
            f = cupy().asnumpy(f)
        host_v = np.zeros_like(rho) if isinstance(v, cupy().ndarray) else v
        result = _PAWCiderContribs.get_paw_atom_contribs_en(self, e, rho, host_v, f)
        if host_v is not v:
            v[:] = cupy().asarray(host_v)
        return result

    def potential(self, rho, v, x, theta_index=None):
        from ciderpress.gpaw.atom_utils import _PAWCiderContribs

        if isinstance(self.plan, _DevicePlan) and self.plan._supported():
            from ciderpress.gpu.pasdw_reductions import spherical
            from ciderpress.gpu.pasdw_exponents import theta_potential

            plan = self.base.plan
            for s in range(len(rho)):
                grid = spherical(self.base.grids_indexer, cupy().asarray(x[s]), False)
                if self.theta_cache is not None and theta_index in self.theta_cache:
                    arg, da, fun, df = self.theta_cache[theta_index][s]
                else:
                    rt = plan.get_rho_tuple(rho[s])
                    arg, da = plan.get_interpolation_arguments(rt, i=-1)
                    fun, df = plan.get_function_to_convolve(rt)
                target = v[s] if isinstance(v, cupy().ndarray) else cupy().asarray(v[s])
                theta_potential(plan, grid, rho[s], arg, fun, da, df, target)
                if not isinstance(v, cupy().ndarray):
                    v[s] = cupy().asnumpy(target)
            return
        if isinstance(x, cupy().ndarray):
            x = cupy().asnumpy(x)
        host_v = cupy().asnumpy(v) if isinstance(v, cupy().ndarray) else v
        result = _PAWCiderContribs.get_paw_atom_contribs_pot(self, rho, host_v, x)
        if host_v is not v:
            v[:] = cupy().asarray(host_v)
        return result


def _theta_table(plan):
    table = plan._local_alpha_transform[-1]
    return _cached(plan, ('spline', -1), _signature(table), lambda cp: cp.asarray(table))


def theta_grid(plan, arg, fun, _device=False):
    cp = cupy()
    out = cp.empty((arg.size, plan.nalpha))
    _launch('theta_grid', out.size, (cp.asarray(arg), cp.asarray(fun), _theta_table(plan),
                                   out, np.int64(arg.size), np.int64(plan.nalpha)))
    return out if _device else cp.asnumpy(out)


def theta_adjoint(plan, x, arg, fun):
    cp = cupy()
    da, df = cp.empty(arg.size), cp.empty(arg.size)
    _launch('theta_adjoint', arg.size, (cp.asarray(x), cp.asarray(arg), cp.asarray(fun),
            _theta_table(plan), da, df, np.int64(arg.size), np.int64(plan.nalpha)))
    return cp.asnumpy(da), cp.asnumpy(df)


class _DevicePlan:
    """Version-j dense interpolation, retaining the NumPy GP API.

    Fits of the spline/radial tables are left on the host. The large g,q
    arrays stay on the device between feature and potential evaluation.
    """

    def __init__(self, base, device_output=False):
        self.base, self.p = base, {}
        self.arguments = {}
        self.device_output = device_output

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _supported(self):
        from ciderpress.dft.plans import NLDFSplinePlan

        return (isinstance(self.base, NLDFSplinePlan) and
                self.base.nldf_settings.nldf_type == 'j' and
                self.base.coef_order == 'gq' and self.base.local_nalpha == self.base.nalpha)

    def eval_rho_full(self, f, rho_data, spin=0, feat=None, dfeat=None,
                      cache_p=True, apply_transformation=False, coeff_multipliers=None):
        if not self._supported() or not cache_p or apply_transformation or coeff_multipliers is not None:
            self.p.pop(spin, None)
            return self.base.eval_rho_full(f, rho_data, spin, feat, dfeat, cache_p,
                                          apply_transformation, coeff_multipliers)
        cp = cupy()
        ng, nq = f.shape
        assert nq == self.base.nalpha
        num = self.base.nldf_settings.num_feat_param_sets
        if feat is None:
            feat = np.empty((num, ng))
        if dfeat is None:
            dfeat = np.empty((num, ng))
        dx = _device_array(f)
        values, derivs = cp.empty((num, ng)), cp.empty((num, ng))
        rho_tuple = None if self.device_output else self.base.get_rho_tuple(rho_data)
        if self.device_output:
            from ciderpress.gpu.pasdw_exponents import arguments

            if self.base.nldf_settings.sl_level == 'MGGA':
                rho_data[4, rho_data[0] < self.base.rhocut] = 0.
            device_rho = cp.asarray(rho_data)
            self.arguments[spin] = []
        self.p[spin] = []
        for i in range(num):
            if self.device_output:
                arg, da = arguments(self.base, device_rho, i)
                self.arguments[spin].append(da)
            else:
                arg = self.base.get_interpolation_arguments(rho_tuple, i=i)[0]
            table = self.base._local_alpha_transform[i]
            device_table = _cached(self.base, ('spline', i), _signature(table),
                                   lambda cp: cp.asarray(table))
            p = cp.empty_like(dx)
            _launch('spline_features', ng, (dx, cp.asarray(arg), device_table, p,
                    values[i], derivs[i], np.int64(ng), np.int64(nq), np.int64(self.base.nspin)))
            self.p[spin].append(p)
        feat[:] = cp.asnumpy(values)
        dfeat[:] = cp.asnumpy(derivs)
        return feat, dfeat

    def eval_vxc_full(self, vfeat, vrho_data, dfeat, rho_data, spin=0, p_i_qg=None, vf=None):
        if not self._supported() or spin not in self.p or p_i_qg is not None:
            return self.base.eval_vxc_full(vfeat, vrho_data, dfeat, rho_data, spin, p_i_qg, vf)
        cp = cupy()
        vfeat[:] *= self.base.nspin
        out = cp.asarray(vf).copy() if vf is not None else cp.zeros_like(self.p[spin][0])
        v, df = cp.asarray(vfeat), cp.asarray(dfeat)
        vrho, drho = cp.array(vrho_data), cp.asarray(rho_data[1:4])
        rho_tuple = None if self.device_output else self.base.get_rho_tuple(rho_data)
        for i, p in enumerate(self.p[spin]):
            out += p * v[i, :, None]
            da = (self.arguments[spin][i] if self.device_output else
                  self.base.get_interpolation_arguments(rho_tuple, i=i)[1])
            vrho[0] += cp.asarray(da[0]) * df[i] * v[i]
            vrho[1:4] += 2 * drho * cp.asarray(da[1]) * df[i] * v[i]
            if self.base.nldf_settings.sl_level == 'MGGA':
                vrho[4] += cp.asarray(da[2]) * df[i] * v[i]
        vrho_data[:] = vrho if isinstance(vrho_data, cp.ndarray) else cp.asnumpy(vrho)
        if self.device_output:
            return out
        result = cp.asnumpy(out)
        if vf is not None:
            vf[:] = result
            return vf
        return result


def _pack_rho(rho):
    # a,s,x,g -> s,x,(a*g): atom identity is independent of spin.
    a, s, x, g = rho.shape
    return np.ascontiguousarray(rho.transpose(1, 2, 0, 3).reshape(s, x, a*g))


def _unpack_rho(rho, count):
    s, x, g = rho.shape
    return rho.reshape(s, x, count, g//count).transpose(2, 0, 1, 3)


def _rhos(kernel, setup, atoms, D):
    """Density is evaluated once per input D, reused by all three stages.

    This is an XC-cycle cache, cleared at every feature call. Setup tables
    must stay fixed during the forward/energy/backward stages, just like the
    saved fr/df intermediates. Replaced tables and changed D invalidate it.
    Persistent device table caches separately check full array contents.
    """
    from ciderpress.gpaw.atom_utils import CiderRadialExpansion, calculate_cider_paw_correction

    c = setup.cider_contribs
    class RhoCalculator:
        mode = 'feature'
        xc = c
        def __call__(self, rgd, n, y, dn, dy, ae=True):
            return c.vec_radial_vars(n, y, dn, dy, ae)[1]
    names = ('B_pqL', 'n_qg', 'nt_qg', 'd_qg', 'dt_qg', 'nc_g', 'nct_g',
             'dc_g', 'dct_g', 'tau_pg', 'taut_pg', 'tauc_g', 'tauct_g',
             'nc_corehole_g', 'dc_corehole_g')
    tables = [getattr(c.xcc, name, None) for name in names]
    tables = [x for x in tables if isinstance(x, np.ndarray)]
    tables.extend((c.xcc.rgd.r_g, c.xcc.rgd.dr_g, c.xcc.rgd.dv_g))
    signature = (tuple(atoms), _signature(*(D[a] for a in atoms)),
                 tuple((id(x), x.shape, x.dtype.str) for x in tables))
    cache = getattr(kernel, '_pasdw_rho_cache', None)
    if cache is None:
        cache = kernel._pasdw_rho_cache = {}
    if id(setup) not in cache or cache[id(setup)][0] != signature:
        from ciderpress.gpu import use_gpu

        if use_gpu('pasdw') and c.is_mgga:
            from ciderpress.gpu.pasdw_reductions import density

            values = density(c, np.stack([D[a] for a in atoms]))
        else:
            pairs = [calculate_cider_paw_correction(CiderRadialExpansion(RhoCalculator()),
                        setup, D[a], separate_ae_ps=True) for a in atoms]
            values = tuple(np.stack([p[i] for p in pairs]) for i in (0, 1))
        cache[id(setup)] = signature, values
    return cache[id(setup)][1]


def _batch_energy(kernel, setup, atoms, D, projections, rhos):
    from ciderpress.gpaw.fit_paw_gauss_pot import get_dv

    contrib, p = setup.cider_contribs, setup.pa_setup
    count, ns = len(atoms), len(D[atoms[0]])
    na = projections[atoms[0]].shape[-1]
    dv = get_dv(setup.xc_correction.rgd)
    cp = cupy()
    fr = _join_device([kernel.fr_asgLq[a] for a in atoms])
    df = _join_device([kernel.df_asgLq[a] for a in atoms])
    shape = (count*ns,) + fr.shape[2:]
    def project(x, to_radial, use_ffuncs=False):
        return coefficients(p, x, None, to_radial, check_q=False, use_ffuncs=use_ffuncs,
                            _nlm=fr.shape[-2])
    weighted = fr.reshape(shape)[..., :na]*cp.asarray(dv)[:, None, None]
    ref = project(weighted, False)
    delta = cp.asarray(np.stack([projections[a] for a in atoms])).reshape(count*ns, p.ni, na)-ref
    ft = cp.zeros_like(fr).reshape(shape)
    ft[..., :na] = project(delta, True, True)
    ft = ft.reshape(fr.shape)
    ft += fr
    batch = _BatchContribs(contrib, count)
    values = []
    for rho, f in zip(rhos, (ft+df, ft)):
        rho = _pack_rho(rho)
        v = cp.zeros(rho.shape, dtype=cp.float64)
        e = np.zeros(rho.shape[-1])
        f = _device_array(f.swapaxes(0, 1))
        vf = batch.energy(e, rho, v, f)
        values.append((e.reshape(count, -1), _unpack_rho(v, count), vf.swapaxes(0, 1)))
    from ciderpress.gpu.pasdw_reductions import finish

    energy, dh = finish(contrib, [(v[0], v[1]) for v in values], energy=True,
                        _tables=getattr(contrib, '_pasdw_cycle_tables', None))
    vf = cp.asarray(values[0][2]).reshape(shape)
    vfr = vf - cp.asarray(values[1][2]).reshape(shape)
    # All projectors see the unchanged AE-minus-PS difference. The original
    # subsequent subtraction is a linear reconstruction of those coefficients.
    d = project(vfr[..., :na], False, True)
    vfr[..., :na] -= project(d, True)*cp.asarray(dv)[:, None, None]
    d = cp.asnumpy(d).reshape(count, ns, p.ni, na)
    vfr = vfr.reshape(fr.shape)
    vf = vf.reshape(fr.shape)
    result, energies, potentials = {}, {}, {}
    for index, a in enumerate(atoms):
        energies[a], potentials[a] = energy[index], dh[index]
        result[a] = d[index]
        kernel.vfr_asgLq[a], kernel.vdf_asgLq[a] = vfr[index], vf[index]
        kernel.fr_asgLq[a], kernel.df_asgLq[a] = None, None
    return result, energies, potentials


def calculate_corrections(kernel, setups, D, projections, vc):
    """Batch equal-setup atoms in the dense forward/backward transforms.

    Atom and spin axes are folded only where routines treat them independently.
    Density is reused between stages; the original spin-aware XC routines
    evaluate one contiguous point batch. Spherical BLAS transforms stay on
    the CPU to preserve their reduction order before the sensitive PAW fits.
    """
    if not D:
        return ({}, {}, {}) if projections is not None else {}
    result = {}
    if projections is not None:
        kernel.vfr_asgLq, kernel.vdf_asgLq = {}, {}
        energies, potentials = {}, {}
    elif vc is None:
        kernel.fr_asgLq, kernel.df_asgLq = {}, {}
        # A feature call begins a new XC cycle. Recompute density once even
        # for repeated timing inputs; reuse it only in energy/potential stages.
        kernel._pasdw_rho_cache = {}
        kernel._pasdw_theta_cache = {}
    for atoms in _groups(setups, D):
        setup = setups[atoms[0]]
        contrib = setup.cider_contribs
        ns = D[atoms[0]].shape[0]
        rhos = _rhos(kernel, setup, atoms, D)
        theta_cache = getattr(kernel, '_pasdw_theta_cache', None)
        if theta_cache is None:
            theta_cache = kernel._pasdw_theta_cache = {}
        if id(setup) not in theta_cache or theta_cache[id(setup)][0] is not rhos:
            theta_cache[id(setup)] = rhos, {}
        batch = _BatchContribs(contrib, len(atoms), theta_cache[id(setup)][1])
        if projections is not None:
            r, e, v = _batch_energy(kernel, setup, atoms, D, projections, rhos)
            result.update(r); energies.update(e); potentials.update(v)
            continue
        if vc is None:
            x = [batch.features(_pack_rho(rho), i).swapaxes(0, 1) for i, rho in enumerate(rhos)]
            shape = (-1,) + x[0].shape[2:]
            xt = _device_array(x[1].reshape(shape))
            dx = _device_array((x[0]-x[1]).reshape(shape))
            from ciderpress.gpaw.atom_utils import USE_GAUSSIAN_SBT

            if kernel._paw_algo == 'v1' and USE_GAUSSIAN_SBT:
                from ciderpress.gpu.pasdw_pipeline import linear

                fr, df, c = linear(contrib, setup.ps_setup, xt, dx)
            else:
                fr, df, c = contrib.calculate_y_terms(cupy().asnumpy(xt), cupy().asnumpy(dx), setup.ps_setup)
            for i, a in enumerate(atoms):
                sl = slice(i*ns, (i+1)*ns)
                kernel.fr_asgLq[a], kernel.df_asgLq[a], result[a] = fr[sl], df[sl], c[sl]
        else:
            vfr = _join_device([kernel.vfr_asgLq[a] for a in atoms], stack=False)
            vdf = _join_device([kernel.vdf_asgLq[a] for a in atoms], stack=False)
            vcoef = np.concatenate([vc[a] for a in atoms])
            from ciderpress.gpaw.atom_utils import USE_GAUSSIAN_SBT

            if kernel._paw_algo == 'v1' and USE_GAUSSIAN_SBT:
                from ciderpress.gpu.pasdw_pipeline import linear

                vxt, vdx = linear(contrib, setup.ps_setup, vfr, vdf, vcoef)
            else:
                vxt, vdx = contrib.calculate_vx_terms(cupy().asnumpy(vfr), cupy().asnumpy(vdf), vcoef, setup.ps_setup)
            values = []
            for index, (rho, x) in enumerate(zip(rhos, (vdx, vdx-vxt))):
                rho = _pack_rho(rho)
                v = cupy().zeros(rho.shape, dtype=cupy().float64)
                x = _device_array(x.reshape(len(atoms), ns, *x.shape[1:]).swapaxes(0, 1))
                batch.potential(rho, v, x, index)
                values.append((None, _unpack_rho(v, len(atoms))))
            from ciderpress.gpu.pasdw_reductions import finish

            dh = finish(contrib, values, _tables=getattr(contrib, '_pasdw_cycle_tables', None))
            for i, a in enumerate(atoms):
                sl = slice(i*ns, (i+1)*ns)
                # v1 updates the caller's coefficient arrays in place.
                if kernel._paw_algo == 'v1':
                    vc[a][:] = vcoef[sl]
                result[a] = dh[i]
    # Preserve the input dictionary's atom order across mixed-setup groups.
    result = {a: result[a] for a in D}
    if projections is not None:
        return result, {a: energies[a] for a in D}, {a: potentials[a] for a in D}
    return result
