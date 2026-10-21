"""Batched PASDW dense transforms behind their existing NumPy interfaces.

All radial tables are evaluated by the original C/libm expressions on the
host. Reductions preserve the C loop order (CUDA FMA contraction is disabled).
The ill-conditioned PAW basis fits and all Cholesky solves remain on the CPU.
"""

import ctypes
from functools import lru_cache
from hashlib import sha256

import numpy as np

from ciderpress.gpu import cupy


_SOURCE = r'''
extern "C" __global__ void rad2orb(
    const double* x, const double* table, const int* shell, const int* lm,
    double* out, int ns, int nr, int nlm, int nu, int nq) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nu * nq) return;
    int q = t % nq, u = t / nq % nu, s = t / (nq * nu);
    double v = out[t];
    for (int r = 0; r < nr; r++)
        v += table[shell[u] * nr + r] * x[((s * nr + r) * nlm + lm[u]) * nq + q];
    out[t] = v;
}
extern "C" __global__ void orb2rad(
    const double* x, const double* table, const int* start, const int* end,
    const int* ao, double* out, int ns, int nr, int nlm, int nu, int nq) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nr * nlm * nq) return;
    int q = t % nq, L = t / nq % nlm, r = t / (nq * nlm) % nr;
    int s = t / (nq * nlm * nr), m = L - start[nlm + L];
    double v = out[t];
    for (int j = start[L]; j < end[L]; j++)
        v += table[j * nr + r] * x[(s * nu + ao[j] + m) * nq + q];
    out[t] = v;
}
extern "C" __global__ void convolve(
    const double* x, const double* table, double* out,
    int ns, int nk, int nlm, int nq) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nk * nlm * nq) return;
    int q2 = t % nq, k = t / (nq * nlm) % nk;
    int base = t - q2;
    double v = 0.0;
    for (int q1 = 0; q1 < nq; q1++)
        v += table[(k * nq + q2) * nq + q1] * x[base + q1];
    out[t] = v;
}
extern "C" __global__ void coef2rad(
    const double* c, const double* table, const int* lm, double* out,
    int ns, int nr, int nlm, int ni, int nq) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * nr * nlm * nq) return;
    int q = t % nq, L = t / nq % nlm, r = t / (nq * nlm) % nr;
    int s = t / (nq * nlm * nr);
    double v = out[t];
    for (int i = 0; i < ni; i++)
        if (lm[i] == L) v += c[(s * ni + i) * nq + q] * table[i * nr + r];
    out[t] = v;
}
extern "C" __global__ void rad2coef(
    const double* x, const double* table, const int* lm, double* out,
    int ns, int nr, int nlm, int ni, int nq) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ns * ni * nq) return;
    int q = t % nq, i = t / nq % ni, s = t / (nq * ni);
    double v = 0.0;
    for (int r = 0; r < nr; r++)
        v += table[i * nr + r] * x[((s * nr + r) * nlm + lm[i]) * nq + q];
    out[t] += v;
}
extern "C" __global__ void spline_features(
    const double* x, const double* arg, const double* table, double* p,
    double* feat, double* dfeat, int ng, int nq, int nspin) {
    int g = blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    int cell = (int)arg[g];
    double d = arg[g] - cell, v = 0.0, dv = 0.0;
    for (int q = 0; q < nq; q++) {
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
    double* out, int ng, int nq) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ng*nq) return;
    int g = t/nq, q = t%nq, cell = (int)arg[g];
    double d = arg[g]-cell;
    const double* w = table + (cell*nq+q)*4;
    out[t] = (w[0] + d*(w[1] + d*(w[2] + d*w[3]))) * fun[g];
}
extern "C" __global__ void theta_adjoint(
    const double* x, const double* arg, const double* fun, const double* table,
    double* dedarg, double* dedfun, int ng, int nq) {
    int g = blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    int cell = (int)arg[g];
    double d = arg[g]-cell, v = 0.0, dv = 0.0;
    for (int q = 0; q < nq; q++) {
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


def _cached(owner, name, signature, build):
    cp = cupy()
    # Device selection and CPU table contents are part of every cache key.
    key = cp.cuda.Device().id, signature
    cache = getattr(owner, '_pasdw_device_cache', None)
    if cache is None:
        cache = owner._pasdw_device_cache = {}
    if name not in cache or cache[name][0] != key:
        cache[name] = key, build(cp)
    return cache[name][1]


@lru_cache(None)
def _kernel(name):
    return cupy().RawKernel(_SOURCE, name, options=('--fmad=false',))


def _launch(name, size, args):
    if size:
        _kernel(name)(((size + 255) // 256,), (256,), args)


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
    dx = cp.asarray(np.ascontiguousarray(x))
    shape = (ns, nu, nq) if to_orb else (ns, nr, nlm, nq)
    out = cp.zeros(shape, dtype=cp.float64) if initial is None else cp.array(initial)
    ints = tuple(np.int32(i) for i in (ns, nr, nlm, nu, nq))
    if to_orb:
        _launch('rad2orb', out.size, (dx, table, shell, lm, out, *ints))
    else:
        _launch('orb2rad', out.size, (dx, table, start, end, ao, out, *ints))
    return cp.asnumpy(out)


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
        libcider.tabulate_pasdw_radial(atco.atco_c_ptr, _ptr(r),
                                      ctypes.c_int(len(r)), _ptr(table))
        return _tables(cp, table, ls, nlm)
    # Several atco/radial grids coexist on each contribution object.
    tables = _cached(owner, ('radial', id(atco), len(r)), signature, build)
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


def coefficients(owner, x, out, to_radial, check_q=True, use_ffuncs=False):
    """PASDW projectors, including additive output semantics."""
    cp = cupy()
    nr, nlm, ni = owner.slrgd.r_g.size, owner.pcol.nlm, owner.ni
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
    dx = cp.asarray(np.ascontiguousarray(x))
    _launch('coef2rad' if to_radial else 'rad2coef', dest.size,
            (dx, *tables, dest, *(np.int32(i) for i in (ns, nr, nlm, ni, nq))))
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
        owner, ('gradient', id(n_qg), id(dndr_qg), nlm), signature, build
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
    solve(owner.atco_inp, p)
    return radial(owner, ao, ro, nlm, p, False)


def reciprocal_convolution(owner, x, rgd, alphas, norms):
    from ciderpress.dft.plans import libcider

    k = np.ascontiguousarray(rgd.k_g)
    signature = _signature(k, alphas, norms)
    def build(cp):
        table = np.empty((len(k), len(alphas), len(alphas)))
        libcider.tabulate_pasdw_convolution(_ptr(k), _ptr(alphas), _ptr(norms),
            ctypes.c_int(len(k)), ctypes.c_int(len(alphas)), _ptr(table))
        return cp.asarray(table)
    table = _cached(owner, ('convolution', len(alphas)), signature, build)
    cp = cupy()
    dx = cp.asarray(np.ascontiguousarray(x))
    out = cp.empty_like(dx)
    assert x.shape[1] == len(k) and x.shape[-1] == len(alphas)
    _launch('convolve', out.size, (dx, table, out,
            *(np.int32(i) for i in x.shape)))
    return cp.asnumpy(out)


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
    def __init__(self, base, count):
        from ciderpress.gpu import use_gpu

        self.base = base
        self.grids_indexer = _BatchIndexer(base.grids_indexer, count)
        self.w_g = np.tile(base.w_g, count)
        self.plan = _DevicePlan(base.plan) if use_gpu('pasdw') else base.plan

    def __getattr__(self, name):
        return getattr(self.base, name)

    def features(self, rho):
        from ciderpress.gpaw.atom_utils import _PAWCiderContribs

        if isinstance(self.plan, _DevicePlan) and self.plan._supported():
            plan = self.base.plan
            out = np.stack([self.grids_indexer.empty_rlmq(plan.nalpha) for _ in rho])
            for s in range(len(rho)):
                rt = plan.get_rho_tuple(rho[s])
                arg = plan.get_interpolation_arguments(rt, i=-1)[0]
                fun = plan.get_function_to_convolve(rt)[0]
                fun *= self.w_g
                x = theta_grid(plan, arg, fun)
                self.grids_indexer.reduce_angc_ylm_(out[s], x, a2y=True)
            return out
        return _PAWCiderContribs.get_paw_atom_contribs(self, rho)

    def energy(self, e, rho, v, f):
        from ciderpress.gpaw.atom_utils import _PAWCiderContribs

        return _PAWCiderContribs.get_paw_atom_contribs_en(self, e, rho, v, f)

    def potential(self, rho, v, x):
        from ciderpress.gpaw.atom_utils import _PAWCiderContribs

        if isinstance(self.plan, _DevicePlan) and self.plan._supported():
            plan = self.base.plan
            grid = self.grids_indexer.empty_gq(plan.nalpha)
            for s in range(len(rho)):
                self.grids_indexer.reduce_angc_ylm_(x[s], grid, a2y=False)
                rt = plan.get_rho_tuple(rho[s])
                arg, da = plan.get_interpolation_arguments(rt, i=-1)
                fun, df = plan.get_function_to_convolve(rt)
                va, vf = theta_adjoint(plan, grid, arg, fun)
                v[s, 0] += va*da[0] + vf*df[0]
                tmp = va*da[1] + vf*df[1]
                v[s, 1:4] += 2*rho[s, 1:4]*tmp
                if len(da) == 3:
                    v[s, 4] += va*da[2] + vf*df[2]
            return
        return _PAWCiderContribs.get_paw_atom_contribs_pot(self, rho, v, x)


def _theta_table(plan):
    table = plan._local_alpha_transform[-1]
    return _cached(plan, ('spline', -1), _signature(table), lambda cp: cp.asarray(table))


def theta_grid(plan, arg, fun):
    cp = cupy()
    out = cp.empty((arg.size, plan.nalpha))
    _launch('theta_grid', out.size, (cp.asarray(arg), cp.asarray(fun), _theta_table(plan),
                                   out, np.int32(arg.size), np.int32(plan.nalpha)))
    return cp.asnumpy(out)


def theta_adjoint(plan, x, arg, fun):
    cp = cupy()
    da, df = cp.empty(arg.size), cp.empty(arg.size)
    _launch('theta_adjoint', arg.size, (cp.asarray(x), cp.asarray(arg), cp.asarray(fun),
            _theta_table(plan), da, df, np.int32(arg.size), np.int32(plan.nalpha)))
    return cp.asnumpy(da), cp.asnumpy(df)


class _DevicePlan:
    """Version-j dense interpolation; CPU exponent arguments and CPU GP API.

    Fits of the spline/radial tables are left on the host. The large g,q
    arrays stay on the device between feature and potential evaluation.
    """

    def __init__(self, base):
        self.base, self.p = base, {}

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
        dx = cp.asarray(np.ascontiguousarray(f))
        values, derivs = cp.empty((num, ng)), cp.empty((num, ng))
        rho_tuple = self.base.get_rho_tuple(rho_data)
        self.p[spin] = []
        for i in range(num):
            arg = self.base.get_interpolation_arguments(rho_tuple, i=i)[0]
            table = self.base._local_alpha_transform[i]
            device_table = _cached(self.base, ('spline', i), _signature(table),
                                   lambda cp: cp.asarray(table))
            p = cp.empty_like(dx)
            _launch('spline_features', ng, (dx, cp.asarray(arg), device_table, p,
                    values[i], derivs[i], np.int32(ng), np.int32(nq), np.int32(self.base.nspin)))
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
        rho_tuple = self.base.get_rho_tuple(rho_data)
        for i, p in enumerate(self.p[spin]):
            out += p * v[i, :, None]
            da = self.base.get_interpolation_arguments(rho_tuple, i=i)[1]
            vrho[0] += cp.asarray(da[0]) * df[i] * v[i]
            vrho[1:4] += 2 * drho * cp.asarray(da[1]) * df[i] * v[i]
            if self.base.nldf_settings.sl_level == 'MGGA':
                vrho[4] += cp.asarray(da[2]) * df[i] * v[i]
        vrho_data[:] = cp.asnumpy(vrho)
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
        pairs = [calculate_cider_paw_correction(CiderRadialExpansion(RhoCalculator()),
                    setup, D[a], separate_ae_ps=True) for a in atoms]
        cache[id(setup)] = signature, tuple(np.stack([p[i] for p in pairs]) for i in (0, 1))
    return cache[id(setup)][1]


class _PrecomputedCalculator:
    def __init__(self, contrib, mode, index, values):
        self.xc, self.mode, self.index, self.values = contrib, mode, index, values

    def __call__(self, rgd, n, y, dn, dy, f, ae=True):
        data = self.values[0 if ae else 1]
        v = data[1][self.index]
        if v.shape[1] == 5:
            self.xc.contract_kinetic_potential(v[:, 4], ae)
        if self.mode == 'energy':
            return data[0][self.index], v[:, 0], v[:, 1:4], data[2][self.index]
        return v[:, 0], v[:, 1:4]


def _batch_energy(kernel, setup, atoms, D, projections, rhos):
    from ciderpress.gpaw.atom_utils import CiderRadialExpansion, calculate_cider_paw_correction
    from ciderpress.gpaw.fit_paw_gauss_pot import get_dv

    contrib, p = setup.cider_contribs, setup.pa_setup
    count, ns = len(atoms), len(D[atoms[0]])
    na = projections[atoms[0]].shape[-1]
    dv = get_dv(setup.xc_correction.rgd)
    # Preserve the original projector summation and dot ordering.
    ft = []
    for a in atoms:
        fr = kernel.fr_asgLq[a]
        f = np.zeros_like(kernel.df_asgLq[a])
        for i in range(p.ni):
            L, n, j = p.lmlist_i[i], p.nlist_i[i], p.jlist_i[i]
            ref = (p.pfuncs_ng[n] * dv).dot(fr[:, :, L, :na])
            for s in range(ns):
                f[s, :, L, :na] += (projections[a][s, i] - ref[s]) * p.ffuncs_jg[j][:, None]
        f += fr
        ft.append(f)
    ft = np.stack(ft)
    df = np.stack([kernel.df_asgLq[a] for a in atoms])
    batch = _BatchContribs(contrib, count)
    values = []
    for rho, f in zip(rhos, (ft+df, ft)):
        rho = _pack_rho(rho)
        v = np.zeros_like(rho)
        e = np.zeros(rho.shape[-1])
        f = np.ascontiguousarray(f.swapaxes(0, 1))
        vf = batch.energy(e, rho, v, f)
        values.append((e.reshape(count, -1), _unpack_rho(v, count), vf.swapaxes(0, 1)))
    result, energies, potentials = {}, {}, {}
    for index, a in enumerate(atoms):
        calc = _PrecomputedCalculator(contrib, 'energy', index, values)
        expansion = CiderRadialExpansion(calc, ft[index], df[index])
        energies[a], potentials[a], vf, vft = calculate_cider_paw_correction(
            expansion, setup, D[a], separate_ae_ps=True)
        vfr = vf - vft
        vft[:] = vfr
        result[a] = np.zeros((ns, p.ni, na))
        for i in range(p.ni):
            L, n, j = p.lmlist_i[i], p.nlist_i[i], p.jlist_i[i]
            d = np.einsum('sgq,g->sq', vft[:, :, L, :na], p.ffuncs_jg[j])
            result[a][:, i] = d
            vfr[:, :, L, :na] -= (d[:, None, :] * p.pfuncs_ng[n][:, None]) * dv[:, None]
        kernel.vfr_asgLq[a], kernel.vdf_asgLq[a] = vfr, vf
        kernel.fr_asgLq[a], kernel.df_asgLq[a] = None, None
    return result, energies, potentials


def calculate_corrections(kernel, setups, D, projections, vc):
    """Batch equal-setup atoms in the dense forward/backward transforms.

    Atom and spin axes are folded only where routines treat them independently.
    Density is reused between stages; the original spin-aware XC routines
    evaluate one contiguous point batch. Spherical BLAS transforms stay on
    the CPU to preserve their reduction order before the sensitive PAW fits.
    """
    from ciderpress.gpaw.atom_utils import (
        CiderRadialExpansion, calculate_cider_paw_correction)

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
    for atoms in _groups(setups, D):
        setup = setups[atoms[0]]
        contrib = setup.cider_contribs
        ns = D[atoms[0]].shape[0]
        rhos = _rhos(kernel, setup, atoms, D)
        batch = _BatchContribs(contrib, len(atoms))
        if projections is not None:
            r, e, v = _batch_energy(kernel, setup, atoms, D, projections, rhos)
            result.update(r); energies.update(e); potentials.update(v)
            continue
        if vc is None:
            x = [batch.features(_pack_rho(rho)).swapaxes(0, 1) for rho in rhos]
            shape = (-1,) + x[0].shape[2:]
            xt = np.ascontiguousarray(x[1].reshape(shape))
            dx = np.ascontiguousarray((x[0]-x[1]).reshape(shape))
            fr, df, c = contrib.calculate_y_terms(xt, dx, setup.ps_setup)
            for i, a in enumerate(atoms):
                sl = slice(i*ns, (i+1)*ns)
                kernel.fr_asgLq[a], kernel.df_asgLq[a], result[a] = fr[sl], df[sl], c[sl]
        else:
            vfr = np.concatenate([kernel.vfr_asgLq[a] for a in atoms])
            vdf = np.concatenate([kernel.vdf_asgLq[a] for a in atoms])
            vcoef = np.concatenate([vc[a] for a in atoms])
            vxt, vdx = contrib.calculate_vx_terms(vfr, vdf, vcoef, setup.ps_setup)
            values = []
            for rho, x in zip(rhos, (vdx, vdx-vxt)):
                rho = _pack_rho(rho)
                v = np.zeros_like(rho)
                x = np.ascontiguousarray(x.reshape(len(atoms), ns, *x.shape[1:]).swapaxes(0, 1))
                batch.potential(rho, v, x)
                values.append((None, _unpack_rho(v, len(atoms))))
            for i, a in enumerate(atoms):
                sl = slice(i*ns, (i+1)*ns)
                # v1 updates the caller's coefficient arrays in place.
                if kernel._paw_algo == 'v1':
                    vc[a][:] = vcoef[sl]
                expansion = CiderRadialExpansion(_PrecomputedCalculator(contrib, 'potential', i, values),
                    vdx[sl] - vxt[sl], vxt[sl].copy())
                result[a] = calculate_cider_paw_correction(expansion, setup, D[a],
                                                          separate_ae_ps=True)
    # Preserve the input dictionary's atom order across mixed-setup groups.
    result = {a: result[a] for a in D}
    if projections is not None:
        return result, {a: energies[a] for a in D}, {a: potentials[a] for a in D}
    return result
