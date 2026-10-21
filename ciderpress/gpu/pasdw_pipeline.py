"""Private device-resident linear PASDW chains; CPU fits/solves are retained."""
import numpy as np
from scipy.linalg import cho_solve
from scipy.linalg.lapack import dpotrs

from ciderpress.gpu import cupy
from ciderpress.gpu.pasdw import (_cached, _signature, _table_revision, radial, solve, r2k,
                                  reciprocal_convolution, numeric_basis, coefficients)


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


def constant(owner, name, x):
    return _cached(owner, ('pipeline', name), _signature(x), lambda cp: cp.asarray(x))


def k2orb(p, x):
    from ciderpress.gpaw.fit_paw_gauss_pot import get_dvk

    x = x * constant(p, 'dvk', get_dvk(p.sbt_rgd))[:, None, None]
    a = radial(p, p.pcol.atco_recip, p.sbt_rgd.k_g, x.shape[-2], x, True)
    return solve(p.pcol.atco, a)


def orb2k(p, x):
    return radial(p, p.pcol.atco_recip, p.sbt_rgd.k_g, p._grid_nlm,
                  solve(p.pcol.atco, x), False)


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
    out1, out2 = np.zeros_like(b1), np.zeros_like(b2)
    for l in range(p.lmax+1):
        pairs = [_slices(p, l, m) for m in range(2*l+1)]
        for s in range(ns):
            for i, j in pairs:
                rhs = np.append(b1[s, i].flatten(), b2[s, j].flatten())
                answer = _solve_cpu(p.p_cl_l_ii[l], rhs)
                n1 = b1[s, i].size
                out1[s, i] = answer[:n1].reshape(-1, nb)
                out2[s, j] = answer[n1:].reshape(-1, na)
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
