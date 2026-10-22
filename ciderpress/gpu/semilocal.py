"""MappedXC2's ordered descriptor/baseline pipeline behind its NumPy API.

Supported models use the ordinary RBF evaluator, SEP/NPOL spin handling,
SLX/SLT/SLN/V maps and the three libxc 7.0.0 baselines. Other models fall
through to the unchanged CPU implementation. No model/feature cache is
maintained here; the existing RBF cache detects mutable model arrays.
"""
import numpy as np

from ciderpress.gpu import cupy
from ciderpress.gpu.descriptors import backward, forward, signature
from ciderpress.gpu.libxc import evaluate_device as _libxc_device
from ciderpress.gpu.rbf import _kernel, _model

_CODES = {"LDA_X": 1, "GGA_X_PBE": 101, "GGA_C_PBE": 130}


def supported(model, nraw=None):
    from ciderpress.dft.xc_evaluator import RBFEvaluator
    from ciderpress.dft.xc_evaluator2 import MappedDFTKernel2
    return bool(model.kernels) and all(type(k) is MappedDFTKernel2 and k.mode in ("SEP", "NPOL") and
               signature(k.feature_list, nraw) is not None and abs(k.omega) <= 1e-11 and
               k._mul_basefunc in _CODES and k._add_basefunc in (*_CODES, None) and
               all(type(ev) is RBFEvaluator and
                   np.array_equal(ev._indexes, np.arange(k.N1)) for ev in k.fevals)
               for k in model.kernels)


def _baseline(name, rho_tuple, sep):
    cp = cupy()
    r = rho_tuple[0]
    ns, n = r.shape
    code = _CODES[name]

    def evaluate(rho, sigma):
        e, v, w = _libxc_device(code, rho, sigma)
        e *= rho.sum(0)
        return (e, v) if code == 1 else (e, v, w)

    if sep and ns == 2:
        energy, vrho = cp.zeros_like(r), cp.zeros_like(r)
        vsigma = None if code == 1 else cp.zeros_like(rho_tuple[1])
        for s in range(ns):
            rs = 2 * r[s:s+1]
            sg = None if code == 1 else 4 * rho_tuple[1][2*s:2*s+1]
            res = evaluate(rs, sg)
            energy[s] = .5 * res[0]
            vrho[s] = res[1][0]
            if code != 1:
                vsigma[2*s] = 2 * res[2][0]
        return (energy, vrho) if code == 1 else (energy, vrho, vsigma)
    res = evaluate(r, None if code == 1 else rho_tuple[1])
    return (res[0][None], *res[1:]) if sep else res


def evaluate(model, X0T, rho_tuple, rhocut):
    cp = cupy()
    x = cp.asarray(np.ascontiguousarray(X0T))
    rho = tuple(cp.asarray(r) for r in rho_tuple)
    res, dres, vrho = evaluate_device(model, x, rho, rhocut)
    # The NumPy interface includes shape/order and independent returned arrays.
    host_dres = np.empty_like(X0T)
    host_dres[...] = dres.get()
    return res.get(), host_dres, tuple(np.asfortranarray(v.get()) for v in vrho)


def evaluate_device(model, x, rho, rhocut):
    """Private resident pipeline; public evaluate retains its NumPy API."""
    cp = cupy()
    vrho = tuple(cp.zeros_like(r, order="F") for r in rho)
    ns, nraw, n = x.shape
    res, dres = 0, 0
    for k in model.kernels:
        sep = k.mode == "SEP"
        raw = [x[s] for s in range(ns)] if sep else [x.mean(0)]
        # Keep spin-major sample order and FeatureList map order.
        desc = cp.concatenate([forward(k.feature_list, a) for a in raw], axis=0)
        f, df = cp.zeros(desc.shape[0]), cp.zeros_like(desc)
        for ev in k.fevals:
            c, alpha, exps = _model(ev)
            if len(f):
                _kernel(ev._nfeat, tiled=True)(((len(f)+127)//128,), (128,),
                     (desc, c, alpha, exps, f, df, np.int32(len(f)), np.int32(c.shape[0])))
        if sep:
            f = f.reshape(ns, n)
            df = df.reshape(ns, n, k.N1)
        if rhocut > 0:
            cond = ns * rho[0] < rhocut if sep else rho[0].sum(0) < rhocut
            f[cond] = 0.
            df[cond] = 0.
        m = _baseline(k._mul_basefunc, rho, sep)
        for v, vm in zip(vrho, m[1:]):
            if sep and v.shape[0] == 3:
                v[::2] += vm[::2] * f
            else:
                v[:] += vm * f
        df *= m[0][..., None]
        f *= m[0]
        if k._add_basefunc is not None:
            a = _baseline(k._add_basefunc, rho, sep)
            f += a[0]
            for v, va in zip(vrho, a[1:]):
                v += va
        if sep:
            dtmp = cp.stack([backward(k.feature_list, raw[s], cp.ascontiguousarray(df[s])) for s in range(ns)])
            f = f.sum(0)
        else:
            dtmp = cp.empty_like(x)
            dtmp[0] = backward(k.feature_list, raw[0], df)
            for s in range(1, ns):
                dtmp[s] = dtmp[0]
            dtmp /= ns
        res += f
        dres += dtmp
    return res, dres, vrho
