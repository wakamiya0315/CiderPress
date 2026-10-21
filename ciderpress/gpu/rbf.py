"""FP64 squared-exponential evaluation behind the NumPy RBF interface.

One thread owns each sample and accumulates control points in C order,
including the caller's initial outputs. No atomics, distance expansion,
parallel reduction, or fast math. The small, read-only model is cached per
evaluator and CUDA device; host snapshots detect replacement AND in-place
edits. Weak keys keep the cache out of model serialization and release it
when the evaluator is destroyed.
"""

from functools import lru_cache
from weakref import WeakKeyDictionary

import numpy as np

from ciderpress.gpu import cupy

_models = WeakKeyDictionary()


@lru_cache(maxsize=None)
def _kernel(nfeat, tiled=False):
    # Both alternatives deliberately have identical arithmetic and control
    # point ordering. ``tiled`` exists for the benchmark, not a public knob.
    source = r"""
    extern "C" __global__ void rbf(
        const double* x, const double* c, const double* a, const double* e,
        double* y, double* dy, const int n, const int nc) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        double xi[NF], grad[NF], value = 0.0;
        if (i < n) {
            value = y[i];
            for (int j = 0; j < NF; ++j) {
                xi[j] = x[(long long)i * NF + j];
                grad[j] = dy[(long long)i * NF + j];
            }
        }
        TILE_DECL
        for (int base = 0; base < nc; base += TILE) {
            TILE_LOAD
            const int end = min(TILE, nc - base);
            if (i < n) {
                for (int t = 0; t < end; ++t) {
                    double dist = 0.0;
                    for (int j = 0; j < NF; ++j) {
                        const double diff = xi[j] - CTRL;
                        dist += e[j] * diff * diff;
                    }
                    const double fac = ALPHA * exp(-dist);
                    value += fac;
                    for (int j = 0; j < NF; ++j)
                        grad[j] += 2 * e[j] * (CTRL - xi[j]) * fac;
                }
            }
            TILE_SYNC
        }
        if (i < n) {
            y[i] = value;
            for (int j = 0; j < NF; ++j)
                dy[(long long)i * NF + j] = grad[j];
        }
    }
    """
    replacements = dict(NF=str(nfeat), TILE="32" if tiled else "1")
    if tiled:
        replacements.update(
            TILE_DECL="__shared__ double sc[32 * NF], sa[32];",
            TILE_LOAD=r"""
                for (int k = threadIdx.x; k < 32 * NF; k += blockDim.x)
                    if ((long long)base * NF + k < (long long)nc * NF)
                        sc[k] = c[(long long)base * NF + k];
                if (threadIdx.x < 32 && base + threadIdx.x < nc)
                    sa[threadIdx.x] = a[base + threadIdx.x];
                __syncthreads();
            """,
            CTRL="sc[t * NF + j]",
            ALPHA="sa[t]",
            TILE_SYNC="__syncthreads();",
        )
    else:
        replacements.update(
            TILE_DECL="",
            TILE_LOAD="",
            TILE_SYNC="",
            CTRL="c[(long long)(base + t) * NF + j]",
            ALPHA="a[base + t]",
        )
    # Expand longer tokens first (TILE before NF, but after TILE_LOAD etc.).
    for token in sorted(replacements, key=len, reverse=True):
        source = source.replace(token, replacements[token])
    return cupy().RawKernel(source, "rbf", options=("--fmad=false",))


def _model(evaluator):
    cp = cupy()
    device = cp.cuda.Device().id
    host = (evaluator._X1ctrl, evaluator._alpha, evaluator._exps)
    devices = _models.setdefault(evaluator, {})
    cached = devices.get(device)
    if cached is None or any(
        old.shape != new.shape or old.dtype != new.dtype or not np.array_equal(old, new)
        for old, new in zip(cached[0], host)
    ):
        snapshots = tuple(np.array(a, copy=True, order="C") for a in host)
        cached = snapshots, tuple(cp.asarray(a) for a in snapshots)
        devices[device] = cached
    return cached[1]


def evaluate(evaluator, x, res, dres, *, tiled=True):
    """Add the RBF value/gradient to contiguous float64 NumPy outputs."""
    cp = cupy()
    if any(a.dtype != np.float64 for a in (x, res, dres)):
        raise TypeError("GPU RBF requires float64 arrays")
    if x.ndim != 2 or evaluator._X1ctrl.ndim != 2:
        raise ValueError("GPU RBF requires ordinary two-dimensional inputs")
    if x.shape[1] != evaluator._nfeat:
        raise ValueError("GPU RBF feature count mismatch")
    c, a, e = _model(evaluator)
    if x.shape[0] == 0:
        return res, dres
    xd, yd, dyd = (cp.asarray(v) for v in (x, res, dres))
    threads = 128
    _kernel(x.shape[1], tiled=tiled)(
        ((x.shape[0] + threads - 1) // threads,),
        (threads,),
        (xd, c, a, e, yd, dyd, np.int32(x.shape[0]), np.int32(c.shape[0])),
    )
    # Blocking get preserves the NumPy interface and synchronizes timing.
    yd.get(out=res)
    dyd.get(out=dres)
    return res, dres
