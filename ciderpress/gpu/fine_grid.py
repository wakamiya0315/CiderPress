"""Serial FP64 finite differences using the CPU operators' exact stencils."""
from functools import lru_cache

import numpy as np

from ciderpress.gpu import cupy
from ciderpress.gpu.pasdw import _cached, _signature, _checked_size


_SOURCE = r'''
extern "C" __global__ void gradient(
    const double* x, double* y, const double* coefficients,
    const long long* offsets, const long long* periodic,
    long long nx, long long ny, long long nz, long long np, long long total) {
    long long t = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (t >= total) return;
    long long g = t%(nx*ny*nz), spin = t/(nx*ny*nz);
    long long ix = g/(ny*nz), iy = g/nz%ny, iz = g%nz;
    double result = 0;
    for (long long p = 0; p < np; p++) {
        long long a = ix+offsets[3*p], b = iy+offsets[3*p+1], c = iz+offsets[3*p+2];
        if (a < 0 || a >= nx) { if (!periodic[0]) continue; a = (a%nx+nx)%nx; }
        if (b < 0 || b >= ny) { if (!periodic[1]) continue; b = (b%ny+ny)%ny; }
        if (c < 0 || c >= nz) { if (!periodic[2]) continue; c = (c%nz+nz)%nz; }
        result += coefficients[p]*x[((spin*nx+a)*ny+b)*nz+c];
    }
    y[t] = result;
}
'''


@lru_cache(None)
def _kernel():
    return cupy().RawKernel(_SOURCE, 'gradient', options=('--fmad=false',))


def apply(operator, x, out):
    cp = cupy()
    assert x.dtype == out.dtype == np.float64
    assert x.flags.c_contiguous and out.flags.c_contiguous
    offsets = np.asarray(operator.offset_pc, dtype=np.int64)
    periodic = np.asarray(operator.gd.pbc_c, dtype=np.int64)
    signature = _signature(operator.coef_p, offsets, periodic)
    coefficients, offsets, periodic = _cached(operator, 'fine_grid', signature,
        lambda cp: (cp.asarray(operator.coef_p), cp.asarray(offsets), cp.asarray(periodic)))
    nx, ny, nz = x.shape[-3:]
    _checked_size(*x.shape)
    if (x.size+255)//256 > 2**31-1:
        raise ValueError('Fine-grid gradient launch exceeds CUDA grid size')
    if x.size:
        _kernel()(((x.size+255)//256,), (256,),
                  (x, out, coefficients, offsets, periodic, np.int64(nx), np.int64(ny),
                   np.int64(nz), np.int64(operator.npoints), np.int64(x.size)))


def get_rho(n, gradients, tau):
    """NumPy in/out, with the same channel layout as get_rho_sxg."""
    cp = cupy()
    device_n = cp.ascontiguousarray(cp.asarray(n, dtype=cp.float64))
    result = cp.empty((len(n), 5) + n.shape[1:], dtype=cp.float64)
    result[:, 0] = device_n
    # Each output spin/channel slice is contiguous in the supported nspin=1
    # backend; use independent compact arrays to retain the general spin API.
    for v, gradient in enumerate(gradients):
        out = cp.empty_like(device_n)
        apply(gradient.__self__, device_n, out)
        result[:, v+1] = out
    result[:, 4] = cp.asarray(tau, dtype=cp.float64)
    return cp.asnumpy(result)


def add_correction(gradients, dedrho, v):
    """Subtract x, y, z derivatives in the original order; update v in place."""
    cp = cupy()
    out = cp.ascontiguousarray(cp.asarray(v, dtype=cp.float64))
    scratch = cp.empty_like(out)
    for direction, gradient in enumerate(gradients):
        x = cp.ascontiguousarray(cp.asarray(dedrho[:, direction+1], dtype=cp.float64))
        apply(gradient.__self__, x, scratch)
        out -= scratch
    v[:] = cp.asnumpy(out)
