"""FP64 libxc 7.0.0 LDA_X / PBE first derivatives (NumPy in/out).

The generated algebra and work_gga/work_lda screening rules are retained.
In particular changing dens_threshold does not change sigma_threshold.
The latter is derived from the functional's *default* density threshold.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from ciderpress.gpu import cupy

_FUNCTIONS = {1: "lda_x", 101: "gga_x_pbe", 130: "gga_c_pbe"}


@lru_cache(maxsize=None)
def _kernel(xcid, nspin):
    cp = cupy()
    name = _FUNCTIONS[xcid] + ("_unpol" if nspin == 1 else "_pol")
    source = Path(__file__).with_name("libxc_7.cuh").read_text()
    # Same default-sigma calculation as xc_func_init; set_dens_threshold
    # subsequently changes only the density threshold.
    sigma_floor = (float(1e-12 if xcid == 130 else 1e-15) ** (4.0 / 3.0)) ** 2
    source += r'''
extern "C" __global__ void baseline(
    const double* r, const double* s, double* e, double* v, double* w,
    const long long n, const double dt) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double rho[2] = {0, 0}, sigma[3] = {0, 0, 0};
    double zk[1] = {0}, vrho[2] = {0, 0}, vsigma[3] = {0, 0, 0};
    const double dens = NS == 1 ? r[i] : r[i] + r[n+i];
    if (!(dens < dt)) {
        rho[0] = dt < r[i] ? r[i] : dt;
        sigma[0] = SF < s[i] ? s[i] : SF;
        if (NS == 2) {
            rho[1] = dt < r[n+i] ? r[n+i] : dt;
            sigma[2] = SF < s[2*n+i] ? s[2*n+i] : SF;
            const double ave = 0.5 * (sigma[0] + sigma[2]);
            sigma[1] = s[n+i] >= -ave ? s[n+i] : -ave;
            sigma[1] = sigma[1] <= ave ? sigma[1] : ave;
        }
        FUNCTION(rho, sigma, zk, vrho, vsigma, dt);
    }
    e[i] = zk[0];
    for (int j=0; j<NS; ++j) v[j*n+i] = vrho[j];
    for (int j=0; j<2*NS-1; ++j) w[j*n+i] = vsigma[j];
}
'''
    source = source.replace("NS", str(nspin)).replace("SF", repr(sigma_floor)).replace("FUNCTION", name)
    return cp.RawKernel(source, "baseline", options=("--fmad=false",))


def evaluate_device(xcid, rho, sigma=None, *, dens_threshold=1e-12):
    """Return device (exc per particle, vrho, vsigma), channels first."""
    cp = cupy()
    nspin, size = rho.shape
    if xcid not in _FUNCTIONS or nspin not in (1, 2):
        raise ValueError("Unsupported libxc GPU functional/spin count")
    if rho.dtype != np.float64 or (sigma is not None and sigma.dtype != np.float64):
        raise TypeError("GPU libxc requires float64 inputs")
    if sigma is not None and sigma.shape != (2 * nspin - 1, size):
        raise ValueError("Invalid libxc sigma shape")
    r = cp.ascontiguousarray(rho)
    s = cp.zeros((2 * nspin - 1, size)) if sigma is None else cp.ascontiguousarray(sigma)
    e, v, w = cp.empty(size), cp.empty_like(r), cp.empty_like(s)
    if size:
        _kernel(xcid, nspin)(((size + 127) // 128,), (128,),
                           (r, s, e, v, w, np.int64(size), np.float64(dens_threshold)))
    return e, v, w


def evaluate(xcid, rho, sigma=None, *, dens_threshold=1e-12):
    """libxc per-particle energy; potentials differentiate energy/volume.

    Matches get_libxc_{lda,gga}_baseline including Fortran output order.
    This helper intentionally does not multiply exc by the original density;
    get_libxc_baseline does so after dispatch, exactly as on the CPU.
    """
    cp = cupy()
    e, v, w = evaluate_device(xcid, cp.asarray(rho), None if sigma is None else cp.asarray(sigma),
                              dens_threshold=dens_threshold)
    result = (e.get(), np.asfortranarray(v.get()))
    if sigma is not None:
        result += (np.asfortranarray(w.get()),)
    return result
