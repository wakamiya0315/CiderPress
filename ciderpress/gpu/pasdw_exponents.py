"""Version-j spline arguments on device, using the host plan's formulas."""
import numpy as np
from functools import lru_cache

from ciderpress.gpu import cupy
from ciderpress.gpu.pasdw import _checked_size


_SOURCE = r'''
__device__ double saturate_exponent(double a, double cut, double &derivative) {
    double x = a/cut, power = 1, total = 0, dtotal = 0;
    for (long long m = 1; m <= 12; m++) {
        dtotal += power;
        power *= x;
        total += power/m;
    }
    derivative = exp(-total)*dtotal;
    return -cut*expm1(-total);
}
extern "C" __global__ void arguments(
    const double* rho, double* out, int* large, long long ng, long long mgga,
    double rhocut, double b, double c, double grad, double cut, long long spin_scale,
    long long smooth, long long etb, double ratio, double inva, double scale, long long sizem1) {
    long long g = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    double raw = rho[g];
    bool low = raw < rhocut;
    double n = (double)spin_scale*(low ? rhocut : raw);
    double sigma = rho[ng+g]*rho[ng+g] + rho[2*ng+g]*rho[2*ng+g];
    sigma += rho[3*ng+g]*rho[3*ng+g];
    if (spin_scale == 2) sigma *= 4;
    if (low) sigma = 0;
    double a = b*pow(n, 2.0/3.0), dn = 2*b/(3*pow(n, 1.0/3.0));
    double ds = 0, dt = 0;
    if (mgga) {
        double tau = low ? 0 : (double)spin_scale*rho[4*ng+g];
        a += c*tau/n;
        dn -= (c*tau/n)/n;
        dt = c/n;
    }
    if (grad != 0) {
        a += grad*sigma/(n*n);
        ds = grad/(n*n);
        dn -= 2*grad*sigma/(n*n*n);
    }
    if (spin_scale == 2) { dn *= 2; ds *= 4; dt *= 2; }
    if (low) dn = ds = dt = 0;
    if (smooth) {
        double derivative;
        a = saturate_exponent(a, cut, derivative);
        dn *= derivative; ds *= derivative; dt *= derivative;
    }
    large[g] = raw > rhocut && a > cut;
    double arg, derivative;
    if (etb) {
        derivative = ratio/a;
        arg = log(a*inva)*ratio;
    } else {
        derivative = ratio*inva/(a*inva+1);
        arg = log(a*inva+1)*ratio;
    }
    arg *= scale;
    derivative *= scale;
    if (!(arg > 0)) { arg = 0; derivative = 0; }
    if (!(arg < sizem1)) { arg = sizem1-1e-10; derivative = 0; }
    out[g] = arg;
    out[ng+g] = dn*derivative;
    out[2*ng+g] = ds*derivative;
    if (mgga) out[3*ng+g] = dt*derivative;
}
extern "C" __global__ void theta_potential(
    const double* x, const double* rho, const double* arg, const double* fun,
    const double* da, const double* df, const double* table, double* v,
    long long ng, long long nq, long long nx) {
    long long g = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    long long cell = (long long)arg[g];
    double d = arg[g]-cell, va = 0, vf = 0;
    for (long long q = 0; q < nq; q++) {
        const double* w = table + (cell*nq+q)*4;
        double p = w[0]+d*(w[1]+d*(w[2]+d*w[3]));
        double dp = w[1]+d*(2*w[2]+d*3*w[3]);
        vf += p*x[g*nq+q];
        va += dp*x[g*nq+q];
    }
    va *= fun[g];
    v[g] += va*da[g] + vf*df[g];
    double tmp = va*da[ng+g] + vf*df[ng+g];
    for (long long j = 1; j < 4; j++) v[j*ng+g] += (2*rho[j*ng+g])*tmp;
    if (nx == 5) v[4*ng+g] += va*da[2*ng+g] + vf*df[2*ng+g];
}
'''


@lru_cache(None)
def _kernel(name='arguments'):
    return cupy().RawKernel(_SOURCE, name, options=('--fmad=false',))


def arguments(plan, rho, i):
    """Device argument and all density/sigma/tau derivatives, in FP64."""
    from ciderpress.dft.settings import CFC

    cp = cupy()
    rho = cp.ascontiguousarray(cp.asarray(rho))
    params = plan.nldf_settings.theta_params if i == -1 else plan.nldf_settings.feat_params[i]
    mgga = plan.nldf_settings.sl_level == 'MGGA'
    a0, grad_mul = params[:2]
    tau_mul = params[2] if mgga else 0.
    tau_fac = tau_mul*1.2*(6*np.pi**2)**(2./3)/np.pi
    b = np.pi/2**(2./3)*(a0-tau_fac)
    c = np.pi/2**(2./3)*tau_fac/CFC
    spin_scale = plan.nspin if mgga else 1
    if not mgga:
        b = np.pi/(2**(2./3) if plan.nspin == 1 else 1.)*a0
    g = 0.
    if grad_mul != 0:
        g = grad_mul*1.2*(6*np.pi**2)**(2./3)/np.pi
        g *= np.pi/2**(2./3)*.125/CFC
    ratio, inva = 1./np.log(plan.lambd), 1./plan.alpha0
    scale = (plan._spline_size-1)/(plan.nalpha-1)
    ng = rho.shape[-1]
    out = cp.empty((4 if mgga else 3, ng), dtype=cp.float64)
    large = cp.empty(ng, dtype=cp.int32)
    _checked_size(*rho.shape)
    _checked_size(ng, plan.nalpha, 4)
    if (ng+255)//256 > 2**31-1:
        raise ValueError("PASDW launch exceeds CUDA grid size")
    if ng:
        _kernel()(((ng+255)//256,), (256,), (
            rho, out, large, np.int64(ng), np.int64(mgga),
            *(np.float64(v) for v in (plan.rhocut, b, c, g, np.max(plan.alphas))),
            np.int64(spin_scale), np.int64(plan._use_smooth_expnt_cutoff), np.int64(plan.alpha_formula == 'etb'),
            *(np.float64(v) for v in (ratio, inva, scale)), np.int64(plan._spline_size-1)))
    if plan._raise_large_expnt_error and bool(cp.any(large)):
        raise RuntimeError('NLDF exponent is too large! Please increase nalpha/alpha_max.')
    return out[0], tuple(out[j] for j in range(1, len(out)))


def theta_potential(plan, grid, rho, arg, fun, da, df, out):
    """Add the theta adjoint directly to a private device potential buffer."""
    from ciderpress.gpu.pasdw import _theta_table

    cp = cupy()
    ng, nq = grid.shape
    nx = rho.shape[0]
    assert out.shape == rho.shape and out.dtype == np.float64 and out.flags.c_contiguous
    _checked_size(*rho.shape)
    _checked_size(ng, plan.nalpha, 4)
    if (ng+255)//256 > 2**31-1:
        raise ValueError("PASDW launch exceeds CUDA grid size")
    if ng:
        _kernel('theta_potential')(((ng+255)//256,), (256,), (
            cp.ascontiguousarray(grid), cp.asarray(rho), cp.asarray(arg), cp.asarray(fun),
            cp.asarray(np.stack(da)), cp.asarray(np.stack(df)), _theta_table(plan), out,
            np.int64(ng), np.int64(nq), np.int64(nx)))
    return out
