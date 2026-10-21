"""Fused fine-grid spline interpolation behind the NumPy engine interface.

Only private padded FFT storage and transient coefficients live on device.
Arguments, features, derivative outputs and returned scratch remain NumPy.
The CPU engine and its PAW stages are unchanged.
"""
from functools import lru_cache

import numpy as np

from ciderpress.gpu import cupy, use_gpu
from ciderpress.gpu.pasdw import _cached, _checked_size, _signature


_SOURCE = r'''
extern "C" __global__ void fill(
    double* work, const double* arg, const double* fun, const double* table,
    long long ng, long long nq, long long nz, long long lda, int add) {
    long long t = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (t >= ng*nq) return;
    long long g = t/nq, q = t%nq, cell = (long long)arg[g];
    double d = arg[g]-cell;
    const double* w = table+(cell*nq+q)*4;
    double p = w[0]+d*(w[1]+d*(w[2]+d*w[3]));
    long long k = (g/nz*lda+g%nz)*nq+q;
    double x = fun[g]*p;
    if (add) work[k] += x;
    else work[k] = x;
}
extern "C" __global__ void contract(
    const double* work, const double* arg, const double* table,
    double* feat, double* dfeat, long long ng, long long nq,
    long long nz, long long lda) {
    long long g = (long long)blockDim.x*blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    long long cell = (long long)arg[g];
    double d = arg[g]-cell, v = 0, dv = 0;
    long long k = (g/nz*lda+g%nz)*nq;
    for (long long q = 0; q < nq; q++) {
        const double* w = table+(cell*nq+q)*4;
        double p = w[0]+d*(w[1]+d*(w[2]+d*w[3]));
        double dp = w[1]+d*(2*w[2]+d*3*w[3]);
        v += work[k+q]*p;
        dv += work[k+q]*dp;
    }
    feat[g] = v;
    dfeat[g] = dv;
}
'''


@lru_cache(None)
def _kernel(name):
    return cupy().RawKernel(_SOURCE, name, options=('--fmad=false',))


def _table(plan, i):
    table = plan._local_alpha_transform[i]
    return _cached(plan, ('spline', i), _signature(table), lambda cp: cp.asarray(table))


def _blocks(size):
    blocks = (int(size)+255)//256
    if blocks > 2**31-1:
        raise ValueError('Grid interpolation launch exceeds CUDA grid size')
    return blocks


def fill(work, plan, arg, fun, i=-1, add=False):
    """Generate spline coefficients and fill the private padded FFT buffer."""
    cp = cupy()
    n = arg.size
    _checked_size(n, work.nq)
    if n:
        _kernel('fill')((_blocks(n*work.nq),), (256,),
            (work.raw, cp.asarray(arg, dtype=cp.float64).ravel(),
             cp.asarray(fun, dtype=cp.float64).ravel(), _table(plan, i),
             np.int64(n), np.int64(work.nq), np.int64(work.shape[2]),
             np.int64(work.lda), np.int32(add)))


def contract(work, plan, arg, i):
    """Return feature and argument derivative, with ordered alpha reductions."""
    cp = cupy()
    n = arg.size
    value, derivative = cp.empty(arg.shape), cp.empty(arg.shape)
    _checked_size(n, work.nq)
    if n:
        _kernel('contract')((_blocks(n),), (256,),
            (work.raw, cp.asarray(arg, dtype=cp.float64).ravel(), _table(plan, i), value, derivative,
             np.int64(n), np.int64(work.nq), np.int64(work.shape[2]), np.int64(work.lda)))
    return cp.asnumpy(value), cp.asnumpy(derivative)


def supported(engine, grad_mode):
    from ciderpress.dft.plans import NLDFSplinePlan
    p = engine._plan
    return (grad_mode == 0 and use_gpu('grid_nldf') and use_gpu('nldf_fft')
            and isinstance(p, NLDFSplinePlan) and p.coef_order == 'gq'
            and p.nldf_settings.nldf_type == 'j' and p.local_nalpha == p.nalpha
            and p.num_vj > 0 and p.num_vj == p.nldf_settings.nfeat
            and engine.fft_obj.comm.size == 1)


def forward(engine, rho_tuple):
    plan, fft = engine._plan, engine.fft_obj
    engine._set_paw_terms()
    arg, darg = plan.get_interpolation_arguments(rho_tuple, i=-1)
    fun, dfun = plan.get_function_to_convolve(rho_tuple)
    features = np.empty((plan.nspin, plan.nldf_settings.nfeat) + arg.shape[1:])
    derivatives = np.empty((plan.nspin, plan.num_vj) + arg.shape[1:])
    work = fft._gpu_work()
    p = dp = None
    arguments = {}
    # Forward and reverse consume the same immutable density tuple within
    # calc_cider. Retain its CPU exponent derivatives for this one evaluation.
    engine._grid_nldf_arguments = (rho_tuple, arguments)
    for s in range(plan.nspin):
        fill(work, plan, arg[s], fun[s])
        engine.set_fft_work(s, pot=False)
        fft.compute_forward_convolution()
        engine.get_fft_work(s, pot=False)
        for i in range(plan.num_vj):
            a, deriv = plan.get_interpolation_arguments(tuple(x[s] for x in rho_tuple), i=i)
            arguments[s, i] = a, deriv
            features[s, i], derivatives[s, i] = contract(work, plan, a, i)
        # Preserve the existing NumPy scratch return values, including layout.
        # One CPU coefficient evaluation replaces all intermediate large copies.
        p, dp = plan.get_interpolation_coefficients(a, i=i, vbuf=p, dbuf=dp)
        p.shape = dp.shape = a.shape + (plan.nalpha,)
    return features, derivatives, arg, darg, fun, dfun, p, dp


def backward(engine, rho_tuple, vfeat, dfeat, arg, fun, v, vsigma, vtau,
             p_gq=None, dp_gq=None):
    plan, fft = engine._plan, engine.fft_obj
    da, df = np.empty(v.shape), np.empty(v.shape)
    engine._calculate_paw_energy_and_potential()
    work = fft._gpu_work()
    saved = engine.__dict__.pop('_grid_nldf_arguments', None)
    arguments = saved[1] if saved is not None and saved[0] is rho_tuple else {}
    for s in range(plan.nspin):
        fft.reset_work()
        for i in range(plan.num_vj):
            if (s, i) in arguments:
                a, deriv = arguments[s, i]
            else:
                a, deriv = plan.get_interpolation_arguments(tuple(x[s] for x in rho_tuple), i=i)
            fill(work, plan, a, vfeat[s, i], i=i, add=True)
            tmp = dfeat[s, i] * vfeat[s, i]
            v[s] += deriv[0] * tmp
            vsigma[2*s] += deriv[1] * tmp
            if len(rho_tuple) == 3:
                vtau[s] += deriv[2] * tmp
        engine.set_fft_work(s, pot=True)
        fft.compute_backward_convolution()
        engine.get_fft_work(s, pot=True)
        df[s], da[s] = contract(work, plan, arg[s], -1)
        # Retain the incoming public NumPy scratch buffers' in-place semantics.
        p, dp = plan.get_interpolation_coefficients(arg[s], i=-1,
                                                   vbuf=p_gq, dbuf=dp_gq)
        p.shape = dp.shape = fun.shape[-3:] + (plan.nalpha,)
    da *= fun
    engine._get_dH_asp()
    return da, df
