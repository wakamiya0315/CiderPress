"""Single-domain FP64 NLDF convolution with a persistent padded work buffer.

The buffer has the C real-space layout (x, y, padded z, alpha). cuFFT uses
channel-major batches internally. C puts 1/N into the Gaussian kernel and
uses an unnormalised inverse; CuPy's normalised inverse supplies that factor.
No alpha-pair-by-grid table is materialised.
"""

from functools import lru_cache

import numpy as np

from ciderpress.gpu import cupy


_SOURCE = r'''
extern "C" __global__ void convolve(
    const double2* x, const double* k2, const double* norms,
    const double* exps, double2* out, long long nk, int nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nk*nq) return;
    int b = t/nk;
    long long k = t%nk;
    double re = 0.0, im = 0.0;
    for (int a = 0; a < nq; a++) {
        // Match compute_kernels_sym: upper triangle of norms_ba.
        int hi = a > b ? a : b, lo = a < b ? a : b;
        int ab = hi*nq + lo;
        double w = norms[ab] * exp(-exps[ab]*k2[k]);
        double2 v = x[(long long)a*nk+k];
        re += v.x*w;
        im += v.y*w;
    }
    out[t] = make_double2(re, im);
}
extern "C" __global__ void feature(
    const double* work, const double* p, double* out,
    long long ng, int nq, int nz, int lda) {
    long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= ng) return;
    long long padded = (g/nz*lda + g%nz)*nq;
    double v = 0.0;
    for (int q = 0; q < nq; q++) v += work[padded+q]*p[g*nq+q];
    out[g] = v;
}
extern "C" __global__ void scatter(
    double* work, const double* f, const long long* inds,
    const long long* order, const long long* starts,
    long long nu, int nq) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= nu*nq) return;
    long long u = t/nq, q = t%nq;
    long long target = inds[order[starts[u]]]*nq+q;
    double v = work[target];
    // Stable grouping retains input order even for periodic duplicate grids.
    for (long long j = starts[u]; j < starts[u+1]; j++)
        v += f[order[j]*nq+q];
    work[target] = v;
}
'''


@lru_cache(None)
def _kernel(name):
    return cupy().RawKernel(_SOURCE, name, options=("--fmad=false",))


def _launch(name, size, args):
    if size:
        _kernel(name)(((int(size)+255)//256,), (256,), args)


class DeviceWork:
    def __init__(self, owner, host):
        cp = cupy()
        self.device = cp.cuda.Device().id
        self.shape = owner.shape
        self.nq = owner.nalpha
        self.lda = int(owner._glda)
        self.raw = cp.asarray(host)
        nx, ny, nz = self.shape
        self.padded = self.raw[:nx*ny*self.lda*self.nq].reshape(
            nx, ny, self.lda, self.nq)
        self.real = self.padded[:, :, :nz]
        self.norms = cp.asarray(owner._fft_norms)
        self.exps = cp.asarray(owner._fft_expnts)
        # C wraps *all* axes, including the even-z Nyquist plane to -Nz/2.
        ks = [(np.arange(n//2+1 if i == 2 else n)+n//2)%n-n//2
              for i, n in enumerate(self.shape)]
        reciprocal = 2*np.pi*np.linalg.inv(owner._cell_cv).T
        k = np.meshgrid(*ks, indexing="ij")
        xyz = [sum(reciprocal[i, v]*k[i] for i in range(3)) for v in range(3)]
        self.k2 = cp.asarray(sum(v*v for v in xyz))
        self.memory_samples = []

    def _sample_memory(self, stage):
        cp = cupy()
        free, total = cp.cuda.runtime.memGetInfo()
        self.memory_samples.append(dict(stage=stage, device_used_bytes=total-free,
            pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
            pool_reserved_bytes=cp.get_default_memory_pool().total_bytes()))

    def sync(self):
        cupy().cuda.get_current_stream().synchronize()

    def convolve(self):
        cp = cupy()
        measure = not self.memory_samples
        if measure:
            self._sample_memory("before_fft")
        x = cp.fft.rfftn(cp.ascontiguousarray(self.real.transpose(3, 0, 1, 2)),
                         axes=(1, 2, 3))
        out = cp.empty_like(x)
        if measure:
            self._sample_memory("spectra_allocated")
        _launch("convolve", x.size, (x, self.k2, self.norms, self.exps, out,
                np.int64(self.k2.size), np.int32(self.nq)))
        real = cp.fft.irfftn(out, s=self.shape, axes=(1, 2, 3))
        if measure:
            self._sample_memory("inverse_allocated")
        self.real[:] = real.transpose(1, 2, 3, 0)
        self.sync()

    def set_work(self, fun, p, add=False):
        value = cupy().asarray(fun)[..., None]*cupy().asarray(p)
        if add:
            self.real += value
        else:
            self.real[:] = value
        self.sync()

    def feature(self, out, p):
        cp = cupy()
        dest = cp.empty(out.shape, dtype=cp.float64)
        _launch("feature", out.size, (self.raw, cp.asarray(p), dest,
                np.int64(out.size), np.int32(self.nq),
                np.int32(self.shape[2]), np.int32(self.lda)))
        dest.get(out=out)

    def reset(self):
        self.raw.fill(0)
        self.sync()

    def scatter(self, funcs, indices):
        cp = cupy()
        order = np.argsort(indices, kind="stable").astype(np.int64)
        starts = np.r_[0, 1+np.flatnonzero(np.diff(indices[order])), len(indices)]
        if not indices.size:
            return
        _launch("scatter", (len(starts)-1)*self.nq,
                (self.raw, cp.asarray(funcs), cp.asarray(indices), cp.asarray(order),
                 cp.asarray(starts.astype(np.int64)), np.int64(len(starts)-1), np.int32(self.nq)))
        self.sync()

    def gather(self, funcs, indices):
        cupy().asnumpy(self.padded.reshape(-1, self.nq)[cupy().asarray(indices)], out=funcs)
