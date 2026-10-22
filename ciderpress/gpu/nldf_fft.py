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
extern "C" __global__ void strain_weight(
    const double2* theta, const double2* adjoint, const double* k2,
    const double* norms, const double* exps, double* weight,
    long long nk, int nq, int nzhalf) {
    long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= nk) return;
    double value = 0.0;
    for (int b = 0; b < nq; b++) {
        double re = 0.0, im = 0.0;
        for (int a = 0; a < nq; a++) {
            int hi = a > b ? a : b, lo = a < b ? a : b;
            int ab = hi*nq + lo;
            double dK = 2.0*norms[ab]*exps[ab]*exp(-exps[ab]*k2[k]);
            double2 v = theta[(long long)a*nk+k];
            re += v.x*dK;
            im += v.y*dK;
        }
        double2 y = adjoint[(long long)b*nk+k];
        value += y.x*re + y.y*im;
    }
    int z = k % nzhalf;
    // Native convention: the final stored plane has weight one for both
    // parities. New's odd-plane correction is selected separately below.
    weight[k] = value * ((z == 0 || z == nzhalf-1) ? 1.0 : 2.0);
}
extern "C" __global__ void strain_reduce(
    const double* weight, const double* G, double* partial,
    long long nk, int blocks, int odd_correction, int nzhalf) {
    __shared__ double values[256];
    int c = blockIdx.y;
    const int first[6] = {0,0,0,1,1,2};
    const int second[6] = {0,1,2,1,2,2};
    long long k = (long long)blockIdx.x*256 + threadIdx.x;
    double v = 0.0;
    if (k < nk) {
        v = weight[k]*G[(long long)first[c]*nk+k]*G[(long long)second[c]*nk+k];
        if (odd_correction && k % nzhalf == nzhalf-1) v *= 2.0;
    }
    values[threadIdx.x] = v;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[c*blocks+blockIdx.x] = values[0];
}
extern "C" __global__ void strain_finish(
    const double* partial, double* result, int blocks, double scale) {
    int c = threadIdx.x;
    if (c >= 6) return;
    double value = 0.0;
    for (int b = 0; b < blocks; b++) value += partial[c*blocks+b];
    result[c] = value*scale;
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
        self.transposed = owner.has_mpi
        self.volume = abs(np.linalg.det(owner._cell_cv))
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
        self._reciprocal = reciprocal
        k = np.meshgrid(*ks, indexing="ij")
        xyz = [sum(reciprocal[i, v]*k[i] for i in range(3)) for v in range(3)]
        self.k2 = cp.asarray(sum(v*v for v in xyz))
        self._stress_G = None
        self.memory_samples = []

    def fftv2(self, array):
        """In-place native reciprocal layout, including MPI's x/y transpose."""
        cp = cupy()
        nx, ny, nz = self.shape
        count = nx*ny*self.lda*self.nq
        real = cp.asarray(array.view(np.float64)[:count]).reshape(
            nx, ny, self.lda, self.nq)[:, :, :nz]
        spectrum = cp.fft.rfftn(real, axes=(0, 1, 2))
        if self.transposed:
            spectrum = spectrum.transpose(1, 0, 2, 3)
        native = cp.ascontiguousarray(spectrum).ravel()
        native.get(out=array[:native.size])
        self.raw[:2*native.size] = native.view(cp.float64)
        self.sync()

    def convolve_with_stress(self, theta, *, full_rfft=False):
        """Mode-2 dK contraction and reverse convolution, entirely in FP64.

        Theta is the NumPy native reciprocal buffer produced by fftv2.
        Only New's private property engine selects corrected odd-nz weights;
        the public/native interface retains its historical convention.
        """
        cp = cupy()
        nx, ny, nz = self.shape
        nk = int(self.k2.size)
        shape = ((ny, nx, nz//2+1, self.nq) if self.transposed else
                 (nx, ny, nz//2+1, self.nq))
        saved = cp.asarray(theta[:nk*self.nq]).reshape(shape)
        if self.transposed:
            saved = saved.transpose(1, 0, 2, 3)
        saved = cp.ascontiguousarray(saved.transpose(3, 0, 1, 2))
        adjoint = cp.fft.rfftn(cp.ascontiguousarray(self.real.transpose(3, 0, 1, 2)),
                               axes=(1, 2, 3))
        weight = cp.empty(nk, dtype=cp.float64)
        _launch('strain_weight', nk, (saved, adjoint, self.k2, self.norms,
                self.exps, weight, np.int64(nk), np.int32(self.nq), np.int32(nz//2+1)))
        if self._stress_G is None:
            ks = [(np.arange(n//2+1 if i == 2 else n)+n//2)%n-n//2
                  for i, n in enumerate(self.shape)]
            integers = np.stack(np.meshgrid(*ks, indexing='ij'), axis=-1)
            # k2 uses this same wrapped even-z Nyquist convention.
            reciprocal = np.asarray(self._reciprocal)
            self._stress_G = cp.asarray(np.ascontiguousarray(
                (integers @ reciprocal).reshape(-1, 3).T))
        blocks = (nk+255)//256
        partial = cp.empty((6, blocks), dtype=cp.float64)
        result = cp.empty(6, dtype=cp.float64)
        _kernel('strain_reduce')((blocks, 6), (256,),
            (weight, self._stress_G, partial, np.int64(nk), np.int32(blocks),
             np.int32(full_rfft and nz % 2), np.int32(nz//2+1)))
        _kernel('strain_finish')((1,), (32,),
            (partial, result, np.int32(blocks),
             np.float64(self.volume / np.prod(self.shape)**2)))
        out = cp.empty_like(adjoint)
        _launch('convolve', adjoint.size, (adjoint, self.k2, self.norms,
                self.exps, out, np.int64(nk), np.int32(self.nq)))
        self.real[:] = cp.fft.irfftn(out, s=self.shape, axes=(1, 2, 3)).transpose(1, 2, 3, 0)
        six = result.get()
        return six[np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])]

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
