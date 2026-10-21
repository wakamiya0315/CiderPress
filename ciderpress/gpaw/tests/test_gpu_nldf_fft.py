"""LibCiderPW C-oracle tests, including the shared PASDW work buffer."""

import numpy as np
import pytest
from gpaw.mpi import world

from ciderpress import gpu
from ciderpress.gpaw.nldf_interface import LibCiderPW


def make_pw(shape, nq):
    cell = np.array([[7., 0.4, 0.2], [0.2, 8., 0.1], [0.3, 0.2, 9.]])
    pw = LibCiderPW(shape, cell, world)
    pw.nalpha = pw.nbeta = nq
    a = np.geomspace(0.1, 12., nq)
    pw.expnts_ab = np.ascontiguousarray(1/(4*(a[:, None]+a)))
    pw.norms_ab = np.ascontiguousarray((np.pi/(a[:, None]+a))**1.5)
    pw.initialize_backend()
    pw.reset_work()
    return pw


def active_work(pw):
    raw = pw.copy_work_array()[:int(pw.get_work_size())].view(np.float64)
    nx, ny, nz = pw.shape
    return raw[:nx*ny*int(pw._glda)*pw.nalpha].reshape(
        nx, ny, int(pw._glda), pw.nalpha)[:, :, :nz].copy()


@pytest.fixture
def cuda(monkeypatch):
    if gpu._real_cupy() is None:
        pytest.skip("requires real CUDA CuPy")
    gpu.enable()
    monkeypatch.setenv("CIDER_GPU_KERNELS", "nldf_fft")
    yield gpu.cupy()
    gpu.disable()


@pytest.mark.parametrize("shape,nq", [((8, 10, 12), 1), ((7, 9, 11), 3), ((8, 6, 10), 21)])
def test_c_oracle(cuda, monkeypatch, shape, nq):
    rng = np.random.default_rng(20261021)
    fun = rng.normal(size=shape)
    p = rng.normal(size=(*shape, nq))
    # Padded indices, including periodic duplicates in the scatter.
    lda = 2*(shape[2]//2+1)
    indices = np.array([0, lda+2, 0, lda+2, 3, 0], dtype=np.int64)
    funcs = rng.normal(size=(len(indices), nq))
    results = []
    for selection in ("none", "nldf_fft"):
        monkeypatch.setenv("CIDER_GPU_KERNELS", selection)
        pw = make_pw(shape, nq)
        rows = []
        pw.set_work(fun, p)
        rows.append(active_work(pw))
        pw.add_paw2grid(funcs, indices)
        rows.append(active_work(pw))
        gathered = np.empty_like(funcs)
        pw.set_grid2paw(gathered, indices)
        rows.append(gathered)
        theta = pw.copy_work_array()
        pw.compute_forward_convolution()
        rows.append(active_work(pw))
        feat = np.full(shape, np.nan)
        pw.fill_vj_feature_(feat, p)
        rows.append(feat)
        pw.reset_work()
        rows.append(active_work(pw))
        for factor in (1., -0.3, 0.2):
            pw.fill_vj_potential_(fun*factor, p)
        rows.append(active_work(pw))
        pw.compute_backward_convolution()
        rows.append(active_work(pw))
        # Stress and fftv2 use the C backend, with explicit state exchange.
        pw.fftv2(theta)
        pw.set_work(fun, p)
        rows.append(pw.compute_backward_convolution_with_stress(theta))
        rows.append(active_work(pw))
        results.append(rows)
    for i, (actual, expected) in enumerate(zip(results[1], results[0])):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-11,
                                   err_msg=f"operation {i}")


def test_switch_keeps_work(cuda, monkeypatch):
    pw = make_pw((8, 6, 10), 3)
    fun = np.ones(pw.shape)
    p = np.ones((*pw.shape, pw.nalpha))
    pw.set_work(fun, p)
    monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
    np.testing.assert_array_equal(active_work(pw), p)
    pw.fill_vj_potential_(fun, p)
    monkeypatch.setenv("CIDER_GPU_KERNELS", "nldf_fft")
    np.testing.assert_array_equal(active_work(pw), 2*p)
    # Repeated convolution is deterministic and a mutated NumPy input is read.
    pw.set_work(fun, p)
    pw.compute_forward_convolution()
    first = active_work(pw)
    p *= 2
    pw.set_work(fun, p)
    pw.compute_forward_convolution()
    np.testing.assert_array_equal(active_work(pw), 2*first)


@pytest.mark.parametrize("shape", [(8, 10, 12), (7, 9, 11)])
def test_numpy_convention(shape, monkeypatch):
    """Independent CPU check of skew-cell wrapping and inverse normalisation."""
    monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
    pw = make_pw(shape, 3)
    rng = np.random.default_rng(17)
    fun, p = rng.normal(size=shape), rng.normal(size=(*shape, 3))
    pw.set_work(fun, p)
    pw.compute_forward_convolution()
    ks = [(np.arange(n//2+1 if i == 2 else n)+n//2)%n-n//2
          for i, n in enumerate(shape)]
    k = np.stack(np.meshgrid(*ks, indexing="ij"), axis=-1)
    g = k @ (2*np.pi*np.linalg.inv(pw._cell_cv).T)
    k2 = np.sum(g*g, axis=-1)
    x = np.fft.rfftn(fun[..., None]*p, axes=(0, 1, 2))
    out = np.zeros_like(x)
    for b in range(3):
        for a in range(3):
            out[..., b] += x[..., a]*pw.norms_ab[a, b]*np.exp(-pw.expnts_ab[a, b]*k2)
    expected = np.fft.irfftn(out, s=shape, axes=(0, 1, 2))
    np.testing.assert_allclose(active_work(pw), expected, rtol=1e-12, atol=2e-12)
