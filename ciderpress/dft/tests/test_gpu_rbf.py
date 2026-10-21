"""GPU/C oracle checks, including mutable model and output accumulation."""
import numpy as np
import pytest

from ciderpress import gpu
from ciderpress.dft.xc_evaluator import RBFEvaluator
from ciderpress.models.kernels import DiffRBF


def test_cpupy_never_enables(monkeypatch):
    gpu.disable()
    monkeypatch.setenv("GPAW_CPUPY", "1")
    with pytest.raises(RuntimeError, match="CUDA device"):
        gpu.enable()
    assert not gpu.use_gpu("rbf")


@pytest.fixture
def cuda(monkeypatch):
    if gpu._real_cupy() is None:
        pytest.skip("requires real CUDA CuPy")
    monkeypatch.setenv("CIDER_GPU_KERNELS", "rbf")
    gpu.enable()
    yield gpu.cupy()
    gpu.disable()


@pytest.mark.parametrize("nfeat", [5, 6])
@pytest.mark.parametrize("n", [0, 1, 129])
def test_rbf_c_oracle(cuda, monkeypatch, nfeat, n):
    rng = np.random.default_rng(2026)
    ev = RBFEvaluator(
        DiffRBF(np.linspace(0.2, 1.0, nfeat)),
        rng.normal(size=(67, nfeat)),
        rng.normal(size=67),
    )
    x = rng.normal(size=(n, nfeat))
    initial = rng.normal(size=n), rng.normal(size=(n, nfeat))
    monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
    ref = ev(x, *(a.copy() for a in initial))
    monkeypatch.setenv("CIDER_GPU_KERNELS", "rbf")
    a, b = (v.copy() for v in initial)
    out = ev(x, a, b)
    assert out[0] is a and out[1] is b
    for value, expected in zip(out, ref):
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-14)
    repeated = ev(x, *(v.copy() for v in initial))
    assert all(v.tobytes() == w.tobytes() for v, w in zip(out, repeated))


def test_rbf_cache_invalidation(cuda, monkeypatch):
    rng = np.random.default_rng(5)
    ev = RBFEvaluator(DiffRBF(np.ones(5)), rng.random((31, 5)), rng.random(31))
    x = ev._X1ctrl.copy()
    ev(x)
    for name in ("_X1ctrl", "_alpha", "_exps"):
        for replace in (False, True):
            arr = getattr(ev, name)
            if replace:
                setattr(ev, name, arr.copy())
            getattr(ev, name).flat[0] += 0.25
            got = ev(x)
            monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
            ref = ev(x)
            monkeypatch.setenv("CIDER_GPU_KERNELS", "rbf")
            for v, w in zip(got, ref):
                np.testing.assert_allclose(v, w, rtol=1e-12, atol=1e-14)


def test_rbf_large_common_offset(cuda, monkeypatch):
    # An expanded squared distance loses the small separation at this scale.
    rng = np.random.default_rng(81)
    ev = RBFEvaluator(
        DiffRBF(np.ones(5)), 1e8 + rng.random((31, 5)), rng.normal(size=31)
    )
    x = 1e8 + rng.random((129, 5))
    got = ev(x)
    monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
    ref = ev(x)
    for value, expected in zip(got, ref):
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-14)
