"""GPU libxc and mapped descriptors: spin, thresholds and NumPy contracts."""
import numpy as np
import pytest

from ciderpress import gpu


@pytest.fixture
def device(monkeypatch):
    monkeypatch.delenv("GPAW_CPUPY", raising=False)
    if gpu._real_cupy() is None:
        pytest.skip("Real CUDA device required")
    gpu.enable()
    yield gpu.cupy()
    gpu.disable()


def test_fake_device_never_enables(monkeypatch):
    gpu.disable()
    monkeypatch.setenv("GPAW_CPUPY", "1")
    with pytest.raises(RuntimeError, match="CUDA device"):
        gpu.enable()
    assert not gpu.use_gpu("libxc")
    assert not gpu.use_gpu("semilocal")


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize("xcid", [1, 101, 130])
def test_libxc_threshold_and_spin(device, monkeypatch, nspin, xcid):
    from ciderpress.dft.baselines import get_libxc_lda_baseline, get_libxc_gga_baseline
    from ciderpress.gpu.libxc import evaluate
    rng = np.random.default_rng(9)
    n = np.r_[0., 1e-16, np.nextafter(1e-12, 0), 1e-12,
              np.nextafter(2e-12, 0), 2e-12, np.nextafter(2e-12, np.inf),
              10.**rng.uniform(-7, 3, 513)]
    r = np.array([n]) if nspin == 1 else np.array([.3*n, .7*n])
    s = np.full((2*nspin-1, len(n)), 1e-10)
    monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
    ref = get_libxc_lda_baseline(xcid, r) if xcid == 1 else get_libxc_gga_baseline(xcid, r, s)
    got = evaluate(xcid, r, None if xcid == 1 else s)
    for a, b in zip(got, ref):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=1e-12)
        assert a.flags.f_contiguous
        assert np.all(a[..., n < 1e-12] == 0.)
    empty = evaluate(xcid, r[:, :0], None if xcid == 1 else s[:, :0])
    assert all(a.size == 0 for a in empty)


@pytest.mark.parametrize("nspin", [1, 2])
@pytest.mark.parametrize('negative', [False, True])
def test_mapped_all_derivatives_and_model_mutation(device, monkeypatch, nspin, negative):
    from ciderpress.gpaw.new_backend import CiderXC
    model = CiderXC().mlfunc
    if negative:
        for kernel in model.kernels:
            for feature in kernel.feature_list.feat_list:
                feature.i = -1
    rng = np.random.default_rng(91)
    n = 129
    r = rng.uniform(.001, 3., (nspin, n))
    s = rng.uniform(0, 2., (2*nspin-1, n))
    t = rng.uniform(.01, 5., r.shape)
    x = np.concatenate([r[:, None], s[::2, None], t[:, None],
                        rng.uniform(0, 2, (nspin, 3, n))], axis=1)
    for mutate in (False, True):
        if mutate:
            # Existing device control cache must detect in-place changes.
            model.kernels[0].fevals[0]._alpha[0] *= 1.1
            model.kernels[0].feature_list[0].gamma *= 1.2
        monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
        ref = model(x, (r, s, t), rhocut=.1)
        monkeypatch.setenv("CIDER_GPU_KERNELS", "rbf,libxc,semilocal")
        got = model(x, (r, s, t), rhocut=.1)
        repeat = model(x, (r, s, t), rhocut=.1)
        for a, b, c in zip((got[0], got[1], *got[2]),
                           (ref[0], ref[1], *ref[2]),
                           (repeat[0], repeat[1], *repeat[2])):
            np.testing.assert_allclose(a, b, atol=1e-12*max(1, abs(b).max()), rtol=1e-12)
            assert a.tobytes() == c.tobytes()
            assert type(a) is np.ndarray


def test_unsupported_map_falls_back():
    from ciderpress.dft.transform_data import FeatureList, UMap
    from ciderpress.gpu.descriptors import signature
    assert signature(FeatureList([UMap(0, 1.)])) is None


@pytest.mark.parametrize("nspin", [1, 2])
def test_gpu_directional_derivative(device, monkeypatch, nspin):
    from ciderpress.gpaw.new_backend import CiderXC
    model = CiderXC().mlfunc
    monkeypatch.setenv("CIDER_GPU_KERNELS", "rbf,libxc,semilocal")
    rng = np.random.default_rng(623)
    n = 31
    r = rng.uniform(.1, 2, (nspin, n))
    sigma = rng.uniform(.01, .2, (2*nspin-1, n))
    if nspin == 2:
        # Stay inside the physical Gram cone: libxc clips inadmissible
        # cross-gradients but returns the unclipped functional derivatives.
        sigma[1] = .2*np.sqrt(sigma[0]*sigma[2])
    tau = rng.uniform(.1, 2, r.shape)
    x = rng.uniform(.1, 2, (nspin, 6, n))
    arrays = (x, r, sigma, tau)
    directions = tuple(.02*a*rng.normal(size=a.shape) for a in arrays)
    e, dx, vr = model(x, (r, sigma, tau))
    analytic = np.sum(dx*directions[0]) + sum(np.sum(v*d) for v,d in zip(vr,directions[1:]))
    eps = 2e-5
    plus = tuple(a+eps*d for a,d in zip(arrays,directions))
    minus = tuple(a-eps*d for a,d in zip(arrays,directions))
    fd = (model(plus[0], plus[1:])[0].sum() - model(minus[0], minus[1:])[0].sum())/(2*eps)
    np.testing.assert_allclose(fd, analytic, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('negative', [False, True])
def test_descriptor_adjoint(device, negative):
    from ciderpress.gpaw.new_backend import CiderXC
    from ciderpress.gpu.descriptors import forward, backward
    cp = device
    rng = np.random.default_rng(524)
    raw = rng.uniform(.1, 2, (6, 129))
    direction = rng.normal(size=raw.shape)
    for k in CiderXC().mlfunc.kernels:
        if negative:
            for feature in k.feature_list.feat_list:
                feature.i = -1
        dual = rng.normal(size=(129,k.N1))
        expected = k.feature_list(raw.T)
        np.testing.assert_allclose(forward(k.feature_list, cp.asarray(raw)).get(), expected,
                                   atol=1e-12, rtol=1e-12)
        # The aliased negative-index maps have higher curvature on this
        # input: 1e-5 has a measurable O(eps**2) truncation error on CPU too.
        eps = 1e-6
        jdx = (forward(k.feature_list,cp.asarray(raw+eps*direction))-
               forward(k.feature_list,cp.asarray(raw-eps*direction)))/(2*eps)
        jty = backward(k.feature_list,cp.asarray(raw),cp.asarray(dual))
        cpu_jty = np.zeros_like(raw)
        k.feature_list.fill_derivs_(cpu_jty, dual.T, raw)
        np.testing.assert_allclose(jty.get(), cpu_jty, atol=1e-12, rtol=1e-12)
        left, right = float(cp.sum(jdx*cp.asarray(dual))), float(cp.sum(cp.asarray(direction)*jty))
        np.testing.assert_allclose(left, right, atol=1e-8, rtol=1e-8)
