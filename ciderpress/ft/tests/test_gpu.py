"""Optional FT device path: explicit failures, FP64 parity and persistence."""

from dataclasses import replace

import numpy as np
import pytest

from ciderpress.ft import FineTuner
from ciderpress.ft.data import to_hartree
from ciderpress.ft.records import covariance, integrate
from ciderpress.ft.tests.test_ft import add_known_labels, synthetic, unconstrained


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
        cp.zeros(1, dtype=cp.float64)
    except Exception as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    return cp


def test_explicit_device_errors(monkeypatch):
    with pytest.raises(ValueError, match="device"):
        FineTuner(device="auto")
    import ciderpress.ft._gpu as gpu

    monkeypatch.setattr(gpu, "_real_cupy", lambda: None)
    with pytest.raises(RuntimeError, match="CuPy.*CUDA"):
        FineTuner(device="gpu")


def test_cpu_default_is_bitwise_unchanged(tmp_path):
    base, data = synthetic(nspin=2, nsystems=3)
    kwargs = dict(n_control=8, max_candidates=100, constraints=unconstrained())
    default = FineTuner(base, **kwargs)
    add_known_labels(data, default)
    default.fit(data)
    explicit = FineTuner(base, device="cpu", **kwargs).fit(data)
    np.testing.assert_array_equal(
        default.engine.kernels[0].alpha, explicit.engine.kernels[0].alpha
    )
    assert default.predict(data) == explicit.predict(data)
    a, b = tmp_path / "default.yaml", tmp_path / "cpu.yaml"
    default.save(a)
    explicit.save(b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("nspin,components", [(1, "x"), (2, "x"), (2, "c"), (2, "xc")])
def test_gpu_rows_fit_predict_reload(cuda, nspin, components, tmp_path, monkeypatch):
    base, data = synthetic(nspin=nspin, nsystems=4)
    kwargs = dict(
        n_control=10,
        max_candidates=100,
        blocksize=7,
        components=components,
        constraints=unconstrained(),
    )
    cpu = FineTuner(base, **kwargs)
    add_known_labels(data, cpu)
    # Resolve the residual above frozen-base FP64 roundoff (~1e-13 Ha), using
    # a pilot-sized signal and its 0.03 eV noise. The recovery helper's much
    # smaller signal tests CPU interpolation, not relative device fit accuracy.
    means = {k: cpu._base_total(r) for k, r in data.records.items()}
    reactions = []
    for r in data.reactions:
        mean = sum(c * means[k] for k, c in r.terms)
        reactions.append(
            replace(
                r,
                reference_ha=mean + 1000 * (r.reference_ha - mean),
                noise_ha=0.03 * to_hartree("eV"),
            )
        )
    data.reactions = reactions
    cpu.fit(data)
    # Explicit FT dispatch must work even when GPAW's ambient policy disables
    # all kernels. It calls the shared resident pipeline directly.
    monkeypatch.setenv("CIDER_GPU_KERNELS", "none")
    gpu = FineTuner(base, device="gpu", **kwargs).fit(data)
    for ck, gk in zip(cpu.engine.kernels, gpu.engine.kernels):
        np.testing.assert_array_equal(ck.X1ctrl, gk.X1ctrl)
        for key in ck.cov_dict:
            np.testing.assert_allclose(
                gk.cov_dict[key], ck.cov_dict[key], rtol=1e-10, atol=1e-14
            )
        np.testing.assert_allclose(gk.alpha, ck.alpha, rtol=1e-10, atol=1e-13)
        for record in data.records.values():
            row = covariance(ck, record, base.settings, 5, cpu.rhocut, device="gpu")
            ref = covariance(ck, record, base.settings, 5, cpu.rhocut)
            np.testing.assert_allclose(row, ref, rtol=1e-10, atol=1e-14)
    for record in data.records.values():
        np.testing.assert_allclose(
            gpu._base_total(record), cpu._base_total(record), rtol=1e-10, atol=1e-13
        )
        # Both zero cutoff and core=False exercise the direct quadrature API.
        np.testing.assert_allclose(
            integrate(base, record, blocksize=11, rhocut=0, core=False, device="gpu"),
            integrate(base, record, blocksize=11, rhocut=0, core=False),
            rtol=1e-10,
            atol=1e-13,
        )
    before = gpu.predict(data, unit="Hartree")
    cpu_prediction = cpu.predict(data, unit="Hartree")
    path = gpu.save(tmp_path / "gpu.yaml")
    loaded = FineTuner.load(path, device="gpu")
    host_loaded = FineTuner.load(path)
    assert host_loaded.device == "cpu"
    for prediction in (
        cpu_prediction,
        loaded.predict(data, unit="Hartree"),
        host_loaded.predict(data, unit="Hartree"),
    ):
        np.testing.assert_allclose(
            [list(v.values()) for v in prediction.values()],
            [list(v.values()) for v in before.values()],
            rtol=1e-10,
            atol=1e-13,
        )
