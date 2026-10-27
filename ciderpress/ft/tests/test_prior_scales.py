"""Opt-in component priors preserve scalar numerics and trusted YAML reload."""
import numpy as np
import pytest

from ciderpress.ft import FineTuner
from ciderpress.ft.tests.test_ft import synthetic, unconstrained, add_known_labels


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("components", ["x", "c", "xc"])
def test_component_priors_fit_reload(components, device, tmp_path):
    if device == "gpu":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    base, data = synthetic(nspin=2, nsystems=4)
    kwargs = dict(n_control=10, max_candidates=100, components=components,
                  constraints=unconstrained(), device=device)
    scalar = FineTuner(base, prior_scale=.1, **kwargs)
    add_known_labels(data, scalar)
    scalar.fit(data)
    same = FineTuner(base, prior_scale={c: .1 for c in components}, **kwargs).fit(data)
    for a, b in zip(scalar.engine.kernels, same.engine.kernels):
        np.testing.assert_array_equal(a.X1ctrl, b.X1ctrl)
        np.testing.assert_array_equal(a.alpha, b.alpha)
    assert scalar.predict(data) == same.predict(data)
    scales = {c: (.3 if c == "c" else .1) for c in components}
    ft = FineTuner(base, prior_scale=scales, **kwargs).fit(data)
    scales[components[0]] = 9  # constructor owns its mapping
    assert ft.prior_scale[components[0]] != 9
    path = ft.save(tmp_path / "prior.yaml")
    loaded = FineTuner.load(path, device=device)
    assert loaded.prior_scale == ft.prior_scale
    assert loaded.predict(data) == ft.predict(data)
    for a, b in zip(ft.model.kernels, loaded.model.kernels):
        for ea, eb in zip(a.fevals, b.fevals):
            for key in ("_X1ctrl", "_alpha", "_exps"):
                np.testing.assert_array_equal(getattr(ea, key), getattr(eb, key))
    again = FineTuner.load(loaded.save(tmp_path / "again.yaml"), device=device)
    assert again.predict(data) == ft.predict(data)
    loaded.fit(data)
    assert loaded.predict(data) == ft.predict(data)


@pytest.mark.parametrize("scales", [{}, {"x": .1}, {"x": .1, "c": 0},
                                    {"x": .1, "c": np.inf}, {"x": .1, "z": .3}])
def test_invalid_component_priors(scales):
    with pytest.raises(ValueError, match="prior_scale"):
        FineTuner(components="xc", prior_scale=scales)
