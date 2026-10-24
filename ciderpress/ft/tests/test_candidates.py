"""Candidate-only stratification, legacy identity and device coverage."""
import numpy as np
import pytest
from ciderpress.ft import FineTuner, Constraints, ReactionSet
from ciderpress.ft.tests.test_ft import synthetic, unconstrained, add_known_labels


class LegacyTuner(FineTuner):
    def _select_controls(self, kernel, records, record_ids=None):
        budget = max(1, self.max_candidates // len(records) // 2)
        pool = []
        for record in records:
            valid = np.flatnonzero(record["rho_data"][:, 0].sum(0) > self.rhocut)
            if valid.size:
                ids = valid[np.linspace(0, valid.size-1, min(budget, valid.size), dtype=int)]
                sample = self._base.settings.normalizers.get_normalized_feature_vector(record["desc"][..., ids])
                pool.append(kernel.get_descriptors(sample))
        pool = np.concatenate(pool)
        if len(pool) > self.max_candidates:
            pool = pool[np.linspace(0, len(pool)-1, self.max_candidates, dtype=int)]
        kernel.X1ctrl = kernel._reduce_npts(pool)


@pytest.mark.parametrize("component", ["x", "c", "xc"])
def test_default_matches_legacy_fit_bitwise(component, tmp_path):
    base, data = synthetic(nspin=2, nsystems=3)
    kwargs = dict(n_control=8, max_candidates=100, constraints=unconstrained(), components=component)
    legacy = LegacyTuner(base, **kwargs)
    add_known_labels(data, legacy)
    legacy.fit(data)
    default = FineTuner(base, **kwargs).fit(data)
    for a, b in zip(legacy.engine.kernels, default.engine.kernels):
        np.testing.assert_array_equal(a.X1ctrl, b.X1ctrl)
        np.testing.assert_array_equal(a.alpha, b.alpha)
    assert default.predict(data) == legacy.predict(data)
    saved=default.save(tmp_path/'default.yaml')
    from ciderpress.dft.model_utils import load_cider_model
    assert 'candidate_groups' not in load_cider_model(saved).ft_metadata
    assert 'group_fractions' not in load_cider_model(saved).ft_metadata


@pytest.mark.parametrize("nspin,component", [(1,"x"), (2,"x"), (2,"c")])
def test_group_budgets_ueg_and_determinism(nspin, component):
    base, data = synthetic(nspin=nspin, nsystems=6)
    groups = {k: "periodic" if k == "0" else "molecular" for k in data.records}
    ft = FineTuner(base, n_control=8, max_candidates=80, components=component,
                   candidate_groups=groups, group_fractions={"periodic":.5,"molecular":.5})
    ueg = list(ft._ueg_records().values())
    records = list(data.records.values()) + ueg
    ids = list(data.records) + [None]*len(ueg)
    k = ft._kernels()[0]
    info = ft._select_controls(k, records, ids)
    assert info == ft._select_controls(k, records, ids)
    for group in groups.values():
        rr = [r for r in info["records"] if r["group"] == group]
        assert len({r["grid_budget"] for r in rr}) == 1
        assert sum(r["candidate_rows"] for r in rr) <= 40
    assert sum(r["candidate_rows"] for r in info["records"] if r["group"] == "__ueg__") == (2 if component == "x" else 1)*len(ueg)
    assert len(info["selected"]) == k.Nctrl
    assert all(r["group"] in ("periodic","molecular","__ueg__") for r in info["selected"])
    with pytest.raises(ValueError, match="Missing candidate group"):
        ft._select_controls(k, records, ["missing"] + ids[1:])


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_grouped_fit_reload_and_protection(device, tmp_path):
    if device == "gpu":
        cp = pytest.importorskip("cupy")
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    base, data = synthetic(nspin=2, nsystems=3)
    seed = FineTuner(base, n_control=8, max_candidates=100, constraints=unconstrained())
    add_known_labels(data, seed)
    protection = ReactionSet.from_records({"p": data.records["2"]})
    ft = FineTuner(base, n_control=8, max_candidates=100, device=device,
        candidate_groups={"0":"periodic", "1":"molecular", "2":"molecular", "p":"molecular"},
        group_fractions={"periodic":.5,"molecular":.5},
        constraints=Constraints(molecules=protection)).fit(data)
    loaded = FineTuner.load(ft.save(tmp_path / "model.yaml"), device=device)
    assert loaded.candidate_groups == ft.candidate_groups
    assert loaded.group_fractions == ft.group_fractions
    assert loaded.fit_info == ft.fit_info
    assert loaded.predict(data) == ft.predict(data)
    assert "p" in {r["record_id"] for r in ft.fit_info["candidate_pools"][0]["records"]}


@pytest.mark.parametrize("kwargs", [
    {"candidate_groups":{}}, {"group_fractions":{"a":1}},
    {"candidate_groups":{"0":"a"}, "group_fractions":{"a":.5}},
    {"candidate_groups":{"0":"b"}, "group_fractions":{"a":1}},
    {"candidate_groups":{"0":"a"}, "group_fractions":{"a":float("nan")}},
])
def test_invalid_candidate_configuration(kwargs):
    with pytest.raises(ValueError):
        FineTuner(**kwargs)


@pytest.mark.parametrize("component", ["x", "c"])
def test_mixed_spin_records_share_candidate_row_budget(component):
    base, one = synthetic(nspin=1, nsystems=2)
    _, two = synthetic(nspin=2, nsystems=2)
    records = {"a":one.records["0"], "b":two.records["0"],
               "c":one.records["1"], "d":two.records["1"]}
    ft = FineTuner(base, n_control=8, max_candidates=48, components=component,
        candidate_groups=dict(a="periodic",b="periodic",c="molecular",d="molecular"),
        group_fractions=dict(periodic=.25,molecular=.75), constraints=unconstrained())
    info=ft._select_controls(ft._kernels()[0],list(records.values()),list(records))
    for group,limit in (("periodic",12),("molecular",36)):
        rr=[r for r in info["records"] if r["group"]==group]
        assert {r["candidate_budget"] for r in rr}=={limit//2}
        assert sum(r["candidate_rows"] for r in rr)==limit
    assert info["candidate_rows"]==48
