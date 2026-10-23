"""Residual learning, units, safeguards and standard inference serialization."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from ciderpress.dft.model_utils import load_cider_model
from ciderpress.dft.plans import SemilocalPlan
from ciderpress.dft.xc_evaluator import RBFEvaluator
from ciderpress.ft import Constraints, FineTuner, ReactionSet
from ciderpress.ft.data import to_hartree
from ciderpress.ft.records import covariance, integrate
from ciderpress.gpu.semilocal import supported


def unconstrained():
    return Constraints(ueg=False, max_abs_correction=None, absolute_noise=None)


def synthetic(nspin=1, nsystems=7):
    model = load_cider_model("CIDER26XCSURFSCI")
    rng = np.random.default_rng(119)
    records = {}
    for i in range(nsystems):
        raw = rng.uniform(0.1, 0.9, (nspin, 5, 23))
        raw[:, 1:4] *= 0.15
        raw[:, 4] += 2
        raw[:, 0, :2] = (1e-10, 7e-9)  # exercise inference cutoff and spin scaling
        sl = SemilocalPlan(model.settings.sl_settings, nspin).get_feat(raw)
        nl = rng.uniform(0.2, 1, (nspin, model.settings.nldf_settings.nfeat, 23))
        records[str(i)] = {
            "desc": np.concatenate((sl, nl), axis=1),
            "rho_data": nspin * raw,
            "wt": np.r_[np.full(9, 0.2), np.ones(7), -np.ones(7)],
            "nspin": nspin,
            "exc_core": -0.7,
            "e_tot_orig": -10 + i * 0.01,
            "exc_orig": -2,
            "composition": "H2",
        }
    return model, ReactionSet.from_records(records)


def add_known_labels(data, ft, component="x"):
    kernel = ft._kernels()[0]
    ft._select_controls(kernel, list(data.records.values()))
    kernel.alpha = np.linspace(-0.025, 0.035, kernel.Nctrl)
    known = kernel.map(lambda k: RBFEvaluator(k.kernel, k.X1ctrl, k.alpha))
    from ciderpress.dft.xc_evaluator2 import MappedXC2

    correction = MappedXC2([known], ft.base.settings)
    totals = {
        k: ft._base_total(r) + integrate(correction, r, core=False, rhocut=ft.rhocut)
        for k, r in data.records.items()
    }
    for key in list(data.records)[1:]:
        data.add_reaction(
            f"r{key}",
            {key: 1, "0": -1},
            reference=totals[key] - totals["0"],
            unit="Hartree",
            noise=1e-7,
            group="odd" if int(key) % 2 else "even",
        )
    return totals


@pytest.mark.parametrize("nspin,component", [(1, "x"), (2, "x"), (2, "c")])
def test_known_correction_recovery_refit_and_mapping(nspin, component, tmp_path):
    base, data = synthetic(nspin)
    frozen = yaml.dump(base)
    ft = FineTuner(
        base,
        n_control=12,
        components=component,
        constraints=unconstrained(),
        max_candidates=160,
        jitter=1e-14,
        prior_scale=1,
    )
    add_known_labels(data, ft, component)
    ft.fit(data)
    report = ft.evaluate(data, unit="Hartree")
    assert report["overall"]["fine_tuned"]["MAE"] < 2e-7
    assert report["overall"]["base"]["MAE"] > 1e-5
    assert set(report["groups"]) == {"odd", "even"}
    assert yaml.dump(base) == frozen
    assert yaml.dump(ft.model.kernels[:2]) == yaml.dump(base.kernels)
    predictions = ft.predict(data)
    first = [k.alpha.copy() for k in ft.engine.kernels]
    ft.fit(data)
    for k, alpha in zip(ft.engine.kernels, first):
        np.testing.assert_array_equal(k.alpha, alpha)
    assert ft.predict(data) == predictions
    path = ft.save(tmp_path / "ft.yaml")
    mapped = load_cider_model(path)
    assert supported(mapped, base.settings.nfeat)
    assert mapped.kernels[-1]._add_basefunc is None
    loaded = FineTuner.load(path)
    assert loaded.evaluate(data) == ft.evaluate(data)
    for key, record in data.records.items():
        expected = integrate(base, record) + predictions[key][
            "correction"
        ] * to_hartree("eV")
        np.testing.assert_allclose(
            integrate(mapped, record), expected, atol=2e-13, rtol=1e-13
        )
        cov = covariance(ft.engine.kernels[0], record, base.settings, 7, ft.rhocut)
        np.testing.assert_allclose(
            cov @ ft.engine.kernels[0].alpha,
            predictions[key]["correction"] * to_hartree("eV"),
            atol=1e-13,
        )
    loaded.fit(data)
    assert loaded.predict(data) == predictions


def test_constraints_bound_ueg_and_molecular_protection(tmp_path):
    _, data = synthetic(nsystems=4)
    free = FineTuner(n_control=12, constraints=unconstrained(), max_candidates=120)
    data.add_reaction(
        "large",
        {"1": 1, "0": -1},
        reference=25,
        noise=0.001,
        unit="eV",
        group="surface",
    )
    free.fit(data)
    protected = ReactionSet.from_records({"molecule": data.records["3"]})
    constraints = Constraints(molecules=protected, max_abs_correction=0.02)
    ft = FineTuner(n_control=12, constraints=constraints, max_candidates=160).fit(data)
    assert ft.fit_info["max_abs_after_ha_per_atom"] <= 0.02000001 * to_hartree("eV")
    assert ft.fit_info["absolute_scale"] < 1
    assert ft.fit_info["max_ueg_ha_per_electron"] < 2e-4 * to_hartree("eV")
    free_molecule = abs(free.predict(protected)["molecule"]["correction"])
    assert abs(ft.predict(protected)["molecule"]["correction"]) < free_molecule * 0.01
    loaded = FineTuner.load(ft.save(tmp_path / "protected.yaml"))
    with pytest.raises(ValueError, match="molecular records"):
        loaded.fit(data)
    assert loaded.predict(protected) == ft.predict(protected)


def test_high_level_molecular_references():
    _, protect = synthetic(nsystems=1)
    ft = FineTuner(n_control=8, constraints=unconstrained(), max_candidates=100)
    mean = ft._base_total(protect.records["0"])
    protect.add_energy("0", reference=mean + 0.005, unit="Hartree", noise=1e-6)
    _, data = synthetic(nsystems=3)
    add_known_labels(data, ft)
    ft.constraints = Constraints(
        ueg=False, absolute_noise=None, max_abs_correction=None, molecules=protect
    )
    ft.fit(data)
    assert ft.predict(protect, unit="Hartree")["0"]["correction"] == pytest.approx(
        0.005, abs=1e-6
    )


def test_units_errors_and_splits():
    _, data = synthetic(nsystems=4)
    for unit in ("eV", "kcal/mol", "Hartree"):
        data.add_reaction(
            unit,
            {"1": 1, "0": -1},
            reference=0.5 / to_hartree(unit),
            noise=0.01 / to_hartree(unit),
            unit=unit,
            group=unit,
        )
    np.testing.assert_allclose([r.reference_ha for r in data.reactions], 0.5)
    np.testing.assert_allclose([r.noise_ha for r in data.reactions], 0.01)
    data.add_reaction(
        "separate_noise_unit",
        {"1": 1, "0": -1},
        reference=1,
        unit="eV",
        noise=0.001,
        noise_unit="Ha",
    )
    assert data.reactions[-1].noise_ha == 0.001
    for group, train, test in data.leave_one_group_out():
        assert all(r.group != group for r in train.reactions)
        assert all(r.group == group for r in test.reactions)
        assert "3" not in train.records
    parts = data.split_by_group(validation=["eV"], test=["Hartree"])
    assert parts["validation"].reactions[0].split == "validation"
    with pytest.raises(ValueError, match="disjoint"):
        data.split_by_group(validation=["eV"], test=["eV"])
    for terms, message in [
        ({"unknown": 1}, "Missing records"),
        ({"0": 1}, "Unbalanced"),
    ]:
        with pytest.raises(ValueError, match=message):
            data.add_reaction("bad", terms, reference=0, noise=0.1, unit="eV")
    with pytest.raises(ValueError, match="unit"):
        data.add_reaction("bad", {"1": 1, "0": -1}, reference=0, noise=1, unit="joules")
    with pytest.raises(ValueError, match="positive"):
        data.add_reaction("bad", {"1": 1, "0": -1}, reference=0, noise=0, unit="eV")
    bad = dict(data.records["0"], energy_unit="eV")
    with pytest.raises(ValueError, match="must be Hartree"):
        ReactionSet.from_records({"0": bad})
    with pytest.raises(ValueError, match="missing record field"):
        ReactionSet.from_records({"0": {}})
    with pytest.raises(ValueError, match="No records"):
        ReactionSet.from_records({})


def test_only_training_records_choose_controls_and_labels():
    _, data = synthetic(nsystems=5)
    data.add_reaction("train", {"1": 1, "0": -1}, reference=0, noise=0.03, unit="eV")
    data.add_reaction(
        "test", {"3": 1, "2": -1}, reference=1e6, noise=1e-9, unit="eV", split="test"
    )
    ft = FineTuner(n_control=10, max_candidates=120, constraints=unconstrained()).fit(
        data
    )
    alpha = ft.engine.kernels[0].alpha.copy()
    ft.fit(data.select(split="train"))
    np.testing.assert_array_equal(ft.engine.kernels[0].alpha, alpha)
    assert ft.fit_info["n_train"] == 1


def test_self_consistent_labels_stay_relative_to_frozen_base():
    _, data = synthetic(nsystems=4)
    ft = FineTuner(n_control=10, constraints=unconstrained(), max_candidates=120)
    add_known_labels(data, ft)
    ft.fit(data)
    before = ft.predict(data)
    original_labels = np.array(ft.engine.rxn_ref_list)

    def recompute(model, dataset, step):
        assert step in (1, 2)
        assert len(model.kernels) == 3
        # A current-model SCF export at unchanged density: the same added XC
        # appears in BOTH total and XC, so it must cancel out of the new mean.
        records = deepcopy(dataset.records)
        for key, record in records.items():
            delta = before[key]["correction"] * to_hartree("eV")
            record["e_tot_orig"] += delta
            record["exc_orig"] += delta
        return records

    history = ft.fit_self_consistent(data, recompute, n_steps=2)
    assert len(history) == 3
    np.testing.assert_allclose(ft.engine.rxn_ref_list, original_labels, atol=1e-14)
    assert len(ft.model.kernels) == 3
    for key, value in ft.predict(data).items():
        assert value["correction"] == pytest.approx(
            before[key]["correction"], abs=1e-10
        )


def test_two_components_and_hdf5(tmp_path):
    from pyscf.lib import chkfile

    _, data = synthetic(nspin=2, nsystems=3)
    ft = FineTuner(
        n_control=8, components="xc", constraints=unconstrained(), max_candidates=100
    )
    add_known_labels(data, ft)
    ft.fit(data)
    assert len(ft.model.kernels) == 4
    assert supported(ft.model, 6)
    record = data.records["0"]
    directories = dict.fromkeys(("REF", "SL", "NLDF", "NLOF", "SDMX", "HYB"))
    for group, fields in (
        ("REF", {k: v for k, v in record.items() if k not in ("desc", "composition")}),
        ("SL", {"desc": record["desc"][:, :3]}),
        ("NLDF", {"desc": record["desc"][:, 3:]}),
    ):
        directory = tmp_path / group
        directory.mkdir()
        directories[group] = str(directory)
        chkfile.save(str(directory / "zero.hdf5"), "train_data", fields)
    loaded = ReactionSet.from_hdf5(directories, ["zero"], compositions={"zero": "H2"})
    np.testing.assert_array_equal(loaded.records["zero"]["desc"], record["desc"])
    np.testing.assert_array_equal(loaded.records["zero"]["wt"], record["wt"])
