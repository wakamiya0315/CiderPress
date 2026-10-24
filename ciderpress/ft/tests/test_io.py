"""PySCF grid, GPAW-free serialization and provenance contracts (CPU only)."""

import os
import subprocess
import sys
from copy import copy

import h5py
import numpy as np
import pytest
from pyscf import dft, gto

from ciderpress.dft.model_utils import load_cider_model
from ciderpress.ft import FineTuner, ReactionSet, from_pyscf, write_records
from ciderpress.models.train import MOLGP2
from ciderpress.pyscf.analyzers import ElectronAnalyzer
from ciderpress.pyscf.descriptors import get_descriptors
from ciderpress.pyscf.dft import make_cider_calc
from ciderpress.pyscf.gen_cider_grid import CiderGrids


@pytest.fixture(params=[False, True], ids=["rks", "uks"])
def calc(request):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = dft.UKS(mol) if request.param else dft.RKS(mol)
    mf.xc = "PBE"
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.kernel()
    assert mf.converged
    return mf


def descriptor_arrays(analyzer):
    settings = load_cider_model("CIDER26XCSURFSCI").settings
    return dict(
        desc=np.concatenate(
            [
                get_descriptors(analyzer, settings.sl_settings),
                get_descriptors(analyzer, settings.nldf_settings),
            ],
            axis=1,
        ),
        rho_data=get_descriptors(analyzer, "l"),
        wt=analyzer.grids.weights,
    )


def fail_energy(*args, **kwargs):
    pytest.fail("Matched-grid export evaluated the SCF energy/integrator")


def test_scf_grid_matches_existing_arrays_without_energy(calc, monkeypatch):
    analyzer = ElectronAnalyzer.from_calc(calc)
    expected = descriptor_arrays(analyzer)
    np.testing.assert_array_equal(analyzer.grids.coords, calc.grids.coords)
    np.testing.assert_array_equal(expected["wt"], calc.grids.weights)
    default = from_pyscf(calc)
    for key in expected:
        np.testing.assert_array_equal(default[key], expected[key])
    monkeypatch.setattr(calc, "energy_tot", fail_energy)
    monkeypatch.setattr(calc._numint, "nr_rks", fail_energy)
    monkeypatch.setattr(calc._numint, "nr_uks", fail_energy)
    # An array copy matches even if the grid's level label differs.
    grids = copy(calc.grids)
    grids.level += 1
    grids.coords = calc.grids.coords.copy()
    grids.weights = calc.grids.weights.copy()
    for options in (
        dict(grids=grids), dict(grid_spec={"level": 0}), dict(analyzer=analyzer)
    ):
        record = from_pyscf(calc, **options)
        for key in expected:
            np.testing.assert_array_equal(record[key], expected[key])
        assert record["e_tot_orig"] == calc.e_tot
        assert record["exc_orig"] == calc.scf_summary["exc"]


def test_custom_grid_order_analyzer_and_generator(calc):
    analyzer = ElectronAnalyzer.from_calc(calc, store_energy_orig=False)
    analyzer.grids = copy(calc.grids)
    analyzer.grids.coords = calc.grids.coords[::-1].copy()
    analyzer.grids.weights = calc.grids.weights[::-1].copy()
    analyzer.grids.non0tab = None
    expected = descriptor_arrays(analyzer)
    # Reordering changes no quadrature, but is deliberately not an exact match.
    calls = []

    def generate(ana, settings):
        assert ana is analyzer
        calls.append(settings)
        return get_descriptors(ana, settings)

    record = from_pyscf(calc, analyzer=analyzer, feature_generator=generate)
    assert len(calls) == 3 and calls[-1] == "l"
    for key in expected:
        np.testing.assert_array_equal(record[key], expected[key])
    np.testing.assert_allclose(record["e_tot_orig"], calc.e_tot, atol=1e-12, rtol=0)
    analyzer.dm = np.zeros_like(analyzer.dm)
    with pytest.raises(ValueError, match="molecule and density"):
        from_pyscf(calc, analyzer=analyzer)


def test_same_level_different_grid_replaces_xc_without_mutation(calc):
    grids = copy(calc.grids)
    grids.weights = calc.grids.weights * 0.99
    summary = calc.scf_summary.copy()
    original_grid = calc.grids
    # Independent full fixed-density energy on the selected grid.
    reference = calc.copy()
    reference.grids = grids
    reference.scf_summary = summary.copy()
    expected = reference.energy_tot(dm=calc.make_rdm1())
    record = from_pyscf(calc, grids=grids)
    assert record["exc_orig"] == pytest.approx(reference.scf_summary["exc"], abs=1e-12)
    assert record["e_tot_orig"] == pytest.approx(expected, abs=1e-12)
    assert abs(record["exc_orig"] - summary["exc"]) > 1e-5
    assert calc.grids is original_grid and calc.scf_summary == summary


def test_cider_scf_grid_does_not_evaluate_cpu_energy(monkeypatch):
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    calc = make_cider_calc(dft.RKS(mol), "CIDER26XCSURFSCI")
    calc.grids.level = 0
    calc.small_rho_cutoff = 0
    calc.kernel()
    assert calc.converged
    # A different explicit specification must keep the CIDER grid indexer,
    # use the original XC functional, and leave the live integrator untouched.
    alternate = CiderGrids(mol)
    alternate.level = 1
    alternate.build()
    reference = calc.copy()
    reference.grids = alternate
    reference._numint = copy(calc._numint)
    reference._numint.build(mol)
    reference.scf_summary = calc.scf_summary.copy()
    expected_energy = reference.energy_tot(dm=calc.make_rdm1())
    summary = calc.scf_summary.copy()
    original_generator = calc._numint.nldfgen
    changed = from_pyscf(calc, grid_spec={"level": 1})
    assert changed["e_tot_orig"] == pytest.approx(expected_energy, abs=1e-12)
    assert changed["exc_orig"] == pytest.approx(reference.scf_summary["exc"], abs=1e-12)
    assert calc.scf_summary == summary
    assert calc._numint.nldfgen is original_generator
    analyzer = ElectronAnalyzer.from_calc(calc, store_energy_orig=False)
    analyzer.grids = calc.grids
    expected = descriptor_arrays(analyzer)
    monkeypatch.setattr(calc, "energy_tot", fail_energy)
    monkeypatch.setattr(calc._numint, "nr_rks", fail_energy)
    for options in (dict(grids=calc.grids), dict(grid_spec={"level": 0})):
        record = from_pyscf(calc, **options)
        for key in expected:
            np.testing.assert_array_equal(record[key], expected[key])
        assert record["exc_orig"] == calc.scf_summary["exc"]


def test_pyscf_hdf5_roundtrip_and_metadata_ignored_by_fit(calc, tmp_path):
    provenance = {
        "df": {"enabled": True, "auxbasis": "def2-universal-jfit", "resolved": None},
        "grid": {"level": 0, "prune": None, "axes": [1, 2, 3]},
        "model_sha256": "test-digest",
        "versions": {"pyscf": "test-version", "code/branch": "日本語"},
    }
    record = from_pyscf(calc, grids=calc.grids, provenance=provenance)
    provenance["grid"]["level"] = 99
    assert record["provenance"]["grid"]["level"] == 0
    data = ReactionSet.from_records({"H2": record})
    directories = data.to_hdf5(tmp_path)
    loaded = ReactionSet.from_hdf5(directories, ["H2"])
    molgp = MOLGP2.load_data(directories, "H2", False)
    for key in ("desc", "rho_data", "wt", "numbers"):
        np.testing.assert_array_equal(loaded.records["H2"][key], record[key])
        np.testing.assert_array_equal(molgp[key], record[key])
    for key in (
        "e_tot_orig", "exc_orig", "exc_core", "nspin", "nfeat_sl",
        "feature_settings_yaml", "provenance",
    ):
        assert loaded.records["H2"][key] == record[key]
    with h5py.File(tmp_path / "REF/H2.hdf5") as h5:
        assert "val" not in h5["train_data"]
        assert "provenance_json" in h5["train_data"]
    data.add_energy("H2", reference=calc.e_tot, unit="Ha", noise=0.01)
    options = dict(n_control=4, max_candidates=24)
    with_metadata = FineTuner(**options).fit(data)
    plain_record = dict(record)
    plain_record.pop("provenance")
    plain = data.with_records({"H2": plain_record})
    without_metadata = FineTuner(**options).fit(plain)
    assert with_metadata.predict(data) == without_metadata.predict(plain)


def simple_record():
    return dict(
        desc=np.ones((2, 6, 4)), rho_data=np.ones((2, 5, 4)),
        wt=np.array([1., 2., -1., -2.]), e_tot_orig=-2., exc_orig=-1.,
        nfeat_sl=3, symbols=["H", "H"], val=np.arange(4.),
    )


def test_writer_signed_weights_optional_val_and_validation(tmp_path):
    record = simple_record()
    record["symbols"] = ["O", "H", "O"]
    record["provenance"] = {"array": np.array([1, 2]), "level": np.int64(0)}
    directories = write_records({"sample": record}, tmp_path)
    loaded = ReactionSet.from_hdf5(directories, ["sample"]).records["sample"]
    np.testing.assert_array_equal(loaded["wt"], record["wt"])
    np.testing.assert_array_equal(loaded["numbers"], [8, 1, 8])
    assert loaded["provenance"] == {"array": [1, 2], "level": 0}
    assert "val" not in loaded
    write_records({"sample": record}, tmp_path, include_val=True)
    np.testing.assert_array_equal(
        MOLGP2.load_data(directories, "sample", False)["val"], record["val"]
    )
    # Replacing a file/group must remove old optional data, too.
    write_records({"sample": record}, tmp_path)
    assert "val" not in MOLGP2.load_data(directories, "sample", False)
    for name in ("", "../escape", ".", "..", "/absolute"):
        with pytest.raises(ValueError, match="filename"):
            write_records({name: record}, tmp_path)
    for change, match in (
        ({"nfeat_sl": 9}, "nfeat_sl"),
        ({"val": np.zeros(2)}, "val"),
        ({"provenance": []}, "provenance"),
        ({"numbers": [8, 8]}, "disagree"),
    ):
        with pytest.raises(ValueError, match=match):
            write_records({"bad": dict(record, **change)}, tmp_path, include_val=True)
    assert not (tmp_path / "REF/bad.hdf5").exists()


def test_grid_option_validation(calc):
    for options in (
        dict(grids=calc.grids, grid_spec={}),
        dict(grids=calc.grids, grids_level=0),
        dict(analyzer=object(), grid_spec={}),
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            from_pyscf(calc, **options)
    with pytest.raises(ValueError, match="coordinates and weights"):
        from_pyscf(calc, grids=dft.gen_grid.Grids(calc.mol))


def test_gpaw_free_import_and_roundtrip(tmp_path):
    script = r'''
import importlib.abc
import sys
class NoGPAW(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gpaw" or fullname.startswith(("gpaw.", "ciderpress.gpaw")):
            raise ModuleNotFoundError("GPAW deliberately unavailable")
sys.meta_path.insert(0, NoGPAW())
from ciderpress.ft import from_pyscf, write_records, ReactionSet
from ciderpress.models.train import MOLGP2
from pyscf import dft, gto
import numpy as np
mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
calc = dft.RKS(mol)
calc.xc = "PBE"
calc.grids.level = 0
calc.kernel()
record = from_pyscf(calc, grids=calc.grids, provenance={"gpaw": None})
dirs = write_records({"He": record}, sys.argv[1])
loaded = ReactionSet.from_hdf5(dirs, ["He"]).records["He"]
other = MOLGP2.load_data(dirs, "He", False)
for key in ("desc", "rho_data", "wt", "numbers"):
    np.testing.assert_array_equal(record[key], loaded[key])
    np.testing.assert_array_equal(record[key], other[key])
assert loaded["provenance"] == {"gpaw": None}
assert not any(
    k == "gpaw" or k.startswith(("gpaw.", "ciderpress.gpaw")) for k in sys.modules
)
print("GPAW-free PySCF export, write and both readers passed")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_physical_atomic_numbers_with_ecp():
    from ciderpress.ft.records import physical_atomic_numbers

    mol = gto.M(atom="I 0 0 0; H 0 0 1.61", basis="def2-svp", ecp="def2-svp",
                verbose=0)
    assert list(mol.atom_charges()) == [25, 1]
    assert list(physical_atomic_numbers(mol)) == [53, 1]
