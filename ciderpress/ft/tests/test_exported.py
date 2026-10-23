"""CPU qualification on real exported records; no cluster/GPU work.

P12_FT_EXPORT_DIR selects a p11-export REF/SL/NLDF directory (sample.hdf5).
Without saved data, a small New-GPAW He record is generated if GPAW is present.
P12_FT_SMOKE_DIR selects legacy p11-cmr exports; the Pt/O subset is sufficient
to expose the weakly constrained absolute direction. Missing smoke files skip
only that historical regression, not the real-record serialization test.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase.units import Hartree

from ciderpress.dft.model_utils import load_cider_model
from ciderpress.ft import Constraints, FineTuner, ReactionSet, from_pyscf
from ciderpress.ft.records import integrate
from ciderpress.gpu.semilocal import supported


def test_real_exported_record_roundtrip(tmp_path):
    root = Path(
        os.environ.get("P12_FT_EXPORT_DIR", "/private/tmp/p11-local/final/ni/training")
    )
    if (root / "REF/sample.hdf5").is_file():
        dirs = {
            name: str(root / name) if (root / name).is_dir() else None
            for name in ("REF", "SL", "NLDF", "NLOF", "SDMX", "HYB")
        }
        data = ReactionSet.from_hdf5(dirs, ["sample"], compositions={"sample": "Ni"})
    else:
        pytest.importorskip("gpaw")
        from ase import Atoms
        from gpaw import GPAW
        from gpaw.fftw import ESTIMATE, FFTWPlans

        from ciderpress.gpaw.new_backend import CiderXC
        from ciderpress.gpaw.new_export import export_training_data

        FFTWPlans._overwrite_flags = ESTIMATE
        atoms = Atoms("He", positions=[[2, 2, 2]], cell=[4, 4, 4], pbc=True)
        atoms.calc = GPAW(
            mode={"name": "pw", "ecut": 150},
            h=0.35,
            xc=CiderXC(),
            legacy_gpaw=False,
            parallel={"gpu": False},
            txt=str(tmp_path / "he.txt"),
            maxiter=100,
        )
        atoms.get_potential_energy()
        record = export_training_data(atoms.calc)
        record["symbols"] = atoms.get_chemical_symbols()
        data = ReactionSet.from_records({"sample": record})
    record = data.records["sample"]
    assert np.any(record["wt"] < 0)
    ft = FineTuner(n_control=12, max_candidates=150)
    # This absolute label exercises the wrapper; it is a synthetic correction
    # on a real density, not a new physical energy reference.
    base_total = ft._base_total(record)
    data.add_energy("sample", reference=base_total + 0.002, unit="Hartree", noise=0.001)
    ft.fit(data)
    saved = ft.save(tmp_path / "real-ft.yaml")
    mapped = load_cider_model(saved)
    correction = ft.engine.kernels[0]
    delta = correction.alpha @ correction.cov_dict["train:sample"]
    base_xc = integrate(ft.base, record)
    saved_xc = integrate(mapped, record)
    np.testing.assert_allclose(saved_xc, base_xc + delta, atol=2e-10, rtol=1e-12)
    assert supported(mapped, record["desc"].shape[1])
    # Exercise all three ordinary calculator factories without an SCF.
    from pyscf import dft, gto

    from ciderpress.gpaw.calculator import get_cider_functional
    from ciderpress.gpaw.new_backend import CiderXC
    from ciderpress.pyscf.dft import make_cider_calc

    assert CiderXC(saved).mlfunc.ft_metadata["version"] == 1
    assert get_cider_functional(saved, xmix=1, xkernel=None, ckernel=None) is not None
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    assert make_cider_calc(dft.RKS(mol), saved) is not None


def legacy_record(folder):
    """Bridge Classic p11 arrays using the explicitly measured quadrature offset.

    Stored GPW scalars are eV; feature integrals are Ha and omit core constants.
    The common core/PAW quadrature bridge is fixed by the p11 PBE readback,
    never fitted to the residual reference. All original points/weights remain.
    """
    from ase.io.ulm import Reader

    metadata = json.loads((folder / "result.json").read_text())
    with (
        np.load(folder / "features.npz") as arrays,
        Reader(folder / "classic-pbe.gpw") as gpw,
    ):
        exc_orig = gpw.hamiltonian.e_xc / Hartree
        return {
            "desc": arrays["desc"],
            "rho_data": arrays["rho"],
            "wt": arrays["wt"],
            "nspin": metadata["nspin"],
            "e_tot_orig": gpw.hamiltonian.e_total_extrapolated / Hartree,
            "exc_orig": exc_orig,
            "numbers": gpw.atoms.numbers,
            "exc_core": exc_orig
            - metadata["XC_PBE_integral_Ha"]
            + metadata["DiffPBE_minus_standard_PBE_XC_eV"] / Hartree,
        }


def test_p11_pt_smoke_absolute_direction(tmp_path):
    root = Path(
        os.environ.get(
            "P12_FT_SMOKE_DIR", "/private/tmp/p11-cmr-classic-reconvergence/exports"
        )
    )
    if not all(
        (root / key / "features.npz").is_file()
        for key in ("Pt-O", "Pt-clean", "gas-O2")
    ):
        pytest.skip(
            "Historical p11 Pt/O feature records unavailable; synthetic safeguards remain tested"
        )
    records = {key: legacy_record(root / key) for key in ("Pt-O", "Pt-clean", "gas-O2")}
    data = ReactionSet.from_records(records)
    data.add_reaction(
        "O/Pt",
        {"Pt-O": 1, "Pt-clean": -1, "gas-O2": -0.5},
        reference=0.015826,
        unit="eV",
        noise=0.03,
        group="Pt",
    )
    protection = ReactionSet.from_records({"O2": records["gas-O2"]})
    # Reproduce the original smoke's molecular zero anchor while removing its
    # missing UEG/absolute constraints. This is a one-reaction historical subset.
    free = FineTuner(
        n_control=32,
        prior_scale=1,
        max_candidates=400,
        constraints=Constraints(
            ueg=False,
            absolute_noise=None,
            max_abs_correction=None,
            molecules=protection,
        ),
    ).fit(data)
    safeguarded = FineTuner(
        n_control=32,
        prior_scale=1,
        max_candidates=400,
        constraints=Constraints(molecules=protection),
    ).fit(data)
    unbounded = free.predict(data)
    bounded = safeguarded.predict(data)
    before = abs(unbounded["Pt-clean"]["correction"])
    after = abs(bounded["Pt-clean"]["correction"])
    assert (
        before > 10.0
    )  # exposes a large absolute direction despite a small adsorption label
    assert after < before * 0.1
    for key, record in data.records.items():
        natoms = len(record["numbers"])
        assert abs(bounded[key]["correction"]) <= 0.500001 * natoms
    model = load_cider_model(safeguarded.save(tmp_path / "pt-ft.yaml"))
    errors = []
    for key, record in data.records.items():
        # Full signed real records, including the complete PAW point set.
        actual = integrate(model, record)
        expected = (
            integrate(safeguarded.base, record) + bounded[key]["correction"] / Hartree
        )
        errors.append(abs(actual - expected))
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=1e-12)
    result = {
        "scope": "p11 Pt/O independently converged Classic subset, not full nine-state smoke",
        "unconstrained_pt_clean_eV": before,
        "constrained_pt_clean_eV": after,
        "max_roundtrip_error_ha": max(errors),
        "fit_info": safeguarded.fit_info,
        "free_report": free.evaluate(data),
        "constrained_report": safeguarded.evaluate(data),
    }
    (tmp_path / "smoke-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


@pytest.mark.parametrize("unrestricted", [False, True])
def test_pyscf_molecular_adapter(unrestricted):
    from pyscf import dft, gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    calc = dft.UKS(mol) if unrestricted else dft.RKS(mol)
    calc.xc = "PBE"
    calc.grids.level = 0
    calc.kernel()
    record = from_pyscf(calc)
    assert record["desc"].shape[:2] == (2 if unrestricted else 1, 6)
    assert record["rho_data"].shape[1] == 5
    data = ReactionSet.from_records({"H2": record})
    ft = FineTuner(n_control=8, max_candidates=100)
    data.add_energy("H2", reference=ft._base_total(record), unit="Hartree", noise=0.01)
    ft.fit(data)
    assert abs(ft.predict(data)["H2"]["correction"]) < 1e-9
    calc.xc = "PBE0"
    with pytest.raises(ValueError, match="pure KS"):
        from_pyscf(calc)
