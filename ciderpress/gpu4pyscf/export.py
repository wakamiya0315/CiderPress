"""Qualified GPU SCF -> official, signed molecular FT records (E0).

This module transfers a final state explicitly; it neither runs SCF nor
implements train_gen. Import the GPU factory before importing gpu4pyscf.
State selection is the caller's responsibility and must be attested through
``state``. A saved-state mapping is an optional consistency check, never a
way to turn an unconverged calculator into a converged one.
"""
from copy import copy, deepcopy
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
from types import MethodType

import numpy as np

MODEL = "CIDER26XCSURFSCI"


def molecular_composition(mol):
    """Physical atomic numbers, distinct from ECP integral nuclear charges.

    PySCF atom_charges() returns effective charges for ECP atoms. The current
    ft.from_pyscf adapter stores those as numbers; repair that metadata here,
    without changing the Mole, density, integrals or existing FT code.
    """
    from pyscf import gto
    physical = np.asarray([gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)], dtype=int)
    effective = np.asarray(mol.atom_charges(), dtype=int)
    if not len(physical) or np.any(physical <= 0):
        raise ValueError("Physical composition is required; ghost atoms unsupported")
    return dict(numbers=physical.tolist(), integral_nuclear_charges=effective.tolist(),
                ecp_core_electrons=int(physical.sum()-effective.sum()))


def array_sha256(value):
    """Digest C-order bytes; shape and dtype are recorded separately."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _host(value):
    import cupy as cp
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def _finite(value, name):
    result = np.asarray(_host(value))
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name}: real finite arrays required")
    return result.copy()


def _equal(left, right, name):
    if (left.dtype != right.dtype or left.shape != right.shape or
            left.tobytes() != right.tobytes()):
        raise ValueError(f"{name}: must equal the frozen GPU SCF state exactly")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _state_check(mf, state, source_geometry, provenance):
    if not isinstance(state, dict) or not state.get("id"):
        raise ValueError("An explicit state id and acceptance decision are required")
    if state.get("accepted") is not True or state.get("resolved") is not True:
        raise ValueError("Unaccepted or unresolved states cannot be exported")
    if mf.mol.spin or np.ndim(_host(mf.mo_occ)) == 2:
        if not state.get("qualification", {}).get("open_shell_accepted", False):
            raise ValueError("Open shells require an explicit qualification decision")
    if not isinstance(source_geometry, dict) or not all(source_geometry.get(k) for k in
            ("database", "subset", "system", "files")):
        raise ValueError("Source database/subset/system and file SHA-256 are required")
    for entry in source_geometry["files"].values():
        digest = entry.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Source file SHA-256 is missing or malformed")
    if not isinstance(provenance, dict) or not all(
            provenance.get(k, {}).get("commit") for k in ("ciderpress", "bench")):
        raise ValueError("CiderPress and bench commit provenance is required")


def cpu_shell_from_gpu(mf, *, saved_state=None):
    """Freeze DM/MOs and exact SCF quadrature in a genuine CPU CIDER shell.

    No CPU energy, XC, J or SCF evaluations are permitted on this shell.
    Descriptors may still be evaluated by the official FT adapter (E1).
    ``saved_state``, when given, contains dm/mo_coeff/mo_occ/mo_energy,
    coords/weights/e_tot/exc_orig and is compared with the live calculator.
    """
    from pyscf import dft
    from ciderpress.pyscf.dft import make_cider_calc
    from .numint import CiderNumInt, molecule_identity

    if not mf.converged or not isinstance(mf._numint, CiderNumInt):
        raise ValueError("A converged make_cider_calc_gpu calculator is required")
    if mf.cider_model != MODEL or mf.xc != "R2SCAN":
        raise ValueError("Only the frozen full-XC GPU model is supported")
    ni = mf._numint
    if molecule_identity(mf.mol) != ni._mol_identity:
        raise ValueError("Molecule or AO order changed since SCF setup")
    numbers = _finite(mf.mol.atom_charges(), "numbers")
    if not numbers.size or np.any(numbers <= 0):
        raise ValueError("Missing composition or unsupported ghost atoms")
    arrays = {k: _finite(mf.make_rdm1() if k == "dm" else getattr(mf, k), k)
              for k in ("dm", "mo_coeff", "mo_occ", "mo_energy")}
    arrays.update({k: _finite(getattr(mf.grids, attr), k)
                   for k, attr in (("coords", "coords"), ("weights", "weights"))})
    n, nao = len(arrays["weights"]), mf.mol.nao_nr()
    ns = 2 if isinstance(ni.cpu_shell, dft.uks.UKS) else 1
    leading = (2,) if ns == 2 else ()
    if (arrays["dm"].shape != leading + (nao, nao) or
            arrays["mo_coeff"].shape != leading + (nao, nao) or
            arrays["mo_occ"].shape != leading + (nao,) or
            arrays["mo_energy"].shape != leading + (nao,) or
            arrays["coords"].shape != (n, 3) or not n):
        raise ValueError("DM/MO/spin/grid shape or AO order mismatch")
    occ = arrays["mo_occ"]
    if not np.all(np.isin(occ, (0, 1) if ns == 2 else (0, 2))):
        raise ValueError("Only integer occupations are supported")
    counts = occ.sum(axis=-1)
    if not np.array_equal(counts, mf.mol.nelec if ns == 2 else mf.mol.nelectron):
        raise ValueError("Occupation electron counts disagree with charge/spin")
    for key in ("coords", "weights"):
        _equal(arrays[key], np.asarray(getattr(ni.cpu_shell.grids, key)), key)
        if ni._grid is not None:
            _equal(arrays[key], _host(getattr(ni, "_" + key)), key)
    energies = dict(e_tot=float(mf.e_tot), exc_orig=float(mf.scf_summary["exc"]))
    if not np.isfinite(list(energies.values())).all():
        raise ValueError("Nonfinite saved SCF energies")
    if saved_state is not None:
        for key, value in arrays.items():
            if key not in saved_state:
                raise ValueError(f"Saved state missing {key}")
            _equal(value, _finite(saved_state[key], key), "saved " + key)
        if any(float(saved_state[k]) != v for k, v in energies.items()):
            raise ValueError("Saved state energy mismatch")
    shell = make_cider_calc((dft.UKS if ns == 2 else dft.RKS)(mf.mol.copy()), MODEL,
                            xmix=1., xkernel=None, ckernel=None)
    if mf.cider_use_df:
        if mf.with_df.auxbasis != "def2-universal-jfit" or not mf.only_dfj:
            raise ValueError("DF must be explicit def2-universal-jfit J-only")
        shell = shell.density_fit(auxbasis="def2-universal-jfit", only_dfj=True)
    shell.grids = copy(ni.cpu_shell.grids)
    shell.grids.coords, shell.grids.weights = arrays["coords"].copy(), arrays["weights"].copy()
    shell.max_memory, shell.small_rho_cutoff = mf.max_memory, 0
    for key in ("mo_coeff", "mo_occ", "mo_energy"):
        setattr(shell, key, arrays[key].copy())
    np.testing.assert_allclose(shell.make_rdm1(), arrays["dm"], rtol=2e-12, atol=2e-12)
    shell.make_rdm1 = MethodType(lambda self: arrays["dm"].copy(), shell)
    shell.e_tot, shell.scf_summary["exc"], shell.converged = energies["e_tot"], energies["exc_orig"], True
    calls = {}
    def guard(owner, names, prefix=""):
        for name in names:
            key = prefix + name
            calls[key] = 0
            def forbidden(*args, _key=key, **kwargs):
                calls[_key] += 1
                raise RuntimeError("CPU shell must not recompute " + _key)
            setattr(owner, name, forbidden)
    guard(shell, ("energy_tot", "energy_elec", "kernel", "scf", "get_veff", "get_j", "get_k"))
    guard(shell._numint, ("build", "nr_rks", "nr_uks", "eval_xc", "eval_xc_eff"), "numint.")
    shell.e0_forbidden_calls = calls
    shell.e0_snapshot = arrays
    return shell


def _grid_spec(mf):
    grid = mf._numint.cpu_shell.grids
    spec = {k: _jsonable(getattr(grid, k)) for k in
            ("level", "alignment", "atom_grid", "atomic_radii", "lmax")}
    for key in ("radi_method", "prune", "becke_scheme", "radii_adjust"):
        fun = getattr(grid, key)
        spec[key] = None if fun is None else fun.__module__ + "." + fun.__name__
    spec.update(sort_grids=False, small_rho_cutoff=mf.small_rho_cutoff,
                coords_sha256=array_sha256(grid.coords), weights_sha256=array_sha256(grid.weights),
                points=len(grid.weights), negative_weights=int(np.count_nonzero(grid.weights < 0)))
    return spec


def from_gpu_scf(mf, *, state, source_geometry, provenance, saved_state=None):
    """Return the official raw SL(3)+NLDF(3) E0 record, with provenance.

    ``state`` requires id/accepted/resolved and, for UKS, a qualification
    dictionary with open_shell_accepted=True. Set cpu_adopted explicitly when
    a CPU-selected state was used. ``provenance`` supplies ciderpress.commit
    and bench.commit; optional additional JSON fields are retained. Source
    files are a mapping of names to {sha256: ...}. No pointwise val is made.
    """
    from .descriptors import SCFFeatureGenerator
    from ciderpress.ft import from_pyscf
    import cupy as cp
    _state_check(mf, state, source_geometry, provenance)
    shell = cpu_shell_from_gpu(mf, saved_state=saved_state)
    arrays = shell.e0_snapshot
    generate = SCFFeatureGenerator(mf, arrays["dm"])
    ns, n = generate.raw.shape[0], len(arrays["weights"])
    for name, shape in (("raw", (ns, 5, n)), ("sl", (ns, 3, n)), ("nldf", (ns, 3, n))):
        value = _finite(getattr(generate, name), name)
        if value.shape != shape:
            raise ValueError("E0 feature shape/order mismatch: " + name)
    expected_sl = np.stack((generate.raw[:, 0], (generate.raw[:, 1:4]**2).sum(axis=1),
                            generate.raw[:, 4]), axis=1)
    np.testing.assert_allclose(generate.sl, expected_sl, rtol=1e-14, atol=0)
    model = resources.files("ciderpress.data.functionals").joinpath(MODEL + ".yaml")
    prov = deepcopy(provenance)
    prov.update(features="scf-direct-gpu", model=dict(name=MODEL,
                sha256=hashlib.sha256(model.read_bytes()).hexdigest()),
                source_geometry=deepcopy(source_geometry), state=deepcopy(state),
                charge=mf.mol.charge, spin=mf.mol.spin, grid=_grid_spec(mf),
                composition=molecular_composition(mf.mol),
                stage1=dict(reduction=mf._numint.engine.nldf.transforms.reduction,
                    schedule=mf._numint.schedule, tile_size=mf._numint.tile_size),
                j=dict(mode="strict deterministic" if hasattr(mf, "cider_j_schedule") else "stock",
                    df=mf.cider_use_df, auxbasis="def2-universal-jfit" if mf.cider_use_df else None,
                    resolved_auxbasis=None),
                scf=dict(converged=True, cycles=int(mf.cycles), conv_tol=mf.conv_tol,
                    conv_tol_grad=mf.conv_tol_grad, max_cycle=mf.max_cycle,
                    summary=dict(mf.scf_summary)),
                versions={name: metadata.version(name) for name in ("pyscf", "gpu4pyscf-cuda13x", "cupy-cuda13x")},
                device=dict(id=cp.cuda.Device().id,
                    properties=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion()),
                arrays={k: dict(sha256=array_sha256(v), shape=list(v.shape), dtype=str(v.dtype))
                        for k, v in arrays.items()})
    prov["ciderpress"].update(pyg_scf="49986ed", pyg_nldf="6dbecae")
    if mf.cider_use_df:
        from pyscf.df.addons import make_auxmol
        prov["j"]["resolved_auxbasis"] = make_auxmol(mf.mol, "def2-universal-jfit")._basis
    prov["state"].setdefault("cpu_adopted", False)
    prov = _jsonable(prov)
    json.dumps(prov, allow_nan=False)
    record = from_pyscf(shell, base=MODEL, grids=shell.grids,
                       feature_generator=generate, provenance=prov)
    # ft/records.py uses atom_charges(), which are effective for ECP atoms.
    record["numbers"] = np.asarray(prov["composition"]["numbers"], dtype=int)
    _validate_e0(record)
    _equal(record["wt"], arrays["weights"], "record weights")
    if any(shell.e0_forbidden_calls.values()):
        raise RuntimeError("Export unexpectedly recomputed CPU energy")
    return record


def _validate_e0(record):
    from ciderpress.ft.records import validate_record
    value = validate_record(record)
    if (value["desc"].shape[1] != 6 or value.get("nfeat_sl") != 3 or
            value["exc_core"] != 0 or not value.get("feature_settings_yaml") or "val" in value):
        raise ValueError("Expected label-free molecular E0 SL(3)+NLDF(3) record")
    numbers = np.asarray(value.get("numbers", []))
    if (numbers.ndim != 1 or not len(numbers) or numbers.dtype.kind not in "iuf"
            or not np.isfinite(numbers).all() or np.any(numbers <= 0)
            or not np.all(numbers == numbers.astype(int))):
        raise ValueError("Composition is required")
    prov = value.get("provenance", {})
    if prov.get("features") != "scf-direct-gpu":
        raise ValueError("Missing E0 provenance")
    if "composition" in prov and not np.array_equal(numbers, prov["composition"]["numbers"]):
        raise ValueError("Physical atomic numbers disagree with provenance")
    if array_sha256(value["wt"]) != prov["grid"]["weights_sha256"]:
        raise ValueError("Signed SCF quadrature weights changed")
    if prov["state"].get("accepted") is not True or prov["state"].get("resolved") is not True:
        raise ValueError("Unaccepted state")
    json.dumps(_jsonable(prov), allow_nan=False)
    return value


def write_e0_records(records, directory, *, manifest_directory=None):
    """Stock FT writer plus one JSON audit manifest per numerical state.

    Returns its six-key directory mapping. JSON manifests deliberately keep
    reactions/labels separate; large arrays remain in HDF5 on task scratch.
    """
    from ciderpress.ft import write_records
    checked = {}
    for name, record in records.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("State IDs must be single path components")
        checked[name] = _validate_e0(record)
    destination = Path(directory)
    manifests = destination / "manifests" if manifest_directory is None else Path(manifest_directory)
    dirs = write_records(checked, destination)
    manifests.mkdir(parents=True, exist_ok=True)
    for name, record in checked.items():
        paths = [Path(dirs[k]) / (name + ".hdf5") for k in ("REF", "SL", "NLDF")]
        manifest = dict(schema="ciderpress-gpu-e0-v1", numerical_id=name,
            chemical_id=record["provenance"]["source_geometry"], provenance=record["provenance"],
            numbers=record["numbers"], e_tot_orig=record["e_tot_orig"], exc_orig=record["exc_orig"],
            exc_core=0., energy_unit="Hartree", nfeat_sl=3,
            feature_settings_yaml=record["feature_settings_yaml"], val_present=False,
            arrays={k: dict(shape=list(record[k].shape), dtype=str(record[k].dtype),
                           sha256=array_sha256(record[k])) for k in ("desc", "rho_data", "wt")},
            files=[dict(path=str(p.resolve()), bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
            reaction=dict(label="NONE; reactions are supplied separately to ReactionSet", fold=None, round=0))
        text = json.dumps(_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary = manifests / (name + ".json.tmp")
        temporary.write_text(text)
        temporary.replace(manifests / (name + ".json"))
    return dirs
