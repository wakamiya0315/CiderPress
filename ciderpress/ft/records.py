"""Record validation, signed quadrature and a molecular PySCF adapter."""

import json
from copy import copy, deepcopy

import numpy as np

from ciderpress.dft.plans import get_rho_tuple_with_grad_cross


def validate_record(record, name="record"):
    result = dict(record)
    if "provenance_json" in result:
        result["provenance"] = json.loads(result.pop("provenance_json"))
    if "provenance" in result:
        if not isinstance(result["provenance"], dict):
            raise ValueError(f"{name}: provenance must be a dictionary")
        result["provenance"] = deepcopy(result["provenance"])
    if isinstance(result.get("feature_settings_yaml"), bytes):
        result["feature_settings_yaml"] = result["feature_settings_yaml"].decode()
    for key in ("desc", "rho_data", "wt", "e_tot_orig", "exc_orig"):
        if key not in result:
            raise ValueError(f"{name}: missing record field {key!r}")
    unit = result.get("energy_unit", result.get("units", {}).get("energy", "Hartree"))
    if isinstance(unit, bytes):
        unit = unit.decode()
    if unit not in ("Hartree", "Ha"):
        raise ValueError(
            f"{name}: record energies must be Hartree, got {unit!r}; convert before import"
        )
    for key in ("desc", "rho_data", "wt"):
        result[key] = np.asarray(result[key], dtype=float)
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{name}: {key} contains nonfinite values")
    desc, raw, wt = (result[k] for k in ("desc", "rho_data", "wt"))
    if (
        desc.ndim != 3
        or desc.shape[0] not in (1, 2)
        or raw.shape != (desc.shape[0], 5, desc.shape[-1])
        or wt.shape != (desc.shape[-1],)
        or wt.size == 0
    ):
        raise ValueError(
            f"{name}: expected desc=(1|2,nfeat,npoint), rho_data=(nspin,5,npoint), wt=(npoint,)"
        )
    if result.get("nspin", desc.shape[0]) != desc.shape[0]:
        raise ValueError(f"{name}: nspin differs from descriptor shape")
    result["nspin"] = desc.shape[0]
    for key in ("e_tot_orig", "exc_orig", "exc_core"):
        result[key] = float(result.get(key, 0.0))
        if not np.isfinite(result[key]):
            raise ValueError(f"{name}: {key} must be finite Hartree")
    # Negative PAW weights are physical and are deliberately retained.
    return result


def blocks(record, settings, blocksize):
    if record["desc"].shape[1] != settings.nfeat:
        raise ValueError("Record feature count differs from frozen model settings")
    for start in range(0, record["wt"].size, blocksize):
        end = start + blocksize
        x = settings.normalizers.get_normalized_feature_vector(
            record["desc"][..., start:end]
        )
        if not np.isfinite(x).all():
            raise ValueError(
                "Normalized features are not finite; check density and feature settings"
            )
        rho = get_rho_tuple_with_grad_cross(
            record["rho_data"][..., start:end] / record["nspin"],
            is_mgga=settings.sl_settings.level == "MGGA",
        )
        yield x, rho, record["wt"][start:end]


def integrate(model, record, *, blocksize=4000, rhocut=1e-8, core=True):
    """MappedXC2 integral in Ha; the unchanged core constant is added once."""
    energy = record["exc_core"] if core else 0.0
    for x, rho, wt in blocks(record, model.settings, blocksize):
        energy += float(model(x, rho, rhocut=rhocut)[0] @ wt)
    return energy


def covariance(kernel, record, settings, blocksize, rhocut):
    """Use DFTKernel2 for the same signed integral and cutoff as MappedXC2.

    The training loader's fixed 1e-6 mask differs from inference, so the wrapper
    constructs energy covariances explicitly with the inference cutoff.
    """
    result = np.zeros(kernel.Nctrl)
    for x, rho, wt in blocks(record, settings, blocksize):
        m = kernel.multiplicative_baseline(rho)[0]
        if kernel.mode == "SEP":
            m[record["nspin"] * rho[0] < rhocut] = 0
            km = (kernel.get_k(x) * m).sum(axis=1)
        else:
            m[rho[0].sum(axis=0) < rhocut] = 0
            km = kernel.get_k(x) * m
        result += km @ wt
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite integrated kernel covariance")
    return result


def _same_grid(left, right):
    return all(
        getattr(left, key, None) is not None
        and getattr(right, key, None) is not None
        and np.array_equal(getattr(left, key), getattr(right, key))
        for key in ("coords", "weights")
    )


def _pyscf_grid_energies(calc, analyzer):
    """Reuse SCF scalars on an identical grid; otherwise replace only XC."""
    exc = calc.scf_summary.get("exc")
    if exc is None or calc.e_tot is None:
        raise ValueError("Explicit-grid export requires saved SCF e_tot and exc")
    if _same_grid(analyzer.grids, calc.grids):
        return calc.e_tot, exc
    # Pure KS: the non-XC part is unchanged at fixed density. A private
    # integrator avoids changing the SCF grids, summary or CIDER generators.
    ni = copy(calc._numint)
    if hasattr(ni, "build"):
        ni.build(calc.mol)
    evaluate = ni.nr_uks if analyzer.dm.ndim == 3 else ni.nr_rks
    new_exc = evaluate(
        calc.mol, analyzer.grids, calc.xc, analyzer.dm, max_memory=calc.max_memory
    )[1]
    return calc.e_tot - exc + new_exc, new_exc


def from_pyscf(
    calc,
    base="CIDER26XCSURFSCI",
    *,
    grids_level=None,
    grid_spec=None,
    grids=None,
    analyzer=None,
    feature_generator=None,
    provenance=None,
):
    """Export a converged pure KS molecule using CiderPress analyzers.

    No SCF is run here. Hybrid/HF, NLC and external dispersion energies are
    refused because e_tot-exc is not the required pure-KS frozen-density mean.
    By default the analyzer uses its own grid, preserving the historical
    behavior. ``grid_spec`` configures that grid; ``grids=calc.grids`` retains
    the exact SCF points/order. Alternatively supply a matching-density
    ``analyzer`` (mutually exclusive with grid options). Explicit grids reuse
    saved SCF energies when coordinates AND weights match; otherwise only XC
    is reevaluated at fixed density, with the calculation's own integrator.

    ``feature_generator(analyzer, settings)`` defaults to ``get_descriptors``;
    it is called for SL, NLDF (if present), and raw density (settings="l").
    Use a wrapper to configure the inner NLDF grid or a custom implementation.
    ``provenance`` is an optional dictionary, ignored by the fit.
    """
    import yaml
    from pyscf import dft
    from pyscf.dft import libxc

    from ciderpress.dft.model_utils import load_cider_model
    from ciderpress.pyscf.analyzers import ElectronAnalyzer
    from ciderpress.pyscf.descriptors import get_descriptors

    if not isinstance(calc, dft.rks.KohnShamDFT) or not calc.converged:
        raise ValueError("PySCF export requires a converged pure KS calculation")
    if (
        libxc.is_hybrid_xc(calc.xc)
        or libxc.is_nlc(calc.xc)
        or getattr(calc, "nlc", "")
        or getattr(calc, "disp", None)
    ):
        raise ValueError("PySCF adapter supports pure KS without NLC or dispersion")
    model = load_cider_model(base)
    settings = model.settings
    if any(
        not s.is_empty
        for s in (settings.nlof_settings, settings.sdmx_settings, settings.hyb_settings)
    ):
        raise ValueError("PySCF adapter currently supports SL + NLDF features only")
    if analyzer is not None and any(
        option is not None for option in (grids_level, grid_spec, grids)
    ):
        raise ValueError("analyzer and grid options are mutually exclusive")
    if grids is not None and (grid_spec is not None or grids_level is not None):
        raise ValueError("grids and grid_spec/grids_level are mutually exclusive")
    explicit_grid = analyzer is not None or grid_spec is not None or grids is not None
    if analyzer is None:
        analyzer = ElectronAnalyzer.from_calc(
            calc,
            grids_level=grids_level,
            grid_spec=grid_spec,
            store_energy_orig=not explicit_grid,
        )
        if grid_spec is not None and hasattr(calc.grids, "grids_indexer"):
            # CIDER's XC integrator needs the atomic-grid indexer on a changed
            # quadrature. Preserve that grid family when building a spec.
            grids = type(calc.grids)(
                calc.mol, lmax=grid_spec.get("lmax", calc.grids.lmax)
            )
            grids.level = calc.grids.level if grids_level is None else grids_level
            grids.__dict__.update(deepcopy(grid_spec))
            if grids.level is None:
                grids.level = analyzer.grids_level
            grids.build(with_non0tab=False)
        if grids is not None:
            if grids.coords is None or grids.weights is None:
                raise ValueError("grids must already contain coordinates and weights")
            analyzer.grids = grids
    elif (
        analyzer.mol.nao_nr() != calc.mol.nao_nr()
        or not np.array_equal(analyzer.mol.atom_charges(), calc.mol.atom_charges())
        or not np.array_equal(analyzer.mol.atom_coords(), calc.mol.atom_coords())
        or not np.array_equal(analyzer.dm, calc.make_rdm1())
    ):
        raise ValueError(
            "analyzer must describe the calculation's molecule and density"
        )
    if explicit_grid:
        e_tot, exc = _pyscf_grid_energies(calc, analyzer)
    else:
        e_tot, exc = analyzer.get("e_tot_orig"), analyzer.get("exc_orig")
    generate = get_descriptors if feature_generator is None else feature_generator
    sl = generate(analyzer, settings.sl_settings)
    parts = [sl]
    if not settings.nldf_settings.is_empty:
        parts.append(generate(analyzer, settings.nldf_settings))
    raw = generate(analyzer, "l")
    record = {
        "desc": np.concatenate(parts, axis=1),
        "rho_data": raw,
        "wt": analyzer.grids.weights.copy(),
        "nspin": raw.shape[0],
        "nfeat_sl": sl.shape[1],
        "exc_core": 0.0,
        "e_tot_orig": e_tot,
        "exc_orig": exc,
        "energy_unit": "Hartree",
        "numbers": calc.mol.atom_charges(),
        "feature_settings_yaml": yaml.dump(settings),
    }
    if provenance is not None:
        record["provenance"] = provenance
    return validate_record(record)
