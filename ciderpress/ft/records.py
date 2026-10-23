"""Record validation, signed quadrature and a molecular PySCF adapter."""

import numpy as np

from ciderpress.dft.plans import get_rho_tuple_with_grad_cross


def validate_record(record, name="record"):
    result = dict(record)
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


def from_pyscf(calc, base="CIDER26XCSURFSCI", *, grids_level=None):
    """Export a converged pure KS molecule using CiderPress analyzers.

    No SCF is run here. Hybrid/HF, NLC and external dispersion energies are
    refused because e_tot-exc is not the required pure-KS frozen-density mean.
    The analyzer uses a separate atom-centred integration grid; qualify its
    quadrature before interpreting small energy differences.
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
    analyzer = ElectronAnalyzer.from_calc(calc, grids_level=grids_level)
    sl = get_descriptors(analyzer, settings.sl_settings)
    parts = [sl]
    if not settings.nldf_settings.is_empty:
        parts.append(get_descriptors(analyzer, settings.nldf_settings))
    raw = get_descriptors(analyzer, "l")
    record = {
        "desc": np.concatenate(parts, axis=1),
        "rho_data": raw,
        "wt": analyzer.grids.weights.copy(),
        "nspin": raw.shape[0],
        "nfeat_sl": sl.shape[1],
        "exc_core": 0.0,
        "e_tot_orig": analyzer.get("e_tot_orig"),
        "exc_orig": analyzer.get("exc_orig"),
        "energy_unit": "Hartree",
        "numbers": calc.mol.atom_charges(),
        "feature_settings_yaml": yaml.dump(settings),
    }
    return validate_record(record)
