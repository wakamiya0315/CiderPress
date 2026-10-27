"""MOLGP2 orchestration with a frozen mean and finite-set safeguards."""

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml

from ciderpress.dft.model_utils import _is_full_xc_model, load_cider_model
from ciderpress.dft.settings import CFC
from ciderpress.dft.xc_evaluator import RBFEvaluator
from ciderpress.dft.xc_evaluator2 import MappedXC2
from ciderpress.gpu import policy
from ciderpress.gpu.semilocal import supported
from ciderpress.models.dft_kernel import DFTKernel2
from ciderpress.models.kernels import DiffRBF
from ciderpress.models.train import MOLGP2

from .data import ReactionSet, _composition, to_hartree
from .records import covariance, integrate


@dataclass
class Constraints:
    """Finite-set regularisation, with all energy scales in ``unit``.

    Absolute scales are per atom; UEG noise is per electron. UEG and absolute
    protection are enabled by default. Supply molecules as a ReactionSet:
    with no observations it is distilled to zero correction; otherwise its
    training observations (including add_energy references) are fitted.

    The hard absolute cap applies only to training/protection records. It is
    enforced by uniformly shrinking the fitted residual coefficients if
    necessary. This can sacrifice reaction accuracy and is reported explicitly.
    None disables the respective absolute cap or zero-centred soft penalty.
    These safeguards do not bound potentials or guarantee SCF stability.
    """

    ueg: bool = True
    molecules: ReactionSet | None = field(default=None, repr=False)
    max_abs_correction: float | None = 0.5
    absolute_noise: float | None = 0.05
    molecular_noise: float = 0.01
    ueg_noise: float = 0.001
    unit: str = "eV"
    densities: tuple = tuple(np.geomspace(1e-4, 10.0, 9))
    spin_polarisations: tuple = (0.0, 0.5, 0.99)

    def __post_init__(self):
        to_hartree(self.unit)
        for key in (
            "max_abs_correction",
            "absolute_noise",
            "molecular_noise",
            "ueg_noise",
        ):
            value = getattr(self, key)
            if value is None and key in ("max_abs_correction", "absolute_noise"):
                continue
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.molecules is not None and not isinstance(self.molecules, ReactionSet):
            raise TypeError("molecules must be a ReactionSet")
        if (
            not len(self.densities)
            or not np.isfinite(self.densities).all()
            or np.any(np.asarray(self.densities) <= 0)
        ):
            raise ValueError(
                "UEG densities must be positive finite atomic-unit densities"
            )
        if (
            not len(self.spin_polarisations)
            or not np.isfinite(self.spin_polarisations).all()
            or np.any(np.abs(self.spin_polarisations) >= 1)
        ):
            raise ValueError("UEG spin polarisations must satisfy abs(zeta) < 1")

    def config(self):
        # Avoid dataclasses.asdict deep-copying potentially huge molecular arrays.
        return {
            key: (
                [float(v) for v in value]
                if key in ("densities", "spin_polarisations")
                else value
            )
            for key, value in vars(self).items()
            if key != "molecules"
        }


class FineTuner:
    """Fit a residual GP around an immutable copy of a full-XC CIDER model.

    ``components='x'`` (default) adds a SEP, LDA_X-scaled RBF on the base's
    exchange feature transforms. ``'c'`` adds an NPOL, GGA_C_PBE-scaled RBF;
    ``'xc'`` adds both. All added additive baselines are None. Length scales
    are inherited from the respective base RBF, whose old coefficients and
    control points remain unchanged. ``n_control`` is a cap per component.

    ``prior_scale`` is the standard deviation of the dimensionless residual
    enhancement, either a scalar shared by all added kernels or a mapping
    with exactly the requested component keys (for example x and c).
    Noise is specified per observation by ReactionSet. Candidate
    pools are deterministically subsampled before the engine's dense pivoted
    Cholesky selection, keeping candidate memory bounded by max_candidates.
    """

    def __init__(
        self,
        base="CIDER26XCSURFSCI",
        n_control=200,
        *,
        constraints=None,
        components="x",
        prior_scale=0.1,
        max_candidates=1500,
        candidate_groups=None,
        group_fractions=None,
        blocksize=4000,
        rhocut=1e-8,
        jitter=1e-12,
        device="cpu",
    ):
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        if device == "gpu":
            from ._gpu import require_gpu

            require_gpu()
        self.device = device
        self._base = deepcopy(load_cider_model(base))
        if not isinstance(self._base, MappedXC2) or not supported(
            self._base, self._base.settings.nfeat
        ):
            raise ValueError(
                "FineTuner requires an ordinary RBF SEP/NPOL MappedXC2 supported by the GPU path"
            )
        if self._base.libxc_baseline is not None:
            raise ValueError(
                "Separate libxc_baseline is unsupported; use a self-contained full-XC model"
            )
        if not _is_full_xc_model(self._base):
            raise ValueError("FineTuner requires a full-XC frozen base")
        if components not in ("x", "c", "xc"):
            raise ValueError("components must be x, c or xc")
        for key, value in (
            ("n_control", n_control),
            ("max_candidates", max_candidates),
            ("blocksize", blocksize),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if isinstance(prior_scale, Mapping):
            if set(prior_scale) != set(components):
                raise ValueError("prior_scale mapping must have exactly the component keys")
            prior_scale = dict(prior_scale)
            scales = [(f"prior_scale[{k}]", v) for k, v in prior_scale.items()]
        else:
            scales = [("prior_scale", prior_scale)]
        for key, value in [*scales, ("jitter", jitter)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not np.isfinite(rhocut) or rhocut < 0:
            raise ValueError("rhocut must be finite and nonnegative")
        if max_candidates < n_control:
            raise ValueError("max_candidates must be at least n_control")
        self.n_control, self.components = n_control, components
        self.prior_scale, self.max_candidates = prior_scale, max_candidates
        if (candidate_groups is None) != (group_fractions is None):
            raise ValueError("candidate_groups and group_fractions must be supplied together")
        self.candidate_groups = None if candidate_groups is None else dict(candidate_groups)
        self.group_fractions = None if group_fractions is None else dict(group_fractions)
        if self.candidate_groups is not None:
            if (not self.group_fractions
                or any(not isinstance(k, str) or not isinstance(v, str)
                       for k, v in self.candidate_groups.items())
                or any(not isinstance(k, str) or k == "__ueg__"
                       for k in self.group_fractions)
                or any(not np.isfinite(v) or v <= 0 for v in self.group_fractions.values())
                or not np.isclose(sum(self.group_fractions.values()), 1, rtol=0, atol=1e-12)
                or not set(self.candidate_groups.values()) <= self.group_fractions.keys()):
                raise ValueError("Candidate groups need named positive fractions summing to one")
        self.blocksize, self.rhocut, self.jitter = blocksize, rhocut, jitter
        self.constraints = Constraints() if constraints is None else constraints
        if not isinstance(self.constraints, Constraints):
            raise TypeError("constraints must be Constraints")
        self._base_yaml = yaml.dump(self._base)
        self._settings_yaml = yaml.dump(self._base.settings)
        self._protection_missing = False
        self._model = None
        self.engine = None
        self.fit_info = None

    @property
    def base(self):
        """Independent copy of the frozen model."""
        return deepcopy(self._base)

    @property
    def model(self):
        """Independent inference model, suitable for a calculator or SCF hook."""
        self._require_fit()
        return deepcopy(self._model)

    def _require_fit(self):
        if self._model is None:
            raise RuntimeError("Call fit() or FineTuner.load() first")

    def _validate_settings(self, record):
        if record["desc"].shape[1] != self._base.settings.nfeat:
            raise ValueError("Record feature count differs from frozen base")
        declared = record.get("feature_settings_yaml")
        if isinstance(declared, bytes):
            declared = declared.decode()
        if declared is not None and declared != self._settings_yaml:
            raise ValueError("Record feature settings differ from frozen base")
        # The export digest describes the SCF model, which may be the current
        # fine-tuned model. It must not be required to equal the frozen mean.

    def _kernels(self):
        kernels = []
        for component in self.components:
            mode, baseline = (
                ("SEP", "LDA_X") if component == "x" else ("NPOL", "GGA_C_PBE")
            )
            templates = [
                k
                for k in self._base.kernels
                if k.mode == mode and k._mul_basefunc == baseline
            ]
            if not templates:
                raise ValueError(f"Base has no {mode}/{baseline} feature template")
            template = templates[0]
            if not template.fevals:
                raise ValueError(f"Base {component} template has no RBF evaluator")
            rbf = template.fevals[0]
            length_scale = np.sqrt(0.5 / rbf._exps)
            scale = (self.prior_scale[component]
                     if isinstance(self.prior_scale, Mapping) else self.prior_scale)
            kernels.append(
                DFTKernel2(
                    scale**2 * DiffRBF(length_scale=length_scale),
                    deepcopy(template.feature_list),
                    mode,
                    baseline,
                    additive_baseline=None,
                    component=component,
                    ctrl_nmax=self.n_control,
                    ctrl_tol=1e-8,
                )
            )
        return kernels

    def _ueg_records(self):
        records = {}
        if not self.constraints.ueg:
            return records
        for zeta in self.constraints.spin_polarisations:
            for density in self.constraints.densities:
                rho = 0.5 * float(density) * np.array([1 + zeta, 1 - zeta])
                raw = np.zeros((2, 5, 1))
                raw[:, 0, 0] = 2 * rho
                raw[:, 4, 0] = 2 * CFC * rho * (2 * rho) ** (2 / 3)
                desc = np.stack(
                    [
                        self._base.settings.ueg_vector(
                            rho=2 * r, with_normalizers=False
                        )
                        for r in rho
                    ]
                )[..., None]
                records[f"ueg:{zeta}:{density}"] = {
                    "desc": desc,
                    "rho_data": raw,
                    "wt": np.array([1 / density]),
                    "nspin": 2,
                    "exc_core": 0.0,
                }
        return records

    def _select_controls(self, kernel, records, record_ids=None):
        if self.candidate_groups is not None:
            return self._select_grouped_controls(kernel, records, record_ids)
        # Equal deterministic sampling budgets per system protect small molecules
        # against a much larger surface grid. The reduction itself is the existing
        # DFTKernel pivoted-Cholesky implementation, not random point selection.
        budget = max(1, self.max_candidates // len(records) // 2)
        pool = []
        for record in records:
            valid = np.flatnonzero(record["rho_data"][:, 0].sum(0) > self.rhocut)
            if valid.size:
                ids = valid[
                    np.linspace(0, valid.size - 1, min(budget, valid.size), dtype=int)
                ]
                sample = self._base.settings.normalizers.get_normalized_feature_vector(
                    record["desc"][..., ids]
                )
                pool.append(kernel.get_descriptors(sample))
        if not pool:
            raise ValueError("No control candidates above the density cutoff")
        pool = np.concatenate(pool)
        if len(pool) > self.max_candidates:
            pool = pool[np.linspace(0, len(pool) - 1, self.max_candidates, dtype=int)]
        if not np.isfinite(pool).all():
            raise ValueError("Control candidates contain nonfinite normalized features")
        kernel.X1ctrl = kernel._reduce_npts(pool)

    def _select_grouped_controls(self, kernel, records, record_ids):
        """Stratify only the pool; keep the existing kernel reduction unchanged.

        IDs are the original ReactionSet IDs (also for constraints.molecules).
        None identifies an internally generated UEG record. Fractions budget
        physical records; UEG is appended separately at the legacy per-record
        budget. Unused capacity is not redistributed. Each record receives an
        equal candidate-row budget; its spin multiplicity determines how many
        grid points fit in that budget.
        """
        if record_ids is None or len(record_ids) != len(records):
            raise ValueError("Grouped candidates require matching record IDs")
        groups = []
        for key in record_ids:
            if key is not None and key not in self.candidate_groups:
                raise ValueError(f"Missing candidate group for record {key!r}")
            groups.append("__ueg__" if key is None else self.candidate_groups[key])
        counts = {g: groups.count(g) for g in set(groups)}
        limits = {g: int(self.max_candidates * f) for g, f in self.group_fractions.items()}
        budgets = {g: limits[g] // n for g, n in counts.items() if g != "__ueg__"}
        ueg_grid_budget = max(1, self.max_candidates // len(records) // 2)
        pool, origins = [], []
        per_record = []
        for record, key, group in zip(records, record_ids, groups):
            valid = np.flatnonzero(record["rho_data"][:, 0].sum(0) > self.rhocut)
            rows_per_point = 0
            grid_budget = 0
            if valid.size:
                probe = self._base.settings.normalizers.get_normalized_feature_vector(
                    record["desc"][..., valid[:1]]
                )
                rows_per_point = len(kernel.get_descriptors(probe))
                grid_budget = (ueg_grid_budget if group == "__ueg__"
                               else budgets[group] // rows_per_point)
            n = min(grid_budget, valid.size)
            rows = 0
            if n:
                ids = valid[np.linspace(0, valid.size - 1, n, dtype=int)]
                sample = self._base.settings.normalizers.get_normalized_feature_vector(
                    record["desc"][..., ids]
                )
                candidates = kernel.get_descriptors(sample)
                rows = len(candidates)
                pool.append(candidates)
                origins.extend([(key, group)] * rows)
            per_record.append(dict(record_id=key, group=group, grid_budget=grid_budget,
                                   candidate_budget=(None if group == "__ueg__" else budgets[group]),
                                   rows_per_grid_point=rows_per_point,
                                   sampled_grid_points=int(n), candidate_rows=rows))
        if not pool:
            raise ValueError("No control candidates above the density cutoff")
        pool = np.concatenate(pool)
        if not np.isfinite(pool).all():
            raise ValueError("Control candidates contain nonfinite normalized features")
        kernel.X1ctrl = kernel._reduce_npts(pool)
        # _reduce_npts returns exact input rows. Attribute duplicate descriptors
        # to their first pool occurrence and expose all possible origin groups.
        lookup = {}
        for i, row in enumerate(pool):
            lookup.setdefault(row.tobytes(), []).append(i)
        selected = []
        for row in kernel.X1ctrl:
            ids = lookup[row.tobytes()]
            i = ids[0]
            selected.append(dict(pool_index=i, record_id=origins[i][0], group=origins[i][1],
                                 possible_groups=sorted({origins[j][1] for j in ids})))
        return dict(group_candidate_limits=limits, records=per_record,
                    candidate_rows=len(pool), selected=selected,
                    duplicate_attribution="first pool occurrence; possible_groups lists ambiguity")

    def _base_total(self, record):
        self._validate_settings(record)
        return (
            record["e_tot_orig"]
            - record["exc_orig"]
            + integrate(
                self._base,
                record,
                blocksize=self.blocksize,
                rhocut=self.rhocut,
                device=self.device,
            )
        )

    def fit(self, data):
        """Fit training observations only, returning self; each fit starts fresh.

        Mean = e_tot_orig - exc_orig + integral(frozen base) + exc_core.
        Even after an SCF callback exported a corrected model's energies, this
        formula removes that model's XC before adding the original frozen XC.
        No already-corrected energy is subtracted from the reference label.
        """
        if not isinstance(data, ReactionSet):
            raise TypeError("fit expects a ReactionSet")
        if self._protection_missing and self.constraints.molecules is None:
            raise ValueError(
                "Saved YAML omits molecular records; supply constraints.molecules before refitting"
            )
        train = data.select(split="train")
        protection = self.constraints.molecules
        if protection is not None and protection.reactions:
            protection = protection.select(split="train")
        # Namespace protection IDs so independently supplied sets cannot collide.
        records = {f"train:{k}": r for k, r in train.records.items()}
        if protection is not None:
            records.update({f"protect:{k}": r for k, r in protection.records.items()})
        for record in records.values():
            self._validate_settings(record)
            _composition(record)  # needed even when absolute safeguards are disabled
        ueg = self._ueg_records()
        ueg_records = {f"ueg:{key}": value for key, value in ueg.items()}
        candidate_ids = list(train.records)
        if protection is not None:
            candidate_ids += list(protection.records)
        candidate_ids += [None] * len(ueg)
        candidate_info = []
        kernels = self._kernels()
        engine = MOLGP2(kernels, deepcopy(self._base.settings))
        engine.numerical_epsilon = self.jitter
        # Candidate selection and the dense solve retain their deterministic
        # CPU implementations; explicit device quadratures dispatch separately.
        with policy(None):
            for kernel in kernels:
                candidate_info.append(self._select_controls(
                    kernel, list(records.values()) + list(ueg.values()), candidate_ids
                ))
                for key, record in {**records, **ueg_records}.items():
                    kernel.cov_dict[key] = covariance(
                        kernel,
                        record,
                        self._base.settings,
                        self.blocksize,
                        self.rhocut,
                        device=self.device,
                    )
                    kernel.base_dict[key] = 0.0
                    engine.ks_baseline_dict[key] = 0.0
            means = {key: self._base_total(record) for key, record in records.items()}
            observations = []

            def add(terms, target, noise):
                observations.append(
                    (
                        2,
                        {
                            "structs": list(terms),
                            "counts": list(terms.values()),
                            "energy": target,
                            "unit": 1.0,
                            "noise": noise,
                        },
                    )
                )

            def add_references(dataset, namespace):
                for r in dataset.reactions:
                    terms = {f"{namespace}:{k}": c for k, c in r.terms}
                    mean = sum(c * means[k] for k, c in terms.items())
                    add(terms, r.reference_ha - mean, r.noise_ha)

            add_references(train, "train")
            factor = to_hartree(self.constraints.unit)
            if protection is not None:
                if protection.reactions:
                    add_references(protection, "protect")
                else:
                    for key in protection.records:
                        add(
                            {f"protect:{key}": 1},
                            0.0,
                            self.constraints.molecular_noise * factor,
                        )
            if self.constraints.absolute_noise is not None:
                for key, record in records.items():
                    natoms = sum(_composition(record).values())
                    add(
                        {key: 1}, 0.0, natoms * self.constraints.absolute_noise * factor
                    )
            for key in ueg_records:
                add({key: 1}, 0.0, self.constraints.ueg_noise * factor)
            engine.add_reactions(observations)
            engine.fit()
            per_atom = np.array(
                [
                    sum(k.alpha @ k.cov_dict[key] for k in kernels)
                    / sum(_composition(record).values())
                    for key, record in records.items()
                ]
            )
            before = float(np.max(np.abs(per_atom)))
            scale = 1.0
            bound = self.constraints.max_abs_correction
            if bound is not None and before > bound * factor:
                scale = bound * factor / before
                for kernel in kernels:
                    kernel.alpha *= scale
                engine.alpha_mol_ *= scale
            correction = engine.map(
                [lambda k: RBFEvaluator(k.kernel, k.X1ctrl, k.alpha)] * len(kernels)
            )
        # Always append to the frozen model, never to a previous fitted model.
        model = deepcopy(self._base)
        model.kernels.extend(correction.kernels)
        if not supported(model, model.settings.nfeat):
            raise RuntimeError(
                "Mapped residual unexpectedly failed GPU compatibility predicate"
            )
        if yaml.dump(self._base) != self._base_yaml:
            raise RuntimeError("Frozen base was mutated during training")
        ueg_error = max(
            (
                abs(sum(k.alpha @ k.cov_dict[key] for k in kernels))
                for key in ueg_records
            ),
            default=0.0,
        )
        self.fit_info = {
            "n_train": len(train.reactions),
            "n_observations": len(observations),
            "n_control": [k.Nctrl for k in kernels],
            "absolute_scale": float(scale),
            "max_abs_before_ha_per_atom": before,
            "max_abs_after_ha_per_atom": before * scale,
            "max_ueg_ha_per_electron": float(ueg_error),
            "gpu_compatible": True,
            "base_sha256": sha256(self._base_yaml.encode()).hexdigest(),
        }
        if self.candidate_groups is not None:
            self.fit_info["candidate_configuration"] = {
                "candidate_groups": self.candidate_groups,
                "group_fractions": self.group_fractions,
                "max_candidates": self.max_candidates,
                "ueg_separate": True,
                "budget_unit": "descriptor_rows",
            }
            self.fit_info["candidate_pools"] = candidate_info
        self.engine, self._model = engine, model
        return self

    def predict(self, data, *, unit="eV"):
        """Per-system frozen-base, correction and corrected total energies."""
        self._require_fit()
        factor = to_hartree(unit)
        result = {}
        correction = MappedXC2(
            self._model.kernels[len(self._base.kernels) :], self._base.settings
        )
        with policy(None):
            for key, record in data.records.items():
                base = self._base_total(record)
                delta = integrate(
                    correction,
                    record,
                    core=False,
                    blocksize=self.blocksize,
                    rhocut=self.rhocut,
                    device=self.device,
                )
                result[key] = {
                    "base": base / factor,
                    "correction": delta / factor,
                    "fine_tuned": (base + delta) / factor,
                }
        return result

    def evaluate(self, data, *, unit="eV"):
        """Reaction MAE and signed ME (prediction minus reference), by group."""
        if not data.reactions:
            raise ValueError("Evaluation needs reference observations")
        predictions = self.predict(data, unit=unit)
        factor = to_hartree(unit)
        rows, grouped = [], {}
        for reaction in data.reactions:
            row = {
                "name": reaction.name,
                "group": reaction.group,
                "split": reaction.split,
                "reference": reaction.reference_ha / factor,
            }
            for method in ("base", "fine_tuned"):
                value = sum(c * predictions[k][method] for k, c in reaction.terms)
                row[method] = value
                row[method + "_error"] = value - row["reference"]
            rows.append(row)
            grouped.setdefault(reaction.group, []).append(row)

        def statistics(items):
            return {
                method: {
                    "n": len(items),
                    "MAE": float(np.mean([abs(r[method + "_error"]) for r in items])),
                    "ME": float(np.mean([r[method + "_error"] for r in items])),
                }
                for method in ("base", "fine_tuned")
            }

        return {
            "unit": unit,
            "overall": statistics(rows),
            "groups": {g: statistics(rs) for g, rs in grouped.items()},
            "reactions": rows,
        }

    def save(self, path):
        """Write a trusted-object MappedXC2 YAML, directly usable by calculators.

        Includes small inference/provenance metadata, but no training records
        or posterior uncertainty. Load only trusted YAML files.
        """
        self._require_fit()
        model = self.model
        model.ft_metadata = {
            "version": 1,
            "base_kernel_count": len(self._base.kernels),
            "fit_info": self.fit_info,
            "n_control": self.n_control,
            "components": self.components,
            "prior_scale": self.prior_scale,
            "max_candidates": self.max_candidates,
            "blocksize": self.blocksize,
            "rhocut": self.rhocut,
            "jitter": self.jitter,
            "constraints": self.constraints.config(),
            "needs_molecules": self.constraints.molecules is not None,
        }
        if self.candidate_groups is not None:
            model.ft_metadata.update(
                candidate_groups=self.candidate_groups,
                group_fractions=self.group_fractions,
            )
        Path(path).write_text(yaml.dump(model, Dumper=yaml.CDumper), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path, *, constraints=None, device="cpu"):
        """Reload inference state; supply protection records again before refit."""
        model = load_cider_model(path)
        meta = getattr(model, "ft_metadata", None)
        if not meta or meta.get("version") != 1:
            raise ValueError(
                "YAML has no supported FineTuner metadata; use FineTuner(base=...) instead"
            )
        base = deepcopy(model)
        del base.ft_metadata
        base.kernels = base.kernels[: meta["base_kernel_count"]]
        config = {
            k: meta[k]
            for k in (
                "n_control",
                "components",
                "prior_scale",
                "max_candidates",
                "blocksize",
                "rhocut",
                "jitter",
            )
        }
        result = cls(
            base,
            candidate_groups=meta.get("candidate_groups"),
            group_fractions=meta.get("group_fractions"),
            constraints=constraints or Constraints(**meta["constraints"]),
            device=device,
            **config,
        )
        result._model, result.fit_info = model, meta["fit_info"]
        result._protection_missing = meta["needs_molecules"]
        return result

    def fit_self_consistent(self, data, recompute, *, n_steps=1):
        """Initial fit, then n_steps density/refit cycles; return evaluations.

        ``recompute(model, data, step) -> {system_id: record}`` must perform
        and qualify the user's GPAW/PySCF SCF, then export same-density total
        and XC energies in Ha. Step starts at 1. IDs/compositions/references
        remain fixed. Protection records remain fixed unless the caller
        explicitly replaces constraints.molecules. No SCF is launched by ft.
        Exceptions propagate; no convergence or stability is inferred.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer")
        self.fit(data)
        history = [self.evaluate(data)]
        for step in range(1, n_steps + 1):
            records = recompute(self.model, data, step)
            data = data.with_records(records)
            self.fit(data)
            history.append(self.evaluate(data))
        return history
