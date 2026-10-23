"""Energy observations with explicit units and composition-aware splitting."""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.formula import Formula
from ase.units import Hartree, kcal, mol

from .records import validate_record


def to_hartree(unit):
    """Hartree per input unit; noise is a standard deviation in that same unit."""
    factors = {
        "eV": 1 / Hartree,
        "Hartree": 1.0,
        "Ha": 1.0,
        "kcal/mol": kcal / mol / Hartree,
        "kcal per mol": kcal / mol / Hartree,
    }
    try:
        return factors[unit]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown energy unit {unit!r}; use eV, kcal/mol or Hartree"
        ) from None


def _composition(record):
    if "composition" in record:
        value = record["composition"]
        counts = dict(Formula(value).count()) if isinstance(value, str) else dict(value)
    elif "symbols" in record:
        symbols = record["symbols"]
        counts = (
            dict(Formula(symbols).count())
            if isinstance(symbols, str)
            else dict(Counter(symbols))
        )
    elif "numbers" in record:
        numbers = np.asarray(record["numbers"])
        if (
            numbers.ndim != 1
            or not np.isfinite(numbers).all()
            or np.any(numbers != numbers.astype(int))
            or np.any(numbers <= 0)
            or np.any(numbers >= len(chemical_symbols))
        ):
            raise ValueError(
                "Atomic numbers must be positive integers for real elements"
            )
        counts = dict(Counter(chemical_symbols[int(z)] for z in numbers))
    else:
        raise ValueError(
            "Missing composition: supply composition, symbols or numbers for each record"
        )
    if not counts or any(
        k not in atomic_numbers
        or k == "X"
        or not np.isfinite(v)
        or v <= 0
        or int(v) != v
        for k, v in counts.items()
    ):
        raise ValueError(f"Invalid atomic composition {counts!r}")
    return counts


@dataclass(frozen=True)
class Reaction:
    name: str
    terms: tuple
    reference_ha: float
    noise_ha: float
    group: str
    split: str = "train"
    absolute: bool = False


class ReactionSet:
    """Mapping of exported records and energy observations (references in Ha).

    Arrays use the New GPAW/Classic training schema, in atomic units. The
    exporter does not supply compositions: attach ``record['symbols']`` from
    the calculation's atoms, or pass ``compositions={id: 'Ni4O', ...}``.
    ``unit`` and ``noise_unit`` apply to reaction references/noise only.
    """

    def __init__(self, records, reactions=()):
        if not records:
            raise ValueError("No records supplied")
        if any(not isinstance(k, str) or not k for k in records):
            raise ValueError("Record identifiers must be nonempty strings")
        self.records = {k: validate_record(v, k) for k, v in records.items()}
        self.reactions = list(reactions)
        if len(self.records) != len(records):
            raise ValueError("Record identifiers must be unique strings")
        for reaction in self.reactions:
            self._check_terms(dict(reaction.terms), balanced=not reaction.absolute)

    @classmethod
    def from_records(cls, records, *, compositions=None):
        """Accept ``{system_id: record}``, without changing the input arrays."""
        enriched = {k: dict(v) for k, v in records.items()}
        if compositions is not None:
            for key, composition in compositions.items():
                if key not in enriched:
                    raise ValueError(f"Composition supplied for missing record {key!r}")
                enriched[key]["composition"] = composition
        return cls(enriched)

    @classmethod
    def from_hdf5(cls, directories, system_ids, *, compositions=None):
        """Load REF/SL/NLDF from ft or new_export, including optional provenance."""
        from ciderpress.models.train import MOLGP

        return cls.from_records(
            {key: MOLGP.load_data(directories, key, False) for key in system_ids},
            compositions=compositions,
        )

    def to_hdf5(self, directory, *, include_val=False):
        """Write records only (not reactions); return the MOLGP directory mapping."""
        from .io import write_records

        return write_records(self.records, directory, include_val=include_val)

    def _check_terms(self, terms, *, balanced=True):
        if not terms or any(not np.isfinite(c) or c == 0 for c in terms.values()):
            raise ValueError(
                "Reaction needs finite, nonzero stoichiometric coefficients"
            )
        missing = set(terms) - self.records.keys()
        if missing:
            raise ValueError(f"Missing records: {sorted(missing)}")
        totals = Counter()
        for key, coefficient in terms.items():
            for element, count in _composition(self.records[key]).items():
                totals[element] += coefficient * count
        if balanced and any(abs(v) > 1e-8 for v in totals.values()):
            raise ValueError(
                f"Unbalanced stoichiometry (products minus reactants): {dict(totals)}"
            )

    def add_reaction(
        self,
        name,
        terms,
        *,
        reference,
        unit,
        noise,
        group="default",
        split="train",
        noise_unit=None,
    ):
        """Add a balanced reaction; reference and noise share ``unit`` by default."""
        return self._add(
            name, terms, reference, unit, noise, group, split, noise_unit, False
        )

    def add_energy(
        self,
        system_id,
        *,
        reference,
        unit,
        noise,
        group="molecules",
        split="train",
        noise_unit=None,
    ):
        """Explicit absolute total-energy reference, e.g. a high-level molecule."""
        return self._add(
            f"energy:{system_id}",
            {system_id: 1},
            reference,
            unit,
            noise,
            group,
            split,
            noise_unit,
            True,
        )

    def _add(
        self, name, terms, reference, unit, noise, group, split, noise_unit, absolute
    ):
        if any(r.name == name for r in self.reactions):
            raise ValueError(f"Duplicate observation name {name!r}")
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        factor = to_hartree(unit)
        noise_factor = to_hartree(unit if noise_unit is None else noise_unit)
        if not np.isfinite(reference) or not np.isfinite(noise) or noise <= 0:
            raise ValueError(
                "Reference must be finite and noise a positive standard deviation"
            )
        self._check_terms(terms, balanced=not absolute)
        self.reactions.append(
            Reaction(
                str(name),
                tuple(terms.items()),
                float(reference * factor),
                float(noise * noise_factor),
                str(group),
                split,
                absolute,
            )
        )
        return self

    def _subset(self, reactions):
        used = {key for r in reactions for key, _ in r.terms}
        if not used:
            raise ValueError("Requested split contains no observations")
        return ReactionSet(
            {k: v for k, v in self.records.items() if k in used}, reactions
        )

    def select(self, *, split=None, groups=None):
        """Select observations and only their systems, preventing feature leakage."""
        if isinstance(groups, str):
            groups = {groups}
        return self._subset(
            [
                r
                for r in self.reactions
                if (split is None or r.split == split)
                and (groups is None or r.group in groups)
            ]
        )

    def leave_one_group_out(self):
        """Yield ``(group, train, test)``; make a fresh FineTuner for each fold."""
        groups = sorted({r.group for r in self.reactions})
        if len(groups) < 2:
            raise ValueError("Leave-one-group-out needs at least two groups")
        for group in groups:
            yield (
                group,
                self._subset(
                    [
                        replace(r, split="train")
                        for r in self.reactions
                        if r.group != group
                    ]
                ),
                self._subset(
                    [
                        replace(r, split="test")
                        for r in self.reactions
                        if r.group == group
                    ]
                ),
            )

    def split_by_group(self, *, validation=(), test=()):
        """Return disjoint train/validation/test ReactionSets (empty parts are None)."""
        validation = {validation} if isinstance(validation, str) else set(validation)
        test = {test} if isinstance(test, str) else set(test)
        known = {r.group for r in self.reactions}
        if validation & test or (validation | test) - known:
            raise ValueError(
                "Split groups must exist and validation/test must be disjoint"
            )
        parts = {"train": [], "validation": [], "test": []}
        for r in self.reactions:
            part = (
                "test"
                if r.group in test
                else "validation"
                if r.group in validation
                else "train"
            )
            parts[part].append(
                Reaction(
                    r.name,
                    r.terms,
                    r.reference_ha,
                    r.noise_ha,
                    r.group,
                    part,
                    r.absolute,
                )
            )
        return {k: self._subset(v) if v else None for k, v in parts.items()}

    def with_records(self, records):
        """Replace densities/energies, preserving observations and compositions."""
        if set(records) != set(self.records):
            raise ValueError(
                "Recomputed records must have exactly the original system IDs"
            )
        enriched = {}
        for key, record in records.items():
            enriched[key] = dict(record)
            old = _composition(self.records[key])
            if (
                any(k in record for k in ("composition", "symbols", "numbers"))
                and _composition(record) != old
            ):
                raise ValueError(f"Recomputed composition changed for {key!r}")
            enriched[key]["composition"] = old
        return ReactionSet(enriched, self.reactions)
