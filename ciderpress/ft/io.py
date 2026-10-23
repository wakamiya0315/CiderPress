"""GPAW-free writer for the Classic REF/SL/NLDF training layout."""

import json
from pathlib import Path

import numpy as np

from .records import validate_record


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Provenance value is not JSON serializable: {type(value).__name__}"
    )


def write_records(records, directory, *, include_val=False):
    """Write ``{system_id: record}``; return the MOLGP directory mapping.

    Records need ``nfeat_sl`` to split raw descriptors into SL and NLDF.
    Atomic numbers are retained, or derived from symbols/composition (without
    implying atomic order). ``feature_settings_yaml`` is retained if supplied.
    Free-form ``provenance`` dictionaries are stored as JSON, including nulls
    and nested lists/dictionaries, and restored by ``ReactionSet.from_hdf5``.
    NumPy metadata values are converted to JSON scalars/lists. Pointwise ``val``
    is omitted unless ``include_val=True``; no EXX calculation is performed.
    Existing files for the supplied IDs are replaced. Observations/reactions
    are not serialized.
    """
    from pyscf.lib import chkfile

    from .data import _composition

    if not records:
        raise ValueError("No records supplied")
    prepared = {}
    # Validate the entire batch before creating any files.
    for system_id, record in records.items():
        if (
            not isinstance(system_id, str)
            or not system_id
            or Path(system_id).name != system_id
            or system_id in (".", "..")
        ):
            raise ValueError("system_id must be a single filename component")
        record = validate_record(record, system_id)
        nsl = record.get("nfeat_sl")
        if (
            not isinstance(nsl, (int, np.integer))
            or not 0 <= nsl <= record["desc"].shape[1]
        ):
            raise ValueError(f"{system_id}: nfeat_sl must specify the SL feature count")
        composition = _composition(record)
        ref = {
            k: v
            for k, v in record.items()
            if k not in ("desc", "val", "provenance", "composition", "symbols")
            and v is not None
        }
        if "numbers" not in ref:
            if "symbols" in record:
                from ase.symbols import symbols2numbers

                ref["numbers"] = np.array(symbols2numbers(record["symbols"]))
            else:
                from ase.data import atomic_numbers

                ref["numbers"] = np.array(
                    [
                        atomic_numbers[s]
                        for s, count in composition.items()
                        for _ in range(int(count))
                    ]
                )
        if _composition({"numbers": ref["numbers"]}) != composition:
            raise ValueError(f"{system_id}: numbers disagree with composition/symbols")
        if "provenance" in record:
            ref["provenance_json"] = json.dumps(
                record["provenance"], default=_json_default, allow_nan=False
            )
        if include_val and "val" in record:
            values = np.asarray(record["val"], dtype=float)
            if values.shape != record["wt"].shape or not np.isfinite(values).all():
                raise ValueError(f"{system_id}: val must match wt and be finite")
            ref["val"] = values
        prepared[system_id] = dict(
            REF=ref,
            SL={"desc": record["desc"][:, :nsl]},
            NLDF={"desc": record["desc"][:, nsl:]},
        )
    directories = dict.fromkeys(("REF", "SL", "NLDF", "NLOF", "SDMX", "HYB"))
    for group in ("REF", "SL", "NLDF"):
        path = Path(directory) / group
        path.mkdir(parents=True, exist_ok=True)
        directories[group] = str(path)
        for system_id, groups in prepared.items():
            chkfile.save(str(path / f"{system_id}.hdf5"), "train_data", groups[group])
    return directories
