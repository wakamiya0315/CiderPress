"""Reproduce p12-ftgpu timings on exported grids (no SCF or scheduler calls).

Requires the unchanged bench ``harness.p12_ftC`` on PYTHONPATH. Inputs are a
frozen catalog/config pair; all caches, models and evidence go under --output.
Run CPU and GPU in separate allocations with the same input and thread count.
"""

import argparse
import fcntl
import json
import resource
import time
from pathlib import Path

import numpy as np
import yaml

from ciderpress.ft import FineTuner, ReactionSet


def validate_full_benchmark(catalog, name):
    """Coordinator-approved B2 exclusions for this timing, without relabeling."""
    excluded = {"Fe/CO", "Fe/N2", "Fe/NO", "Co/N2", "Co/O", "Ni/CH", "Ni/NO"}
    metals = ("Fe", "Co", "Ni")
    adsorbates = ("H", "O", "CO", "NO", "N2", "OH", "CH", "N")
    expected = {f"C/{m}/{a}" for m in metals for a in adsorbates} - {
        "C/" + s for s in excluded
    }
    actual = {r["name"] for r in catalog["reactions"] if r["role"] == "magnetic"}
    if actual != expected:
        raise ValueError(
            "Full benchmark requires exactly the 17 coordinator-included magnetic reactions"
        )
    names = {r["name"] for r in catalog["reactions"]}
    if (
        not {f"C/{m}/{a}" for m in ("Cu", "Pt", "Pd", "Rh") for a in adsorbates}
        <= names
    ):
        raise ValueError(
            "Full benchmark lacks protection/validation/evaluation endpoints"
        )
    statuses = catalog.get("molecular_status", {})
    if not statuses or any(
        s not in ("complete", "unresolved") for s in statuses.values()
    ):
        raise ValueError("Full benchmark molecular generation is unfinished")
    if not any(r["role"] == "molecular_fit" for r in catalog["reactions"]):
        raise ValueError("Full benchmark lacks molecular fit observations")
    if (
        name == "FT-2-final"
        and sum(r["role"] == "torres" for r in catalog["reactions"]) != 6
    ):
        raise ValueError("Full FT-2 benchmark requires the six Torres reactions")
    return sorted(excluded)


def main():
    from harness import p12_ftC as h

    import ciderpress

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), required=True)
    p.add_argument("--name", default="dry-run")
    p.add_argument("--n-control", type=int, default=12)
    p.add_argument("--fit-only", action="store_true")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    catalog, config = h.load_inputs(args.run)
    exclusions = (
        validate_full_benchmark(catalog, args.name) if args.name != "dry-run" else []
    )
    names = h.allowed(catalog, args.name)
    used = {k for r in catalog["reactions"] if r["name"] in names for k in r["terms"]}
    keys = used if args.fit_only else None
    # CPU/GPU jobs may start together. Serialize initial cache creation; the
    # unchanged loader writes .npy files before publishing meta.json.
    with (out.parent / "cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = h.load_records(catalog, out.parent / "cache", keys)
    data = h.reaction_set(catalog, records, names)
    template = h.make_tuner(config, args.n_control, 1)
    ft = type(template)(
        base=config["base"]["name"],
        n_control=args.n_control,
        constraints=template.constraints,
        components=config["fit"]["components"],
        prior_scale=config["fit"]["prior_scale"],
        max_candidates=config["fit"]["max_candidates"],
        blocksize=config["fit"]["blocksize"],
        device=args.device,
    )
    ft._pilot_base_totals = {}
    result = dict(
        device=args.device,
        name=args.name,
        n_control=args.n_control,
        length_multiplier=1,
        catalog_sha256=h.digest(args.run / "catalog.json"),
        records_loaded=len(records),
        records_fit=len(data.records),
        reactions_fit=len(data.reactions),
        coordinator_excluded_magnetic=exclusions,
        points_loaded=sum(r["wt"].size for r in records.values()),
        load_seconds=time.perf_counter() - start,
    )
    t = time.perf_counter()
    ft.fit(data)
    result["fit_seconds"] = time.perf_counter() - t
    arrays, order = {}, {}
    for i, kernel in enumerate(ft.engine.kernels):
        order[f"covariance_{i}"] = sorted(kernel.cov_dict)
        arrays[f"covariance_{i}"] = np.stack(
            [kernel.cov_dict[k] for k in order[f"covariance_{i}"]]
        )
        arrays[f"alpha_{i}"] = kernel.alpha
        arrays[f"controls_{i}"] = kernel.X1ctrl
    t = time.perf_counter()
    ft.save(out / "model.yaml")
    result["save_seconds"] = time.perf_counter() - t
    order["base_fit"] = sorted(data.records)
    arrays["base_fit"] = np.array(
        [ft._base_total(data.records[k]) for k in order["base_fit"]]
    )
    if not args.fit_only:
        all_data = ReactionSet.from_records(records)
        t = time.perf_counter()
        before = ft.predict(all_data, unit="Hartree")
        result["predict_seconds"] = time.perf_counter() - t
        order["predictions"] = sorted(before)
        arrays["predictions"] = np.array(
            [list(before[k].values()) for k in order["predictions"]]
        )
        loaded = FineTuner.load(out / "model.yaml", device=args.device)
        assert yaml.dump(loaded.base) == yaml.dump(ft.base)
        # Reuse only means already computed by this device; residual integrals
        # are repeated independently through the deserialized mapped model.
        # ReactionSet copies record dictionaries; key by immutable source digest.
        means = {
            r["_ftC_source_signature"]: ft._base_total(r) for r in records.values()
        }
        loaded._base_total = lambda r: means[r["_ftC_source_signature"]]
        t = time.perf_counter()
        after = loaded.predict(all_data, unit="Hartree")
        result["reload_predict_seconds"] = time.perf_counter() - t
        arrays["reloaded_predictions"] = np.array(
            [list(after[k].values()) for k in order["predictions"]]
        )
        result["reload_max_absolute_ha"] = float(
            np.max(np.abs(arrays["predictions"] - arrays["reloaded_predictions"]))
        )
    result["fit_info"] = ft.fit_info
    result["model_sha256"] = h.digest(out / "model.yaml")
    result["benchmark_sha256"] = h.digest(__file__)
    result["harness_sha256"] = h.digest(h.__file__)
    result["array_order"] = order
    result["process_maxrss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["process_seconds"] = time.perf_counter() - start
    src = Path(ciderpress.__file__).parent
    result["source"] = str(src)
    result["native_sha256"] = {
        str(f): h.digest(f)
        for f in sorted((src / "lib").iterdir())
        if f.suffix in (".so", ".dylib")
    }
    result["ft_source_sha256"] = {
        str(f.relative_to(src)): h.digest(f) for f in sorted((src / "ft").glob("*.py"))
    }
    if args.device == "gpu":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
        result["gpu"] = {
            "name": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
            "cupy": cp.__version__,
            "pool_total_bytes": cp.get_default_memory_pool().total_bytes(),
        }
    np.savez(out / "arrays.npz", **arrays)
    (out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
