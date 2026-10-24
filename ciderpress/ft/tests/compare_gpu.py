"""Compare benchmark_gpu outputs with strict elementwise relative errors."""

import argparse
import json
from pathlib import Path

import numpy as np


def errors(actual, expected):
    delta = np.abs(actual - expected)
    nonzero = expected != 0
    zero_mismatch = int(np.count_nonzero(delta[~nonzero]))
    relative = np.zeros_like(delta)
    np.divide(delta, np.abs(expected), out=relative, where=nonzero)
    maximum = float(np.max(relative, initial=0))
    return dict(
        max_absolute=float(np.max(delta, initial=0)),
        max_relative=maximum if zero_mismatch == 0 else None,
        zero_reference_mismatches=zero_mismatch,
        passed=zero_mismatch == 0 and maximum <= 1e-10,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cpu", type=Path)
    p.add_argument("gpu", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b = (
        json.loads((path / "result.json").read_text()) for path in (args.cpu, args.gpu)
    )
    for key in (
        "catalog_sha256",
        "n_control",
        "length_multiplier",
        "array_order",
        "records_fit",
        "reactions_fit",
        "coordinator_excluded_magnetic",
    ):
        if a[key] != b[key]:
            raise ValueError(f"CPU/GPU settings differ: {key}")
    result = dict(tolerance_relative=1e-10, arrays={}, reload={})
    with np.load(args.cpu / "arrays.npz") as x, np.load(args.gpu / "arrays.npz") as y:
        if set(x.files) != set(y.files):
            raise ValueError("CPU/GPU array sets differ")
        for key in x.files:
            result["arrays"][key] = errors(y[key], x[key])
            if key.startswith("controls_"):
                result["arrays"][key]["bitwise_identical"] = bool(
                    np.array_equal(x[key], y[key])
                )
        if "predictions" in x:
            for device, arrays in (("cpu", x), ("gpu", y)):
                result["reload"][device] = errors(
                    arrays["reloaded_predictions"], arrays["predictions"]
                )
            for col, name in enumerate(("base", "correction", "fine_tuned")):
                result["arrays"]["prediction_" + name] = errors(
                    y["predictions"][:, col], x["predictions"][:, col]
                )
    result["passed"] = all(
        r["passed"] for group in ("arrays", "reload") for r in result[group].values()
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
