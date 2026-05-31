"""Tune reducers parallel grains on the current machine.

The tuner searches powers-of-two grain values for each algorithm class and
reports values that minimize geometric mean runtime over representative
workloads, subject to a per-workload regression guard. It is a benchmark aid,
not a promise of portable universal defaults.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import reducers as rd
from _benchutils import print_environment, timeit

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover - depends on optional benchmark env
    bn = None


GRAIN_ENV = {
    "axis_scan_plain": "REDUCERS_AXIS_SCAN_PLAIN_GRAIN",
    "axis_scan_nan": "REDUCERS_AXIS_SCAN_NAN_GRAIN",
    "axis_scan_var": "REDUCERS_AXIS_SCAN_VAR_GRAIN",
    "axis_weighted": "REDUCERS_AXIS_WEIGHTED_GRAIN",
    "axis_order_median": "REDUCERS_AXIS_ORDER_MEDIAN_GRAIN",
    "axis_order_percentile": "REDUCERS_AXIS_ORDER_PERCENTILE_GRAIN",
    "minmax_1d": "REDUCERS_MINMAX_1D_GRAIN",
}
SEED = 20250311


@dataclass(frozen=True)
class Workload:
    """One callable benchmark item owned by a parallel grain class."""

    label: str
    grain: str
    func: Callable[[], object]


def make_stack(shape: tuple[int, int, int], dtype: str, *, nan: bool) -> np.ndarray:
    """Return a deterministic benchmark stack."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(1000.0, 30.0, size=shape).astype(dtype)
    if nan:
        a.ravel()[::101] = np.nan
    return np.ascontiguousarray(a)


def make_1d(length: int, dtype: str, *, nan: bool) -> np.ndarray:
    """Return a deterministic 1-D benchmark array."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(1000.0, 30.0, size=length).astype(dtype)
    if nan:
        a[:: max(1, length // 97)] = np.nan
    return np.ascontiguousarray(a)


def build_workloads(profile: str) -> list[Workload]:
    """Build representative workloads for the tuning profile."""
    guard_shapes = [(5, 100, 100)]
    small_shapes = [*guard_shapes, (31, 100, 100), (100, 100, 100)]
    full_shapes = [*small_shapes, (11, 512, 512)]
    shapes = small_shapes if profile == "quick" else full_shapes
    dtypes = ["float64"] if profile == "quick" else ["float64", "float32"]
    workloads: list[Workload] = []

    for dtype in dtypes:
        for shape in shapes:
            clean = make_stack(shape, dtype, nan=False)
            nans = make_stack(shape, dtype, nan=True)
            axis = 0
            weights = np.linspace(0.5, 1.5, shape[0]).astype(dtype)
            workloads.extend(
                [
                    Workload(
                        f"{dtype} {shape} axis0 mean",
                        "axis_scan_plain",
                        lambda a=clean, ax=axis: rd.mean(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 min",
                        "axis_scan_plain",
                        lambda a=clean, ax=axis: rd.min(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanmean",
                        "axis_scan_nan",
                        lambda a=nans, ax=axis: rd.nanmean(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanmin",
                        "axis_scan_nan",
                        lambda a=nans, ax=axis: rd.nanmin(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 var",
                        "axis_scan_var",
                        lambda a=clean, ax=axis: rd.var(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanvar",
                        "axis_scan_var",
                        lambda a=nans, ax=axis: rd.nanvar(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 average",
                        "axis_weighted",
                        lambda a=clean, w=weights, ax=axis: rd.average(
                            a, weights=w, axis=ax
                        ),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanaverage",
                        "axis_weighted",
                        lambda a=nans, w=weights, ax=axis: rd.nanaverage(
                            a, weights=w, axis=ax
                        ),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 median",
                        "axis_order_median",
                        lambda a=clean, ax=axis: rd.median(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanmedian",
                        "axis_order_median",
                        lambda a=nans, ax=axis: rd.nanmedian(a, axis=ax),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 percentile",
                        "axis_order_percentile",
                        lambda a=clean, ax=axis: rd.percentile(
                            a, [16, 50, 84], axis=ax
                        ),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanpercentile",
                        "axis_order_percentile",
                        lambda a=nans, ax=axis: rd.nanpercentile(
                            a, [16, 50, 84], axis=ax
                        ),
                    ),
                ]
            )

    lengths = [10_000, 1_000_000]
    if profile == "full":
        lengths.append(10_000_000)
    for dtype in dtypes:
        for length in lengths:
            clean = make_1d(length, dtype, nan=False)
            nans = make_1d(length, dtype, nan=True)
            workloads.extend(
                [
                    Workload(
                        f"{dtype} n={length} minmax",
                        "minmax_1d",
                        lambda a=clean: rd.minmax(a, validate=False),
                    ),
                    Workload(
                        f"{dtype} n={length} nanminmax",
                        "minmax_1d",
                        lambda a=nans: rd.nanminmax(a, validate=False),
                    ),
                ]
            )

    return workloads


def geometric_mean(values: list[float]) -> float:
    """Return geometric mean of positive timings."""
    return math.exp(sum(math.log(value) for value in values) / len(values))


def tune_grain(
    grain: str,
    workloads: list[Workload],
    candidates: list[int],
    *,
    repeats: int,
    warmups: int,
    max_regression: float,
) -> tuple[int, dict[int, float], dict[int, float]]:
    """Tune one grain class and return the best candidate plus scores."""
    owned = [workload for workload in workloads if workload.grain == grain]
    default = rd.get_parallel_grains()[grain]
    rd.set_parallel_grain(grain, default)
    baseline = [
        timeit(workload.func, repeats=repeats, warmups=warmups) for workload in owned
    ]
    scores: dict[int, float] = {}
    worst_regressions: dict[int, float] = {}
    for candidate in candidates:
        rd.set_parallel_grain(grain, candidate)
        times = [timeit(w.func, repeats=repeats, warmups=warmups) for w in owned]
        scores[candidate] = geometric_mean(times)
        worst_regressions[candidate] = max(
            time / base for time, base in zip(times, baseline, strict=True)
        )
    scores[default] = geometric_mean(baseline)
    worst_regressions[default] = 1.0
    valid = [
        candidate
        for candidate in candidates
        if worst_regressions[candidate] <= max_regression
    ]
    if valid:
        best = min(valid, key=scores.__getitem__)
    else:
        if default not in scores:
            rd.set_parallel_grain(grain, default)
            scores[default] = geometric_mean(baseline)
            worst_regressions[default] = 1.0
        best = default
    rd.set_parallel_grain(grain, best)
    return best, scores, worst_regressions


def parse_candidates(text: str) -> list[int]:
    """Parse comma-separated exponents into powers-of-two grain candidates."""
    exponents = [int(part) for part in text.split(",") if part.strip()]
    return [2**exponent for exponent in exponents]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--exponents", default="10,12,14,16,18,20,22,24,30")
    parser.add_argument("--max-regression", type=float, default=1.10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    candidates = parse_candidates(args.exponents)
    initial = rd.get_parallel_grains()
    workloads = build_workloads(args.profile)

    print_environment(
        title=f"reducers parallel-grain tuner ({args.profile})",
        bottleneck_available=bn is not None,
    )
    print(
        "| grain | default | best | default score (ms) | best score (ms) | "
        "speedup | worst regression |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")

    best_values: dict[str, int] = {}
    for grain in GRAIN_ENV:
        rd.set_parallel_grains(initial)
        best, scores, regressions = tune_grain(
            grain,
            workloads,
            candidates,
            repeats=args.repeats,
            warmups=args.warmups,
            max_regression=args.max_regression,
        )
        default = initial[grain]
        default_score = scores.get(default)
        if default_score is None:
            rd.set_parallel_grain(grain, default)
            default_workloads = [
                workload for workload in workloads if workload.grain == grain
            ]
            default_score = geometric_mean(
                [
                    timeit(workload.func, repeats=args.repeats, warmups=args.warmups)
                    for workload in default_workloads
                ]
            )
            regressions[default] = 1.0
        best_score = scores[best]
        best_values[grain] = best
        print(
            f"| `{grain}` | {default} | {best} | {default_score:.3f} | "
            f"{best_score:.3f} | {default_score / best_score:.2f}x | "
            f"{regressions[best]:.2f}x |"
        )

    rd.set_parallel_grains(best_values)
    print("\n## Environment")
    for grain, value in best_values.items():
        print(f"export {GRAIN_ENV[grain]}={value}")

    print("\n## Python")
    print("rd.set_parallel_grains({")
    for grain, value in best_values.items():
        print(f'    "{grain}": {value},')
    print("})")

    rd.set_parallel_grains(initial)


if __name__ == "__main__":
    main()
