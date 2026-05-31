"""Command-line autotuner for persisted parallel-grain settings."""

from __future__ import annotations

import argparse
import math
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

import reducers as rd

SEED = 20250311
GRAIN_ENV = {
    "axis_scan_plain": "REDUCERS_AXIS_SCAN_PLAIN_GRAIN",
    "axis_scan_nan": "REDUCERS_AXIS_SCAN_NAN_GRAIN",
    "axis_scan_var": "REDUCERS_AXIS_SCAN_VAR_GRAIN",
    "axis_weighted": "REDUCERS_AXIS_WEIGHTED_GRAIN",
    "axis_order_median": "REDUCERS_AXIS_ORDER_MEDIAN_GRAIN",
    "axis_order_percentile": "REDUCERS_AXIS_ORDER_PERCENTILE_GRAIN",
    "minmax_1d": "REDUCERS_MINMAX_1D_GRAIN",
}


@dataclass(frozen=True)
class Workload:
    """One representative timing target owned by a grain class."""

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
    small_shapes = [(5, 100, 100), (31, 100, 100), (100, 100, 100)]
    full_shapes = [*small_shapes, (11, 512, 512)]
    shapes = small_shapes if profile == "quick" else full_shapes
    dtypes = ["float64"] if profile == "quick" else ["float64", "float32"]
    workloads: list[Workload] = []

    for dtype in dtypes:
        for shape in shapes:
            clean = make_stack(shape, dtype, nan=False)
            nans = make_stack(shape, dtype, nan=True)
            weights = np.linspace(0.5, 1.5, shape[0]).astype(dtype)
            workloads.extend(
                [
                    Workload(
                        f"{dtype} {shape} axis0 mean",
                        "axis_scan_plain",
                        lambda a=clean: rd.mean(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 min",
                        "axis_scan_plain",
                        lambda a=clean: rd.min(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanmean",
                        "axis_scan_nan",
                        lambda a=nans: rd.nanmean(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanmin",
                        "axis_scan_nan",
                        lambda a=nans: rd.nanmin(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 var",
                        "axis_scan_var",
                        lambda a=clean: rd.var(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanvar",
                        "axis_scan_var",
                        lambda a=nans: rd.nanvar(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 average",
                        "axis_weighted",
                        lambda a=clean, w=weights: rd.average(a, weights=w, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanaverage",
                        "axis_weighted",
                        lambda a=nans, w=weights: rd.nanaverage(a, weights=w, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 median",
                        "axis_order_median",
                        lambda a=clean: rd.median(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanmedian",
                        "axis_order_median",
                        lambda a=nans: rd.nanmedian(a, axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 percentile",
                        "axis_order_percentile",
                        lambda a=clean: rd.percentile(a, [16, 50, 84], axis=0),
                    ),
                    Workload(
                        f"{dtype} {shape} axis0 nanpercentile",
                        "axis_order_percentile",
                        lambda a=nans: rd.nanpercentile(a, [16, 50, 84], axis=0),
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


def timeit(func: Callable[[], object], *, repeats: int, warmups: int) -> float:
    """Return a trimmed median runtime in milliseconds."""
    for _ in range(warmups):
        func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        func()
        times.append((time.perf_counter_ns() - start) / 1_000_000)
    times.sort()
    if len(times) >= 5:
        times = times[1:-1]
    return float(np.median(times))


def geometric_mean(values: list[float]) -> float:
    """Return the geometric mean of positive timings."""
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
    """Tune one grain class with a per-workload regression guard."""
    owned = [workload for workload in workloads if workload.grain == grain]
    default = rd.get_parallel_grains()[grain]
    rd.set_parallel_grain(grain, default)
    baseline = [
        timeit(workload.func, repeats=repeats, warmups=warmups) for workload in owned
    ]
    scores: dict[int, float] = {default: geometric_mean(baseline)}
    worst_regressions: dict[int, float] = {default: 1.0}

    for candidate in candidates:
        if candidate == default:
            continue
        rd.set_parallel_grain(grain, candidate)
        times = [timeit(w.func, repeats=repeats, warmups=warmups) for w in owned]
        scores[candidate] = geometric_mean(times)
        worst_regressions[candidate] = max(
            time / base for time, base in zip(times, baseline, strict=True)
        )

    valid = [
        candidate
        for candidate in candidates
        if worst_regressions[candidate] <= max_regression
    ]
    best = min(valid, key=scores.__getitem__) if valid else default
    rd.set_parallel_grain(grain, best)
    return best, scores, worst_regressions


def parse_candidates(text: str) -> list[int]:
    """Parse comma-separated exponents into powers-of-two candidates."""
    exponents = [int(part) for part in text.split(",") if part.strip()]
    return [2**exponent for exponent in exponents]


def print_environment(profile: str) -> None:
    """Print the tuning context."""
    print(f"# reducers autotuner ({profile})")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"numpy: {np.__version__}")
    print(f"reducers: {rd.__version__}")
    print(f"rayon threads: {rd.get_num_threads()}")
    print(f"seed: {SEED}")


def build_parser() -> argparse.ArgumentParser:
    """Build the autotuner CLI parser."""
    parser = argparse.ArgumentParser(
        description="Tune reducers parallel grains and save them for future imports."
    )
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--exponents", default="10,12,14,16,18,20,22,24,30")
    parser.add_argument("--max-regression", type=float, default=1.10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--config", default=None, help="Config path to write/read.")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the chosen grains without saving them.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete saved grains and restore built-in defaults.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the autotuner CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reset:
        removed = rd.clear_parallel_grains_config(path=args.config)
        status = "removed" if removed else "no saved config found"
        print(f"{status}: {rd.parallel_grains_config_path(args.config)}")
        print("active grains restored to built-in defaults")
        return 0

    candidates = parse_candidates(args.exponents)
    initial = rd.get_parallel_grains()
    workloads = build_workloads(args.profile)

    print_environment(args.profile)
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
        default_score = scores[default]
        best_score = scores[best]
        best_values[grain] = best
        print(
            f"| `{grain}` | {default} | {best} | {default_score:.3f} | "
            f"{best_score:.3f} | {default_score / best_score:.2f}x | "
            f"{regressions[best]:.2f}x |"
        )

    rd.set_parallel_grains(best_values)
    if args.no_save:
        print("\nchosen grains were not saved")
    else:
        path = rd.save_parallel_grains_config(best_values, path=args.config)
        print(f"\nsaved: {path}")
        print("future `import reducers` calls will apply these grains automatically")

    print("\n## Python")
    print("rd.set_parallel_grains({")
    for grain, value in best_values.items():
        print(f'    "{grain}": {value},')
    print("})")

    print("\n## Environment")
    for grain, value in best_values.items():
        print(f"export {GRAIN_ENV[grain]}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
