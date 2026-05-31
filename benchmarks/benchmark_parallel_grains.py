"""Compare axis parallel-grain settings on representative stack shapes.

This benchmark is intentionally narrower than ``benchmark_axis.py``. It focuses
on cheap scan reducers where Rayon overhead can dominate shallow, small-output
stacks but helps once either the reducing axis or output image is large enough.
"""

from __future__ import annotations

import argparse

import numpy as np
import reducers as rd
from _benchutils import print_environment, ratio_cell, timeit

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover - depends on optional benchmark env
    bn = None


CASES = [
    ("shallow_small", (5, 100, 100), 0),
    ("deep_same_output", (100, 100, 100), 0),
    ("shallow_wide", (5, 2000, 2000), 0),
]

OPS = {
    "mean": (np.mean, None, rd.mean, False),
    "min": (np.min, None, rd.min, False),
    "nanmean": (np.nanmean, "nanmean", rd.nanmean, True),
    "nanmin": (np.nanmin, "nanmin", rd.nanmin, True),
}
SEED = 20250311


def make_stack(shape, dtype, *, include_nan):
    rng = np.random.default_rng(SEED)
    a = rng.normal(1000.0, 30.0, size=shape).astype(dtype)
    if include_nan:
        flat = a.ravel()
        flat[::101] = np.nan
    return a


def apply_setting(name, initial):
    if name == "default":
        rd.set_parallel_grains(initial)
    elif name == "serial":
        rd.set_parallel_grains(
            {
                key: (1_000_000_000 if key.startswith("axis_scan") else value)
                for key, value in initial.items()
            }
        )
    elif name == "forced":
        rd.set_parallel_grains(
            {
                key: (1 if key.startswith("axis_scan") else value)
                for key, value in initial.items()
            }
        )
    else:
        msg = f"unknown setting: {name}"
        raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["float64"])
    parser.add_argument(
        "--ops", nargs="+", choices=OPS, default=["mean", "min", "nanmean", "nanmin"]
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=["default", "serial", "forced"],
        default=["default", "serial", "forced"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()

    print_environment(
        title="reducers axis parallel-grain benchmark",
        bottleneck_available=bn is not None,
    )
    initial = rd.get_parallel_grains()

    print(
        "| setting | case | shape | dtype | function | np (ms) | bn (ms) | "
        "rd (ms) | np/rd | bn/rd |"
    )
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for setting in args.settings:
        apply_setting(setting, initial)
        for label, shape, axis in CASES:
            for dtype in args.dtypes:
                arrays = {}
                for op in args.ops:
                    np_fn, bn_name, rd_fn, include_nan = OPS[op]
                    if include_nan not in arrays:
                        arrays[include_nan] = make_stack(
                            shape, dtype, include_nan=include_nan
                        )
                    a = arrays[include_nan]
                    npt = timeit(
                        lambda arr=a, fn=np_fn, ax=axis: fn(arr, axis=ax),
                        repeats=args.repeats,
                        warmups=args.warmups,
                    )
                    bnt = None
                    if bn is not None and bn_name is not None:
                        bn_fn = getattr(bn, bn_name)
                        bnt = timeit(
                            lambda arr=a, fn=bn_fn, ax=axis: fn(arr, axis=ax),
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                    rdt = timeit(
                        lambda arr=a, fn=rd_fn, ax=axis: fn(arr, axis=ax),
                        repeats=args.repeats,
                        warmups=args.warmups,
                    )
                    bn_cell = f"{bnt:.2f}" if bnt is not None else "-"
                    print(
                        f"| {setting} | {label} | {shape} | {dtype} | `{op}` | "
                        f"{npt:.2f} | {bn_cell} | {rdt:.2f} | "
                        f"{ratio_cell(npt, rdt)} | {ratio_cell(bnt, rdt)} |"
                    )

    rd.set_parallel_grains(initial)


if __name__ == "__main__":
    main()
