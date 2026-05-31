"""Benchmark axis reductions (axis=0 and axis=-1) vs NumPy and Bottleneck.

    python benchmarks/benchmark_axis.py

Cases mimic stack reductions over a cube. A general ``(100, 100, 100)`` cube is
run both ``axis=0`` (reduce each same-position reducing-axis slice across frames)
and ``axis=-1`` (contiguous trailing reducing axis). Short ``(11, 100, 100)`` and
``(31, 100, 100)`` stacks with ``axis=0`` mimic frame combining, where
order-statistic overhead is visible. ``(11, 512, 512)``
repeats the depth-11 stack with many more output elements, to show how
`reducers`' per-output parallelism widens the gap. Prints plain finite-data
reducers first, then NaN-aware reducers.

``percentile``/``quantile`` request three positions, ``[16, 50, 84]`` (the
median and +/-1 sigma). The NaN-aware percentile/quantile rows are skipped: NumPy
uses per-slice NaN removal plus general quantile machinery for axis
``nanpercentile``/``nanquantile``, so timing it just stalls the benchmark for a
ratio that is always ~100-1000x.
"""

from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import reducers as rd
from _benchutils import assert_equivalent, print_environment, ratio_cell, timeit

try:
    import bottleneck as bn
except ImportError:
    bn = None

OPS = (
    "mean",
    "average",
    "median",
    "var",
    "std",
    "min",
    "max",
    "minmax",
    "sum",
    "percentile",
    "quantile",
)
_PCT_Q = [16.0, 50.0, 84.0]
_QTL_Q = [0.16, 0.50, 0.84]
# (label, shape, axis)
CASES = (
    ("cube_axis0", (100, 100, 100), 0),
    ("cube_axislast", (100, 100, 100), -1),
    ("stack11_axis0", (11, 100, 100), 0),
    ("stack31_axis0", (31, 100, 100), 0),
    ("stack11_big_axis0", (11, 512, 512), 0),
)

_NP_NAN = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "var": np.nanvar,
    "std": lambda arr, axis: np.nanstd(arr, axis=axis, ddof=1),
    "min": np.nanmin,
    "max": np.nanmax,
    "sum": np.nansum,
}
_RD_NAN = {
    "mean": rd.nanmean,
    "median": rd.nanmedian,
    "var": rd.nanvar,
    "std": lambda arr, axis: rd.nanstd(arr, axis=axis, ddof=1),
    "min": rd.nanmin,
    "max": rd.nanmax,
    "sum": rd.nansum,
}
_NP_PLAIN = {
    "mean": np.mean,
    "median": np.median,
    "var": np.var,
    "std": lambda arr, axis: np.std(arr, axis=axis, ddof=1),
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
}
_RD_PLAIN = {
    "mean": rd.mean,
    "median": rd.median,
    "var": rd.var,
    "std": lambda arr, axis: rd.std(arr, axis=axis, ddof=1),
    "min": rd.min,
    "max": rd.max,
    "sum": rd.sum,
}
_BN_NAN = {
    "mean": getattr(bn, "nanmean", None) if bn is not None else None,
    "median": getattr(bn, "nanmedian", None) if bn is not None else None,
    "var": getattr(bn, "nanvar", None) if bn is not None else None,
    "std": (
        (lambda arr, axis: bn.nanstd(arr, axis=axis, ddof=1))
        if bn is not None and getattr(bn, "nanstd", None) is not None
        else None
    ),
    "min": getattr(bn, "nanmin", None) if bn is not None else None,
    "max": getattr(bn, "nanmax", None) if bn is not None else None,
    "sum": getattr(bn, "nansum", None) if bn is not None else None,
}
_BN_PLAIN = {
    "mean": None,
    "median": getattr(bn, "median", None) if bn is not None else None,
    "var": None,
    "std": None,
    "min": None,
    "max": None,
    "sum": None,
}
_NAN_DISPLAY_OP = {
    "mean": "nanmean",
    "average": "nanaverage",
    "median": "nanmedian",
    "var": "nanvar",
    "std": "nanstd",
    "min": "nanmin",
    "max": "nanmax",
    "minmax": "nanminmax",
    "sum": "nansum",
    "percentile": "nanpercentile",
    "quantile": "nanquantile",
}
SEED = 20250311


def make_stack(shape, dtype, *, include_nan):
    rng = np.random.default_rng(SEED)
    a = rng.normal(1000.0, 30.0, size=shape).astype(dtype)
    if include_nan:
        flat = a.reshape(-1)
        flat[::97] = np.nan
    return np.ascontiguousarray(a)


def make_weights(shape, axis, dtype, *, full_shape):
    n = shape[axis]
    rng = np.random.default_rng(SEED)
    if full_shape:
        weights = rng.uniform(0.5, 1.5, size=shape).astype(dtype)
    else:
        weights = rng.uniform(0.5, 1.5, size=n).astype(dtype)
    return np.ascontiguousarray(weights)


def np_minmax(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    return np.min(arr, axis=axis), np.max(arr, axis=axis)


def np_nanminmax(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmin(arr, axis=axis), np.nanmax(arr, axis=axis)


def bn_nanminmax(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    return bn.nanmin(arr, axis=axis), bn.nanmax(arr, axis=axis)


def axis_call(fn, arr: np.ndarray, axis: int):
    return fn(arr, axis=axis)


def axis_q_call(fn, arr: np.ndarray, q: list[float], axis: int):
    return fn(arr, q, axis=axis)


def np_average_axis(arr: np.ndarray, weights: np.ndarray, axis: int):
    return np.average(arr, weights=weights, axis=axis)


def np_masked_average_axis(arr: np.ma.MaskedArray, weights: np.ndarray, axis: int):
    return np.ma.average(arr, weights=weights, axis=axis).filled(np.nan)


def rd_average_axis(arr: np.ndarray, weights: np.ndarray, axis: int):
    return rd.average(arr, weights=weights, axis=axis)


def rd_nanaverage_axis(arr: np.ndarray, weights: np.ndarray, axis: int):
    return rd.nanaverage(arr, weights=weights, axis=axis)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    p.add_argument("--dtypes", nargs="+", default=["float64", "float32"])
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmups", type=int, default=3)
    args = p.parse_args()

    print_environment(
        title=f"reducers axis benchmark; bottleneck={'yes' if bn else 'no'}",
        bottleneck_available=bn is not None,
    )
    for plain in (True, False):
        np_funcs = _NP_PLAIN if plain else _NP_NAN
        rd_funcs = _RD_PLAIN if plain else _RD_NAN
        bn_funcs = _BN_PLAIN if plain else _BN_NAN

        print(f"## {'Plain finite-data' if plain else 'NaN-aware'}")
        print(
            "| case | shape | axis | dtype | function | np (ms) | bn (ms) | "
            "rd (ms) | np/rd | bn/rd |"
        )
        print("|---|---|---:|---|---|---:|---:|---:|---:|---:|")
        for dtype in args.dtypes:
            for label, shape, axis in CASES:
                a = make_stack(shape, dtype, include_nan=not plain)
                for op in args.ops:
                    # np.nanpercentile/nanquantile use per-slice NaN removal
                    # plus general quantile machinery on axis reductions.
                    if not plain and op in ("percentile", "quantile"):
                        continue
                    if op == "minmax":
                        if plain:
                            np_call = partial(np_minmax, a, axis)
                            rd_call = partial(axis_call, rd.minmax, a, axis)
                            assert_equivalent(
                                rd_call(),
                                np_call(),
                                dtype=dtype,
                                label=f"{op} {label} axis={axis} dtype={dtype}",
                            )
                            npt = timeit(
                                np_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                rd_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            bnt = None
                        else:
                            np_call = partial(np_nanminmax, a, axis)
                            rd_call = partial(axis_call, rd.nanminmax, a, axis)
                            assert_equivalent(
                                rd_call(),
                                np_call(),
                                dtype=dtype,
                                label=(
                                    f"{_NAN_DISPLAY_OP[op]} {label} "
                                    f"axis={axis} dtype={dtype}"
                                ),
                            )
                            npt = timeit(
                                np_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                rd_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            bnt = (
                                timeit(
                                    partial(bn_nanminmax, a, axis),
                                    repeats=args.repeats,
                                    warmups=args.warmups,
                                )
                                if bn is not None
                                else None
                            )
                    elif op == "average":
                        weights = make_weights(shape, axis, dtype, full_shape=False)
                        if plain:
                            np_call = partial(np_average_axis, a, weights, axis)
                            rd_call = partial(rd_average_axis, a, weights, axis)
                            assert_equivalent(
                                rd_call(),
                                np_call(),
                                dtype=dtype,
                                label=f"{op} {label} axis={axis} dtype={dtype}",
                            )
                            npt = timeit(
                                np_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                rd_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                        else:
                            masked = np.ma.array(a, mask=np.isnan(a))
                            np_call = partial(
                                np_masked_average_axis, masked, weights, axis
                            )
                            rd_call = partial(rd_nanaverage_axis, a, weights, axis)
                            assert_equivalent(
                                rd_call(),
                                np_call(),
                                dtype=dtype,
                                label=(
                                    f"{_NAN_DISPLAY_OP[op]} {label} "
                                    f"axis={axis} dtype={dtype}"
                                ),
                            )
                            npt = timeit(
                                np_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                rd_call,
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                        bnt = None
                    elif op in ("percentile", "quantile"):
                        q = _PCT_Q if op == "percentile" else _QTL_Q
                        if op == "percentile":
                            np_fn = np.percentile if plain else np.nanpercentile
                            rd_fn = rd.percentile if plain else rd.nanpercentile
                        else:
                            np_fn = np.quantile if plain else np.nanquantile
                            rd_fn = rd.quantile if plain else rd.nanquantile
                        bnt = None
                        np_call = partial(axis_q_call, np_fn, a, q, axis)
                        rd_call = partial(axis_q_call, rd_fn, a, q, axis)
                        assert_equivalent(
                            rd_call(),
                            np_call(),
                            dtype=dtype,
                            label=f"{op} {label} axis={axis} dtype={dtype}",
                        )
                        npt = timeit(
                            np_call,
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                        rdt = timeit(
                            rd_call,
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                    else:
                        np_fn = np_funcs[op]
                        rd_fn = rd_funcs[op]
                        bn_fn = bn_funcs[op]
                        np_call = partial(axis_call, np_fn, a, axis)
                        rd_call = partial(axis_call, rd_fn, a, axis)
                        assert_equivalent(
                            rd_call(),
                            np_call(),
                            dtype=dtype,
                            label=f"{op} {label} axis={axis} dtype={dtype}",
                        )
                        npt = timeit(
                            np_call,
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                        rdt = timeit(
                            rd_call,
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                        bnt = (
                            timeit(
                                lambda fn=bn_fn, arr=a, ax=axis: fn(arr, axis=ax),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            if bn_fn is not None
                            else None
                        )

                    npt_s = "-" if npt is None else f"{npt:.2f}"
                    bnt_s = "-" if bnt is None else f"{bnt:.2f}"
                    rdt_s = "-" if rdt is None else f"{rdt:.2f}"
                    npt_ratio = ratio_cell(npt, rdt)
                    bnt_ratio = ratio_cell(bnt, rdt)

                    print(
                        f"| {label} | {shape} | {axis} | {dtype} | "
                        f"`{_NAN_DISPLAY_OP[op] if not plain else op}` | "
                        f"{npt_s} | {bnt_s} | {rdt_s} | {npt_ratio} | "
                        f"{bnt_ratio} |"
                    )
        print()


if __name__ == "__main__":
    main()
