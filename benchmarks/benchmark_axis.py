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

import numpy as np
import reducers as rd
from _benchutils import print_environment, ratio_cell, timeit

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
                            npt = timeit(
                                lambda arr=a, ax=axis: (
                                    np.min(arr, axis=ax),
                                    np.max(arr, axis=ax),
                                ),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                lambda arr=a, ax=axis: rd.minmax(arr, axis=ax),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            bnt = None
                        else:
                            npt = timeit(
                                lambda arr=a, ax=axis: (
                                    np.nanmin(arr, axis=ax),
                                    np.nanmax(arr, axis=ax),
                                ),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                lambda arr=a, ax=axis: rd.nanminmax(arr, axis=ax),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            bnt = (
                                timeit(
                                    lambda arr=a, ax=axis: (
                                        bn.nanmin(arr, axis=ax),
                                        bn.nanmax(arr, axis=ax),
                                    ),
                                    repeats=args.repeats,
                                    warmups=args.warmups,
                                )
                                if bn is not None
                                else None
                            )
                    elif op == "average":
                        weights = make_weights(shape, axis, dtype, full_shape=False)
                        if plain:
                            npt = timeit(
                                lambda arr=a, w=weights, ax=axis: np.average(
                                    arr, weights=w, axis=ax
                                ),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                lambda arr=a, w=weights, ax=axis: rd.average(
                                    arr, weights=w, axis=ax
                                ),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                        else:
                            masked = np.ma.array(a, mask=np.isnan(a))
                            npt = timeit(
                                lambda arr=masked, w=weights, ax=axis: np.ma.average(
                                    arr, weights=w, axis=ax
                                ).filled(np.nan),
                                repeats=args.repeats,
                                warmups=args.warmups,
                            )
                            rdt = timeit(
                                lambda arr=a, w=weights, ax=axis: rd.nanaverage(
                                    arr, weights=w, axis=ax
                                ),
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
                        npt = timeit(
                            lambda fn=np_fn, arr=a, q=q, ax=axis: fn(arr, q, axis=ax),
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                        rdt = timeit(
                            lambda fn=rd_fn, arr=a, q=q, ax=axis: fn(arr, q, axis=ax),
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                    else:
                        np_fn = np_funcs[op]
                        rd_fn = rd_funcs[op]
                        bn_fn = bn_funcs[op]
                        npt = timeit(
                            lambda fn=np_fn, arr=a, ax=axis: fn(arr, axis=ax),
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                        rdt = timeit(
                            lambda fn=rd_fn, arr=a, ax=axis: fn(arr, axis=ax),
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
