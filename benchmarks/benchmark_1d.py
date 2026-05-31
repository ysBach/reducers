"""Benchmark 1-D reducers kernels against NumPy and Bottleneck.

    python benchmarks/benchmark_1d.py --lengths 10000 10000000

Prints plain finite-data reducers first, then NaN-aware reducers.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

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
    "sum",
    "min",
    "max",
    "minmax",
    "var",
    "std",
    "percentile",
    "quantile",
)
DEFAULT_LENGTHS = (30, 100, 10_000, 10_000_000)
# Multi-rank positions exercise the selection-budget path.
PERCENTILE_Q = [16.0, 50.0, 84.0]
QUANTILE_Q = [0.16, 0.50, 0.84]

_NAN_DISPLAY_OP = {
    "mean": "nanmean",
    "average": "nanaverage",
    "median": "nanmedian",
    "sum": "nansum",
    "min": "nanmin",
    "max": "nanmax",
    "minmax": "nanminmax",
    "var": "nanvar",
    "std": "nanstd",
    "percentile": "nanpercentile",
    "quantile": "nanquantile",
}
SEED = 20250311


def make_values(length: int, dtype: str, *, include_nan: bool) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    values = rng.normal(1000.0, 30.0, size=length).astype(dtype)
    if include_nan and length >= 10:
        values[0] = np.nan
    if include_nan and length >= 1_000:
        values[:: max(1, length // 97)] = np.nan
    return np.ascontiguousarray(values)


def make_weights(length: int, dtype: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    weights = rng.uniform(0.5, 1.5, size=length).astype(dtype)
    return np.ascontiguousarray(weights)


def calls(length: int) -> int:
    return max(1, min(100_000, 1_000_000 // length))


def funcs_for(
    v: np.ndarray, op: str, *, plain: bool
) -> dict[str, Callable[[], object]]:
    f: dict[str, Callable[[], object]] = {}
    if op == "minmax":
        if plain:
            f["numpy"] = lambda: (np.min(v), np.max(v))
            f["reducers"] = lambda: rd.minmax(v, validate=False)
        else:
            f["numpy"] = lambda: (np.nanmin(v), np.nanmax(v))
            f["reducers"] = lambda: rd.nanminmax(v, validate=False)
            if bn is not None:
                f["bottleneck"] = lambda: (bn.nanmin(v), bn.nanmax(v))
        return f
    if op == "average":
        w = make_weights(v.size, str(v.dtype))
        if plain:
            f["numpy"] = lambda: np.average(v, weights=w)
            f["reducers"] = lambda: rd.average(v, weights=w, validate=False)
        else:
            mask = ~np.isnan(v)
            f["numpy"] = lambda: np.average(v[mask], weights=w[mask])
            f["reducers"] = lambda: rd.nanaverage(v, weights=w, validate=False)
        return f
    if op in ("percentile", "quantile"):
        q = PERCENTILE_Q if op == "percentile" else QUANTILE_Q
        if plain:
            np_fn = np.percentile if op == "percentile" else np.quantile
            rd_fn = rd.percentile if op == "percentile" else rd.quantile
        else:
            np_fn = np.nanpercentile if op == "percentile" else np.nanquantile
            rd_fn = rd.nanpercentile if op == "percentile" else rd.nanquantile
        f["numpy"] = lambda: np_fn(v, q)
        f["reducers"] = lambda: rd_fn(v, q, validate=False)
        return f
    if plain:
        np_fn = {
            "mean": np.mean,
            "median": np.median,
            "sum": np.sum,
            "min": np.min,
            "max": np.max,
            "var": np.var,
            "std": lambda x: np.std(x, ddof=1),
        }[op]
        rd_fn = {
            "mean": rd.mean,
            "median": rd.median,
            "sum": rd.sum,
            "min": rd.min,
            "max": rd.max,
            "var": rd.var,
            "std": lambda x, *, validate: rd.std(x, ddof=1, validate=validate),
        }[op]
        f["numpy"] = lambda: np_fn(v)
        if bn is not None and op in ("median",):
            f["bottleneck"] = lambda: bn.median(v)
        f["reducers"] = lambda: rd_fn(v, validate=False)
    else:
        np_fn = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "sum": np.nansum,
            "min": np.nanmin,
            "max": np.nanmax,
            "var": np.nanvar,
            "std": lambda x: np.nanstd(x, ddof=1),
        }[op]
        rd_fn = {
            "mean": rd.nanmean,
            "median": rd.nanmedian,
            "sum": rd.nansum,
            "min": rd.nanmin,
            "max": rd.nanmax,
            "var": rd.nanvar,
            "std": lambda x, *, validate: rd.nanstd(x, ddof=1, validate=validate),
        }[op]
        bn_fn = None
        if bn is not None:
            bn_fn = {
                "mean": bn.nanmean,
                "median": bn.nanmedian,
                "sum": bn.nansum,
                "min": bn.nanmin,
                "max": bn.nanmax,
                "var": getattr(bn, "nanvar", None),
                "std": (
                    (lambda x: bn.nanstd(x, ddof=1))
                    if getattr(bn, "nanstd", None) is not None
                    else None
                ),
            }[op]
        f["numpy"] = lambda: np_fn(v)
        if bn_fn is not None:
            f["bottleneck"] = lambda: bn_fn(v)
        f["reducers"] = lambda: rd_fn(v, validate=False)
    return f


def fmt_len(n: int) -> str:
    return {100: "10^2", 10_000: "10^4", 1_000_000: "10^6", 10_000_000: "10^7"}.get(
        n, f"{n:,}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    p.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    p.add_argument("--dtypes", nargs="+", default=["float64", "float32"])
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmups", type=int, default=3)
    args = p.parse_args()

    print_environment(
        title=f"reducers 1-D benchmark; bottleneck={'yes' if bn else 'no'}",
        bottleneck_available=bn is not None,
    )
    for plain in (True, False):
        print(f"## {'Plain finite-data' if plain else 'NaN-aware'}")
        print(
            "| length | dtype | function | np (µs) | bn (µs) | rd (µs) | "
            "np/rd | bn/rd |"
        )
        print("|---:|---|---|---:|---:|---:|---:|---:|")
        for dtype in args.dtypes:
            for length in args.lengths:
                v = make_values(length, dtype, include_nan=not plain)
                inner = calls(length)
                for op in args.ops:
                    funcs = funcs_for(v, op, plain=plain)
                    expected = funcs["numpy"]()
                    assert_equivalent(
                        funcs["reducers"](),
                        expected,
                        dtype=dtype,
                        label=f"{op} length={length} dtype={dtype}",
                    )
                    t = {
                        n: timeit(
                            fn,
                            repeats=args.repeats,
                            warmups=args.warmups,
                            inner=inner,
                        )
                        for n, fn in funcs.items()
                    }
                    npt = t.get("numpy")
                    bnt = t.get("bottleneck")
                    rdt = t.get("reducers")

                    npt_s = "-" if npt is None else f"{npt * 1000:.2f}"
                    bnt_s = "-" if bnt is None else f"{bnt * 1000:.2f}"
                    rdt_s = "-" if rdt is None else f"{rdt * 1000:.2f}"
                    npt_ratio = ratio_cell(npt, rdt)
                    bnt_ratio = ratio_cell(bnt, rdt)

                    display_op = op if plain else _NAN_DISPLAY_OP[op]
                    print(
                        f"| {fmt_len(length)} | {dtype} | `{display_op}` | "
                        f"{npt_s} | {bnt_s} | {rdt_s} | {npt_ratio} | "
                        f"{bnt_ratio} |"
                    )
        print()


if __name__ == "__main__":
    main()
