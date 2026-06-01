"""Trusted-buffer reducers for callers that already normalized their data.

This module exposes the fast Rust kernels with an explicit low-level contract:
Inputs must already have the dimensionality, C-contiguity, supported real NumPy
dtype, and paired-buffer invariants expected by the called kernel. Most scalar
and axis helpers accept `copy=True` as an explicit convenience; the bare
weighted 1-D kernels do not. Functions named ``*_valid`` do not validate or
filter non-finite values. Functions named ``*_skip_nan`` or
``*_skip_nonfinite`` apply only that stated filtering policy.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from . import _core
from ._validation import prepare_q

_ALL_FIN = 1
_SKIP_NAN = 2
_SKIP_NONFIN = 3

_KIND_CODES = {
    "mean": 0,
    "sum": 1,
    "min": 2,
    "max": 3,
    "median": 4,
    "lmedian": 5,
    "var": 6,
    "std": 7,
    "count_finite": 8,
}

AxisOp = Literal[
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "lmedian",
    "var",
    "std",
    "count_finite",
]


def _axis_kind(op: AxisOp | str) -> int:
    try:
        return _KIND_CODES[op]
    except KeyError as exc:
        allowed = ", ".join(sorted(_KIND_CODES))
        raise ValueError(
            f"unknown reducer op {op!r}; expected one of {allowed}"
        ) from exc


def _std_from_var_mean(var_mean: tuple[float, float]) -> tuple[float, float]:
    var, mean = var_mean
    return float(np.sqrt(var)), mean


def _axis_reduce(
    a: object,
    op: AxisOp | str,
    *,
    axis_last: bool,
    policy: int,
    ddof: int = 0,
    copy: bool = False,
):
    arr = np.ascontiguousarray(a) if copy else a
    return _core.reduce_axis(arr, _axis_kind(op), axis_last, int(ddof), policy)


def _axis_percentile(
    a: object,
    q: object,
    *,
    axis_last: bool,
    policy: int,
    percent: bool,
    copy: bool = False,
):
    arr = np.ascontiguousarray(a) if copy else a
    q_arr, scalar = prepare_q(q, percent=percent)
    flat = _core.percentile_axis(arr, q_arr, axis_last, policy)
    outer = flat.size // q_arr.size
    out = flat.reshape((q_arr.size, outer))
    return out[0] if scalar else out


def sum_valid(a: object, *, copy: bool = False):
    return _core.sum_1d(np.ascontiguousarray(a) if copy else a, _ALL_FIN)


def mean_valid(a: object, *, copy: bool = False):
    return _core.mean_1d(np.ascontiguousarray(a) if copy else a, _ALL_FIN)


def min_valid(a: object, *, copy: bool = False):
    return _core.min_1d(np.ascontiguousarray(a) if copy else a, _ALL_FIN)


def max_valid(a: object, *, copy: bool = False):
    return _core.max_1d(np.ascontiguousarray(a) if copy else a, _ALL_FIN)


def minmax_valid(a: object, *, copy: bool = False):
    return _core.minmax_1d(np.ascontiguousarray(a) if copy else a, _ALL_FIN)


def var_mean_valid(a: object, ddof: int = 0, *, copy: bool = False):
    return _core.var_1d(np.ascontiguousarray(a) if copy else a, int(ddof), _ALL_FIN)


def var_valid(a: object, ddof: int = 0, *, copy: bool = False):
    return var_mean_valid(a, ddof, copy=copy)[0]


def std_mean_valid(a: object, ddof: int = 0, *, copy: bool = False):
    return _std_from_var_mean(var_mean_valid(a, ddof, copy=copy))


def std_valid(a: object, ddof: int = 0, *, copy: bool = False):
    return std_mean_valid(a, ddof, copy=copy)[0]


weighted_sum_valid = _core.weighted_sum_valid_1d
weighted_sum_only_valid = _core.weighted_sum_only_valid_1d
weighted_sum_and_weights_valid = _core.weighted_sum_and_weights_valid_1d
weighted_sum_and_unweighted_valid = _core.weighted_sum_and_unweighted_valid_1d
weighted_average_valid = _core.weighted_average_valid_1d


def median_valid(a: object, *, copy: bool = False):
    return _core.median_1d(np.ascontiguousarray(a) if copy else a, _ALL_FIN)


def lmedian_valid(a: object, *, copy: bool = False):
    return _core.lmedian_1d(np.ascontiguousarray(a) if copy else a, _ALL_FIN)


def percentile_valid(a: object, q: object, *, copy: bool = False):
    q_arr, scalar = prepare_q(q, percent=True)
    out = _core.percentile_1d(np.ascontiguousarray(a) if copy else a, q_arr, _ALL_FIN)
    return out[0] if scalar else out


def quantile_valid(a: object, q: object, *, copy: bool = False):
    q_arr, scalar = prepare_q(q, percent=False)
    out = _core.percentile_1d(np.ascontiguousarray(a) if copy else a, q_arr, _ALL_FIN)
    return out[0] if scalar else out


def percentiles_valid(a: object, q: object, *, copy: bool = False):
    q_arr, _ = prepare_q(q, percent=True)
    return _core.percentile_1d(np.ascontiguousarray(a) if copy else a, q_arr, _ALL_FIN)


def median_valid_in_place(a: object):
    return _core.median_valid_in_place_1d(a)


def lmedian_valid_in_place(a: object):
    return _core.lmedian_valid_in_place_1d(a)


def percentile_valid_in_place(a: object, q: object):
    q_arr, scalar = prepare_q(q, percent=True)
    if scalar:
        return _core.percentile_valid_in_place_1d(a, float(q_arr[0]))
    return _core.percentiles_valid_in_place_1d(a, q_arr)


def quantile_valid_in_place(a: object, q: object):
    q_arr, scalar = prepare_q(q, percent=False)
    if scalar:
        return _core.quantile_valid_in_place_1d(a, float(q_arr[0] / 100.0))
    return _core.percentiles_valid_in_place_1d(a, q_arr)


def percentiles_valid_in_place(a: object, q: object):
    q_arr, _ = prepare_q(q, percent=True)
    return _core.percentiles_valid_in_place_1d(a, q_arr)


def sum_skip_nan(a: object, *, copy: bool = False):
    return _core.sum_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NAN)


def mean_skip_nan(a: object, *, copy: bool = False):
    return _core.mean_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NAN)


def min_skip_nan(a: object, *, copy: bool = False):
    return _core.min_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NAN)


def max_skip_nan(a: object, *, copy: bool = False):
    return _core.max_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NAN)


def minmax_skip_nan(a: object, *, copy: bool = False):
    return _core.minmax_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NAN)


def var_mean_skip_nan(a: object, ddof: int = 0, *, copy: bool = False):
    return _core.var_1d(np.ascontiguousarray(a) if copy else a, int(ddof), _SKIP_NAN)


def var_skip_nan(a: object, ddof: int = 0, *, copy: bool = False):
    return var_mean_skip_nan(a, ddof, copy=copy)[0]


def std_mean_skip_nan(a: object, ddof: int = 0, *, copy: bool = False):
    return _std_from_var_mean(var_mean_skip_nan(a, ddof, copy=copy))


def std_skip_nan(a: object, ddof: int = 0, *, copy: bool = False):
    return std_mean_skip_nan(a, ddof, copy=copy)[0]


weighted_sum_skip_nan = _core.weighted_sum_skip_nan_1d
weighted_sum_only_skip_nan = _core.weighted_sum_only_skip_nan_1d
weighted_sum_and_weights_skip_nan = _core.weighted_sum_and_weights_skip_nan_1d
weighted_sum_and_unweighted_skip_nan = _core.weighted_sum_and_unweighted_skip_nan_1d
weighted_average_skip_nan = _core.weighted_average_skip_nan_1d


def median_skip_nan(a: object, *, copy: bool = False):
    return _core.median_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NAN)


def lmedian_skip_nan(a: object, *, copy: bool = False):
    return _core.lmedian_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NAN)


def percentile_skip_nan(a: object, q: object, *, copy: bool = False):
    q_arr, scalar = prepare_q(q, percent=True)
    out = _core.percentile_1d(np.ascontiguousarray(a) if copy else a, q_arr, _SKIP_NAN)
    return out[0] if scalar else out


def quantile_skip_nan(a: object, q: object, *, copy: bool = False):
    q_arr, scalar = prepare_q(q, percent=False)
    out = _core.percentile_1d(np.ascontiguousarray(a) if copy else a, q_arr, _SKIP_NAN)
    return out[0] if scalar else out


def sum_skip_nonfinite(a: object, *, copy: bool = False):
    return _core.sum_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NONFIN)


def mean_skip_nonfinite(a: object, *, copy: bool = False):
    return _core.mean_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NONFIN)


def min_skip_nonfinite(a: object, *, copy: bool = False):
    return _core.min_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NONFIN)


def max_skip_nonfinite(a: object, *, copy: bool = False):
    return _core.max_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NONFIN)


def minmax_skip_nonfinite(a: object, *, copy: bool = False):
    return _core.minmax_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NONFIN)


def var_mean_skip_nonfinite(a: object, ddof: int = 0, *, copy: bool = False):
    return _core.var_1d(np.ascontiguousarray(a) if copy else a, int(ddof), _SKIP_NONFIN)


def var_skip_nonfinite(a: object, ddof: int = 0, *, copy: bool = False):
    return var_mean_skip_nonfinite(a, ddof, copy=copy)[0]


def std_mean_skip_nonfinite(a: object, ddof: int = 0, *, copy: bool = False):
    return _std_from_var_mean(var_mean_skip_nonfinite(a, ddof, copy=copy))


def std_skip_nonfinite(a: object, ddof: int = 0, *, copy: bool = False):
    return std_mean_skip_nonfinite(a, ddof, copy=copy)[0]


weighted_sum_skip_nonfinite = _core.weighted_sum_skip_nonfinite_1d
weighted_sum_only_skip_nonfinite = _core.weighted_sum_only_skip_nonfinite_1d
weighted_sum_and_weights_skip_nonfinite = (
    _core.weighted_sum_and_weights_skip_nonfinite_1d
)
weighted_sum_and_unweighted_skip_nonfinite = (
    _core.weighted_sum_and_unweighted_skip_nonfinite_1d
)
weighted_average_skip_nonfinite = _core.weighted_average_skip_nonfinite_1d


def median_skip_nonfinite(a: object, *, copy: bool = False):
    return _core.median_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NONFIN)


def lmedian_skip_nonfinite(a: object, *, copy: bool = False):
    return _core.lmedian_1d(np.ascontiguousarray(a) if copy else a, _SKIP_NONFIN)


def percentile_skip_nonfinite(a: object, q: object, *, copy: bool = False):
    q_arr, scalar = prepare_q(q, percent=True)
    out = _core.percentile_1d(
        np.ascontiguousarray(a) if copy else a, q_arr, _SKIP_NONFIN
    )
    return out[0] if scalar else out


def quantile_skip_nonfinite(a: object, q: object, *, copy: bool = False):
    q_arr, scalar = prepare_q(q, percent=False)
    out = _core.percentile_1d(
        np.ascontiguousarray(a) if copy else a, q_arr, _SKIP_NONFIN
    )
    return out[0] if scalar else out


def count_finite(a: object, *, copy: bool = False) -> int:
    return int(_core.count_finite_1d(np.ascontiguousarray(a) if copy else a))


def reduce_axis0_valid(
    a: object, op: AxisOp | str, ddof: int = 0, *, copy: bool = False
):
    return _axis_reduce(a, op, axis_last=False, policy=_ALL_FIN, ddof=ddof, copy=copy)


def reduce_axis_last_valid(
    a: object, op: AxisOp | str, ddof: int = 0, *, copy: bool = False
):
    return _axis_reduce(a, op, axis_last=True, policy=_ALL_FIN, ddof=ddof, copy=copy)


def reduce_axis0_skip_nan(
    a: object, op: AxisOp | str, ddof: int = 0, *, copy: bool = False
):
    return _axis_reduce(a, op, axis_last=False, policy=_SKIP_NAN, ddof=ddof, copy=copy)


def reduce_axis_last_skip_nan(
    a: object, op: AxisOp | str, ddof: int = 0, *, copy: bool = False
):
    return _axis_reduce(a, op, axis_last=True, policy=_SKIP_NAN, ddof=ddof, copy=copy)


def reduce_axis0_skip_nonfinite(
    a: object, op: AxisOp | str, ddof: int = 0, *, copy: bool = False
):
    return _axis_reduce(
        a, op, axis_last=False, policy=_SKIP_NONFIN, ddof=ddof, copy=copy
    )


def reduce_axis_last_skip_nonfinite(
    a: object, op: AxisOp | str, ddof: int = 0, *, copy: bool = False
):
    return _axis_reduce(
        a, op, axis_last=True, policy=_SKIP_NONFIN, ddof=ddof, copy=copy
    )


def percentile_axis0_valid(a: object, q: object, *, copy: bool = False):
    return _axis_percentile(
        a, q, axis_last=False, policy=_ALL_FIN, percent=True, copy=copy
    )


def percentile_axis_last_valid(a: object, q: object, *, copy: bool = False):
    return _axis_percentile(
        a, q, axis_last=True, policy=_ALL_FIN, percent=True, copy=copy
    )


def quantile_axis0_valid(a: object, q: object, *, copy: bool = False):
    return _axis_percentile(
        a, q, axis_last=False, policy=_ALL_FIN, percent=False, copy=copy
    )


def quantile_axis_last_valid(a: object, q: object, *, copy: bool = False):
    return _axis_percentile(
        a, q, axis_last=True, policy=_ALL_FIN, percent=False, copy=copy
    )


def percentile_axis0_skip_nonfinite(a: object, q: object, *, copy: bool = False):
    return _axis_percentile(
        a, q, axis_last=False, policy=_SKIP_NONFIN, percent=True, copy=copy
    )


def percentile_axis_last_skip_nonfinite(a: object, q: object, *, copy: bool = False):
    return _axis_percentile(
        a, q, axis_last=True, policy=_SKIP_NONFIN, percent=True, copy=copy
    )


__all__ = sorted(
    name
    for name, value in globals().items()
    if not name.startswith("_")
    and name not in {"AxisOp", "Literal", "np", "prepare_q"}
    and callable(value)
)
