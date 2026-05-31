"""Public Python API: numpy-like plain reducers and ``nan*`` reducers.

Two user entry points per operation:

- ``mean(a)`` includes all values; NaN/inf propagate as ordinary IEEE floats
  (numpy ``np.mean`` semantics). This is the fastest path for known-clean data.
- ``nanmean(a, ignore_inf=False)`` skips NaN (``np.nanmean`` parity);
  ``ignore_inf=True`` additionally drops ``+/-inf`` (finite-only).

``validate=False`` is for trusted hot loops where the caller already has a
contiguous supported kernel dtype (`float32`, `float64`, bool, or a NumPy
integer dtype). It skips normalization; integer and bool arrays are reduced
directly and return the same public result types as the validated path.
Exact selection reducers (``min``, ``nanmin``, ``max``, ``nanmax``,
``lmedian``) preserve integer/bool dtype when the reducing-axis slice is
non-empty.
``axis=None`` (default) reduces the whole array; ``axis`` may also be ``0``,
``-1``, or ``ndim-1``.
"""

from __future__ import annotations

import numpy as np

from . import _core, _doc
from ._validation import (
    prepare_1d,
    prepare_axis,
    prepare_q,
    prepare_weighted_1d,
    prepare_weighted_axis,
)

# ScanPolicy codes mirrored from src/finite.rs.
_ALL_VALUES = 0
_SKIP_NAN = 2
_SKIP_NONFINITE = 3

# Kind codes mirrored from reducers_1d::Kind.
_K_MEAN = 0
_K_SUM = 1
_K_MIN = 2
_K_MAX = 3
_K_MEDIAN = 4
_K_LMEDIAN = 5
_K_VAR = 6
_K_STD = 7
_K_COUNT_FINITE = 8


def _nan_policy(ignore_inf: bool) -> int:
    return _SKIP_NONFINITE if ignore_inf else _SKIP_NAN


def _axis_scalar(a, axis, kind, policy, *, ddof=0, validate=True, preserve_dtype=False):
    arr2, axis_last, out_shape = prepare_axis(
        a, axis, validate=validate, preserve_dtype=preserve_dtype
    )
    out = _core.reduce_axis(arr2, kind, axis_last, int(ddof), policy)
    return out.reshape(out_shape)


def _axis_pct(a, q, axis, policy, *, percent, validate):
    arr2, axis_last, out_shape = prepare_axis(a, axis, validate=validate)
    q_arr, scalar = prepare_q(q, percent=percent)
    flat = _core.percentile_axis(arr2, q_arr, axis_last, policy)
    res = flat.reshape((q_arr.size, *out_shape))
    return res[0] if scalar else res


def _weighted(a, weights, axis, policy, validate):
    if weights is None:
        if policy == _ALL_VALUES:
            return mean(a, axis=axis, validate=validate)
        return nanmean(
            a, axis=axis, ignore_inf=policy == _SKIP_NONFINITE, validate=validate
        )
    if axis is not None:
        arr2, w_arr, weights_1d, axis_last, out_shape = prepare_weighted_axis(
            a, weights, axis, validate=validate
        )
        out = _core.average_axis(arr2, w_arr, weights_1d, axis_last, policy)
        return out.reshape(out_shape)
    arr, w_arr = prepare_weighted_1d(a, weights, validate=validate)
    return _core.average_1d(arr, w_arr, policy)


# ---- mean / sum ---------------------------------------------------------------


def mean(a, axis=None, *, validate=True):
    """Arithmetic mean; NaN/inf propagate (numpy ``mean`` semantics)."""
    if axis is not None:
        return _axis_scalar(a, axis, _K_MEAN, _ALL_VALUES, validate=validate)
    return _core.mean_1d(prepare_1d(a, validate=validate), _ALL_VALUES)


def nanmean(a, axis=None, *, ignore_inf=False, validate=True):
    """Mean ignoring NaN (``np.nanmean`` parity); ``ignore_inf`` also drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_scalar(a, axis, _K_MEAN, pol, validate=validate)
    return _core.mean_1d(prepare_1d(a, validate=validate), pol)


def average(a, weights=None, axis=None, *, validate=True):
    """Weighted average; with ``weights=None`` this is equivalent to ``mean``."""
    return _weighted(a, weights, axis, _ALL_VALUES, validate)


def nanaverage(a, weights=None, axis=None, *, ignore_inf=False, validate=True):
    """Weighted average ignoring NaN values; ``ignore_inf`` also drops inf."""
    return _weighted(a, weights, axis, _nan_policy(ignore_inf), validate)


def sum(a, axis=None, *, validate=True):  # noqa: A001 - numpy-like name
    """Sum; NaN/inf propagate."""
    if axis is not None:
        return _axis_scalar(a, axis, _K_SUM, _ALL_VALUES, validate=validate)
    return _core.sum_1d(prepare_1d(a, validate=validate), _ALL_VALUES)


def nansum(a, axis=None, *, ignore_inf=False, validate=True):
    """Sum ignoring NaN; ``ignore_inf`` also drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_scalar(a, axis, _K_SUM, pol, validate=validate)
    return _core.sum_1d(prepare_1d(a, validate=validate), pol)


# ---- min / max / minmax -------------------------------------------------------


def min(a, axis=None, *, validate=True):  # noqa: A001 - numpy-like name
    """Minimum; NaN propagates (numpy ``min`` semantics)."""
    if axis is not None:
        return _axis_scalar(
            a, axis, _K_MIN, _ALL_VALUES, validate=validate, preserve_dtype=True
        )
    return _core.min_1d(
        prepare_1d(a, validate=validate, preserve_dtype=True), _ALL_VALUES
    )


def nanmin(a, axis=None, *, ignore_inf=False, validate=True):
    """Minimum ignoring NaN (``np.nanmin`` parity); ``ignore_inf`` also drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_scalar(
            a, axis, _K_MIN, pol, validate=validate, preserve_dtype=True
        )
    return _core.min_1d(prepare_1d(a, validate=validate, preserve_dtype=True), pol)


def max(a, axis=None, *, validate=True):  # noqa: A001 - numpy-like name
    """Maximum; NaN propagates."""
    if axis is not None:
        return _axis_scalar(
            a, axis, _K_MAX, _ALL_VALUES, validate=validate, preserve_dtype=True
        )
    return _core.max_1d(
        prepare_1d(a, validate=validate, preserve_dtype=True), _ALL_VALUES
    )


def nanmax(a, axis=None, *, ignore_inf=False, validate=True):
    """Maximum ignoring NaN (``np.nanmax`` parity); ``ignore_inf`` also drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_scalar(
            a, axis, _K_MAX, pol, validate=validate, preserve_dtype=True
        )
    return _core.max_1d(prepare_1d(a, validate=validate, preserve_dtype=True), pol)


def minmax(a, axis=None, *, validate=True):
    """Return ``(min, max)`` with NaN propagation."""
    if axis is not None:
        return (
            _axis_scalar(
                a, axis, _K_MIN, _ALL_VALUES, validate=validate, preserve_dtype=True
            ),
            _axis_scalar(
                a, axis, _K_MAX, _ALL_VALUES, validate=validate, preserve_dtype=True
            ),
        )
    return _core.minmax_1d(
        prepare_1d(a, validate=validate, preserve_dtype=True), _ALL_VALUES
    )


def nanminmax(a, axis=None, *, ignore_inf=False, validate=True):
    """Return ``(nanmin, nanmax)`` in one pass; ``ignore_inf`` drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return (
            _axis_scalar(a, axis, _K_MIN, pol, validate=validate, preserve_dtype=True),
            _axis_scalar(a, axis, _K_MAX, pol, validate=validate, preserve_dtype=True),
        )
    return _core.minmax_1d(prepare_1d(a, validate=validate, preserve_dtype=True), pol)


# ---- variance / std -----------------------------------------------------------


def _var(a, axis, ddof, return_mean, policy, validate):
    if axis is not None:
        v = _axis_scalar(a, axis, _K_VAR, policy, ddof=ddof, validate=validate)
        if return_mean:
            m = _axis_scalar(a, axis, _K_MEAN, policy, validate=validate)
            return v, m
        return v
    v, m = _core.var_1d(prepare_1d(a, validate=validate), int(ddof), policy)
    return (v, m) if return_mean else v


def var(a, axis=None, ddof=0, *, return_mean=False, validate=True):
    """Variance; NaN/inf propagate. If ``return_mean``, return ``(var, mean)``."""
    return _var(a, axis, ddof, return_mean, _ALL_VALUES, validate)


def nanvar(a, axis=None, ddof=0, *, return_mean=False, ignore_inf=False, validate=True):
    """Variance ignoring NaN; ``ignore_inf`` also drops inf."""
    return _var(a, axis, ddof, return_mean, _nan_policy(ignore_inf), validate)


def std(a, axis=None, ddof=0, *, return_mean=False, validate=True):
    """Standard deviation; NaN/inf propagate."""
    if axis is not None:
        s = _axis_scalar(a, axis, _K_STD, _ALL_VALUES, ddof=ddof, validate=validate)
        if return_mean:
            m = _axis_scalar(a, axis, _K_MEAN, _ALL_VALUES, validate=validate)
            return s, m
        return s
    v, m = _core.var_1d(prepare_1d(a, validate=validate), int(ddof), _ALL_VALUES)
    s = float(np.sqrt(v))
    return (s, m) if return_mean else s


def nanstd(a, axis=None, ddof=0, *, return_mean=False, ignore_inf=False, validate=True):
    """Standard deviation ignoring NaN; ``ignore_inf`` also drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        s = _axis_scalar(a, axis, _K_STD, pol, ddof=ddof, validate=validate)
        if return_mean:
            m = _axis_scalar(a, axis, _K_MEAN, pol, validate=validate)
            return s, m
        return s
    v, m = _core.var_1d(prepare_1d(a, validate=validate), int(ddof), pol)
    s = float(np.sqrt(v))
    return (s, m) if return_mean else s


# ---- median / lmedian ---------------------------------------------------------


def median(a, axis=None, *, validate=True):
    """Median; NaN propagates (numpy ``median`` semantics)."""
    if axis is not None:
        return _axis_scalar(a, axis, _K_MEDIAN, _ALL_VALUES, validate=validate)
    return _core.median_1d(prepare_1d(a, validate=validate), _ALL_VALUES)


def nanmedian(a, axis=None, *, ignore_inf=False, validate=True):
    """Median ignoring NaN (``np.nanmedian`` parity); ``ignore_inf`` drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_scalar(a, axis, _K_MEDIAN, pol, validate=validate)
    return _core.median_1d(prepare_1d(a, validate=validate), pol)


def lmedian(a, axis=None, *, ignore_inf=False, validate=True):
    """Lower value-selecting median ignoring NaN; ``ignore_inf`` also drops inf."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_scalar(
            a, axis, _K_LMEDIAN, pol, validate=validate, preserve_dtype=True
        )
    return _core.lmedian_1d(prepare_1d(a, validate=validate, preserve_dtype=True), pol)


# ---- percentile / quantile ----------------------------------------------------


def percentile(a, q, axis=None, *, validate=True):
    """Linear-interpolation percentile(s) in ``[0, 100]``; NaN propagates."""
    if axis is not None:
        return _axis_pct(a, q, axis, _ALL_VALUES, percent=True, validate=validate)
    q_arr, scalar = prepare_q(q, percent=True)
    out = _core.percentile_1d(prepare_1d(a, validate=validate), q_arr, _ALL_VALUES)
    return out[0] if scalar else out


def nanpercentile(a, q, axis=None, *, ignore_inf=False, validate=True):
    """Percentile(s) ignoring NaN (``np.nanpercentile`` parity)."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_pct(a, q, axis, pol, percent=True, validate=validate)
    q_arr, scalar = prepare_q(q, percent=True)
    out = _core.percentile_1d(prepare_1d(a, validate=validate), q_arr, pol)
    return out[0] if scalar else out


def quantile(a, q, axis=None, *, validate=True):
    """Linear-interpolation quantile(s) in ``[0, 1]``; NaN propagates."""
    if axis is not None:
        return _axis_pct(a, q, axis, _ALL_VALUES, percent=False, validate=validate)
    q_arr, scalar = prepare_q(q, percent=False)
    out = _core.percentile_1d(prepare_1d(a, validate=validate), q_arr, _ALL_VALUES)
    return out[0] if scalar else out


def nanquantile(a, q, axis=None, *, ignore_inf=False, validate=True):
    """Quantile(s) ignoring NaN (``np.nanquantile`` parity)."""
    pol = _nan_policy(ignore_inf)
    if axis is not None:
        return _axis_pct(a, q, axis, pol, percent=False, validate=validate)
    q_arr, scalar = prepare_q(q, percent=False)
    out = _core.percentile_1d(prepare_1d(a, validate=validate), q_arr, pol)
    return out[0] if scalar else out


# ---- count_finite -------------------------------------------------------------


def count_finite(a, axis=None, *, validate=True):
    """Number of finite (non-NaN, non-inf) values."""
    if axis is not None:
        # count_finite ignores policy; pass AllValues. Cast float counts to int.
        out = _axis_scalar(a, axis, _K_COUNT_FINITE, _ALL_VALUES, validate=validate)
        return out.astype(np.intp)
    return _core.count_finite_1d(prepare_1d(a, validate=validate))


_doc.install_docstrings(globals())
