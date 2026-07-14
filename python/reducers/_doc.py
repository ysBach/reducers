"""Generated docstrings for the public Python API.

Only call-site behavior (NaN/inf policy, errors, dtype, ``ddof``) lives here.
Kernel design, parallelization, and benchmarks are documented once in the
project docs (the "How It Gets Fast" page), not repeated per function.
"""

from __future__ import annotations

from collections.abc import MutableMapping

_A_PARAM = """a : array_like
    Input data. Supported numeric inputs are normalized to a contiguous kernel
    array when `validate=True`. Computed reducers promote integer and bool
    inputs to ``float64``; exact selection reducers keep integer and bool dtypes
    where the selected value can be returned exactly. Complex and object arrays
    are not supported."""

_AXIS_PARAM = """axis : None, 0, -1, or int, optional
    Axis to reduce. ``None`` reduces the whole array. ``0`` reduces strided
    reducing-axis slices into the remaining shape. ``-1`` and ``ndim - 1``
    reduce contiguous slices. Other axes raise ``NotImplementedError``."""

_VALIDATE = """validate : bool, optional
    If `True`, check dtype, dimensionality, contiguity, and axis validity
    before entering the Rust kernel. If `False`, the caller must provide a
    contiguous supported kernel dtype (``float32``, ``float64``, bool, or a
    NumPy integer dtype). `validate=False` skips dtype promotion: integer and
    bool arrays are reduced directly, while complex and object arrays remain
    unsupported."""

_IGNORE_INF = """ignore_inf : bool, optional
    If `False`, skip only NaN values and keep ``+/-inf`` values, matching
    NumPy's ``nan*`` reducers. If `True`, skip all non-finite values."""

_WEIGHTS = """weights : None or array_like, optional
    Weights for a weighted reduction. With ``axis=None``, weights must have the
    same shape as `a`. With an axis reduction, weights may either have the same
    shape as `a` or be 1-D with length equal to the reducing axis."""

_WEIGHTED_SUM_RETURNS = """weights : None or array_like, optional
    If provided, return ``sum(a * weights)`` instead of ``sum(a)``. With
    ``axis=None``, weights must have the same shape as `a`. With an axis
    reduction, weights may either have the same shape as `a` or be 1-D with
    length equal to the reducing axis.
return_sum_weights : bool, optional
    If `True` and `weights` is provided, also return the sum of retained
    weights.
return_unweighted_sum : bool, optional
    If `True` and `weights` is provided, also return the unweighted sum of
    retained `a` values."""

_DDOF_RETURN_MEAN = """ddof : int, optional
    Delta degrees of freedom. Results with ``nvalid <= ddof`` are ``NaN``.
return_mean : bool, optional
    If `True`, return ``(variance, mean)``, reusing the mean already computed
    by the variance kernel."""

_DDOF_RETURN_STD_MEAN = """ddof : int, optional
    Delta degrees of freedom. Results with ``nvalid <= ddof`` are ``NaN``.
return_mean : bool, optional
    If `True`, return ``(standard_deviation, mean)``, reusing the mean already
    computed by the variance kernel."""

_Q_PERCENTILE = """q : scalar or array_like
    Percentile or percentiles in ``[0, 100]``. A scalar `q` returns a scalar for
    ``axis=None`` or one output array for axis reductions. Multiple `q` values
    are returned on a leading output axis."""

_Q_QUANTILE = """q : scalar or array_like
    Quantile or quantiles in ``[0, 1]``. A scalar `q` returns a scalar for
    ``axis=None`` or one output array for axis reductions. Multiple `q` values
    are returned on a leading output axis."""

_SCALAR_OR_AXIS_RETURNS = """out : float or ndarray
    Reduction result. ``axis=None`` returns a Python ``float``. Axis reductions
    return an array with the reduced axis removed."""

_SUM_RETURNS = """out : float, ndarray, or tuple
    Reduction result. ``axis=None`` returns a Python ``float``. Axis reductions
    return an array with the reduced axis removed. With `weights` and optional
    return flags, returns ``(weighted_sum, sum_weights)``,
    ``(weighted_sum, unweighted_sum)``, or
    ``(weighted_sum, sum_weights, unweighted_sum)``."""

_EXACT_SELECT_RETURNS = """out : scalar or ndarray
    Reduction result. Float inputs return float results. Integer and bool inputs
    return an exact selected value for non-empty reducing-axis slices:
    ``axis=None`` returns a Python scalar and axis reductions preserve the input
    dtype."""

_INT_COUNT_RETURNS = """out : int or ndarray of int
    Number of finite values. ``axis=None`` returns a Python ``int``. Axis
    reductions return an integer array with the reduced axis removed."""

_MINMAX_RETURNS = """minimum, maximum : scalar or ndarray
    Pair of reduction results. ``axis=None`` uses a fused one-pass Rust kernel.
    Axis reductions currently compute the minimum and maximum as separate
    reductions."""

_PERCENTILE_RETURNS = """out : float or ndarray
    Interpolated percentile result. Multiple `q` values are returned on a
    leading axis, matching NumPy's output layout."""

# ---- behavior notes (semantics only; mechanism lives in the docs site) -------

_PLAIN_NOTE = (
    "Plain reducers include every value, so NaN and inf propagate with "
    "IEEE / NumPy-like semantics. This is the fastest path for known-clean "
    "finite data."
)

_NAN_NOTE = (
    "NaN-aware reducers skip NaN values directly, without building a filtered "
    "copy of the input."
)

_WEIGHTED_NOTE = (
    "Weighted averages return floating results, so integer and bool inputs are "
    "promoted. A zero sum of retained weights raises ``ZeroDivisionError``. For "
    "``nanaverage``, NaN values in `a` are skipped (with ``ignore_inf=True``, "
    "all non-finite values in `a` are skipped), and the weights attached to "
    "skipped values are skipped with them; a NaN weight on a retained value "
    "propagates to the result."
)

_WEIGHTED_SUM_NOTE = (
    "Weighted sums compute ``sum(a * weights)`` in one pass. For ``nansum``, "
    "weights attached to skipped values are skipped with them; optional "
    "``sum_weights`` and unweighted sums use the same retained-value policy. "
    "Like ``return_mean`` for variance and standard deviation, these optional "
    "returns expose quantities already available in the fused kernel and avoid "
    "separate reductions at the call site."
)

_EXACT_INT_NOTE = (
    "For integer and bool arrays there is no NaN to skip, so the ``nan*`` forms "
    "return the same result as their plain counterparts. ``min``, ``nanmin``, "
    "``max``, ``nanmax``, and ``lmedian`` preserve the integer or bool dtype for "
    "non-empty reducing-axis slices."
)

_COUNT_NOTE = (
    "Counts are always finite-only: this reducer counts values for which "
    "``isfinite`` is true, regardless of the policy used by other reducers."
)


def _params(*extra: str) -> str:
    parts = [_A_PARAM, _AXIS_PARAM, *[item for item in extra if item], _VALIDATE]
    return "\n".join(parts)


def _params_q(q_param: str, *extra: str) -> str:
    parts = [
        _A_PARAM,
        q_param,
        _AXIS_PARAM,
        *[item for item in extra if item],
        _VALIDATE,
    ]
    return "\n".join(parts)


_SPECS = {
    "mean": (
        "Return the arithmetic mean with NaN/inf propagation.",
        _params(),
        _SCALAR_OR_AXIS_RETURNS,
        _PLAIN_NOTE,
    ),
    "nanmean": (
        "Return the arithmetic mean while skipping NaN values.",
        _params(_IGNORE_INF),
        _SCALAR_OR_AXIS_RETURNS,
        _NAN_NOTE,
    ),
    "average": (
        "Return the weighted average with NaN/inf propagation.",
        _params(_WEIGHTS),
        _SCALAR_OR_AXIS_RETURNS,
        f"{_PLAIN_NOTE}\n\n{_WEIGHTED_NOTE}",
    ),
    "nanaverage": (
        "Return the weighted average while skipping NaN values.",
        _params(_WEIGHTS, _IGNORE_INF),
        _SCALAR_OR_AXIS_RETURNS,
        f"{_NAN_NOTE}\n\n{_WEIGHTED_NOTE}",
    ),
    "sum": (
        "Return the sum with NaN/inf propagation.",
        _params(_WEIGHTED_SUM_RETURNS),
        _SUM_RETURNS,
        f"{_PLAIN_NOTE}\n\n{_WEIGHTED_SUM_NOTE}",
    ),
    "nansum": (
        "Return the sum while skipping NaN values.",
        _params(_WEIGHTED_SUM_RETURNS, _IGNORE_INF),
        _SUM_RETURNS,
        f"{_NAN_NOTE}\n\n{_WEIGHTED_SUM_NOTE}",
    ),
    "min": (
        "Return the minimum with NaN propagation.",
        _params(),
        _EXACT_SELECT_RETURNS,
        f"{_PLAIN_NOTE}\n\n{_EXACT_INT_NOTE}",
    ),
    "nanmin": (
        "Return the minimum while skipping NaN values.",
        _params(_IGNORE_INF),
        _EXACT_SELECT_RETURNS,
        f"{_NAN_NOTE}\n\n{_EXACT_INT_NOTE}",
    ),
    "max": (
        "Return the maximum with NaN propagation.",
        _params(),
        _EXACT_SELECT_RETURNS,
        f"{_PLAIN_NOTE}\n\n{_EXACT_INT_NOTE}",
    ),
    "nanmax": (
        "Return the maximum while skipping NaN values.",
        _params(_IGNORE_INF),
        _EXACT_SELECT_RETURNS,
        f"{_NAN_NOTE}\n\n{_EXACT_INT_NOTE}",
    ),
    "minmax": (
        "Return the minimum and maximum with NaN propagation.",
        _params(),
        _MINMAX_RETURNS,
        _PLAIN_NOTE,
    ),
    "nanminmax": (
        "Return the minimum and maximum while skipping NaN values.",
        _params(_IGNORE_INF),
        _MINMAX_RETURNS,
        _NAN_NOTE,
    ),
    "var": (
        "Return the variance with NaN/inf propagation.",
        _params(_DDOF_RETURN_MEAN),
        _SCALAR_OR_AXIS_RETURNS,
        _PLAIN_NOTE,
    ),
    "nanvar": (
        "Return the variance while skipping NaN values.",
        _params(_DDOF_RETURN_MEAN, _IGNORE_INF),
        _SCALAR_OR_AXIS_RETURNS,
        _NAN_NOTE,
    ),
    "std": (
        "Return the standard deviation with NaN/inf propagation.",
        _params(_DDOF_RETURN_STD_MEAN),
        _SCALAR_OR_AXIS_RETURNS,
        _PLAIN_NOTE,
    ),
    "nanstd": (
        "Return the standard deviation while skipping NaN values.",
        _params(_DDOF_RETURN_STD_MEAN, _IGNORE_INF),
        _SCALAR_OR_AXIS_RETURNS,
        _NAN_NOTE,
    ),
    "median": (
        "Return the median with NaN propagation.",
        _params(),
        _SCALAR_OR_AXIS_RETURNS,
        _PLAIN_NOTE,
    ),
    "nanmedian": (
        "Return the median while skipping NaN values.",
        _params(_IGNORE_INF),
        _SCALAR_OR_AXIS_RETURNS,
        _NAN_NOTE,
    ),
    "lmedian": (
        "Return the lower value-selecting median while skipping NaN values.",
        _params(_IGNORE_INF),
        _EXACT_SELECT_RETURNS,
        "For an even number of retained values, `lmedian` returns the lower of "
        "the two middle values instead of averaging them. For an odd number of "
        f"retained values it matches `median`.\n\n{_NAN_NOTE}\n\n{_EXACT_INT_NOTE}",
    ),
    "percentile": (
        "Return linear-interpolation percentiles with NaN propagation.",
        _params_q(_Q_PERCENTILE),
        _PERCENTILE_RETURNS,
        _PLAIN_NOTE,
    ),
    "nanpercentile": (
        "Return linear-interpolation percentiles while skipping NaN values.",
        _params_q(_Q_PERCENTILE, _IGNORE_INF),
        _PERCENTILE_RETURNS,
        _NAN_NOTE,
    ),
    "quantile": (
        "Return linear-interpolation quantiles with NaN propagation.",
        _params_q(_Q_QUANTILE),
        _PERCENTILE_RETURNS,
        _PLAIN_NOTE,
    ),
    "nanquantile": (
        "Return linear-interpolation quantiles while skipping NaN values.",
        _params_q(_Q_QUANTILE, _IGNORE_INF),
        _PERCENTILE_RETURNS,
        _NAN_NOTE,
    ),
    "count_finite": (
        "Return the number of finite values.",
        _params(),
        _INT_COUNT_RETURNS,
        _COUNT_NOTE,
    ),
}


def install_docstrings(namespace: MutableMapping[str, object]) -> None:
    """Install generated docstrings into public API functions."""
    for name, (summary, params, returns, notes) in _SPECS.items():
        obj = namespace[name]
        obj.__doc__ = f"""{summary}

Parameters
----------
{params}

Returns
-------
{returns}

Notes
-----
{notes}
"""
