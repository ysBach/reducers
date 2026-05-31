"""Input validation and dtype/axis preparation for the public reducers API."""

from __future__ import annotations

import numpy as np

_FLOAT_DTYPES = (np.float32, np.float64)


def _as_real_numeric(a: object) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype.kind not in "biuf":
        raise TypeError(
            f"reducers supports real numeric dtypes only; got dtype {arr.dtype!s}"
        )
    return arr


def _as_float(a: object) -> np.ndarray:
    arr = _as_real_numeric(a)
    if arr.dtype not in _FLOAT_DTYPES:
        # Promote integers/bools to float64 (matches np.nanmean/np.nanvar).
        arr = arr.astype(np.float64, copy=False)
    return arr


def prepare_1d(
    a: object, *, validate: bool, preserve_dtype: bool = False
) -> np.ndarray:
    """Return a contiguous 1-D array for a whole-array reduction.

    Most computed reducers promote integer/bool inputs to float64 before
    entering Rust. Median-style reducers may set ``preserve_dtype=True`` and
    still return their public float result; that avoids a whole-array float copy
    when Rust can select from integer values directly. Exact-selection reducers
    also set ``preserve_dtype=True`` to keep integer/bool output dtype. ``ravel``
    of a C-contiguous array is a free view. With ``validate=False`` the caller
    guarantees a contiguous 1-D supported kernel dtype; no dtype promotion or
    dimensionality normalization is performed.
    """
    if not validate:
        return a  # type: ignore[return-value]
    arr = _as_real_numeric(a) if preserve_dtype else _as_float(a)
    return np.ascontiguousarray(arr).ravel()


def prepare_axis(
    a: object, axis: int, *, validate: bool, preserve_dtype: bool = False
) -> tuple[np.ndarray, bool, tuple[int, ...]]:
    """Normalize an `axis=0`/`axis=-1` reduction to a 2-D contiguous array.

    Returns ``(arr2d, axis_last, out_shape)`` where ``arr2d`` is ``(outer, n)``
    for the last axis or ``(n, outer)`` for axis 0 - both free reshapes of a
    C-contiguous input. Unsupported axes raise ``NotImplementedError`` (they are
    never silently copied/moved).
    """
    if validate:
        arr = _as_real_numeric(a) if preserve_dtype else _as_float(a)
    else:
        arr = np.asarray(a)
    ndim = arr.ndim
    if ndim == 0:
        raise ValueError("axis reduction requires at least a 1-D array")
    ax = axis + ndim if axis < 0 else axis
    arr = np.ascontiguousarray(arr)
    if ax == ndim - 1:
        n = arr.shape[-1]
        return arr.reshape(-1, n), True, arr.shape[:-1]
    if ax == 0:
        n = arr.shape[0]
        return arr.reshape(n, -1), False, arr.shape[1:]
    raise NotImplementedError(
        "reducers supports axis in {None, 0, -1, ndim-1} only; "
        f"got axis={axis} for a {ndim}-D array. Move/copy the target axis "
        "explicitly if you need another reduction axis."
    )


def prepare_weighted_1d(
    a: object, weights: object, *, validate: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous 1-D data and weights for a weighted reduction."""
    if validate:
        arr = _as_float(a)
        w = _as_float(weights)
        if w.shape != arr.shape:
            raise ValueError(
                "weights must have the same shape as a when axis=None; "
                f"got weights shape {w.shape} and a shape {arr.shape}"
            )
        return np.ascontiguousarray(arr).ravel(), np.ascontiguousarray(w).ravel()

    arr = np.asarray(a)
    w = np.asarray(weights)
    if w.shape != arr.shape:
        raise ValueError(
            "weights must have the same shape as a when axis=None; "
            f"got weights shape {w.shape} and a shape {arr.shape}"
        )
    return arr, w


def prepare_weighted_axis(
    a: object, weights: object, axis: int, *, validate: bool
) -> tuple[np.ndarray, np.ndarray, bool, bool, tuple[int, ...]]:
    """Normalize data and weights for supported weighted axis reductions."""
    arr = _as_float(a) if validate else np.asarray(a)
    w = _as_float(weights) if validate else np.asarray(weights)
    ndim = arr.ndim
    if ndim == 0:
        raise ValueError("axis reduction requires at least a 1-D array")
    ax = axis + ndim if axis < 0 else axis
    if ax not in (0, ndim - 1):
        raise NotImplementedError(
            "reducers supports axis in {None, 0, -1, ndim-1} only; "
            f"got axis={axis} for a {ndim}-D array. Move/copy the target axis "
            "explicitly if you need another reduction axis."
        )

    if w.shape == arr.shape:
        arr = np.ascontiguousarray(arr)
        w = np.ascontiguousarray(w)
        if ax == ndim - 1:
            n = arr.shape[-1]
            return arr.reshape(-1, n), w.reshape(-1), False, True, arr.shape[:-1]
        n = arr.shape[0]
        return arr.reshape(n, -1), w.reshape(-1), False, False, arr.shape[1:]

    n = arr.shape[ax]
    if w.ndim == 1 and w.shape[0] == n:
        arr = np.ascontiguousarray(arr)
        w = np.ascontiguousarray(w)
        if ax == ndim - 1:
            return arr.reshape(-1, n), w, True, True, arr.shape[:-1]
        return arr.reshape(n, -1), w, True, False, arr.shape[1:]

    raise ValueError(
        "weights must either match a.shape or be 1-D with length equal to "
        f"the reduction axis; got weights shape {w.shape}, a shape {arr.shape}, "
        f"axis={axis}"
    )


def prepare_q(q: object, *, percent: bool) -> tuple[np.ndarray, bool]:
    """Validate scalar or 1-D quantile/percentile positions.

    Returns ``(q_array_in_percent_units, was_scalar)``.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    scalar = q_arr.ndim == 0
    if scalar:
        q_arr = q_arr.reshape(1)
    elif q_arr.ndim != 1:
        raise ValueError(f"q must be scalar or 1-D; got shape {q_arr.shape}")
    lo, hi = (0.0, 100.0) if percent else (0.0, 1.0)
    if np.any(~np.isfinite(q_arr)) or np.any((q_arr < lo) | (q_arr > hi)):
        raise ValueError(f"q must be in [{lo:g}, {hi:g}]")
    if not percent:
        q_arr = q_arr * 100.0
    return np.ascontiguousarray(q_arr), scalar
