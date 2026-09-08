"""Finite-only sigma clipping for vectors and axis-0 stacks.

All clipping uses the shared Rust engine. NaN, infinities, and input masks are
excluded; accepted rejections accumulate. The generic defaults are `nkeep=0`
and `revert_on_nkeep=False`.
"""

# Adapted from imcombiners, Copyright 2026 @ysBach, BSD-3-Clause.
# Full notice: licenses/imcombiners-BSD-3-Clause.txt in this distribution.

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from . import _core

RejectionResult = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
VectorResult = tuple[np.ndarray, object, object, object, object, object]

_FLOAT_DTYPES = (np.float32, np.float64)
_PROMOTE_TO_FLOAT32_DTYPES = (np.uint8, np.uint16, np.int16)

__all__ = [
    "SigClip",
    "grow_mask",
    "sigclip",
    "sigclip_1d",
    "sigclip_mask",
    "sigclip_mask_1d",
    "sigclip_combine",
    "sigclip_combine_1d",
    "sigclip_restored_flags",
]


def _floating_input(arr: np.ndarray) -> np.ndarray:
    """Convert supported integer inputs to the floating workspace dtype."""
    if arr.dtype in _PROMOTE_TO_FLOAT32_DTYPES:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.dtype == np.int32:
        return np.ascontiguousarray(arr, dtype=np.float64)
    if arr.dtype not in _FLOAT_DTYPES:
        raise TypeError(
            "sigma clipping supports uint8, uint16, int16, int32, float32, "
            f"or float64; got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def _validate_stack(arr: np.ndarray) -> np.ndarray:
    """Normalize axis-0 stacks, including empty axes, to the Rust 3-D layout."""
    if arr.ndim < 2:
        raise ValueError(f"arr must have at least 2 dimensions; got {arr.shape}")
    arr = _floating_input(arr)
    if arr.ndim != 3:
        arr = arr.reshape(arr.shape[0], prod(arr.shape[1:]), 1)
    return arr


def _validate_values_1d(values: np.ndarray) -> np.ndarray:
    """Normalize vectors without changing the dtype-dependent clipping path."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D; got shape {values.shape}")
    return _floating_input(values)


def _validate_mask(
    mask: np.ndarray | None, shape: tuple[int, ...]
) -> np.ndarray | None:
    """Convert masks to contiguous boolean arrays and enforce original shape."""
    if mask is None:
        return None
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match arr shape {shape}")
    return mask


def grow_mask(mask: np.ndarray, grow: float, *, validate: bool = True) -> np.ndarray:
    """Dilate a stack mask over spatial axes by an exact Euclidean radius.

    Axis 0 is the stack axis and is never grown across. For each input plane,
    every `True` sample marks all spatial samples whose integer-grid Euclidean
    distance is less than or equal to `grow`.

    Parameters
    ----------
    mask : ndarray of bool, shape (N, *spatial)
        Rejection mask to grow. `True` values are expanded within each plane.
    grow : float
        Non-negative radius in pixels. `0` returns an unchanged copy. Values
        below `1` grow no additional integer-grid samples.
    validate : bool, optional
        If `True`, validate dimensionality, dtype, finiteness, and contiguity
        before entering the Rust kernel. If `False`, callers must provide a
        C-contiguous boolean array with at least two dimensions and a finite
        non-negative radius.

    Returns
    -------
    grown : ndarray of bool, shape (N, *spatial)
        Grown mask with the same shape as `mask`.
    """
    if validate:
        grow = float(grow)
        if not np.isfinite(grow) or grow < 0.0:
            raise ValueError("grow must be a finite non-negative radius")
        mask = np.asarray(mask)
        if mask.ndim < 2:
            raise ValueError(
                f"grow mask must have shape (N, *spatial); got {mask.shape}"
            )
        if mask.dtype != np.bool_:
            raise TypeError(f"grow mask must have dtype bool; got {mask.dtype}")
        mask = np.ascontiguousarray(mask)
    return _core.grow_mask(mask, grow)


def _grow_rejection_mask(
    mask_rej: np.ndarray,
    output_flags: np.ndarray,
    grow: float | None,
    *,
    validate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow a rejection mask and set output_flags bit 16 where growth added samples."""
    if grow is None:
        return mask_rej, output_flags
    grown = grow_mask(mask_rej, grow, validate=validate)
    added = grown & ~mask_rej
    if np.any(added):
        output_flags = output_flags.copy()
        output_flags[np.any(added, axis=0)] |= np.uint8(16)
    return grown, output_flags


def _prepare_1d_values(values: np.ndarray, *, validate: bool) -> np.ndarray:
    """Validate the public 1-D rejection contract and return Rust input."""
    if validate:
        return _validate_values_1d(values)

    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D; got shape {values.shape}")
    return np.ascontiguousarray(values)


def _prepare_1d_rejection_inputs(
    values: np.ndarray, mask: np.ndarray | None, *, validate: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate a public 1-D vector/mask pair for direct Rust slice kernels."""
    values = _prepare_1d_values(values, validate=validate)
    mask = _validate_mask(mask, values.shape)
    return values, mask


def _scalar(value: object) -> object:
    """Return a Python/NumPy scalar from a scalar-like result."""
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0]
    return value


def _rejection_1d_result(
    result: RejectionResult,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """Unwrap a stack rejection result to the 1-D public shape."""
    mask_rej, std, low, upp, nit, output_flags = result
    return (
        mask_rej.reshape(-1),
        _scalar(std),
        _scalar(low),
        _scalar(upp),
        _scalar(nit),
        _scalar(output_flags),
    )


def _sigma_pair(sigma: float | tuple[float, float]) -> tuple[float, float]:
    """Normalize scalar or asymmetric clipping thresholds."""
    if isinstance(sigma, (tuple, list)):
        return float(sigma[0]), float(sigma[1])
    if isinstance(sigma, (int, float, np.generic)):
        value = float(sigma)
        return value, value
    if np.isscalar(sigma):
        value = float(sigma)
        return value, value
    return float(sigma[0]), float(sigma[1])


def sigclip(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
    grow: float | None = None,
    validate: bool = True,
) -> RejectionResult:
    """Sigma-clipping rejection."""
    arr = np.asarray(arr)
    orig_shape = arr.shape
    trailing = arr.shape[1:]
    if validate:
        arr = _validate_stack(arr)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != arr.shape and mask.shape == orig_shape:
                mask = mask.reshape(arr.shape)
        mask = _validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    mask_rej, std, low, upp, nit, output_flags = _core.sigclip(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    )
    mask_rej = mask_rej.reshape(orig_shape)
    output_flags = output_flags.reshape(trailing)
    if grow is not None:
        mask_rej, output_flags = _grow_rejection_mask(
            mask_rej, output_flags, grow, validate=validate
        )
    return (
        mask_rej,
        std.reshape(trailing),
        low.reshape(trailing),
        upp.reshape(trailing),
        nit.reshape(trailing),
        output_flags,
    )


def sigclip_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
    validate: bool = True,
) -> tuple[np.ndarray, object, object, object, object, object]:
    """Sigma-clipping rejection for a 1-D value vector."""
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _rejection_1d_result(
        _core.sigclip_1d(
            values,
            mask=mask,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            maxiters=int(maxiters),
            ddof=int(ddof),
            nkeep=int(nkeep),
            maxrej=maxrej,
            cenfunc=str(cenfunc),
            clip_cen=str(_clip_cen),
            stdfunc=str(stdfunc),
            revert_on_nkeep=bool(revert_on_nkeep),
            validate=False,
        )
    )


def sigclip_mask(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
    grow: float | None = None,
    validate: bool = True,
) -> np.ndarray:
    """Return only the sigma-clipping rejection mask."""
    arr = np.asarray(arr)
    orig_shape = arr.shape
    if validate:
        arr = _validate_stack(arr)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != arr.shape and mask.shape == orig_shape:
                mask = mask.reshape(arr.shape)
        mask = _validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    mask_rej = _core.sigclip_mask(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    ).reshape(orig_shape)
    return mask_rej if grow is None else grow_mask(mask_rej, grow, validate=validate)


def sigclip_mask_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
    validate: bool = True,
) -> np.ndarray:
    """Return only the sigma-clipping rejection mask for a 1-D value vector."""
    values = _prepare_1d_values(values, validate=validate)
    mask = _validate_mask(mask, values.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _core.sigclip_mask_1d(
        values,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    )


def sigclip_restored_flags(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
    validate: bool = True,
) -> np.ndarray:
    """Return per-sample restored-candidate flags for sigma clipping."""
    arr = np.asarray(arr)
    orig_shape = arr.shape
    if validate:
        arr = _validate_stack(arr)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != arr.shape and mask.shape == orig_shape:
                mask = mask.reshape(arr.shape)
        mask = _validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    return _core.sigclip_restored_flags(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    ).reshape(orig_shape)


def sigclip_combine(
    arr: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
    validate: bool = True,
) -> np.ndarray:
    """Return an output-only sigma-clipped mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused sigclip currently supports mean and median")
    arr = np.asarray(arr)
    orig_shape = arr.shape
    trailing = orig_shape[1:]
    if validate:
        arr = _validate_stack(arr)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != arr.shape and mask.shape == orig_shape:
                mask = mask.reshape(arr.shape)
        mask = _validate_mask(mask, arr.shape)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    kernel = _core.sigclip_median if cb in ("median", "med") else _core.sigclip_mean
    out = kernel(
        arr,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=bool(validate),
    )
    return out.reshape(trailing)


def sigclip_combine_1d(
    values: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    combine: str,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
    validate: bool = True,
) -> object:
    """Return a 1-D sigma-clipped mean or median."""
    cb = combine.lower()
    if cb not in ("mean", "average", "avg", "median", "med"):
        raise NotImplementedError("fused sigclip currently supports mean and median")
    values, mask = _prepare_1d_rejection_inputs(values, mask, validate=validate)
    sigma_lower, sigma_upper = _sigma_pair(sigma)
    _clip_cen = cenfunc if clip_cen is None else clip_cen
    kernel = (
        _core.sigclip_median_1d if cb in ("median", "med") else _core.sigclip_mean_1d
    )
    return kernel(
        values,
        mask=mask,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=int(maxiters),
        ddof=int(ddof),
        nkeep=int(nkeep),
        maxrej=maxrej,
        cenfunc=str(cenfunc),
        clip_cen=str(_clip_cen),
        stdfunc=str(stdfunc),
        revert_on_nkeep=bool(revert_on_nkeep),
        validate=False,
    )


@dataclass(frozen=True)
class SigClip:
    """Reusable, immutable settings for axis-0 sigma clipping.

    Parameters
    ----------
    sigma : float or (float, float), optional
        Lower/upper thresholds; default `(3.0, 3.0)`.
    maxiters : int, optional
        Maximum attempted passes; default `5`.
    cenfunc : str, optional
        Rejection center: `"median"` (default), `"lmedian"`, or `"mean"`.
    clip_cen : str or None, optional
        Dispersion center; `None` follows `cenfunc`.
    stdfunc : str, optional
        `"std"` (default) or scaled `"mad"`.
    ddof : int, optional
        Degrees of freedom for standard deviation; default `0`.
    nkeep : int, optional
        Survivor rollback minimum; default `0`.
    maxrej : int or None, optional
        Maximum total exclusions before rollback; default `None` (unlimited).
    revert_on_nkeep : bool, optional
        Enable survivor rollback; default `False`.
    grow : float or None, optional
        Euclidean mask-growth radius within each plane; default `None`.

    Notes
    -----
    This object stores settings, not mutable Rust workspace. Its ``apply()``
    method returns the same six outputs as ``sigclip()``.
    """

    sigma: float | tuple[float, float] = (3.0, 3.0)
    maxiters: int = 5
    cenfunc: str = "median"
    clip_cen: str | None = None
    stdfunc: str = "std"
    ddof: int = 0
    nkeep: int = 0
    maxrej: int | None = None
    revert_on_nkeep: bool = False
    grow: float | None = None

    def apply(
        self,
        arr: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        validate: bool = True,
    ) -> RejectionResult:
        """Clip an axis-0 stack using this object's stored settings.

        Parameters
        ----------
        arr : ndarray, shape (N, *spatial)
            Input values; the sample axis is axis 0.
        mask : ndarray or None, optional
            Same-shape input exclusions; `True` means excluded.
        validate : bool, optional
            Validate and normalize arrays; default `True`.

        Returns
        -------
        result : tuple
            ``(mask_rej, std, low, upp, nit, output_flags)`` from ``sigclip()``.
        """
        return sigclip(
            arr,
            mask=mask,
            sigma=self.sigma,
            maxiters=self.maxiters,
            ddof=self.ddof,
            nkeep=self.nkeep,
            maxrej=self.maxrej,
            cenfunc=self.cenfunc,
            clip_cen=self.clip_cen,
            stdfunc=self.stdfunc,
            revert_on_nkeep=self.revert_on_nkeep,
            grow=self.grow,
            validate=validate,
        )


_COMMON_PARAMS = """mask : ndarray or None, optional
    Same shape as the input; `True` excludes a sample. Default `None`.
sigma : float or (float, float), optional
    Finite, nonnegative lower and upper multipliers; default `(3.0, 3.0)`.
    Samples exactly on a threshold survive.
maxiters : int, optional
    Maximum attempted passes; default `5`. `0` applies initial exclusions only.
ddof : int, optional
    Divide squared deviations by ``n - ddof``; default `0`. Ignored for MAD.
    Undefined statistics stop the current pass without changing the mask.
nkeep : int, optional
    Rollback minimum, active only with `revert_on_nkeep=True`; default `0`.
    This is not a final minimum-count requirement.
maxrej : int or None, optional
    Roll back a pass if total exclusions exceed this count. Counts input masks
    and nonfinite samples as well as clipping; default `None` (unlimited).
cenfunc : str, optional
    Rejection center: `"median"` (default), `"lmedian"`, or `"mean"`.
clip_cen : str or None, optional
    Dispersion center: `"mean"`, `"median"`, `"lmedian"`, or `"cenfunc"` to
    use the computed rejection center directly. `None` follows `cenfunc`'s
    estimator, preserving its arithmetic; default `None`.
stdfunc : str, optional
    `"std"` (default) or `"mad"` (median absolute deviation times 1.4826).
revert_on_nkeep : bool, optional
    Enable whole-pass survivor rollback; default `False`. Earlier accepted
    rejections and initial exclusions remain after rollback.
"""

_COMMON_NOTES = """NaN, infinities, and input masks are excluded before clipping.
Inputs are never modified. Accepted rejections accumulate. Empty/all-excluded inputs
have all-excluded masks, NaN extrema/spread, and NaN fused reductions.

With `validate=True`, `float32` and `float64` are preserved; `uint8`, `uint16`,
and `int16` use `float32`, and `int32` uses `float64`. Other dtypes are rejected.
Noncontiguous input is copied only when needed. Centers and MAD round to the
input dtype before threshold comparisons. Squared residuals and limits can
overflow or underflow.
"""


def _install_docstrings() -> None:
    """Share the parameter contract while documenting each distinct output path."""
    for function in (
        sigclip,
        sigclip_1d,
        sigclip_mask,
        sigclip_mask_1d,
        sigclip_combine,
        sigclip_combine_1d,
        sigclip_restored_flags,
    ):
        name = function.__name__
        vector = name.endswith("_1d")
        fused = "combine" in name
        growth = name in ("sigclip", "sigclip_mask")
        input_doc = (
            "values : ndarray, shape (N,)\n    One sample vector.\n"
            if vector
            else "arr : ndarray, shape (N, *spatial)\n"
            "    Samples along axis 0; at least two dimensions.\n"
        )
        extras = ""
        if fused:
            extras += (
                "combine : str\n"
                '    Required statistic: `"mean"`, `"average"`, `"avg"`,\n'
                '    `"median"`, or `"med"`.\n'
            )
        if growth:
            extras += (
                "grow : float or None, optional\n"
                "    Nonnegative Euclidean dilation radius within each plane.\n"
                "    Axis 0 is never grown across; default `None`.\n"
            )
        extras += (
            "validate : bool, optional\n    Validate/promote arrays; default `True`. "
        )
        extras += (
            "With `False`, supply a 1-D float32/float64 vector.\n"
            if vector
            else "With `False`, supply a C-contiguous\n"
            "    3-D float32/float64 stack and same-shape boolean mask.\n"
        )
        if fused:
            returns = (
                "value : float\n    Clipped mean or median; NaN if none survive.\n"
                if vector
                else "value : ndarray, shape (*spatial,)\n"
                "    Clipped mean or median in workspace dtype; NaN if empty.\n"
            )
        elif "restored" in name:
            returns = (
                "restored_flags : ndarray of uint8, same shape as input\n"
                "    Per-sample rollback bits: 64 for `nkeep`, 128 for `maxrej`.\n"
            )
        elif "mask" in name:
            returns = (
                "mask_rej : ndarray of bool, same shape as input\n"
                "    Final exclusions, including input masks and nonfinite samples.\n"
            )
        else:
            returns = (
                "mask_rej : ndarray of bool, same shape as input\n"
                "    Final exclusions, including input masks and nonfinite samples.\n"
                "std, low, upp : scalars or ndarrays\n"
                "    Last valid attempted spread and final retained extrema.\n"
                "    Extrema are not clipping bounds. Scalars for vectors;\n"
                "    arrays of shape `(*spatial,)` in workspace dtype for stacks.\n"
                "nit, output_flags : scalars or ndarrays\n"
                "    Legacy counter and flags; uint8 arrays for stacks. Starts at\n"
                "    one and saturates at 255. Bits: 1 input mask, 2 iteration limit,\n"
                "    4 `nkeep` rollback, 8 `maxrej` rollback, 16 added growth.\n"
            )
        notes = _COMMON_NOTES
        if growth:
            notes += (
                "\nGrowth includes input masks and nonfinite values.\n"
                "Scalar diagnostics describe clipping before growth.\n"
            )
        function.__doc__ = (
            f"{function.__doc__}\n\nParameters\n----------\n"
            f"{input_doc}{_COMMON_PARAMS}{extras}\nReturns\n-------\n{returns}"
            f"\nNotes\n-----\n{notes}"
        )


_install_docstrings()
