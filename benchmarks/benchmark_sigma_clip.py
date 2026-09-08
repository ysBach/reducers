"""Measure the public sigma-clipping API; a NumPy model checks correctness.

The reference model intentionally covers only the benchmark contract: finite-only
clipping with ``sigma=3``, ``maxiters=5``, ``ddof=0``, no survivor rollback, and
matching center and spread estimators. It is used for correctness checks before
any timing starts; it is never included in the timed call.

Examples
--------
Run the complete vector and stack matrix and save rows for later reporting::

    uv run --no-sync python benchmarks/benchmark_sigma_clip.py \
        --csv benchmarks/sigma_clip.csv

Check the matrix without timing it::

    uv run --no-sync python benchmarks/benchmark_sigma_clip.py --check-only
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import reducers as rd
from _benchutils import environment_lines, package_version, timeit

SIGMA = 3.0
MAXITERS = 5
DDOF = 0
NKEEP = 0
REVERT_ON_NKEEP = False
STACK_SHAPE = (31, 64, 64)
DEFAULT_LENGTHS = (31, 257, 4096)
DEFAULT_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class Profile:
    """One center/spread pair exercised by the benchmark."""

    name: str
    cenfunc: str
    stdfunc: str


PROFILES = {
    "mean_std": Profile("mean_std", "mean", "std"),
    "median_mad": Profile("median_mad", "median", "mad"),
}


@dataclass(frozen=True)
class ReferenceColumn:
    """Reference result for one reducing-axis column."""

    rejected: np.ndarray
    std: object
    low: object
    upp: object
    nit: np.uint8
    flags: np.uint8


@dataclass(frozen=True)
class ReferenceResult:
    """Reference result with the public stack or vector shapes."""

    rejected: np.ndarray
    std: object
    low: object
    upp: object
    nit: object
    flags: object


def _center(values: np.ndarray, name: str) -> object:
    """Compute a reference center in f64, then narrow it to input dtype."""
    if name == "mean":
        value = np.sum(values, dtype=np.float64) / values.size
    elif name == "median":
        value = np.median(values)
    else:  # pragma: no cover - profiles are declared above
        raise AssertionError(f"unsupported reference center {name!r}")
    return np.asarray(value, dtype=values.dtype)[()]


def _reference_column(
    values: np.ndarray,
    input_mask: np.ndarray | None,
    profile: Profile,
) -> ReferenceColumn:
    """Apply the restricted finite-only reference algorithm to one column."""
    values = np.asarray(values)
    rejected = ~np.isfinite(values)
    if input_mask is not None:
        rejected = rejected | np.asarray(input_mask, dtype=bool)
    retained = int(np.count_nonzero(~rejected))
    std_out = np.asarray(np.nan, dtype=values.dtype)[()]
    legacy_nit = 1
    stopped_at_limit = MAXITERS == 0

    for iteration in range(MAXITERS):
        stopped_at_limit = iteration + 1 == MAXITERS
        active = values[~rejected]
        if active.size == 0:
            break

        center = _center(active, profile.cenfunc)
        spread_center = center
        if profile.stdfunc == "std":
            if active.size <= DDOF:
                break
            # Mean-centered dispersion uses the unrounded f64 mean; only the
            # rejection center is narrowed to the input dtype.
            variance_center = (
                np.mean(active, dtype=np.float64)
                if profile.cenfunc == "mean"
                else float(spread_center)
            )
            delta = active.astype(np.float64) - variance_center
            with np.errstate(over="ignore", invalid="ignore"):
                variance = np.sum(delta * delta, dtype=np.float64) / (
                    active.size - DDOF
                )
            if not np.isfinite(variance):
                break
            spread_squared = float(variance)
            std_out = np.asarray(np.sqrt(variance), dtype=values.dtype)[()]
        elif profile.stdfunc == "mad":
            delta = np.abs(active.astype(np.float64) - float(spread_center))
            spread = np.asarray(1.4826 * np.median(delta), dtype=values.dtype)[()]
            spread_squared = float(spread) * float(spread)
            std_out = spread
        else:  # pragma: no cover - profiles are declared above
            raise AssertionError(f"unsupported reference spread {profile.stdfunc!r}")

        lower_squared = SIGMA * SIGMA * spread_squared
        upper_squared = lower_squared
        newly_rejected = np.zeros(values.shape, dtype=bool)
        for index, value in enumerate(values):
            if rejected[index]:
                continue
            delta = float(value) - float(center)
            delta_squared = delta * delta
            if (delta < 0.0 and delta_squared > lower_squared) or (
                delta > 0.0 and delta_squared > upper_squared
            ):
                newly_rejected[index] = True

        if not np.any(newly_rejected):
            stopped_at_limit = False
            break

        rejected |= newly_rejected
        kept = int(np.count_nonzero(~rejected))
        if retained == kept:  # Defensive; a nonempty candidate should change it.
            stopped_at_limit = False
            break
        retained = kept
        legacy_nit = min(255, legacy_nit + 1)

    final_values = values[~rejected]
    if final_values.size:
        low = np.min(final_values)
        upp = np.max(final_values)
    else:
        low = np.asarray(np.nan, dtype=values.dtype)[()]
        upp = np.asarray(np.nan, dtype=values.dtype)[()]
    flags = np.uint8(0)
    if input_mask is not None and np.any(input_mask):
        flags |= np.uint8(1)
    if stopped_at_limit:
        flags |= np.uint8(2)
    return ReferenceColumn(
        rejected=rejected,
        std=std_out,
        low=low,
        upp=upp,
        nit=np.uint8(legacy_nit),
        flags=flags,
    )


def reference_clip(
    values: np.ndarray,
    input_mask: np.ndarray | None,
    profile: Profile,
) -> ReferenceResult:
    """Return the independent reference result for a vector or axis-0 stack."""
    values = np.asarray(values)
    if values.ndim == 1:
        column = _reference_column(values, input_mask, profile)
        return ReferenceResult(
            column.rejected,
            column.std,
            column.low,
            column.upp,
            column.nit,
            column.flags,
        )

    trailing = values.shape[1:]
    rejected = np.empty(values.shape, dtype=bool)
    std = np.empty(trailing, dtype=values.dtype)
    low = np.empty(trailing, dtype=values.dtype)
    upp = np.empty(trailing, dtype=values.dtype)
    nit = np.empty(trailing, dtype=np.uint8)
    flags = np.empty(trailing, dtype=np.uint8)
    for index in np.ndindex(trailing):
        selector = (slice(None),) + index
        column_mask = None if input_mask is None else input_mask[selector]
        column = _reference_column(values[selector], column_mask, profile)
        rejected[selector] = column.rejected
        std[index] = column.std
        low[index] = column.low
        upp[index] = column.upp
        nit[index] = column.nit
        flags[index] = column.flags
    return ReferenceResult(rejected, std, low, upp, nit, flags)


def _make_values(length: int, dtype: str, *, stack: bool) -> np.ndarray:
    """Create deterministic finite, nonfinite, masked, and outlier samples."""
    shape = (length, 64, 64) if stack else (length,)
    sample_axis = np.arange(length, dtype=np.float64)
    if stack:
        column_axis = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        values = (
            100.0
            + 0.35 * np.sin(sample_axis[:, None, None] * 0.71)
            + 0.15 * np.cos(column_axis[None, :, :] * 0.017)
            + 0.03
            * np.sin(sample_axis[:, None, None] * (column_axis[None, :, :] % 11 + 1))
        )
    else:
        values = (
            100.0
            + 0.35 * np.sin(sample_axis * 0.71)
            + 0.03 * np.cos(sample_axis * 0.19)
        )
    values = np.asarray(values, dtype=dtype).reshape(shape)
    values[1, ...] = np.asarray(np.nan, dtype=dtype)
    values[2, ...] = np.asarray(np.inf, dtype=dtype)
    values[3, ...] = np.asarray(-np.inf, dtype=dtype)
    high = length // 3
    low = (2 * length) // 3
    if stack:
        column_axis = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        values[high] += np.asarray(1000.0 + (column_axis % 7) * 5.0, dtype=dtype)
        values[low] -= np.asarray(1000.0 + (column_axis % 5) * 7.0, dtype=dtype)
    else:
        values[high] += np.asarray(1000.0, dtype=dtype)
        values[low] -= np.asarray(1000.0, dtype=dtype)
    return np.ascontiguousarray(values)


def _make_mask(values: np.ndarray, *, enabled: bool) -> np.ndarray | None:
    """Mark a clean sample so input-mask handling is exercised separately."""
    if not enabled:
        return None
    mask = np.zeros(values.shape, dtype=bool)
    mask[4, ...] = True
    return mask


def _params(profile: Profile) -> dict[str, object]:
    """Return the explicit, fixed parameter set used by every timed call."""
    return {
        "sigma": SIGMA,
        "maxiters": MAXITERS,
        "ddof": DDOF,
        "nkeep": NKEEP,
        "revert_on_nkeep": REVERT_ON_NKEEP,
        "cenfunc": profile.cenfunc,
        "clip_cen": profile.cenfunc,
        "stdfunc": profile.stdfunc,
    }


def _operations(
    values: np.ndarray,
    input_mask: np.ndarray | None,
    profile: Profile,
) -> dict[str, Callable[[], object]]:
    """Build the public operations whose results are checked and timed."""
    kwargs = _params(profile)
    if values.ndim == 1:
        return {
            "full": lambda: rd.sigclip_1d(values, mask=input_mask, **kwargs),
            "mask": lambda: rd.sigclip_mask_1d(values, mask=input_mask, **kwargs),
            "fused_mean": lambda: rd.sigclip_combine_1d(
                values, mask=input_mask, combine="mean", **kwargs
            ),
            "fused_median": lambda: rd.sigclip_combine_1d(
                values, mask=input_mask, combine="median", **kwargs
            ),
        }
    return {
        "full": lambda: rd.sigclip(values, mask=input_mask, **kwargs),
        "mask": lambda: rd.sigclip_mask(values, mask=input_mask, **kwargs),
        "restored_flags": lambda: rd.sigclip_restored_flags(
            values, mask=input_mask, **kwargs
        ),
        "fused_mean": lambda: rd.sigclip_combine(
            values, mask=input_mask, combine="mean", **kwargs
        ),
        "fused_median": lambda: rd.sigclip_combine(
            values, mask=input_mask, combine="median", **kwargs
        ),
    }


def _assert_close(actual: object, expected: object, *, dtype: str, label: str) -> None:
    """Compare diagnostic or fused floating outputs at dtype precision."""
    if np.dtype(dtype) == np.float32:
        rtol, atol = 2e-5, 2e-5
    else:
        rtol, atol = 2e-12, 2e-12
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=label,
    )


def _expected_fused(
    values: np.ndarray,
    rejected: np.ndarray,
    combine: str,
) -> np.ndarray | object:
    """Reduce retained original values in the same output dtype as the input."""
    if values.ndim == 1:
        retained = values[~rejected].astype(np.float64)
        if retained.size == 0:
            return np.asarray(np.nan, dtype=values.dtype)[()]
        if combine == "mean":
            result = np.sum(retained, dtype=np.float64) / retained.size
        else:
            result = np.median(retained)
        return np.asarray(result, dtype=values.dtype)[()]

    output = np.empty(values.shape[1:], dtype=values.dtype)
    for index in np.ndindex(output.shape):
        selector = (slice(None),) + index
        retained = values[selector][~rejected[selector]].astype(np.float64)
        if retained.size == 0:
            output[index] = np.asarray(np.nan, dtype=values.dtype)
        elif combine == "mean":
            output[index] = np.asarray(
                np.sum(retained, dtype=np.float64) / retained.size, dtype=values.dtype
            )
        else:
            output[index] = np.asarray(np.median(retained), dtype=values.dtype)
    return output


def check_case(
    values: np.ndarray, input_mask: np.ndarray | None, profile: Profile
) -> None:
    """Check every operation in one case before timing it."""
    expected = reference_clip(values, input_mask, profile)
    operations = _operations(values, input_mask, profile)
    actual_mask, actual_std, actual_low, actual_upp, actual_nit, actual_flags = (
        operations["full"]()
    )
    np.testing.assert_array_equal(actual_mask, expected.rejected)
    _assert_close(actual_std, expected.std, dtype=str(values.dtype), label="std")
    _assert_close(actual_low, expected.low, dtype=str(values.dtype), label="low")
    _assert_close(actual_upp, expected.upp, dtype=str(values.dtype), label="upp")
    np.testing.assert_array_equal(actual_nit, expected.nit)
    np.testing.assert_array_equal(actual_flags, expected.flags)
    np.testing.assert_array_equal(operations["mask"](), expected.rejected)

    if values.ndim != 1:
        np.testing.assert_array_equal(
            operations["restored_flags"](), np.zeros(values.shape, dtype=np.uint8)
        )
    for combine in ("mean", "median"):
        _assert_close(
            operations[f"fused_{combine}"](),
            _expected_fused(values, expected.rejected, combine),
            dtype=str(values.dtype),
            label=f"fused_{combine}",
        )


def _inner_calls(size: int, requested: int) -> int:
    """Choose equal per-operation work for a case unless explicitly overridden."""
    if requested > 0:
        return requested
    return max(1, min(100_000, 1_000_000 // max(1, size)))


def _shape_label(values: np.ndarray) -> str:
    """Format a stable shape label for tables and CSV rows."""
    return "x".join(str(size) for size in values.shape)


def _metadata() -> list[str]:
    """Return benchmark metadata, including the fixed clipping contract."""
    return [
        "contract: sigma=3,maxiters=5,ddof=0,nkeep=0,revert_on_nkeep=False",
        "reference: independent NumPy finite-only cumulative mean/std or median/MAD",
        *environment_lines(
            bottleneck_available=package_version("bottleneck") != "not installed"
        ),
    ]


def _write_csv(
    path: Path, metadata: Iterable[str], rows: list[dict[str, object]]
) -> None:
    """Write metadata comments followed by raw timing rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layout",
        "shape",
        "dtype",
        "profile",
        "input_mask",
        "operation",
        "elements",
        "inner_calls",
        "repeats",
        "warmups",
        "median_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        for line in metadata:
            stream.write(f"# {line}\n")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run correctness checks and, unless requested, the timing matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--dtypes", nargs="+", choices=DEFAULT_DTYPES, default=list(DEFAULT_DTYPES)
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILES),
        default=list(PROFILES),
    )
    parser.add_argument("--without-stack", action="store_true")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--inner",
        type=int,
        default=0,
        help="calls per timing sample; 0 chooses a size-based common count",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0 or args.inner < 0:
        parser.error("repeats must be positive; warmups and inner must be nonnegative")
    if any(length < 15 for length in args.lengths):
        parser.error(
            "lengths must be at least 15 to separate outliers "
            "and masked/nonfinite samples"
        )

    metadata = _metadata()
    for line in metadata:
        print(f"# {line}")
    rows: list[dict[str, object]] = []
    cases: list[tuple[str, np.ndarray]] = []
    for dtype in args.dtypes:
        for length in args.lengths:
            cases.append((f"vector_{length}", _make_values(length, dtype, stack=False)))
        if not args.without_stack:
            cases.append(
                ("stack_31x64x64", _make_values(STACK_SHAPE[0], dtype, stack=True))
            )

    print("| layout | shape | dtype | profile | input mask | operation | median ms |")
    print("|---|---|---|---|---:|---|---:|")
    for layout, values in cases:
        for profile_name in args.profiles:
            profile = PROFILES[profile_name]
            for mask_enabled in (False, True):
                input_mask = _make_mask(values, enabled=mask_enabled)
                check_case(values, input_mask, profile)
                if args.check_only:
                    continue
                operations = _operations(values, input_mask, profile)
                inner = _inner_calls(values.size, args.inner)
                for operation, function in operations.items():
                    median_ms = timeit(
                        function,
                        repeats=args.repeats,
                        warmups=args.warmups,
                        inner=inner,
                    )
                    row = {
                        "layout": layout.split("_")[0],
                        "shape": _shape_label(values),
                        "dtype": str(values.dtype),
                        "profile": profile.name,
                        "input_mask": mask_enabled,
                        "operation": operation,
                        "elements": values.size,
                        "inner_calls": inner,
                        "repeats": args.repeats,
                        "warmups": args.warmups,
                        "median_ms": median_ms,
                    }
                    rows.append(row)
                    print(
                        f"| {row['layout']} | {row['shape']} | {row['dtype']} | "
                        f"{row['profile']} | {str(mask_enabled).lower()} | "
                        f"`{operation}` | {median_ms:.6f} |"
                    )
    if args.csv is not None:
        _write_csv(args.csv, metadata, rows)
        print(f"# wrote {args.csv}")


if __name__ == "__main__":
    main()
