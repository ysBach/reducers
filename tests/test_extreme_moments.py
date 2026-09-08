"""Regression tests for recovering finite moments after intermediate overflow."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import reducers as rd


def _fraction_mean(values: np.ndarray) -> float:
    exact = sum((Fraction.from_float(float(value)) for value in values), Fraction())
    return float(exact / values.size)


def _decimal_variance(values: np.ndarray, ddof: int = 0) -> float:
    # Decimal keeps the reference independent of the reducer's floating-point
    # summation order. The precision covers the products of binary float values
    # near 1e154 before converting the final representable result to float.
    with localcontext() as context:
        context.prec = 512
        exact = [Decimal.from_float(float(value)) for value in values]
        mean = sum(exact, Decimal(0)) / Decimal(len(exact))
        denominator = Decimal(len(exact) - ddof)
        return float(sum((value - mean) ** 2 for value in exact) / denominator)


def test_overflowing_moments_match_independent_references() -> None:
    constant = np.full(8, 1e308, dtype=np.float64)
    assert rd.mean(constant) == _fraction_mean(constant)
    assert rd.var(constant) == 0.0
    assert rd.std(constant) == 0.0

    alternating = np.array([-1e154, 1e154, -1e154, 1e154])
    expected = _decimal_variance(alternating)
    assert rd.mean(alternating) == 0.0
    assert rd.var(alternating) == expected
    assert rd.var(alternating, ddof=1) == _decimal_variance(alternating, ddof=1)
    assert rd.std(alternating) == np.sqrt(expected)


def test_overflow_recovery_preserves_cancellation_and_subnormal_residuals() -> None:
    residual = np.array(
        [
            1e308,
            1e308,
            -1e308,
            -1e308,
            1e308,
            1e308,
            -1e308,
            -1e308,
            9e-100,
        ],
        dtype=np.float64,
    )
    assert rd.mean(residual) == _fraction_mean(residual)

    smallest = np.nextafter(0.0, 1.0)
    subnormal_residual = np.concatenate(
        [
            residual[:8],
            np.full(16, smallest, dtype=np.float64),
        ]
    )
    assert rd.mean(subnormal_residual) == _fraction_mean(subnormal_residual)
    assert rd.mean(subnormal_residual) == smallest


def test_mean_recovers_when_an_unscaled_partial_exceeds_float64() -> None:
    values = np.array(
        [
            8.988465674311569e307,
            1.7976931348623193e307,
            -7.190772539449259e307,
            7.190772539449254e307,
            7.190772539449275e307,
        ]
    )
    expected = _fraction_mean(values)
    np.testing.assert_allclose(rd.mean(values), expected, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        rd.mean(values[:, None], axis=0), [expected], rtol=2e-15, atol=0
    )


def test_nonfinite_policies_empty_inputs_and_ddof_remain_distinct() -> None:
    values = np.array([1e308, np.nan, 1e308, np.inf, -np.inf])
    assert np.isnan(rd.mean(values))
    assert np.isnan(rd.var(values))
    assert np.isnan(rd.std(values))

    # nan* keeps infinities, so an infinity participates in the statistic.
    keeps_inf = np.array([1e308, np.nan, np.inf])
    assert rd.nanmean(keeps_inf) == np.inf
    assert np.isnan(rd.nanvar(keeps_inf))

    finite = rd.nanvar(values, ignore_inf=True, return_mean=True)
    assert finite[0] == 0.0
    assert finite[1] == 1e308
    insufficient = rd.nanvar(
        values,
        ignore_inf=True,
        ddof=2,
        return_mean=True,
    )
    assert np.isnan(insufficient[0])
    assert insufficient[1] == 1e308

    empty = np.array([], dtype=np.float64)
    for reducer in (rd.mean, rd.var, rd.std, rd.nanmean, rd.nanvar, rd.nanstd):
        assert np.isnan(reducer(empty))
    all_skipped = np.array([np.nan, np.inf, -np.inf])
    assert np.isnan(rd.nanmean(all_skipped, ignore_inf=True))
    assert np.isnan(rd.nanvar(all_skipped, ignore_inf=True))


def test_overflow_recovery_handles_strided_axis0_and_general_axis() -> None:
    base = np.empty((4, 4), dtype=np.float64)
    base[:, 0] = 1e308
    base[:, 1] = 0.0
    base[:, 2] = [-1e154, 1e154, -1e154, 1e154]
    base[:, 3] = 0.0
    strided = base[:, ::2]
    assert not strided.flags.c_contiguous

    np.testing.assert_array_equal(
        rd.mean(strided, axis=0), np.array([1e308, 0.0], dtype=np.float64)
    )
    np.testing.assert_array_equal(
        rd.var(strided, axis=0), np.array([0.0, 1e308], dtype=np.float64)
    )

    cube = np.array(
        [
            [[1e308, 1e308, 1e308, 1e308], [-1e154, 1e154, -1e154, 1e154]],
            [[1e308, 1e308, 1e308, 1e308], [-1e154, 1e154, -1e154, 1e154]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        rd.mean(cube, axis=-1), np.array([[1e308, 0.0], [1e308, 0.0]])
    )
    np.testing.assert_array_equal(
        rd.var(cube, axis=-1), np.array([[0.0, 1e308], [0.0, 1e308]])
    )


def test_axis_mean_recovery_keeps_small_residual_after_large_cancellation() -> None:
    values = np.array(
        [1.0, 1.0, 1e308, 1e308, -1e308, 1e308, -1e308, -1e308],
        dtype=np.float64,
    ).reshape(8, 1)

    np.testing.assert_array_equal(rd.mean(values, axis=0), np.array([0.25]))
    variance, mean = rd.var(values, axis=0, ddof=8, return_mean=True)
    assert np.isnan(variance[0])
    assert mean[0] == 0.25


def test_lowlevel_moment_helpers_share_overflow_recovery() -> None:
    constant = np.full(8, 1e308, dtype=np.float64)
    assert rd.lowlevel.mean_valid(constant) == 1e308
    assert rd.lowlevel.var_valid(constant) == 0.0
    assert rd.lowlevel.std_mean_valid(constant) == (0.0, 1e308)

    values = np.array([1e308, np.nan, 1e308, np.inf, -np.inf])
    assert rd.lowlevel.mean_skip_nonfinite(values) == 1e308
    assert rd.lowlevel.var_mean_skip_nonfinite(values) == (0.0, 1e308)


def test_f32_axis_results_keep_dtype_when_true_variance_exceeds_f32() -> None:
    largest = np.finfo(np.float32).max
    values = np.array([[-largest], [largest], [-largest], [largest]], dtype=np.float32)

    mean = rd.mean(values, axis=0)
    variance, returned_mean = rd.var(values, axis=0, return_mean=True)
    standard_deviation = rd.std(values, axis=0)

    assert mean.dtype == np.dtype(np.float32)
    assert variance.dtype == np.dtype(np.float32)
    assert returned_mean.dtype == np.dtype(np.float32)
    assert standard_deviation.dtype == np.dtype(np.float32)
    assert mean[0] == 0.0
    assert returned_mean[0] == 0.0
    assert np.isinf(variance[0])
    assert standard_deviation[0] == largest
