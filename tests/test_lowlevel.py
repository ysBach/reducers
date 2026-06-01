"""Low-level trusted-buffer Python API tests."""

from __future__ import annotations

import numpy as np
import pytest
import reducers as rd


def test_lowlevel_namespace_is_exported():
    assert rd.lowlevel.mean_valid(np.array([1.0, 2.0, 3.0])) == 2.0


def test_1d_valid_scan_primitives_use_trusted_buffer():
    a = np.array([1.0, 2.0, 3.0, np.inf], dtype=np.float64)
    finite = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert rd.lowlevel.sum_valid(a) == np.inf
    assert rd.lowlevel.mean_valid(a) == np.inf
    assert rd.lowlevel.sum_skip_nonfinite(a) == 6.0
    assert rd.lowlevel.mean_skip_nonfinite(a) == 2.0

    assert rd.lowlevel.var_valid(finite, ddof=1) == pytest.approx(
        np.var(finite, ddof=1)
    )
    std, mean = rd.lowlevel.std_mean_valid(finite, ddof=1)
    assert std == pytest.approx(np.std(finite, ddof=1))
    assert mean == pytest.approx(np.mean(finite))


def test_1d_order_primitives_and_integer_exact_selection():
    values = np.array([9, 1, 5, 3], dtype=np.uint16)

    assert rd.lowlevel.lmedian_valid(values) == np.uint16(3)
    assert rd.lowlevel.median_valid(values) == 4.0
    assert rd.lowlevel.percentile_valid(values, [25.0, 50.0, 75.0]).tolist() == [
        2.5,
        4.0,
        6.0,
    ]
    assert rd.lowlevel.quantile_valid(values, 0.5) == 4.0


def test_order_primitives_have_mutating_in_place_variants():
    values = np.array([9.0, 1.0, 5.0, 3.0], dtype=np.float64)
    ints = np.array([9, 1, 5, 3], dtype=np.uint16)

    assert rd.lowlevel.median_valid_in_place(values) == 4.0
    assert sorted(values.tolist()) == [1.0, 3.0, 5.0, 9.0]

    assert rd.lowlevel.lmedian_valid_in_place(ints) == np.uint16(3)
    assert sorted(ints.tolist()) == [1, 3, 5, 9]

    pct_values = np.array([9.0, 1.0, 5.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(
        rd.lowlevel.percentiles_valid_in_place(pct_values, [25.0, 50.0, 75.0]),
        np.percentile([9.0, 1.0, 5.0, 3.0], [25.0, 50.0, 75.0]),
    )


def test_1d_copy_false_delegates_input_contract_to_core():
    base = np.arange(10, dtype=np.float64)
    view = base[::2]

    with pytest.raises(TypeError, match="not contiguous or is misaligned"):
        rd.lowlevel.sum_valid(view)

    assert rd.lowlevel.sum_valid(view, copy=True) == np.sum(view)


def test_1d_weighted_sum_primitives_return_raw_triple():
    a = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    w = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    assert rd.lowlevel.weighted_sum_valid(a, w) == (170.0, 60.0, 7.0)
    assert rd.lowlevel.weighted_sum_only_valid(a, w) == 170.0
    assert rd.lowlevel.weighted_sum_and_weights_valid(a, w) == (170.0, 60.0)
    assert rd.lowlevel.weighted_sum_and_unweighted_valid(a, w) == (170.0, 7.0)
    assert rd.lowlevel.weighted_average_valid(a, w) == pytest.approx(170.0 / 60.0)


def test_1d_weighted_sum_skip_policies_filter_values_not_weights():
    a = np.array([1.0, np.nan, np.inf, 4.0], dtype=np.float64)
    w = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)

    weighted, sum_weights, unweighted = rd.lowlevel.weighted_sum_skip_nan(a, w)
    assert weighted == np.inf
    assert sum_weights == 14.0
    assert unweighted == np.inf

    assert rd.lowlevel.weighted_sum_skip_nonfinite(a, w) == (30.0, 9.0, 5.0)
    assert rd.lowlevel.weighted_sum_only_skip_nonfinite(a, w) == 30.0
    assert rd.lowlevel.weighted_sum_and_weights_skip_nonfinite(a, w) == (30.0, 9.0)
    assert rd.lowlevel.weighted_sum_and_unweighted_skip_nonfinite(a, w) == (
        30.0,
        5.0,
    )
    assert rd.lowlevel.weighted_average_skip_nonfinite(a, w) == pytest.approx(
        30.0 / 9.0
    )


def test_1d_weighted_sum_copy_false_delegates_dtype_and_contiguity_to_core():
    a = np.arange(6, dtype=np.float64)[::2]
    w = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(TypeError, match="not contiguous or is misaligned"):
        rd.lowlevel.weighted_sum_valid(a, w)

    with pytest.raises(TypeError, match="copy"):
        rd.lowlevel.weighted_sum_valid(a, w, copy=True)


def test_fixed_axis_valid_reducers_avoid_normalization_when_copy_false():
    rng = np.random.default_rng(20260601)
    a = rng.normal(size=(5, 7)).astype(np.float64)

    np.testing.assert_allclose(
        rd.lowlevel.reduce_axis0_valid(a, "mean"),
        np.mean(a, axis=0),
    )
    np.testing.assert_allclose(
        rd.lowlevel.reduce_axis_last_valid(a, "std", ddof=1),
        np.std(a, axis=-1, ddof=1),
    )
    np.testing.assert_allclose(
        rd.lowlevel.reduce_axis0_valid(a.astype(np.float32), "median"),
        np.median(a.astype(np.float32), axis=0),
        rtol=1e-6,
    )


def test_fixed_axis_skip_nonfinite_policy_and_integer_dtype():
    a = np.array(
        [
            [1.0, np.nan, np.inf],
            [3.0, 5.0, 7.0],
            [np.inf, 9.0, 11.0],
        ],
        dtype=np.float64,
    )
    uints = np.array([[5, 2], [7, 4], [1, 9]], dtype=np.uint16)

    np.testing.assert_allclose(
        rd.lowlevel.reduce_axis0_skip_nonfinite(a, "mean"),
        np.array([2.0, 7.0, 9.0]),
    )
    got = rd.lowlevel.reduce_axis0_valid(uints, "min")
    assert got.dtype == np.uint16
    np.testing.assert_array_equal(got, np.min(uints, axis=0))


def test_fixed_axis_copy_false_delegates_input_contract_to_core():
    view = np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2]

    with pytest.raises(TypeError, match="not contiguous or is misaligned"):
        rd.lowlevel.reduce_axis_last_valid(view, "sum")

    np.testing.assert_allclose(
        rd.lowlevel.reduce_axis_last_valid(view, "sum", copy=True),
        np.sum(view, axis=-1),
    )
