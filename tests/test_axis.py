"""Axis parity tests for reducers vs NumPy (axis=0 and axis=-1)."""

from __future__ import annotations

import numpy as np
import pytest
import reducers as rd

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover - depends on optional test environment
    bn = None

SEED = 20250311


def _arr(dtype):
    rng = np.random.default_rng(SEED)
    a = rng.normal(0.0, 1.0, size=(7, 5, 4)).astype(dtype)
    a[0, 1, 2] = np.nan
    a[3, 0, 1] = np.nan
    return a


@pytest.mark.parametrize("axis", [0, -1, 2])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    ("rd_fn", "np_fn"),
    [
        (rd.nanmean, np.nanmean),
        (rd.nanmedian, np.nanmedian),
        (rd.nansum, np.nansum),
        (rd.nanmin, np.nanmin),
        (rd.nanmax, np.nanmax),
    ],
)
def test_axis_nan_reducers_match_numpy(axis, dtype, rd_fn, np_fn):
    a = _arr(dtype)
    got = rd_fn(a, axis=axis)
    exp = np_fn(a, axis=axis)
    assert got.shape == exp.shape
    # atol guards near-zero means where rd (f64 accum) and np (f32 accum) differ
    # by a large *relative* but tiny absolute amount.
    np.testing.assert_allclose(got, exp, rtol=1e-4, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_axis_nanvar(axis, dtype):
    a = _arr(dtype)
    np.testing.assert_allclose(
        rd.nanvar(a, axis=axis, ddof=1),
        np.nanvar(a, axis=axis, ddof=1),
        rtol=1e-3,
        atol=1e-5,
        equal_nan=True,
    )
    s, m = rd.nanstd(a, axis=axis, ddof=1, return_mean=True)
    np.testing.assert_allclose(
        s, np.nanstd(a, axis=axis, ddof=1), rtol=1e-3, atol=1e-5, equal_nan=True
    )
    np.testing.assert_allclose(
        m, np.nanmean(a, axis=axis), rtol=1e-4, atol=1e-5, equal_nan=True
    )


def test_axis0_variance_is_stable_for_large_offsets():
    a = np.array(
        [
            [1.0e16, 1.0e16 + 4.0],
            [1.0e16 + 2.0, 1.0e16 + 6.0],
            [1.0e16 + 4.0, 1.0e16 + 8.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(rd.var(a, axis=0), np.var(a, axis=0), rtol=1e-12)
    np.testing.assert_allclose(rd.std(a, axis=0), np.std(a, axis=0), rtol=1e-12)


@pytest.mark.parametrize("axis", [0, -1])
def test_axis_plain_reducers_match_numpy(axis):
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=(6, 8)).astype(np.float64)  # finite -> plain == numpy
    for rd_fn, np_fn in [
        (rd.mean, np.mean),
        (rd.median, np.median),
        (rd.sum, np.sum),
        (rd.min, np.min),
        (rd.max, np.max),
        (rd.var, np.var),
        (
            lambda x, axis: rd.std(x, axis=axis, ddof=1),
            lambda x, axis: np.std(x, axis=axis, ddof=1),
        ),
    ]:
        np.testing.assert_allclose(
            rd_fn(a, axis=axis), np_fn(a, axis=axis), rtol=1e-5, atol=1e-6
        )
    s, m = rd.std(a, axis=axis, ddof=1, return_mean=True)
    np.testing.assert_allclose(s, np.std(a, axis=axis, ddof=1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(m, np.mean(a, axis=axis), rtol=1e-5, atol=1e-6)


def test_axis0_min_max_edge_semantics():
    a = np.array(
        [
            [np.nan, 3.0, np.inf, -np.inf],
            [1.0, np.nan, 4.0, -5.0],
            [2.0, 0.0, -np.inf, np.nan],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(rd.min(a, axis=0), np.min(a, axis=0), equal_nan=True)
    np.testing.assert_allclose(rd.max(a, axis=0), np.max(a, axis=0), equal_nan=True)
    np.testing.assert_allclose(
        rd.nanmin(a, axis=0), np.nanmin(a, axis=0), equal_nan=True
    )
    np.testing.assert_allclose(
        rd.nanmax(a, axis=0), np.nanmax(a, axis=0), equal_nan=True
    )
    np.testing.assert_allclose(
        rd.nanmin(a, axis=0, ignore_inf=True),
        np.array([1.0, 0.0, 4.0, -5.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rd.nanmax(a, axis=0, ignore_inf=True),
        np.array([2.0, 3.0, 4.0, -5.0]),
        equal_nan=True,
    )


def test_axis0_mean_sum_edge_semantics():
    a = np.array(
        [
            [np.nan, 3.0, np.inf, -np.inf],
            [1.0, np.nan, 4.0, -5.0],
            [2.0, np.nan, -np.inf, np.nan],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(rd.mean(a, axis=0), np.mean(a, axis=0), equal_nan=True)
    np.testing.assert_allclose(rd.sum(a, axis=0), np.sum(a, axis=0), equal_nan=True)
    np.testing.assert_allclose(
        rd.nanmean(a, axis=0), np.nanmean(a, axis=0), equal_nan=True
    )
    np.testing.assert_allclose(
        rd.nansum(a, axis=0), np.nansum(a, axis=0), equal_nan=True
    )
    np.testing.assert_allclose(
        rd.nanmean(a, axis=0, ignore_inf=True),
        np.array([1.5, 3.0, 4.0, -5.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rd.nansum(a, axis=0, ignore_inf=True),
        np.array([3.0, 3.0, 4.0, -5.0]),
        equal_nan=True,
    )


def test_axis0_order_stats_handle_nan_inf_and_lower_median():
    a = np.array(
        [
            [np.nan, 1.0, np.inf, -np.inf, 4.0],
            [1.0, np.inf, 3.0, 5.0, np.nan],
            [5.0, 7.0, 1.0, np.nan, 2.0],
            [3.0, np.nan, 5.0, -1.0, 8.0],
            [9.0, 9.0, 7.0, 11.0, 10.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        rd.median(a, axis=0),
        np.array([np.nan, np.nan, 5.0, np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rd.nanmedian(a, axis=0),
        np.nanmedian(a, axis=0),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rd.nanmedian(a, axis=0, ignore_inf=True),
        np.array([4.0, 7.0, 4.0, 5.0, 6.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rd.lmedian(a, axis=0, ignore_inf=True),
        np.array([3.0, 7.0, 3.0, 5.0, 4.0], dtype=np.float32),
        equal_nan=True,
    )


def test_axis0_order_stats_small_hardcoded_finite_only_semantics():
    a = np.array(
        [
            [5.0, np.nan, np.inf, 1.0],
            [1.0, 2.0, 7.0, np.inf],
            [4.0, 4.0, np.nan, 9.0],
            [3.0, 8.0, -np.inf, 3.0],
            [2.0, 6.0, 5.0, np.nan],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        rd.nanmedian(a, axis=0, ignore_inf=True),
        np.array([3.0, 5.0, 6.0, 3.0], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rd.lmedian(a, axis=0, ignore_inf=True),
        np.array([3.0, 4.0, 5.0, 3.0], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_axis0_order_stats_longer_hardcoded_medians():
    descending = np.arange(30, -1, -1, dtype=np.float32)
    shifted = np.array(
        [
            10,
            3,
            27,
            14,
            8,
            22,
            30,
            1,
            19,
            6,
            25,
            12,
            17,
            4,
            29,
            0,
            21,
            9,
            15,
            2,
            28,
            7,
            24,
            11,
            18,
            5,
            26,
            13,
            20,
            16,
            23,
        ],
        dtype=np.float32,
    )
    a = np.stack([descending, np.arange(31, dtype=np.float32), shifted], axis=1)

    expected = np.array([15.0, 15.0, 15.0], dtype=np.float32)
    np.testing.assert_allclose(
        rd.nanmedian(a, axis=0, ignore_inf=True), expected, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        rd.lmedian(a, axis=0, ignore_inf=True), expected, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("dtype", [np.uint16, np.int16])
def test_integer_axis0_median_hardcoded_examples_return_float(dtype):
    n5 = np.array(
        [
            [5, 9],
            [1, 7],
            [4, 3],
            [3, 5],
            [2, 1],
        ],
        dtype=dtype,
    )
    got5 = rd.nanmedian(n5, axis=0, validate=False)
    assert got5.dtype == np.float64
    np.testing.assert_allclose(got5, np.array([3.0, 5.0]), rtol=0.0, atol=0.0)

    n31 = np.stack(
        [
            np.arange(30, -1, -1, dtype=dtype),
            np.arange(31, dtype=dtype),
        ],
        axis=1,
    )
    got31 = rd.nanmedian(n31, axis=0, validate=False)
    assert got31.dtype == np.float64
    np.testing.assert_allclose(got31, np.array([15.0, 15.0]), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("axis", [0, -1])
def test_axis_percentile_matches_numpy(axis):
    a = _arr(np.float64)
    got = rd.nanpercentile(a, [25.0, 50.0, 75.0], axis=axis)
    exp = np.nanpercentile(a, [25.0, 50.0, 75.0], axis=axis)
    assert got.shape == exp.shape
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize(
    ("rd_fn", "np_fn", "q"),
    [
        (rd.percentile, np.percentile, [25.0, 50.0, 75.0]),
        (rd.quantile, np.quantile, [0.25, 0.50, 0.75]),
        (rd.nanquantile, np.nanquantile, [0.25, 0.50, 0.75]),
    ],
)
def test_axis_rank_reducers_match_numpy(axis, rd_fn, np_fn, q):
    a = _arr(np.float64)
    got = rd_fn(a, q, axis=axis)
    exp = np_fn(a, q, axis=axis)
    assert got.shape == exp.shape
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_axis_percentile_scalar_q_drops_leading_axis():
    a = _arr(np.float64)
    got = rd.nanpercentile(a, 50.0, axis=0)
    exp = np.nanpercentile(a, 50.0, axis=0)
    assert got.shape == exp.shape == (5, 4)
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("axis", [0, -1])
def test_axis_average_matches_numpy_full_weights(axis):
    a = np.arange(2 * 5 * 4, dtype=np.float64).reshape(2, 5, 4)
    w = np.linspace(1.0, 2.0, a.size).reshape(a.shape)
    np.testing.assert_allclose(
        rd.average(a, weights=w, axis=axis),
        np.average(a, weights=w, axis=axis),
    )


@pytest.mark.parametrize("axis", [0, -1])
def test_axis_average_matches_numpy_1d_weights(axis):
    a = np.arange(2 * 5 * 4, dtype=np.uint16).reshape(2, 5, 4)
    n = a.shape[axis]
    w = np.arange(1, n + 1, dtype=np.int16)
    np.testing.assert_allclose(
        rd.average(a, weights=w, axis=axis),
        np.average(a, weights=w, axis=axis),
    )
    np.testing.assert_allclose(
        rd.average(a, weights=w, axis=axis, validate=False),
        np.average(a, weights=w, axis=axis),
    )


def test_axis_nanaverage_semantics():
    a = np.array(
        [
            [1.0, np.nan, np.inf],
            [3.0, 4.0, 6.0],
        ]
    )
    w = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        rd.nanaverage(a, weights=w, axis=0, ignore_inf=True),
        np.array([7.0 / 3.0, 4.0, 6.0]),
        equal_nan=True,
    )
    got = rd.nanaverage(np.array([[np.nan, 1.0]]), weights=np.array([1.0]), axis=0)
    np.testing.assert_allclose(got, np.array([np.nan, 1.0]), equal_nan=True)


def test_axis0_average_full_shape_weights_nan_semantics():
    a = np.array(
        [
            [1.0, np.nan, np.inf],
            [3.0, 4.0, 6.0],
            [5.0, 8.0, 10.0],
        ],
        dtype=np.float64,
    )
    w = np.array(
        [
            [1.0, 10.0, 10.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        rd.average(a, weights=w, axis=0),
        np.average(a, weights=w, axis=0),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        rd.nanaverage(a, weights=w, axis=0, ignore_inf=True),
        np.array([22.0 / 6.0, 32.0 / 5.0, 42.0 / 5.0]),
        equal_nan=True,
    )


def test_axis_average_errors():
    a = np.ones((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="reduction axis"):
        rd.average(a, weights=np.ones(4), axis=0)
    with pytest.raises(ZeroDivisionError, match="weights sum to zero"):
        rd.average(a, weights=np.zeros(2), axis=0)


@pytest.mark.parametrize("axis", [0, -1])
def test_axis_minmax_matches_nanmin_nanmax(axis):
    a = _arr(np.float64)
    lo, hi = rd.minmax(a, axis=axis)
    np.testing.assert_allclose(lo, np.min(a, axis=axis), equal_nan=True)
    np.testing.assert_allclose(hi, np.max(a, axis=axis), equal_nan=True)
    lo, hi = rd.nanminmax(a, axis=axis)
    np.testing.assert_allclose(lo, np.nanmin(a, axis=axis), equal_nan=True)
    np.testing.assert_allclose(hi, np.nanmax(a, axis=axis), equal_nan=True)


@pytest.mark.skipif(bn is None, reason="bottleneck is not installed")
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize(
    ("rd_fn", "bn_fn", "kwargs"),
    [
        (rd.nanmean, lambda x, axis: bn.nanmean(x, axis=axis), {}),
        (rd.nanmedian, lambda x, axis: bn.nanmedian(x, axis=axis), {}),
        (rd.nansum, lambda x, axis: bn.nansum(x, axis=axis), {}),
        (rd.nanmin, lambda x, axis: bn.nanmin(x, axis=axis), {}),
        (rd.nanmax, lambda x, axis: bn.nanmax(x, axis=axis), {}),
        (rd.nanvar, lambda x, axis: bn.nanvar(x, axis=axis, ddof=1), {"ddof": 1}),
        (rd.nanstd, lambda x, axis: bn.nanstd(x, axis=axis, ddof=1), {"ddof": 1}),
    ],
)
def test_axis_nan_reducers_match_bottleneck_where_available(axis, rd_fn, bn_fn, kwargs):
    a = _arr(np.float64)
    np.testing.assert_allclose(
        rd_fn(a, axis=axis, **kwargs),
        bn_fn(a, axis),
        rtol=1e-4,
        atol=1e-6,
        equal_nan=True,
    )


def test_unsupported_axis_raises():
    a = np.ones((3, 4, 5), dtype=np.float64)
    with pytest.raises(NotImplementedError):
        rd.mean(a, axis=1)


def test_count_finite_axis():
    a = _arr(np.float64)
    got = rd.count_finite(a, axis=0)
    exp = np.isfinite(a).sum(axis=0)
    np.testing.assert_array_equal(got, exp)


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ],
)
@pytest.mark.parametrize("axis", [0, -1])
def test_integer_axis_validate_false_matches_validated_path(dtype, axis):
    a = (np.arange(2 * 5 * 4).reshape(2, 5, 4) % 7).astype(dtype)
    ax = axis if axis >= 0 else a.ndim + axis
    expected = {
        rd.mean: np.mean(a, axis=axis),
        rd.nanmean: np.nanmean(a, axis=axis),
        rd.sum: np.sum(a, axis=axis),
        rd.nansum: np.nansum(a, axis=axis),
        rd.min: np.min(a, axis=axis),
        rd.nanmin: np.nanmin(a, axis=axis),
        rd.max: np.max(a, axis=axis),
        rd.nanmax: np.nanmax(a, axis=axis),
        rd.median: np.median(a, axis=axis),
        rd.nanmedian: np.nanmedian(a, axis=axis),
        rd.var: np.var(a, axis=axis),
        rd.nanvar: np.nanvar(a, axis=axis),
    }
    for rd_fn, want in expected.items():
        np.testing.assert_allclose(
            rd_fn(a, axis=axis, validate=False),
            want,
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            rd_fn(a, axis=axis, validate=True),
            want,
            rtol=1e-12,
            atol=0.0,
        )

    lmedian_expected = np.take(
        np.sort(a, axis=axis),
        (a.shape[ax] - 1) // 2,
        axis=axis,
    )
    np.testing.assert_array_equal(
        rd.lmedian(a, axis=axis, validate=False), lmedian_expected
    )
    np.testing.assert_array_equal(
        rd.lmedian(a, axis=axis, validate=True), lmedian_expected
    )

    pct_expected = np.nanpercentile(a.astype(np.float64), [25.0, 50.0], axis=axis)
    np.testing.assert_allclose(
        rd.nanpercentile(a, [25.0, 50.0], axis=axis, validate=False),
        pct_expected,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rd.nanpercentile(a, [25.0, 50.0], axis=axis, validate=True),
        pct_expected,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        rd.count_finite(a, axis=axis, validate=False),
        np.full(a.shape[:ax] + a.shape[ax + 1 :], a.shape[ax]),
    )
    np.testing.assert_array_equal(
        rd.count_finite(a, axis=axis, validate=True),
        np.full(a.shape[:ax] + a.shape[ax + 1 :], a.shape[ax]),
    )


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ],
)
@pytest.mark.parametrize("axis", [0, -1])
def test_integer_axis_exact_select_reducers_preserve_dtype(dtype, axis, validate):
    a = np.array(
        [
            [[5, 1], [2, 7]],
            [[3, 4], [6, 0]],
            [[8, 9], [1, 2]],
        ],
        dtype=dtype,
    )
    if axis == 0:
        expected = {
            rd.min: np.array([[3, 1], [1, 0]], dtype=dtype),
            rd.nanmin: np.array([[3, 1], [1, 0]], dtype=dtype),
            rd.max: np.array([[8, 9], [6, 7]], dtype=dtype),
            rd.nanmax: np.array([[8, 9], [6, 7]], dtype=dtype),
            rd.lmedian: np.array([[5, 4], [2, 2]], dtype=dtype),
        }
    else:
        expected = {
            rd.min: np.array([[1, 2], [3, 0], [8, 1]], dtype=dtype),
            rd.nanmin: np.array([[1, 2], [3, 0], [8, 1]], dtype=dtype),
            rd.max: np.array([[5, 7], [4, 6], [9, 2]], dtype=dtype),
            rd.nanmax: np.array([[5, 7], [4, 6], [9, 2]], dtype=dtype),
            rd.lmedian: np.array([[1, 2], [3, 0], [8, 1]], dtype=dtype),
        }
    for rd_fn in [rd.min, rd.nanmin, rd.max, rd.nanmax, rd.lmedian]:
        got = rd_fn(a, axis=axis, validate=validate)
        assert got.dtype == a.dtype
        np.testing.assert_array_equal(got, expected[rd_fn])


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32])
@pytest.mark.parametrize("axis", [0, -1])
def test_integer_axis_median_returns_float_without_value_drift(dtype, axis, validate):
    a = np.array(
        [
            [[5, 1], [2, 7]],
            [[3, 4], [6, 0]],
            [[8, 9], [1, 2]],
        ],
        dtype=dtype,
    )
    expected = (
        np.array([[5.0, 4.0], [2.0, 2.0]])
        if axis == 0
        else np.array([[3.0, 4.5], [3.5, 3.0], [8.5, 1.5]])
    )
    for rd_fn in [rd.median, rd.nanmedian]:
        got = rd_fn(a, axis=axis, validate=validate)
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)
