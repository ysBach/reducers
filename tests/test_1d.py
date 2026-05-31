"""1-D parity tests for reducers vs NumPy (plain + nan-aware + inf semantics)."""

from __future__ import annotations

import numpy as np
import pytest
import reducers as rd

try:
    import bottleneck as bn
except ImportError:  # pragma: no cover - depends on optional test environment
    bn = None

SEED = 20250311


def test_runtime_getters_return_positive_integers():
    assert isinstance(rd.get_num_threads(), int)
    assert rd.get_num_threads() > 0
    grains = rd.get_parallel_grains()
    assert set(grains) == {
        "axis_scan_plain",
        "axis_scan_nan",
        "axis_scan_var",
        "axis_weighted",
        "axis_order_median",
        "axis_order_percentile",
        "minmax_1d",
    }
    assert all(isinstance(value, int) and value > 0 for value in grains.values())


def test_runtime_grain_setters_accept_positive_values():
    grains = rd.get_parallel_grains()
    assert (
        rd.set_parallel_grain("axis_scan_plain", grains["axis_scan_plain"])
        == grains["axis_scan_plain"]
    )
    assert rd.set_parallel_grains(grains) == grains
    assert (
        rd.set_axis_scan_grain(grains["axis_scan_plain"]) == grains["axis_scan_plain"]
    )
    assert (
        rd.set_axis_order_grain(grains["axis_order_median"])
        == grains["axis_order_median"]
    )
    assert rd.set_minmax_1d_grain(grains["minmax_1d"]) == grains["minmax_1d"]


def test_runtime_grain_setters_reject_zero():
    with pytest.raises(ValueError, match="positive integers"):
        rd.set_parallel_grain("axis_scan_plain", 0)
    with pytest.raises(ValueError, match="unknown parallel grain"):
        rd.set_parallel_grain("not_a_grain", 1024)


def _data(dtype):
    rng = np.random.default_rng(SEED)
    a = rng.normal(1000.0, 30.0, size=2000).astype(dtype)
    a[::97] = np.nan
    a[3] = np.inf
    a[7] = -np.inf
    return a


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    ("rd_fn", "np_fn", "kwargs"),
    [
        (rd.mean, np.mean, {}),
        (rd.sum, np.sum, {}),
        (rd.min, np.min, {}),
        (rd.max, np.max, {}),
        (rd.median, np.median, {}),
        (rd.var, np.var, {"ddof": 1}),
        (rd.std, np.std, {"ddof": 1}),
    ],
)
def test_plain_reducers_match_numpy_on_finite_data(dtype, rd_fn, np_fn, kwargs):
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=257).astype(dtype)
    np.testing.assert_allclose(
        rd_fn(a, **kwargs), np_fn(a, **kwargs), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    ("rd_fn", "np_fn"),
    [
        (rd.percentile, np.percentile),
        (rd.quantile, np.quantile),
    ],
)
def test_plain_rank_reducers_match_numpy(dtype, rd_fn, np_fn):
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=257).astype(dtype)
    q = [0.16, 0.5, 0.84] if rd_fn is rd.quantile else [16.0, 50.0, 84.0]
    np.testing.assert_allclose(rd_fn(a, q), np_fn(a, q), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_average_matches_numpy_for_float_dtypes(dtype):
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=257).astype(dtype)
    w = rng.uniform(0.5, 1.5, size=a.size).astype(dtype)
    np.testing.assert_allclose(
        rd.average(a, weights=w), np.average(a, weights=w), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(rd.average(a), np.average(a), rtol=1e-5, atol=1e-7)


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
def test_nan_reducers_match_numpy_including_inf(dtype, rd_fn, np_fn):
    a = _data(dtype)
    np.testing.assert_allclose(rd_fn(a), np_fn(a), rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    ("rd_fn", "np_fn", "kwargs"),
    [
        (rd.nanvar, np.nanvar, {"ddof": 1}),
        (rd.nanstd, np.nanstd, {"ddof": 1}),
    ],
)
def test_nan_spread_reducers_match_numpy_on_finite_or_nan(dtype, rd_fn, np_fn, kwargs):
    a = _data(dtype)
    a[~np.isfinite(a)] = np.nan
    np.testing.assert_allclose(
        rd_fn(a, **kwargs), np_fn(a, **kwargs), rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    ("rd_fn", "np_fn"),
    [
        (rd.nanpercentile, np.nanpercentile),
        (rd.nanquantile, np.nanquantile),
    ],
)
def test_nan_rank_reducers_match_numpy(dtype, rd_fn, np_fn):
    a = _data(dtype)
    q = [0.16, 0.5, 0.84] if rd_fn is rd.nanquantile else [16.0, 50.0, 84.0]
    np.testing.assert_allclose(rd_fn(a, q), np_fn(a, q), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nanaverage_matches_masked_numpy(dtype):
    a = _data(dtype)
    w = np.linspace(0.5, 1.5, a.size, dtype=dtype)
    mask = ~np.isnan(a)
    np.testing.assert_allclose(
        rd.nanaverage(a, weights=w),
        np.average(a[mask], weights=w[mask]),
        equal_nan=True,
    )
    finite = np.isfinite(a)
    np.testing.assert_allclose(
        rd.nanaverage(a, weights=w, ignore_inf=True),
        np.average(a[finite], weights=w[finite]),
        equal_nan=True,
    )


@pytest.mark.skipif(bn is None, reason="bottleneck is not installed")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    ("rd_fn", "bn_fn", "kwargs"),
    [
        (rd.nanmean, lambda x: bn.nanmean(x), {}),
        (rd.nanmedian, lambda x: bn.nanmedian(x), {}),
        (rd.nansum, lambda x: bn.nansum(x), {}),
        (rd.nanmin, lambda x: bn.nanmin(x), {}),
        (rd.nanmax, lambda x: bn.nanmax(x), {}),
        (rd.nanvar, lambda x: bn.nanvar(x, ddof=1), {"ddof": 1}),
        (rd.nanstd, lambda x: bn.nanstd(x, ddof=1), {"ddof": 1}),
    ],
)
def test_nan_reducers_match_bottleneck_where_available(dtype, rd_fn, bn_fn, kwargs):
    a = _data(dtype)
    if rd_fn in (rd.nanvar, rd.nanstd):
        a[~np.isfinite(a)] = np.nan
    np.testing.assert_allclose(
        rd_fn(a, **kwargs), bn_fn(a), rtol=1e-4, atol=1e-6, equal_nan=True
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nanvar_matches_numpy(dtype):
    a = _data(dtype)
    # inf in the data makes both NaN-aware var inf/nan; compare on a finite copy
    # for the numeric path and verify inf handling separately.
    finite = a.copy()
    finite[~np.isfinite(finite)] = np.nan
    np.testing.assert_allclose(
        rd.nanvar(finite, ddof=1), np.nanvar(finite, ddof=1), rtol=1e-4
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nanpercentile_matches_numpy(dtype):
    a = _data(dtype)
    got = rd.nanpercentile(a, [10.0, 50.0, 90.0])
    exp = np.nanpercentile(a, [10.0, 50.0, 90.0])
    np.testing.assert_allclose(got, exp, rtol=1e-5)


def test_inf_semantics():
    x = np.array([1.0, 2.0, np.inf], dtype=np.float64)
    # nanmean keeps inf (np parity)
    assert rd.nanmean(x) == np.inf == np.nanmean(x)
    # ignore_inf drops it
    assert rd.nanmean(x, ignore_inf=True) == 1.5
    assert rd.nanmedian(x) == np.nanmedian(x) == 2.0
    assert rd.nanmin(x, ignore_inf=True) == 1.0
    assert rd.nanmax(x, ignore_inf=True) == 2.0


def test_plain_reducers_propagate_nan():
    y = np.array([1.0, np.nan, 3.0])
    for rd_fn, np_fn in [
        (rd.mean, np.mean),
        (rd.sum, np.sum),
        (rd.median, np.median),
        (rd.min, np.min),
        (rd.max, np.max),
        (rd.var, np.var),
    ]:
        np.testing.assert_allclose(rd_fn(y), np_fn(y), equal_nan=True)


def test_plain_order_stats_nan_positions():
    # NaN not in the first slot must still propagate (regression guard).
    for arr in ([1.0, np.nan, 2.0], [np.nan, 1.0, 2.0], [2.0, 1.0, np.nan]):
        a = np.array(arr)
        assert np.isnan(rd.min(a)) and np.isnan(np.min(a))
        assert np.isnan(rd.max(a)) and np.isnan(np.max(a))
        assert np.isnan(rd.median(a)) and np.isnan(np.median(a))


def test_plain_min_max_inf():
    assert rd.min(np.array([1.0, np.inf])) == 1.0
    assert rd.max(np.array([1.0, -np.inf])) == 1.0
    assert rd.min(np.array([-np.inf, 1.0])) == -np.inf


def test_int_promotion_matches_numpy():
    i = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    np.testing.assert_allclose(rd.mean(i), np.mean(i))
    np.testing.assert_allclose(rd.var(i, ddof=1), np.var(i, ddof=1))
    np.testing.assert_allclose(rd.median(i), np.median(i))


def test_average_matches_numpy():
    a = np.array([1.0, 2.0, 5.0, 9.0])
    w = np.array([1.0, 0.5, 2.0, 3.0])
    np.testing.assert_allclose(rd.average(a, weights=w), np.average(a, weights=w))
    np.testing.assert_allclose(rd.average(a), np.average(a))


def test_average_integer_data_and_weights():
    a = np.array([1, 2, 5, 9], dtype=np.uint16)
    w = np.array([1, 2, 3, 4], dtype=np.int16)
    np.testing.assert_allclose(rd.average(a, weights=w), np.average(a, weights=w))
    np.testing.assert_allclose(
        rd.average(a, weights=w, validate=False), np.average(a, weights=w)
    )
    assert isinstance(rd.average(a, weights=w), float)


def test_nanaverage_semantics():
    a = np.array([1.0, np.nan, 3.0, np.inf])
    w = np.array([1.0, 100.0, 3.0, 2.0])
    assert rd.nanaverage(a, weights=w) == np.inf
    np.testing.assert_allclose(
        rd.nanaverage(a, weights=w, ignore_inf=True),
        np.average(np.array([1.0, 3.0]), weights=np.array([1.0, 3.0])),
    )
    assert np.isnan(rd.nanaverage(np.array([np.nan]), weights=np.array([1.0])))


def test_average_weight_nan_and_zero_weight():
    assert np.isnan(rd.average(np.array([1.0]), weights=np.array([np.nan])))
    with pytest.raises(ZeroDivisionError, match="weights sum to zero"):
        rd.average(np.array([1.0]), weights=np.array([0.0]))
    with pytest.raises(ZeroDivisionError, match="weights sum to zero"):
        rd.average(
            np.array([], dtype=np.float64), weights=np.array([], dtype=np.float64)
        )


def test_average_weight_shape_and_dtype_errors():
    with pytest.raises(ValueError, match="same shape"):
        rd.average(np.ones((2, 3)), weights=np.ones(3))
    with pytest.raises(TypeError, match="real numeric dtypes"):
        rd.average(np.ones(3), weights=np.array([1 + 1j, 2 + 0j, 3 + 0j]))


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
def test_integer_validate_false_reducers_match_validated_path(dtype):
    a = np.array([1, 5, 2, 8, 3, 13, 21], dtype=dtype)
    for rd_fn in [
        rd.mean,
        rd.nanmean,
        rd.sum,
        rd.nansum,
        rd.min,
        rd.nanmin,
        rd.max,
        rd.nanmax,
        rd.median,
        rd.nanmedian,
        rd.lmedian,
        rd.var,
        rd.nanvar,
        rd.std,
        rd.nanstd,
    ]:
        np.testing.assert_allclose(
            rd_fn(a, validate=False), rd_fn(a, validate=True), rtol=1e-12
        )

    np.testing.assert_allclose(
        rd.percentile(a, [25.0, 50.0, 75.0], validate=False),
        rd.percentile(a, [25.0, 50.0, 75.0], validate=True),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        rd.nanpercentile(a, [25.0, 50.0, 75.0], validate=False),
        rd.nanpercentile(a, [25.0, 50.0, 75.0], validate=True),
        rtol=1e-12,
    )
    assert rd.count_finite(a, validate=False) == a.size


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
def test_integer_exact_select_reducers_preserve_scalar_type(dtype, validate):
    a = np.array([1, 5, 2, 8, 3, 13, 21], dtype=dtype)
    for rd_fn in [rd.min, rd.nanmin, rd.max, rd.nanmax, rd.lmedian]:
        got = rd_fn(a, validate=validate)
        assert got == np.asarray(rd_fn(a, validate=False)).item()
        if dtype is np.bool_:
            assert isinstance(got, bool)
        else:
            assert isinstance(got, int)


@pytest.mark.parametrize(
    "a",
    [
        np.array([1 + 2j, 3 + 4j]),
        np.array([1, 2, 3], dtype=object),
    ],
)
def test_non_real_dtypes_raise(a):
    with pytest.raises(TypeError, match="real numeric dtypes"):
        rd.mean(a)
    with pytest.raises(TypeError, match="supported real dtype"):
        rd.mean(a, validate=False)


def test_quantile_and_scalar_q():
    a = np.arange(11.0)
    assert rd.quantile(a, 0.25) == np.quantile(a, 0.25)
    np.testing.assert_allclose(
        rd.quantile(a, [0.25, 0.75]), np.quantile(a, [0.25, 0.75])
    )


def test_return_mean():
    a = _data(np.float64)
    finite = a.copy()
    finite[~np.isfinite(finite)] = np.nan
    clean = np.arange(11.0)
    v, m = rd.var(clean, ddof=1, return_mean=True)
    np.testing.assert_allclose(v, np.var(clean, ddof=1))
    np.testing.assert_allclose(m, np.mean(clean))
    s, m = rd.std(clean, ddof=1, return_mean=True)
    np.testing.assert_allclose(s, np.std(clean, ddof=1))
    np.testing.assert_allclose(m, np.mean(clean))
    v, m = rd.nanvar(finite, ddof=1, return_mean=True)
    np.testing.assert_allclose(v, np.nanvar(finite, ddof=1), rtol=1e-4)
    np.testing.assert_allclose(m, np.nanmean(finite), rtol=1e-6)
    s, m = rd.nanstd(finite, ddof=1, return_mean=True)
    np.testing.assert_allclose(s, np.nanstd(finite, ddof=1), rtol=1e-4)
    np.testing.assert_allclose(m, np.nanmean(finite), rtol=1e-6)


def test_count_finite():
    a = np.array([1.0, np.nan, np.inf, 2.0, -np.inf])
    assert rd.count_finite(a) == 2


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e8])
def test_variance_stable_with_large_offset(dtype, offset):
    # One-pass (sumsq - sum*mean) would catastrophically cancel here; two-pass
    # must stay accurate. reducers accumulates in f64, so the correct reference
    # is the true (f64) variance of the actual (possibly float32-quantized)
    # input values -- not np.var's lower-precision float32 path.
    base = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=dtype)
    a = (base + offset).astype(dtype)
    ref = np.var(a.astype(np.float64))
    np.testing.assert_allclose(rd.var(a), ref, rtol=1e-4)
    ref1 = np.var(a.astype(np.float64), ddof=1)
    np.testing.assert_allclose(rd.nanvar(a, ddof=1), ref1, rtol=1e-4)


def test_variance_extreme_offset_f64():
    a = np.array([1e16, 1e16 + 2, 1e16 + 4], dtype=np.float64)
    np.testing.assert_allclose(rd.var(a), np.var(a), rtol=1e-9)


def test_empty_and_all_skipped_edge_cases():
    empty = np.array([], dtype=np.float64)
    assert np.isnan(rd.mean(empty))
    assert rd.sum(empty) == 0.0  # numpy sum of empty is 0.0
    assert np.isnan(rd.median(empty))

    all_nan = np.array([np.nan, np.nan], dtype=np.float64)
    assert np.isnan(rd.nanmean(all_nan))
    assert rd.nansum(all_nan) == 0.0  # np.nansum convention
    assert np.isnan(rd.nanmedian(all_nan))
    assert np.isnan(rd.nanmin(all_nan))
    assert rd.count_finite(all_nan) == 0

    all_inf = np.array([np.inf, np.inf], dtype=np.float64)
    assert np.isnan(rd.nanmin(all_inf, ignore_inf=True))  # nothing finite
    assert rd.nanmin(all_inf) == np.inf  # keep inf -> min is inf


def test_minmax_one_pass():
    a = np.array([3.0, 1.0, np.nan, 5.0, 2.0])
    lo, hi = rd.minmax(a)
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = rd.nanminmax(a)
    assert (lo, hi) == (1.0, 5.0)
    lo, hi = rd.nanminmax(np.array([1.0, np.inf, -2.0]), ignore_inf=True)
    assert (lo, hi) == (-2.0, 1.0)


def test_integer_minmax_matches_numpy_endpoints():
    a = np.array([7, -3, 11, 0, -8, 5, 11, -2, 4], dtype=np.int64)
    assert rd.minmax(a) == (float(np.min(a)), float(np.max(a)))
    assert rd.nanminmax(a) == (float(np.min(a)), float(np.max(a)))
