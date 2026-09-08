"""Public Python sigma-clipping contract tests.

The cases use the reducers API directly. The small reference implementation
covers finite f64 mask semantics, while hand-derived cases cover dtype
narrowing, rollback, diagnostics, and public shape adapters.
"""

from __future__ import annotations

import numpy as np
import pytest
import reducers as rd


def _center(values: np.ndarray, name: str) -> float:
    """Return the f64 reference center for one supported center name."""
    name = name.lower()
    if name in {"mean", "average", "avg"}:
        return float(np.mean(values, dtype=np.float64))
    if name in {"lmedian", "lmed", "lower_median", "lower-median", "lower median"}:
        ordered = np.sort(values)
        return float(ordered[(ordered.size - 1) // 2])
    if name in {"median", "med"}:
        return float(np.median(values))
    raise AssertionError(f"unhandled reference center {name!r}")


def _reference_mask(
    values: np.ndarray,
    input_mask: np.ndarray | None = None,
    *,
    sigma: float | tuple[float, float] = (3.0, 3.0),
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 0,
    maxrej: int | None = None,
    cenfunc: str = "median",
    clip_cen: str | None = None,
    stdfunc: str = "std",
    revert_on_nkeep: bool = False,
) -> np.ndarray:
    """Reference only the cumulative rejection-mask algorithm in f64."""
    values = np.asarray(values, dtype=np.float64)
    lower, upper = (
        (float(sigma), float(sigma))
        if np.isscalar(sigma)
        else (float(sigma[0]), float(sigma[1]))
    )
    rejected = ~np.isfinite(values)
    if input_mask is not None:
        rejected |= np.asarray(input_mask, dtype=bool)
    spread_name = (
        cenfunc if clip_cen is None or clip_cen in {"center", "cenfunc"} else clip_cen
    )

    for _ in range(maxiters):
        active = ~rejected
        sample = values[active]
        if sample.size == 0:
            break
        center = _center(sample, cenfunc)
        spread_center = _center(sample, spread_name)
        if stdfunc.lower() == "std":
            if sample.size <= ddof:
                break
            spread_sq = float(
                np.sum((sample - spread_center) ** 2) / (sample.size - ddof)
            )
            spread = np.sqrt(spread_sq)
        elif stdfunc.lower() == "mad":
            spread = 1.4826 * float(np.median(np.abs(sample - spread_center)))
        else:
            raise AssertionError(f"unhandled reference spread {stdfunc!r}")
        if not np.isfinite(center) or not np.isfinite(spread):
            break

        candidate = (~rejected) & (
            (values < center - lower * spread) | (values > center + upper * spread)
        )
        if not np.any(candidate):
            break
        proposed = rejected | candidate
        kept = int((~proposed).sum())
        min_violation = revert_on_nkeep and kept < nkeep
        max_violation = maxrej is not None and values.size - kept > maxrej
        if min_violation or max_violation:
            break
        rejected = proposed
    return rejected


def _assert_scalar_equal(actual: object, expected: float) -> None:
    """Compare scalar-like values while treating NaNs as equal."""
    actual = np.asarray(actual)
    if np.isnan(expected):
        assert bool(np.isnan(actual))
    else:
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def _assert_result_shapes(result: tuple[object, ...], shape: tuple[int, ...]) -> None:
    """Check the six-tuple shape contract for an axis-0 stack."""
    mask, std, low, upp, nit, flags = result
    assert np.asarray(mask).shape == shape
    for diagnostic in (std, low, upp, nit, flags):
        assert np.asarray(diagnostic).shape == shape[1:]


def test_public_sigma_clip_surface_and_generic_defaults() -> None:
    names = (
        "sigclip",
        "sigclip_1d",
        "sigclip_mask",
        "sigclip_mask_1d",
        "sigclip_combine",
        "sigclip_combine_1d",
        "sigclip_restored_flags",
        "grow_mask",
    )
    for name in names:
        assert callable(getattr(rd, name))

    params = rd.SigClip()
    assert params.maxiters == 5
    assert params.ddof == 0
    assert params.cenfunc == "median"
    assert params.clip_cen is None
    assert params.stdfunc == "std"
    assert params.nkeep == 0
    assert params.revert_on_nkeep is False
    assert params.maxrej is None


def test_generic_default_differs_from_explicit_rollback() -> None:
    values = np.array([0.0, 2.0])
    kwargs = {"sigma": 0.0, "maxiters": 1}

    generic = rd.sigclip_1d(values, **kwargs)
    rollback = rd.sigclip_1d(
        values,
        **kwargs,
        nkeep=1,
        revert_on_nkeep=True,
    )

    np.testing.assert_array_equal(generic[0], [True, True])
    np.testing.assert_array_equal(rollback[0], [False, False])
    _assert_scalar_equal(
        rd.sigclip_combine_1d(values, combine="mean", **kwargs), np.nan
    )
    _assert_scalar_equal(
        rd.sigclip_combine_1d(
            values,
            combine="mean",
            **kwargs,
            nkeep=1,
            revert_on_nkeep=True,
        ),
        1.0,
    )
    assert int(generic[5]) & 2
    assert int(rollback[5]) & 4


def test_hand_derived_rollback_restores_only_last_iteration() -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0, 100.0])
    kwargs = {
        "sigma": 1.5,
        "maxiters": 5,
        "nkeep": 4,
        "revert_on_nkeep": True,
        "cenfunc": "median",
        "clip_cen": "median",
    }

    mask, std, low, upp, nit, flags = rd.sigclip_1d(values, **kwargs)
    np.testing.assert_array_equal(mask, [False, False, False, False, True])
    _assert_scalar_equal(std, 5.0)
    _assert_scalar_equal(low, 0.0)
    _assert_scalar_equal(upp, 10.0)
    assert int(nit) == 2
    assert int(flags) & 4

    restored = rd.sigclip_restored_flags(values.reshape(-1, 1), **kwargs)
    np.testing.assert_array_equal(restored.reshape(-1), [0, 0, 0, 64, 0])


def test_maxrej_restores_last_iteration_and_sets_rejection_limit_flag() -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0, 100.0])
    kwargs = {
        "sigma": 1.5,
        "maxiters": 5,
        "maxrej": 1,
        "cenfunc": "median",
        "clip_cen": "median",
    }

    result = rd.sigclip_1d(values, **kwargs)
    np.testing.assert_array_equal(result[0], [False, False, False, False, True])
    assert int(result[5]) & 8
    restored = rd.sigclip_restored_flags(values.reshape(-1, 1), **kwargs)
    np.testing.assert_array_equal(restored.reshape(-1), [0, 0, 0, 128, 0])


def test_f64_mask_and_fused_outputs_match_independent_reference() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0, -100.0])
    input_mask = np.array([False, False, True, False, False, False, False])
    kwargs = {
        "sigma": (2.5, 3.0),
        "maxiters": 4,
        "ddof": 1,
        "nkeep": 0,
        "revert_on_nkeep": False,
        "cenfunc": "median",
        "clip_cen": "median",
    }
    expected = _reference_mask(values, input_mask, **kwargs)
    result = rd.sigclip_1d(values, mask=input_mask, **kwargs)

    np.testing.assert_array_equal(result[0], expected)
    np.testing.assert_array_equal(
        rd.sigclip_mask_1d(values, mask=input_mask, **kwargs), expected
    )
    for combine, reducer in (("mean", np.mean), ("median", np.median)):
        got = rd.sigclip_combine_1d(values, mask=input_mask, combine=combine, **kwargs)
        expected_value = reducer(values[~expected]) if np.any(~expected) else np.nan
        np.testing.assert_allclose(
            got, expected_value, rtol=0.0, atol=0.0, equal_nan=True
        )


def test_mad_ignores_ddof_and_uses_independent_dispersion_center() -> None:
    values = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 100.0], dtype=np.float32)
    kwargs = {
        "sigma": (10.0, 3.0),
        "maxiters": 1,
        "cenfunc": "median",
        "clip_cen": "median",
        "stdfunc": "mad",
        "nkeep": 0,
        "revert_on_nkeep": False,
    }
    low_ddof = rd.sigclip_1d(values, ddof=0, **kwargs)
    high_ddof = rd.sigclip_1d(values, ddof=99, **kwargs)
    np.testing.assert_array_equal(
        low_ddof[0], [False, False, False, False, False, True]
    )
    np.testing.assert_array_equal(high_ddof[0], low_ddof[0])
    np.testing.assert_allclose(
        low_ddof[1], 1.4826 * np.median(np.abs(values - 10.0)), rtol=1e-6
    )
    np.testing.assert_allclose(high_ddof[1], low_ddof[1], rtol=1e-6)
    np.testing.assert_allclose(
        rd.sigclip_combine_1d(values, combine="mean", ddof=99, **kwargs), 10.0
    )


def test_mean_clip_center_none_is_distinct_from_dtype_rounded_center() -> None:
    values = np.array([2**24, 2**24 + 2], dtype=np.float32)
    implicit_mean = rd.sigclip_1d(
        values,
        sigma=1.5,
        maxiters=1,
        cenfunc="mean",
        clip_cen=None,
        nkeep=0,
        revert_on_nkeep=False,
    )
    rounded_center = rd.sigclip_1d(
        values,
        sigma=1.5,
        maxiters=1,
        cenfunc="mean",
        clip_cen="center",
        nkeep=0,
        revert_on_nkeep=False,
    )
    np.testing.assert_array_equal(implicit_mean[0], [False, True])
    np.testing.assert_array_equal(rounded_center[0], [False, False])
    np.testing.assert_allclose(implicit_mean[1], 1.0)
    np.testing.assert_allclose(rounded_center[1], np.sqrt(2.0), rtol=1e-6)


def test_stack_vector_mask_and_fused_results_match_per_pixel_calls() -> None:
    stack = np.array(
        [
            [[0.0, 1.0], [np.nan, 0.0]],
            [[0.0, 1.0], [2.0, 2.0]],
            [[0.0, 1.0], [3.0, 4.0]],
            [[10.0, 1.0], [4.0, 6.0]],
            [[100.0, 1.0], [5.0, 8.0]],
        ],
        dtype=np.float64,
    )
    input_mask = np.zeros(stack.shape, dtype=bool)
    input_mask[1, 1, 1] = True
    kwargs = {
        "sigma": 1.5,
        "maxiters": 5,
        "nkeep": 4,
        "revert_on_nkeep": True,
        "cenfunc": "median",
        "clip_cen": "median",
    }

    stack_result = rd.sigclip(stack, mask=input_mask, **kwargs)
    _assert_result_shapes(stack_result, stack.shape)
    np.testing.assert_array_equal(
        rd.sigclip_mask(stack, mask=input_mask, **kwargs), stack_result[0]
    )
    np.testing.assert_array_equal(
        rd.sigclip_restored_flags(stack, mask=input_mask, **kwargs),
        np.stack(
            [
                rd.sigclip_restored_flags(
                    stack[:, row, col, None],
                    mask=input_mask[:, row, col, None],
                    **kwargs,
                )[:, 0]
                for row in range(stack.shape[1])
                for col in range(stack.shape[2])
            ],
            axis=1,
        ).reshape(stack.shape),
    )

    for row in range(stack.shape[1]):
        for col in range(stack.shape[2]):
            vector_result = rd.sigclip_1d(
                stack[:, row, col], mask=input_mask[:, row, col], **kwargs
            )
            np.testing.assert_array_equal(
                stack_result[0][:, row, col], vector_result[0]
            )
            for stack_value, vector_value in zip(
                stack_result[1:], vector_result[1:], strict=True
            ):
                _assert_scalar_equal(stack_value[row, col], float(vector_value))

    for combine in ("mean", "median"):
        fused = rd.sigclip_combine(stack, mask=input_mask, combine=combine, **kwargs)
        expected = np.empty(stack.shape[1:], dtype=np.float64)
        for row in range(stack.shape[1]):
            for col in range(stack.shape[2]):
                expected[row, col] = rd.sigclip_combine_1d(
                    stack[:, row, col],
                    mask=input_mask[:, row, col],
                    combine=combine,
                    **kwargs,
                )
        np.testing.assert_allclose(fused, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("with_input_mask", [False, True])
def test_stack_chunked_dispatch_matches_serial_for_all_output_modes(
    dtype: np.dtype, with_input_mask: bool
) -> None:
    if rd.get_num_threads() < 2:
        pytest.skip("chunked dispatch needs at least two Rayon threads")
    rng = np.random.default_rng(20260908)
    values = rng.normal(100.0, 2.0, size=(17, 19, 13)).astype(dtype)
    values[3, 2, 4] = dtype(150.0)
    values[11, 17, 12] = dtype(54.0)
    values[0, 3, 6] = dtype(np.nan)
    values[8, 5, 1] = dtype(np.inf)
    input_mask = np.zeros(values.shape, dtype=bool)
    input_mask[1, 0, 0] = True
    input_mask[14, 18, 12] = True
    kwargs: dict[str, object] = {
        "sigma": (2.2, 2.8),
        "maxiters": 4,
        "nkeep": 2,
        "revert_on_nkeep": True,
        "cenfunc": "median",
        "clip_cen": None,
        "stdfunc": "std",
    }
    if with_input_mask:
        kwargs["mask"] = input_mask

    def output_modes() -> tuple[object, ...]:
        return (
            rd.sigclip(values, **kwargs),
            rd.sigclip_mask(values, **kwargs),
            rd.sigclip_restored_flags(values, **kwargs),
            rd.sigclip_combine(values, combine="mean", **kwargs),
            rd.sigclip_combine(values, combine="median", **kwargs),
        )

    work = values.shape[0] * values.shape[1] * values.shape[2]
    original = rd.get_parallel_grains()["axis_order_median"]
    try:
        rd.set_axis_order_grain(work + 1)
        serial = output_modes()
        # Grain 3 is smaller than the sample axis.  Grain 29 creates blocks
        # spanning multiple samples with an intentionally uneven final block.
        for grain in (1, 3, 29):
            rd.set_axis_order_grain(grain)
            chunked = output_modes()
            for serial_output, chunked_output in zip(serial, chunked, strict=True):
                if isinstance(serial_output, tuple):
                    for serial_item, chunked_item in zip(
                        serial_output, chunked_output, strict=True
                    ):
                        np.testing.assert_array_equal(chunked_item, serial_item)
                else:
                    np.testing.assert_array_equal(chunked_output, serial_output)
    finally:
        rd.set_axis_order_grain(original)


def test_arbitrary_trailing_shape_is_preserved() -> None:
    values = np.arange(5 * 2 * 3 * 2, dtype=np.float32).reshape(5, 2, 3, 2)
    values[-1, 0, 1, 1] = 1000.0
    kwargs = {"sigma": 1.0, "maxiters": 2, "nkeep": 0, "revert_on_nkeep": False}

    result_nd = rd.sigclip(values, **kwargs)
    result_3d = rd.sigclip(values.reshape(5, 6, 2), **kwargs)
    _assert_result_shapes(result_nd, values.shape)
    np.testing.assert_array_equal(result_nd[0], result_3d[0].reshape(values.shape))
    for nd_value, three_d_value in zip(result_nd[1:], result_3d[1:], strict=True):
        np.testing.assert_array_equal(
            nd_value, np.asarray(three_d_value).reshape(values.shape[1:])
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stack_mask_normalization_preserves_all_output_modes(
    dtype: np.dtype,
) -> None:
    values = np.arange(5 * 2 * 3 * 2, dtype=dtype).reshape(5, 2, 3, 2)
    original_mask = np.zeros(values.shape, dtype=bool)
    original_mask[1, 0, 2, 1] = True
    original_mask[4, 1, 0, 0] = True
    normalized_mask = original_mask.reshape(5, 12, 1)
    noncontiguous_mask = np.asfortranarray(original_mask)
    assert not noncontiguous_mask.flags.c_contiguous

    def output_modes(array: np.ndarray, mask: np.ndarray) -> tuple[object, ...]:
        kwargs = {"mask": mask, "sigma": 100.0, "maxiters": 1}
        return (
            rd.sigclip(array, **kwargs),
            rd.sigclip_mask(array, **kwargs),
            rd.sigclip_restored_flags(array, **kwargs),
            rd.sigclip_combine(array, combine="mean", **kwargs),
            rd.sigclip_combine(array, combine="median", **kwargs),
        )

    original_outputs = output_modes(values, original_mask)
    for alternate_mask in (normalized_mask, noncontiguous_mask):
        for expected, actual in zip(
            original_outputs, output_modes(values, alternate_mask), strict=True
        ):
            if isinstance(expected, tuple):
                for expected_item, actual_item in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(actual_item, expected_item)
            else:
                np.testing.assert_array_equal(actual, expected)

    empty = np.empty((0, 2, 3, 2), dtype=dtype)
    empty_original_mask = np.zeros(empty.shape, dtype=bool)
    empty_normalized_mask = empty_original_mask.reshape(0, 12, 1)
    original_empty_outputs = output_modes(empty, empty_original_mask)
    normalized_empty_outputs = output_modes(empty, empty_normalized_mask)
    for expected, actual in zip(
        original_empty_outputs, normalized_empty_outputs, strict=True
    ):
        if isinstance(expected, tuple):
            for expected_item, actual_item in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(actual_item, expected_item)
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "expected_float"),
    [
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_supported_dtypes_follow_documented_promotion(
    dtype: np.dtype, expected_float: np.dtype
) -> None:
    values = np.array([0, 0, 0, 10, 100], dtype=dtype).reshape(5, 1, 1)
    result = rd.sigclip(
        values,
        sigma=1.5,
        maxiters=5,
        nkeep=4,
        revert_on_nkeep=True,
        cenfunc="median",
        clip_cen="median",
    )
    assert result[1].dtype == np.dtype(expected_float)
    assert result[2].dtype == np.dtype(expected_float)
    assert result[3].dtype == np.dtype(expected_float)
    assert result[4].dtype == np.uint8
    assert result[5].dtype == np.uint8
    np.testing.assert_array_equal(
        result[0][:, 0, 0], [False, False, False, False, True]
    )


def test_int32_fused_result_keeps_values_above_float32_precision() -> None:
    values = (np.arange(5, dtype=np.int32) + 2**24 + 1).reshape(5, 1, 1)
    got = rd.sigclip_combine(
        values,
        combine="mean",
        sigma=100.0,
        maxiters=1,
        nkeep=0,
        revert_on_nkeep=False,
    )
    expected = values.astype(np.float64).mean(axis=0)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, expected)


def test_nonfinite_and_input_masked_samples_are_always_rejected() -> None:
    values = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    result = rd.sigclip_1d(values, sigma=100.0, maxiters=1)
    np.testing.assert_array_equal(result[0], [False, True, True, True, False])
    assert not (int(result[5]) & 1)

    input_mask = np.array([False, False, True, False, False])
    masked = rd.sigclip_1d(values, mask=input_mask, sigma=100.0, maxiters=1)
    np.testing.assert_array_equal(masked[0], [False, True, True, True, False])
    assert int(masked[5]) & 1


def test_zero_spread_and_exact_threshold_equality() -> None:
    equal = np.array([0.0, 1.0, 2.0])
    equal_result = rd.sigclip_1d(
        equal,
        sigma=1.0,
        maxiters=1,
        ddof=1,
        cenfunc="mean",
        clip_cen="mean",
    )
    np.testing.assert_array_equal(equal_result[0], [False, False, False])

    zero_spread = rd.sigclip_1d(np.ones(4), sigma=0.0, maxiters=3)
    np.testing.assert_array_equal(zero_spread[0], False)
    np.testing.assert_allclose(zero_spread[1], 0.0)


def test_ddof_can_invalidate_std_but_is_ignored_by_mad() -> None:
    values = np.array([0.0, 1.0, 3.0])
    std0 = rd.sigclip_1d(
        values,
        sigma=100.0,
        maxiters=1,
        ddof=0,
        cenfunc="mean",
        clip_cen="mean",
    )
    std1 = rd.sigclip_1d(
        values,
        sigma=100.0,
        maxiters=1,
        ddof=1,
        cenfunc="mean",
        clip_cen="mean",
    )
    invalid = rd.sigclip_1d(
        values,
        sigma=100.0,
        maxiters=1,
        ddof=3,
        cenfunc="mean",
        clip_cen="mean",
    )
    np.testing.assert_allclose(std0[1], np.sqrt(14.0 / 9.0))
    np.testing.assert_allclose(std1[1], np.sqrt(7.0 / 3.0))
    assert np.isnan(invalid[1])
    np.testing.assert_array_equal(invalid[0], False)

    mad0 = rd.sigclip_1d(
        values,
        sigma=100.0,
        maxiters=1,
        ddof=0,
        cenfunc="median",
        clip_cen="median",
        stdfunc="mad",
    )
    mad3 = rd.sigclip_1d(
        values,
        sigma=100.0,
        maxiters=1,
        ddof=3,
        cenfunc="median",
        clip_cen="median",
        stdfunc="mad",
    )
    np.testing.assert_allclose(mad0[1], mad3[1])


def test_empty_and_all_masked_inputs_have_stable_shapes() -> None:
    empty = np.array([], dtype=np.float64)
    result = rd.sigclip_1d(empty)
    assert result[0].shape == (0,)
    for scalar in result[1:4]:
        assert np.isnan(scalar)
    assert np.isnan(rd.sigclip_combine_1d(empty, combine="mean"))
    np.testing.assert_array_equal(rd.sigclip_mask_1d(empty), [])

    empty_stack = np.empty((0, 2, 3), dtype=np.float64)
    stack_result = rd.sigclip(empty_stack)
    _assert_result_shapes(stack_result, empty_stack.shape)
    assert np.isnan(stack_result[1]).all()
    assert np.isnan(rd.sigclip_combine(empty_stack, combine="median")).all()

    values = np.arange(6, dtype=np.float64).reshape(3, 1, 2)
    input_mask = np.ones(values.shape, dtype=bool)
    all_masked = rd.sigclip(values, mask=input_mask)
    np.testing.assert_array_equal(all_masked[0], input_mask)
    assert np.isnan(all_masked[1]).all()
    assert np.isnan(rd.sigclip_combine(values, mask=input_mask, combine="mean")).all()


def test_inputs_and_masks_are_not_mutated() -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0, 100.0]).reshape(5, 1, 1)
    input_mask = np.zeros(values.shape, dtype=bool)
    values_before = values.copy()
    mask_before = input_mask.copy()
    rd.sigclip(values, mask=input_mask, sigma=1.5, maxiters=2)
    rd.sigclip_mask(values, mask=input_mask, sigma=1.5, maxiters=2)
    rd.sigclip_combine(values, mask=input_mask, combine="mean", sigma=1.5, maxiters=2)
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(input_mask, mask_before)


def test_grow_mask_does_not_cross_stack_axis_and_grow_updates_flags() -> None:
    input_mask = np.zeros((2, 3, 3), dtype=bool)
    input_mask[0, 1, 1] = True
    input_mask[1, 0, 0] = True
    grown = rd.grow_mask(input_mask, 1.0)
    expected = np.zeros_like(input_mask)
    expected[0, 0:3, 1] = True
    expected[0, 1, 0:3] = True
    expected[1, 0:2, 0] = True
    expected[1, 0, 0:2] = True
    np.testing.assert_array_equal(grown, expected)

    values = np.zeros((5, 3, 3), dtype=np.float64)
    values[-1, 1, 1] = 100.0
    result = rd.sigclip(
        values,
        sigma=1.0,
        maxiters=1,
        nkeep=0,
        revert_on_nkeep=False,
        grow=1.0,
    )
    assert result[0][-1, 1, 1]
    assert result[0][-1, 0, 1]
    assert result[0][-1, 1, 0]
    assert np.any((result[5] & 16) != 0)
    np.testing.assert_array_equal(
        rd.sigclip_mask(
            values,
            sigma=1.0,
            maxiters=1,
            nkeep=0,
            revert_on_nkeep=False,
            grow=1.0,
        ),
        result[0],
    )


def test_validation_rejects_bad_shapes_dtypes_parameters_and_combine() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        rd.sigclip(np.ones(4))
    with pytest.raises(ValueError, match="1-D"):
        rd.sigclip_1d(np.ones((2, 2)))
    with pytest.raises(ValueError, match="mask shape"):
        rd.sigclip(np.ones((3, 2, 2)), mask=np.zeros((3, 2), dtype=bool))
    with pytest.raises(TypeError, match="supports"):
        rd.sigclip_1d(np.ones(4, dtype=np.float16))
    with pytest.raises(TypeError, match="supports"):
        rd.sigclip(np.ones((3, 2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="sigma"):
        rd.sigclip_1d(np.ones(4), sigma=(-1.0, 2.0))
    with pytest.raises(ValueError, match="sigma"):
        rd.sigclip_1d(np.ones(4), sigma=(np.nan, 2.0))
    for name in ("maxiters", "ddof", "nkeep", "maxrej"):
        with pytest.raises((OverflowError, ValueError)):
            rd.sigclip_1d(np.ones(4), **{name: -1})
    with pytest.raises(ValueError, match="unknown cenfunc"):
        rd.sigclip_1d(np.ones(4), cenfunc="trimmed")
    with pytest.raises(ValueError, match="unknown clip_cen"):
        rd.sigclip_1d(np.ones(4), clip_cen="trimmed")
    with pytest.raises(ValueError, match="unknown stdfunc"):
        rd.sigclip_1d(np.ones(4), stdfunc="biweight")
    with pytest.raises(NotImplementedError, match="mean and median"):
        rd.sigclip_combine_1d(np.ones(4), combine="sum")
    with pytest.raises(ValueError, match="grow"):
        rd.grow_mask(np.zeros((2, 2), dtype=bool), -1.0)


def test_validate_false_keeps_the_low_level_contiguous_contract() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    checked = rd.sigclip(values, sigma=100.0, maxiters=1)
    unchecked = rd.sigclip(values, sigma=100.0, maxiters=1, validate=False)
    for left, right in zip(checked, unchecked, strict=True):
        np.testing.assert_array_equal(left, right)

    with pytest.raises((TypeError, ValueError)):
        rd.sigclip(values.astype(np.float64)[:, :, ::-1], validate=False)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sigclip_class_apply_matches_function(dtype: np.dtype) -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0, 100.0], dtype=dtype)[:, None]
    params = rd.SigClip(
        sigma=1.5,
        maxiters=5,
        nkeep=4,
        revert_on_nkeep=True,
        cenfunc="median",
        clip_cen="median",
    )
    expected = rd.sigclip(
        values,
        sigma=1.5,
        maxiters=5,
        nkeep=4,
        revert_on_nkeep=True,
        cenfunc="median",
        clip_cen="median",
    )
    actual = params.apply(values)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


def test_scalar_sigma_and_center_aliases_match_explicit_pairs() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    scalar = rd.sigclip_1d(values, sigma=2.0, cenfunc="lmed", clip_cen="lmed")
    pair = rd.sigclip_1d(
        values,
        sigma=(2.0, 2.0),
        cenfunc="lower_median",
        clip_cen="lower-median",
    )
    for left, right in zip(scalar, pair, strict=True):
        np.testing.assert_array_equal(left, right)


def test_result_masks_are_boolean_and_diagnostics_have_uint8_types() -> None:
    result = rd.sigclip(np.arange(12, dtype=np.float64).reshape(3, 2, 2))
    assert result[0].dtype == np.bool_
    assert result[4].dtype == np.uint8
    assert result[5].dtype == np.uint8


def test_independent_reference_covers_asymmetric_lower_median_case() -> None:
    values = np.array([-5.0, -1.0, 0.0, 1.0, 5.0, 100.0])
    expected = _reference_mask(
        values,
        sigma=(0.5, 10.0),
        maxiters=2,
        cenfunc="lmedian",
        clip_cen="lmedian",
    )
    got = rd.sigclip_mask_1d(
        values,
        sigma=(0.5, 10.0),
        maxiters=2,
        cenfunc="lmedian",
        clip_cen="lmedian",
    )
    np.testing.assert_array_equal(got, expected)
