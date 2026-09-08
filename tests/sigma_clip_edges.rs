//! Independent cases with exact masks derived from small samples.
use reducers::reducers_1d::var_mean_valid;
use reducers::sigma_clip::{
    clip, CenFunc, ClipCenter, OutputOptions, Params, Reduction, StdFunc, StopReason, Workspace,
};

fn params() -> Params {
    Params {
        sigma_lower: 1.5,
        sigma_upper: 1.5,
        maxiters: 5,
        ddof: 0,
        cenfunc: CenFunc::Median,
        clip_cen: Some(ClipCenter::Median),
        stdfunc: StdFunc::Std,
        nkeep: 0,
        revert_on_nkeep: false,
        maxrej: None,
        final_min_retained: 0,
    }
}

const OUTPUT: OutputOptions = OutputOptions {
    reduction: Some(Reduction::Mean),
    diagnostics: true,
};

#[test]
fn empty_all_masked_and_nonfinite_inputs_have_no_survivors() {
    let mut ws = Workspace::new();
    for (values, mask, input_flag) in [
        (vec![], None, 0),
        (vec![1.0, 2.0], Some(vec![true, true]), 1),
        (vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY], None, 0),
    ] {
        let result = clip(&values, mask.as_deref(), &params(), OUTPUT, &mut ws).unwrap();
        assert!(result.rejected.iter().all(|&x| x));
        assert_eq!(result.retained, 0);
        assert!(result.minimum_met); // Zero required; empty mean is still NaN.
        assert!(result.value.unwrap().is_nan());
        assert_eq!(result.iterations, 1);
        assert_eq!(result.stop_reason, StopReason::InvalidStatistics);
        let d = result.diagnostics.unwrap();
        assert!(d.low.is_nan() && d.upp.is_nan() && d.std.is_nan());
        assert_eq!(d.legacy_flags, input_flag);
        assert_eq!(d.legacy_nit, 1);
    }
}

#[test]
fn zero_spread_and_exact_threshold_equality() {
    let mut ws = Workspace::new();
    let mut p = params();
    p.sigma_lower = 0.0;
    p.sigma_upper = 0.0;
    let result = clip(&[3.0; 4], None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.retained, 4);
    assert_eq!(result.stop_reason, StopReason::Converged);
    assert_eq!(result.diagnostics.unwrap().std, 0.0);

    // MAD about zero is zero; the unequal tail is rejected at zero spread.
    p.stdfunc = StdFunc::Mad;
    let result = clip(&[0.0, 0.0, 0.0, 10.0], None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false, false, true]);

    // [-1,1] has median=0, population variance=1; both boundaries are exact.
    p.stdfunc = StdFunc::Std;
    p.sigma_lower = 1.0;
    p.sigma_upper = 1.0;
    let result = clip(&[-1.0, 1.0], None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false]);
    assert_eq!(result.diagnostics.unwrap().std, 1.0);

    // The lower limit is tightened independently; the high boundary survives.
    p.sigma_lower = 0.5;
    p.maxiters = 1;
    let result = clip(&[-1.0, 1.0], None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[true, false]);
    assert_eq!(result.value, Some(1.0));
}

#[test]
fn ddof_changes_the_threshold_and_can_invalidate_statistics() {
    let mut p = params();
    p.sigma_lower = 0.75;
    p.sigma_upper = 0.75;
    p.maxiters = 1;
    let mut ws = Workspace::new();
    // ddof=0: variance=1, both rejected; ddof=1: variance=2,
    // squared threshold=0.5625*2=1.125, both retained.
    for (ddof, rejected) in [(0, true), (1, false)] {
        p.ddof = ddof;
        let result = clip(&[-1.0, 1.0], None, &p, OUTPUT, &mut ws).unwrap();
        assert_eq!(result.rejected, &[rejected, rejected]);
    }
    for ddof in [2, 3, usize::MAX] {
        p.ddof = ddof;
        let result = clip(&[-1.0, 1.0], None, &p, OUTPUT, &mut ws).unwrap();
        assert_eq!(result.rejected, &[false, false]);
        assert_eq!(result.stop_reason, StopReason::InvalidStatistics);
        assert_eq!(result.diagnostics.unwrap().legacy_flags, 2);
    }
}

#[test]
fn iteration_limit_is_distinct_from_convergence_and_legacy_counter() {
    let mut p = params();
    let mut ws = Workspace::new();
    p.maxiters = 0;
    let result = clip(&[f64::NAN, 0.0, 100.0], None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[true, false, false]);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.stop_reason, StopReason::MaxIterations);
    let d = result.diagnostics.unwrap();
    assert_eq!(d.legacy_nit, 1);
    assert_eq!(d.legacy_flags, 2);
    assert!(d.std.is_nan());

    let values = [0.0, 0.0, 0.0, 10.0, 100.0];
    p.maxiters = 1;
    let result = clip(&values, None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false, false, false, true]);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.stop_reason, StopReason::MaxIterations);
    let d = result.diagnostics.unwrap();
    assert_eq!(d.legacy_nit, 2);
    assert_eq!(d.legacy_flags, 2);
    assert_eq!(d.std, 2020.0_f64.sqrt()); // Last pass, not final-sample std=5.

    p.maxiters = 3;
    let result = clip(&values, None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false, false, true, true]);
    assert_eq!(result.iterations, 3);
    assert_eq!(result.stop_reason, StopReason::Converged);
    assert_eq!(result.diagnostics.unwrap().legacy_flags, 0);
}

#[test]
fn rollback_preserves_prior_rejections_and_retains_attempted_spread() {
    let mut p = params();
    p.nkeep = 4;
    p.revert_on_nkeep = true;
    let mut ws = Workspace::new();
    // Pass 1: var=2020 rejects 100 only. Pass 2: var=25 tentatively rejects
    // 10, which is restored because only three samples would remain.
    let result = clip(&[0.0, 0.0, 0.0, 10.0, 100.0], None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false, false, false, true]);
    assert_eq!(result.retained, 4);
    assert_eq!(result.iterations, 2);
    let d = result.diagnostics.unwrap();
    assert_eq!(d.restored_flags, &[0, 0, 0, 64, 0]);
    assert_eq!(d.std, 5.0);
    assert_eq!(d.low, 0.0);
    assert_eq!(d.upp, 10.0);
    assert_eq!(d.legacy_nit, 2);
}

#[test]
fn rejection_limits_include_initial_exclusions_and_can_both_trigger() {
    let mut p = params();
    p.maxrej = Some(1);
    p.nkeep = 4;
    p.revert_on_nkeep = true;
    let mut ws = Workspace::new();
    let result = clip(&[f64::NAN, 0.0, 0.0, 0.0, 10.0], None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[true, false, false, false, false]);
    assert_eq!(result.retained, 4);
    let d = result.diagnostics.unwrap();
    assert_eq!(d.legacy_flags, 12);
    assert_eq!(d.restored_flags, &[0, 0, 0, 0, 192]);

    // Initial exclusions exceed the limit before clipping. Even a pass with
    // no changes checks limits; initial exclusions cannot be restored.
    p.revert_on_nkeep = false;
    p.maxrej = Some(0);
    let result = clip(&[9.0, 1.0], Some(&[true, false]), &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[true, false]);
    assert_eq!(
        result.stop_reason,
        StopReason::RejectionLimit {
            min_retained: false,
            max_rejected: true,
        }
    );
    let d = result.diagnostics.unwrap();
    assert_eq!(d.legacy_flags, 9);
    assert_eq!(d.restored_flags, &[0, 0]);
}

#[test]
fn dispersion_center_is_independent_of_rejection_center() {
    let mut p = params();
    p.maxiters = 1;
    let mut ws = Workspace::new();
    // Rejection center=0 either way. About median var=25; about mean var=18.75:
    // 100 < 4.41*25 but 100 > 4.41*18.75.
    p.sigma_lower = 2.1;
    p.sigma_upper = 2.1;
    let values = [0.0, 0.0, 0.0, 10.0];
    let result = clip(&values, None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false, false, false]);
    p.clip_cen = Some(ClipCenter::Mean);
    let result = clip(&values, None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false, false, true]);
}

#[test]
fn f32_mean_dispersion_and_rounded_clipping_center_are_distinct() {
    let mut p = params();
    p.cenfunc = CenFunc::Mean;
    p.sigma_lower = 1.5;
    p.sigma_upper = 1.5;
    p.maxiters = 1;
    let mut ws = Workspace::new();
    // f32 spacing at 2^24 is 2. The f64 mean is 2^24+1; narrowing it
    // gives 2^24. Mean-based variance is 1, whereas variance about the
    // narrowed rejection center is (0+4)/2=2.
    let values = [16_777_216.0_f32, 16_777_218.0];
    p.clip_cen = Some(ClipCenter::Mean);
    let result = clip(&values, None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, true]);
    assert_eq!(result.diagnostics.unwrap().std, 1.0);
    p.clip_cen = Some(ClipCenter::ClippingCenter);
    let result = clip(&values, None, &p, OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[false, false]);
    assert_eq!(result.diagnostics.unwrap().std, 2.0_f32.sqrt());
}

#[test]
fn final_mean_cache_preserves_scalar_sum_at_each_stop_state() {
    let cancellation = [1.0e16_f64, 1.0, -1.0e16, 1.0, 1.0];
    let mut p = params();
    p.cenfunc = CenFunc::Mean;
    p.clip_cen = Some(ClipCenter::Mean);
    p.sigma_lower = 10.0;
    p.sigma_upper = 10.0;
    let mut workspace = Workspace::with_capacity(6);

    // The four-lane variance helper groups the first and fifth values, so its
    // mean is 1/5. The final reduction deliberately keeps the scalar input
    // order: 1e16 + 1 - 1e16 + 1 + 1 = 2, hence the expected mean is 2/5.
    let (_, lane_mean) = var_mean_valid(&cancellation, 0);
    assert_eq!(lane_mean, 0.2);
    {
        let result = clip(&cancellation, None, &p, OUTPUT, &mut workspace).unwrap();
        assert_eq!(result.rejected, &[false; 5]);
        assert_eq!(result.retained, 5);
        assert_eq!(result.value, Some(0.4));
        assert_eq!(result.iterations, 1);
        assert_eq!(result.stop_reason, StopReason::Converged);
    }

    // A second pass tentatively rejects 10, then restores it because nkeep=4.
    // The statistics scratch at the attempted pass already contains the final
    // retained values, so the cached final mean remains valid after rollback.
    let mut rollback = p;
    rollback.sigma_lower = 1.5;
    rollback.sigma_upper = 1.5;
    rollback.nkeep = 4;
    rollback.revert_on_nkeep = true;
    let rollback_values = [0.0, 0.0, 0.0, 10.0, 100.0];
    {
        let result = clip(&rollback_values, None, &rollback, OUTPUT, &mut workspace).unwrap();
        assert_eq!(result.rejected, &[false, false, false, false, true]);
        assert_eq!(result.retained, 4);
        assert_eq!(result.value, Some(2.5));
        assert_eq!(result.iterations, 2);
        assert_eq!(
            result.stop_reason,
            StopReason::RejectionLimit {
                min_retained: true,
                max_rejected: false,
            }
        );
    }

    // An accepted rejection on the final allowed pass invalidates the cache;
    // the reduction must use the post-rejection mask and return 2.5.
    let mut final_pass = rollback;
    final_pass.nkeep = 0;
    final_pass.revert_on_nkeep = false;
    final_pass.maxiters = 1;
    {
        let result = clip(&rollback_values, None, &final_pass, OUTPUT, &mut workspace).unwrap();
        assert_eq!(result.rejected, &[false, false, false, false, true]);
        assert_eq!(result.retained, 4);
        assert_eq!(result.value, Some(2.5));
        assert_eq!(result.iterations, 1);
        assert_eq!(result.stop_reason, StopReason::MaxIterations);
    }

    // With maxiters=0 no statistics pass populates the scratch buffer. Use a
    // different cancellation tail so stale scratch cannot accidentally supply
    // the result; only initial finite-value exclusions apply.
    let zero_values = [1.0e16_f64, 1.0, -1.0e16, 1.0, 2.0];
    let mut zero_pass = p;
    zero_pass.maxiters = 0;
    let result = clip(&zero_values, None, &zero_pass, OUTPUT, &mut workspace).unwrap();
    assert_eq!(result.rejected, &[false; 5]);
    assert_eq!(result.retained, 5);
    assert_eq!(result.value, Some(0.6));
    assert_eq!(result.iterations, 0);
    assert_eq!(result.stop_reason, StopReason::MaxIterations);
}

#[test]
fn input_bits_and_original_indices_are_preserved() {
    let values = [
        f64::from_bits(0x7ff8_0000_0000_1234),
        -0.0,
        100.0,
        0.0,
        0.0,
        0.0,
    ];
    let before = values.map(f64::to_bits);
    let mask = [false, true, false, false, false, false];
    let mut ws = Workspace::new();
    let result = clip(&values, Some(&mask), &params(), OUTPUT, &mut ws).unwrap();
    assert_eq!(result.rejected, &[true, true, true, false, false, false]);
    assert_eq!(values.map(f64::to_bits), before);
    assert_eq!(mask, [false, true, false, false, false, false]);
}

#[test]
fn fused_median_tracks_retained_samples_at_stop_and_limit_boundaries() {
    let options = OutputOptions {
        reduction: Some(Reduction::Median),
        diagnostics: true,
    };
    let values = [-0.0_f64, 1.0, 2.0, 100.0];
    let before = values.map(f64::to_bits);
    let mut workspace = Workspace::with_capacity(values.len());

    // The first pass rejects only 100. The next pass converges on the
    // retained values, whose independent median is exactly 1.
    let result = clip(&values, None, &params(), options, &mut workspace).unwrap();
    assert_eq!(result.rejected, &[false, false, false, true]);
    assert_eq!(result.retained, 3);
    assert_eq!(result.value, Some(1.0));
    assert_eq!(result.stop_reason, StopReason::Converged);
    assert_eq!(values.map(f64::to_bits), before);

    // A rejection accepted on the final allowed iteration still contributes
    // to the fused median, even though the stop reason is MaxIterations.
    let mut one_pass = params();
    one_pass.maxiters = 1;
    let result = clip(&values, None, &one_pass, options, &mut workspace).unwrap();
    assert_eq!(result.rejected, &[false, false, false, true]);
    assert_eq!(result.value, Some(1.0));
    assert_eq!(result.stop_reason, StopReason::MaxIterations);

    // The final minimum gate reports NaN while preserving the accepted mask.
    let mut minimum = one_pass;
    minimum.final_min_retained = 4;
    let result = clip(&values, None, &minimum, options, &mut workspace).unwrap();
    assert_eq!(result.rejected, &[false, false, false, true]);
    assert!(!result.minimum_met);
    assert!(result.value.is_some_and(f64::is_nan));

    // With no iterations, only the initial nonfinite exclusion is present and
    // the fused median is the ordinary median of the two finite values.
    let mut zero_pass = params();
    zero_pass.maxiters = 0;
    let result = clip(
        &[f64::NAN, 1.0, 100.0],
        None,
        &zero_pass,
        options,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(result.rejected, &[true, false, false]);
    assert_eq!(result.retained, 2);
    assert_eq!(result.value, Some(50.5));
    assert_eq!(result.stop_reason, StopReason::MaxIterations);

    // ddof equal to the retained count makes the first statistics estimate
    // invalid; this is distinct from a zero-spread, converged pass.
    let mut invalid = params();
    invalid.maxiters = 1;
    invalid.ddof = 3;
    let result = clip(
        &[1.0_f64, 2.0, 3.0],
        None,
        &invalid,
        options,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(result.rejected, &[false, false, false]);
    assert_eq!(result.value, Some(2.0));
    assert_eq!(result.stop_reason, StopReason::InvalidStatistics);
    assert!(result.diagnostics.unwrap().std.is_nan());

    // A prior rejection remains accepted when a later pass would violate
    // nkeep; only the newly rejected sample gets restoration flag 64.
    let mut nkeep = params();
    nkeep.nkeep = 4;
    nkeep.revert_on_nkeep = true;
    let result = clip(
        &[0.0, 0.0, 0.0, 10.0, 100.0],
        None,
        &nkeep,
        options,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(result.rejected, &[false, false, false, false, true]);
    assert_eq!(result.retained, 4);
    assert_eq!(
        result.stop_reason,
        StopReason::RejectionLimit {
            min_retained: true,
            max_rejected: false,
        }
    );
    assert_eq!(
        result.diagnostics.unwrap().restored_flags,
        &[0, 0, 0, 64, 0]
    );

    // maxrej applies the same whole-iteration rollback with legacy flag 128.
    let mut maxrej = params();
    maxrej.maxrej = Some(1);
    let result = clip(
        &[0.0, 0.0, 0.0, 10.0, 100.0],
        None,
        &maxrej,
        options,
        &mut workspace,
    )
    .unwrap();
    assert_eq!(result.rejected, &[false, false, false, false, true]);
    assert_eq!(result.retained, 4);
    assert_eq!(
        result.stop_reason,
        StopReason::RejectionLimit {
            min_retained: false,
            max_rejected: true,
        }
    );
    assert_eq!(
        result.diagnostics.unwrap().restored_flags,
        &[0, 0, 0, 128, 0]
    );
}

#[test]
fn mad_center_choice_is_distinct_and_reusable_workspace_safe() {
    let options = OutputOptions {
        reduction: Some(Reduction::Median),
        diagnostics: true,
    };
    let values = [0.0_f64, 0.0, 0.0, 5.0, 10.0];
    let mut params = params();
    params.cenfunc = CenFunc::Median;
    params.clip_cen = Some(ClipCenter::Mean);
    params.stdfunc = StdFunc::Mad;
    params.sigma_lower = 10.0;
    params.sigma_upper = 1.5;
    params.maxiters = 1;
    let mut workspace = Workspace::with_capacity(values.len());

    // MAD about the mean is 1.4826 * median([2, 3, 3, 3, 7]) = 4.4478,
    // so 5 remains while 10 is rejected around the median rejection center.
    let mean_center = {
        let result = clip(&values, None, &params, options, &mut workspace).unwrap();
        (
            result.rejected.to_vec(),
            result.retained,
            result.value,
            result.diagnostics.unwrap().std,
        )
    };
    assert_eq!(mean_center.0, &[false, false, false, false, true]);
    assert_eq!(mean_center.1, 4);
    assert_eq!(mean_center.2, Some(0.0));
    assert_eq!(mean_center.3.to_bits(), (1.4826_f64 * 3.0).to_bits());

    // MAD about the clipping (median) center is zero, so both positive
    // outliers are rejected. Reusing the workspace must not retain the prior
    // mean-centered scratch ordering or values.
    params.clip_cen = Some(ClipCenter::ClippingCenter);
    let switched = {
        let result = clip(&values, None, &params, options, &mut workspace).unwrap();
        (
            result.rejected.to_vec(),
            result.retained,
            result.value,
            result.diagnostics.unwrap().std,
        )
    };
    assert_eq!(switched.0, &[false, false, false, true, true]);
    assert_eq!(switched.1, 3);
    assert_eq!(switched.2, Some(0.0));
    assert_eq!(switched.3, 0.0);

    // A fresh workspace must produce exactly the same mask, reduction, and
    // diagnostic spread as the reused workspace call.
    let mut fresh_workspace = Workspace::new();
    let fresh = clip(&values, None, &params, options, &mut fresh_workspace).unwrap();
    assert_eq!(switched.0, fresh.rejected);
    assert_eq!(switched.1, fresh.retained);
    assert_eq!(switched.2, fresh.value);
    assert_eq!(switched.3, fresh.diagnostics.unwrap().std);
}
