//! Hand-derived behavioural tests for the reusable sigma-clipping kernel.
//!
//! The expected masks in this file are deliberately small enough to derive by
//! hand while keeping every case inside the reducers crate.

use reducers::sigma_clip::{
    clip, CenFunc, ClipCenter, ClipError, OutputOptions, Params, Reduction, StdFunc, StopReason,
    Workspace,
};

fn standard_params() -> Params {
    Params {
        sigma_lower: 1.5,
        sigma_upper: 1.5,
        maxiters: 5,
        ddof: 0,
        cenfunc: CenFunc::Median,
        clip_cen: Some(ClipCenter::Median),
        stdfunc: StdFunc::Std,
        nkeep: 1,
        revert_on_nkeep: true,
        maxrej: None,
        final_min_retained: 1,
    }
}

fn diagnostic_options(reduction: Option<Reduction>) -> OutputOptions {
    OutputOptions {
        reduction,
        diagnostics: true,
    }
}

fn assert_close(got: f64, expected: f64, tolerance: f64) {
    assert!(
        (got - expected).abs() <= tolerance,
        "got {got:?}, expected {expected:?} (tol {tolerance})"
    );
}

fn assert_close_f32(got: f32, expected: f32, tolerance: f32) {
    assert!(
        (got - expected).abs() <= tolerance,
        "got {got:?}, expected {expected:?} (tol {tolerance})"
    );
}

#[test]
fn median_standard_deviation_clips_one_high_outlier_and_reports_diagnostics() {
    let samples = [1.0_f64, 2.0, 3.0, 100.0];
    let mut params = standard_params();
    params.sigma_lower = 1.5;
    params.sigma_upper = 1.5;
    let options = diagnostic_options(Some(Reduction::Median));
    let mut workspace = Workspace::with_capacity(samples.len());

    let result = clip(&samples, None, &params, options, &mut workspace).unwrap();

    assert_eq!(result.rejected, &[false, false, false, true]);
    assert_eq!(result.retained, 3);
    assert!(result.minimum_met);
    assert_eq!(result.value, Some(2.0));
    assert_eq!(result.iterations, 2);
    assert!(matches!(result.stop_reason, StopReason::Converged));

    let diagnostics = result.diagnostics.expect("diagnostics requested");
    assert_eq!(diagnostics.low, 1.0);
    assert_eq!(diagnostics.upp, 3.0);
    assert_close(diagnostics.std, (2.0_f64 / 3.0).sqrt(), 1e-14);
    assert_eq!(diagnostics.legacy_nit, 2);
    assert_eq!(diagnostics.legacy_flags, 0);
    assert_eq!(diagnostics.restored_flags, &[0, 0, 0, 0]);
}

#[test]
fn mean_standard_deviation_uses_mean_for_both_center_roles() {
    let samples = [0.0_f64, 0.0, 0.0, 10.0];
    let mut params = standard_params();
    params.sigma_lower = 1.5;
    params.sigma_upper = 1.5;
    params.cenfunc = CenFunc::Mean;
    params.clip_cen = Some(ClipCenter::Mean);
    let options = diagnostic_options(Some(Reduction::Mean));
    let mut workspace = Workspace::new();

    let result = clip(&samples, None, &params, options, &mut workspace).unwrap();

    assert_eq!(result.rejected, &[false, false, false, true]);
    assert_eq!(result.retained, 3);
    assert_eq!(result.value, Some(0.0));
    assert_eq!(result.iterations, 2);
    assert!(matches!(result.stop_reason, StopReason::Converged));
    let diagnostics = result.diagnostics.expect("diagnostics requested");
    assert_eq!(diagnostics.low, 0.0);
    assert_eq!(diagnostics.upp, 0.0);
    assert_eq!(diagnostics.std, 0.0);
    assert_eq!(diagnostics.legacy_nit, 2);
    assert_eq!(diagnostics.legacy_flags, 0);
}

#[test]
fn mad_ignores_ddof_and_rejects_the_high_tail() {
    let samples = [8.0_f64, 9.0, 10.0, 11.0, 12.0, 100.0];
    let mut params = standard_params();
    params.sigma_lower = 10.0;
    params.sigma_upper = 3.0;
    params.ddof = 1;
    params.stdfunc = StdFunc::Mad;
    let options = diagnostic_options(Some(Reduction::Mean));
    let mut workspace = Workspace::with_capacity(samples.len());

    let result = clip(&samples, None, &params, options, &mut workspace).unwrap();

    assert_eq!(result.rejected, &[false, false, false, false, false, true]);
    assert_eq!(result.retained, 5);
    assert_eq!(result.value, Some(10.0));
    assert!(matches!(result.stop_reason, StopReason::Converged));
    let diagnostics = result.diagnostics.expect("diagnostics requested");
    assert_eq!(diagnostics.low, 8.0);
    assert_eq!(diagnostics.upp, 12.0);
    assert_close(diagnostics.std, 1.4826, 1e-12);
    assert_eq!(diagnostics.legacy_nit, 2);
    assert_eq!(diagnostics.legacy_flags, 0);
}

#[test]
fn nonfinite_and_input_masked_samples_remain_rejected_and_are_flagged() {
    let samples = [f64::NAN, 0.0, 0.0, 0.0, 10.0];
    let input_mask = [false, true, false, false, false];
    let mut params = standard_params();
    params.sigma_lower = 1.5;
    params.sigma_upper = 1.5;
    let options = diagnostic_options(None);
    let mut workspace = Workspace::new();

    let result = clip(
        &samples,
        Some(&input_mask),
        &params,
        options,
        &mut workspace,
    )
    .unwrap();

    assert_eq!(result.rejected, &[true, true, false, false, true]);
    assert_eq!(result.retained, 2);
    assert!(result.minimum_met);
    assert_eq!(result.value, None);
    assert!(matches!(result.stop_reason, StopReason::Converged));
    let diagnostics = result.diagnostics.expect("diagnostics requested");
    assert_eq!(diagnostics.legacy_nit, 2);
    assert_eq!(diagnostics.legacy_flags, 1); // Explicit input mask only; NaN alone does not set it.
    assert_eq!(diagnostics.restored_flags, &[0, 0, 0, 0, 0]);
}

#[test]
fn rollback_restores_a_whole_rejection_iteration_and_marks_the_sample() {
    let samples = [0.0_f64, 1.0, 100.0];
    let mut params = standard_params();
    params.sigma_lower = 10.0;
    params.sigma_upper = 0.5;
    params.maxiters = 1;
    params.nkeep = 3;
    params.revert_on_nkeep = true;
    let options = diagnostic_options(Some(Reduction::Mean));
    let mut workspace = Workspace::with_capacity(samples.len());

    let result = clip(&samples, None, &params, options, &mut workspace).unwrap();

    assert_eq!(result.rejected, &[false, false, false]);
    assert_eq!(result.retained, 3);
    assert!(result.minimum_met);
    assert_eq!(result.value, Some(101.0 / 3.0));
    assert_eq!(result.iterations, 1);
    assert!(matches!(
        result.stop_reason,
        StopReason::RejectionLimit {
            min_retained: true,
            max_rejected: false
        }
    ));
    let diagnostics = result.diagnostics.expect("diagnostics requested");
    assert_eq!(diagnostics.legacy_nit, 1);
    assert_eq!(diagnostics.legacy_flags, 4);
    assert_eq!(diagnostics.restored_flags, &[0, 0, 64]);
}

#[test]
fn max_rejected_restores_the_iteration_and_sets_diagnostic_128_flag() {
    let samples = [0.0_f64, 0.0, 0.0, 10.0];
    let mut params = standard_params();
    params.maxiters = 1;
    params.maxrej = Some(0);
    let options = diagnostic_options(None);
    let mut workspace = Workspace::with_capacity(samples.len());

    let result = clip(&samples, None, &params, options, &mut workspace).unwrap();

    assert_eq!(result.rejected, &[false, false, false, false]);
    assert_eq!(result.retained, 4);
    assert!(matches!(
        result.stop_reason,
        StopReason::RejectionLimit {
            min_retained: false,
            max_rejected: true,
        }
    ));
    let diagnostics = result.diagnostics.expect("diagnostics requested");
    assert_eq!(diagnostics.legacy_flags, 8);
    assert_eq!(diagnostics.restored_flags, &[0, 0, 0, 128]);
}

#[test]
fn final_minimum_is_reported_without_rolling_back_the_candidate() {
    let samples = [0.0_f64, 1.0, 100.0];
    let mut params = standard_params();
    params.sigma_lower = 10.0;
    params.sigma_upper = 0.5;
    params.maxiters = 1;
    params.revert_on_nkeep = false;
    params.final_min_retained = 3;
    let options = OutputOptions {
        reduction: Some(Reduction::Mean),
        diagnostics: false,
    };
    let mut workspace = Workspace::new();

    let result = clip(&samples, None, &params, options, &mut workspace).unwrap();

    assert_eq!(result.rejected, &[false, false, true]);
    assert_eq!(result.retained, 2);
    assert!(!result.minimum_met);
    assert!(result.value.is_some_and(f64::is_nan));
    assert!(matches!(result.stop_reason, StopReason::MaxIterations));
    assert!(result.diagnostics.is_none());
}

#[test]
fn lower_median_center_uses_the_lower_even_sample() {
    let samples = [8.0_f64, 10.0, 10.6, 40.0];
    let mut params = standard_params();
    params.sigma_lower = 10.0;
    params.sigma_upper = 0.021;
    params.maxiters = 1;
    params.cenfunc = CenFunc::LowerMedian;
    params.clip_cen = Some(ClipCenter::LowerMedian);
    let options = OutputOptions {
        reduction: None,
        diagnostics: false,
    };
    let mut workspace = Workspace::with_capacity(samples.len());

    let result = clip(&samples, None, &params, options, &mut workspace).unwrap();

    assert_eq!(result.rejected, &[false, false, true, true]);
    assert_eq!(result.retained, 2);
    assert!(result.minimum_met);
    assert!(matches!(result.stop_reason, StopReason::MaxIterations));
}

#[test]
fn f32_and_f64_share_the_same_mask_and_reusable_workspace_is_safe() {
    let samples32 = [0.0_f32, 0.0, 0.0, 10.0];
    let mut params = standard_params();
    params.sigma_lower = 1.5;
    params.sigma_upper = 1.5;
    let options = OutputOptions {
        reduction: Some(Reduction::Mean),
        diagnostics: true,
    };

    let mut workspace32 = Workspace::with_capacity(1);
    let first = clip(&samples32, None, &params, options, &mut workspace32).unwrap();
    assert_eq!(first.rejected, &[false, false, false, true]);
    assert_eq!(first.retained, 3);
    assert_eq!(first.value, Some(0.0));
    assert_close_f32(first.diagnostics.unwrap().std, 0.0, 0.0);

    // A workspace may grow when a later call has a longer vector, then be
    // reused for a shorter vector without retaining stale rejection flags.
    let longer = [0.0_f32, 0.0, 0.0, 10.0, 0.0, 0.0];
    let second = clip(
        &longer,
        None,
        &params,
        OutputOptions {
            reduction: None,
            diagnostics: false,
        },
        &mut workspace32,
    )
    .unwrap();
    assert_eq!(second.rejected, &[false, false, false, true, false, false]);
    assert_eq!(second.retained, 5);

    let shorter = [0.0_f32, 0.0, 10.0];
    let third = clip(
        &shorter,
        None,
        &params,
        OutputOptions {
            reduction: None,
            diagnostics: false,
        },
        &mut workspace32,
    )
    .unwrap();
    assert_eq!(third.rejected, &[false, false, true]);
    assert_eq!(third.retained, 2);
}

#[test]
fn invalid_masks_and_sigmas_are_reported_before_mutating_results() {
    let samples = [1.0_f64, 2.0, 3.0];
    let mask = [false, true];
    let mut params = standard_params();
    let options = OutputOptions::default();
    let mut workspace = Workspace::new();

    assert!(matches!(
        clip(&samples, Some(&mask), &params, options, &mut workspace),
        Err(ClipError::MaskLength {
            samples: 3,
            mask: 2
        })
    ));

    params.sigma_lower = -1.0;
    assert!(matches!(
        clip(&samples, None, &params, options, &mut workspace),
        Err(ClipError::InvalidSigma)
    ));

    params.sigma_lower = f64::NAN;
    assert!(matches!(
        clip(&samples, None, &params, options, &mut workspace),
        Err(ClipError::InvalidSigma)
    ));
}
