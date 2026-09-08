//! Public defaults and center-selection behavior for sigma clipping.

use reducers::sigma_clip::{
    clip, CenFunc, ClipCenter, OutputOptions, Params, Reduction, StdFunc, StopReason, Workspace,
};

fn diagnostic_options() -> OutputOptions {
    OutputOptions {
        reduction: Some(Reduction::Mean),
        diagnostics: true,
    }
}

#[test]
fn default_params_have_documented_values_and_clip_an_outlier() {
    let defaults = Params::default();
    assert_eq!(defaults.sigma_lower, 3.0);
    assert_eq!(defaults.sigma_upper, 3.0);
    assert_eq!(defaults.maxiters, 5);
    assert_eq!(defaults.ddof, 0);
    assert_eq!(defaults.cenfunc, CenFunc::Median);
    assert_eq!(defaults.clip_cen, None);
    assert_eq!(defaults.stdfunc, StdFunc::Std);
    assert_eq!(defaults.nkeep, 0);
    assert!(!defaults.revert_on_nkeep);
    assert_eq!(defaults.maxrej, None);
    assert_eq!(defaults.final_min_retained, 0);

    let values = [0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
    let mut workspace = Workspace::with_capacity(values.len());
    let result = clip(
        &values,
        None,
        &defaults,
        diagnostic_options(),
        &mut workspace,
    )
    .unwrap();

    assert_eq!(
        result.rejected,
        &[false, false, false, false, false, false, false, false, false, true]
    );
    assert_eq!(result.retained, 9);
    assert!(result.minimum_met);
    assert_eq!(result.value, Some(0.0));
    assert_eq!(result.iterations, 2);
    assert_eq!(result.stop_reason, StopReason::Converged);
}

#[test]
fn default_does_not_rollback_without_an_explicit_survivor_limit() {
    let values = [0.0_f64, 2.0];
    let default_params = Params {
        sigma_lower: 0.0,
        sigma_upper: 0.0,
        maxiters: 1,
        ..Params::default()
    };
    let rollback_params = Params {
        nkeep: 1,
        revert_on_nkeep: true,
        ..default_params
    };

    let mut workspace = Workspace::with_capacity(values.len());
    let result = clip(
        &values,
        None,
        &default_params,
        diagnostic_options(),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(result.rejected, &[true, true]);
    assert_eq!(result.retained, 0);
    assert!(result.value.unwrap().is_nan());
    assert_eq!(result.stop_reason, StopReason::MaxIterations);
    assert_eq!(result.diagnostics.unwrap().restored_flags, &[0, 0]);

    let result = clip(
        &values,
        None,
        &rollback_params,
        diagnostic_options(),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(result.rejected, &[false, false]);
    assert_eq!(result.retained, 2);
    assert_eq!(result.value, Some(1.0));
    assert_eq!(
        result.stop_reason,
        StopReason::RejectionLimit {
            min_retained: true,
            max_rejected: false,
        }
    );
    assert_eq!(result.diagnostics.unwrap().restored_flags, &[64, 64]);
}

#[test]
fn mean_center_without_override_uses_unrounded_mean_dispersion() {
    let values = [16_777_216.0_f32, 16_777_218.0];
    let params = Params {
        sigma_lower: 1.5,
        sigma_upper: 1.5,
        maxiters: 1,
        cenfunc: CenFunc::Mean,
        ..Params::default()
    };
    let explicit_mean = Params {
        clip_cen: Some(ClipCenter::Mean),
        ..params
    };
    let rounded_center = Params {
        clip_cen: Some(ClipCenter::ClippingCenter),
        ..params
    };

    let mut implicit_workspace = Workspace::with_capacity(values.len());
    let implicit = clip(
        &values,
        None,
        &params,
        diagnostic_options(),
        &mut implicit_workspace,
    )
    .unwrap();
    let mut explicit_workspace = Workspace::with_capacity(values.len());
    let explicit = clip(
        &values,
        None,
        &explicit_mean,
        diagnostic_options(),
        &mut explicit_workspace,
    )
    .unwrap();
    let mut rounded_workspace = Workspace::with_capacity(values.len());
    let rounded = clip(
        &values,
        None,
        &rounded_center,
        diagnostic_options(),
        &mut rounded_workspace,
    )
    .unwrap();

    assert_eq!(implicit.rejected, &[false, true]);
    assert_eq!(explicit.rejected, implicit.rejected);
    assert_eq!(rounded.rejected, &[false, false]);
    assert_eq!(implicit.value, Some(16_777_216.0));
    assert_eq!(rounded.value, Some(16_777_217.0));
    assert_eq!(implicit.diagnostics.unwrap().std, 1.0);
    assert_eq!(rounded.diagnostics.unwrap().std, 2.0_f32.sqrt());
}
