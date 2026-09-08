use reducers::axis::{
    reduce_axis0, reduce_axis_last, variance_mean_axis0, variance_mean_axis_last,
};
use reducers::reducers_1d::{mean, variance_mean, Kind};
use reducers::ScanPolicy;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        actual.is_finite(),
        "{actual} is not finite; expected {expected}"
    );
    assert!(
        (actual - expected).abs() <= expected.abs() * 2e-14,
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn finite_mean_survives_intermediate_sum_overflow() {
    for sign in [-1.0, 1.0] {
        let values = [sign * 1e308; 8];
        assert_close(mean(&values, ScanPolicy::AllFinite), values[0]);
        assert_eq!(
            variance_mean(&values, 0, ScanPolicy::AllFinite),
            (0.0, values[0])
        );
    }
    let values = [1e308, 1e308, -1e308, -1e308, 1e308, 1e308, -1e308, -1e308];
    assert_eq!(mean(&values, ScanPolicy::AllFinite), 0.0);
    assert_eq!(
        variance_mean(&values, 0, ScanPolicy::AllFinite),
        (f64::INFINITY, 0.0)
    );
}

#[test]
fn finite_variance_survives_square_sum_overflow() {
    // Even a finite rounded sum can displace the mean enough to overflow
    // squared deviations of a constant input. Overflow recovery must refine
    // the center as well as rescale the deviations.
    assert_eq!(
        variance_mean(&[1e307; 7], 0, ScanPolicy::AllFinite),
        (0.0, 1e307)
    );
    let values = [-1e154, 1e154, -1e154, 1e154];
    for ddof in [0, 1] {
        assert_close(
            variance_mean(&values, ddof, ScanPolicy::AllFinite).0,
            1e308 * (4.0 / (4 - ddof) as f64),
        );
    }
    // Individual squared deviations overflow, but division by count makes
    // the population variance representable: 2 * (2e154)^2 / 8 = 1e308.
    let values = [-2e154, 2e154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_close(variance_mean(&values, 0, ScanPolicy::AllFinite).0, 1e308);
}

#[test]
fn overflowing_moments_obey_scan_policy_and_ddof() {
    let values = [1e308, f64::NAN, 1e308, f64::INFINITY, f64::NEG_INFINITY];
    assert_eq!(
        variance_mean(&values, 0, ScanPolicy::SkipNonFinite),
        (0.0, 1e308)
    );
    assert!(mean(&values, ScanPolicy::AllValues).is_nan());
    assert!(mean(&values, ScanPolicy::SkipNan).is_nan());
    let retained_inf = [1e308, 1e308, f64::INFINITY];
    assert_eq!(mean(&retained_inf, ScanPolicy::SkipNan), f64::INFINITY);
    assert!(variance_mean(&retained_inf, 0, ScanPolicy::SkipNan)
        .0
        .is_nan());
    let (var, avg) = variance_mean(&[1e308, 1e308], 2, ScanPolicy::AllFinite);
    assert!(var.is_nan());
    assert_eq!(avg, 1e308);
    assert!(variance_mean::<f64>(&[], 0, ScanPolicy::AllFinite)
        .0
        .is_nan());
    let small = f64::from_bits(1);
    assert_eq!(mean(&[small; 4], ScanPolicy::AllFinite), small);
    assert_eq!(variance_mean(&[small; 4], 0, ScanPolicy::AllFinite).0, 0.0);
}

#[test]
fn axis_moments_share_overflow_handling() {
    let data = [1e308, -1e154, 1e308, 1e154, 1e308, -1e154, 1e308, 1e154];
    let means = reduce_axis0(&data, 4, 2, Kind::Mean, 0, ScanPolicy::SkipNonFinite);
    assert_eq!(means, vec![1e308, 0.0]);
    let (vars, means) = variance_mean_axis0(&data, 4, 2, 0, ScanPolicy::SkipNonFinite);
    assert_eq!(means, vec![1e308, 0.0]);
    assert_eq!(vars[0], 0.0);
    assert_close(vars[1], 1e308);
    let last = [1e308, 1e308, 1e308, 1e308, -1e154, 1e154, -1e154, 1e154];
    let vars = reduce_axis_last(&last, 2, 4, Kind::Var, 0, ScanPolicy::SkipNonFinite);
    assert_eq!(vars[0], 0.0);
    assert_close(vars[1], 1e308);
}

#[test]
fn overflowing_mean_keeps_small_cancellation_residual() {
    let values = [
        1e308, 1e308, -1e308, -1e308, 1e308, 1e308, -1e308, -1e308, 9e-100,
    ];
    assert_close(mean(&values, ScanPolicy::AllFinite), 1e-100);
}

#[test]
fn axis_overflow_recovers_even_when_a_different_sum_order_is_finite() {
    // Sequential axis accumulation overflows; a four-lane ordinary sum loses
    // the two unit terms and returns zero. Recovery must preserve their mean.
    let values = [1.0_f64, 1.0, 1e308, 1e308, -1e308, 1e308, -1e308, -1e308];
    assert_eq!(
        reduce_axis0(&values, 8, 1, Kind::Mean, 0, ScanPolicy::AllFinite),
        [0.25]
    );
    let (vars, means) = variance_mean_axis0(&values, 8, 1, 8, ScanPolicy::AllFinite);
    assert!(vars[0].is_nan());
    assert_eq!(means, [0.25]);
}

#[test]
fn mean_recovers_when_restoring_a_partial_would_overflow() {
    let values = [
        8.988465674311569e307,
        1.7976931348623193e307,
        -7.190772539449259e307,
        7.190772539449254e307,
        7.190772539449275e307,
    ];
    // Exact rational summation of the input binary floats, divided by five.
    assert_close(mean(&values, ScanPolicy::AllFinite), 3.5953862697246315e307);
    assert_close(
        reduce_axis0(&values, 5, 1, Kind::Mean, 0, ScanPolicy::AllFinite)[0],
        3.5953862697246315e307,
    );
}

#[test]
fn overflowing_mean_recovers_subnormal_residuals() {
    let small = f64::from_bits(1);
    let mut values = vec![1e308, 1e308, -1e308, -1e308, 1e308, 1e308, -1e308, -1e308];
    values.extend([small; 16]);
    assert_eq!(mean(&values, ScanPolicy::AllFinite), small);
    for n in [3, 7, 16, 33] {
        let values = vec![f64::MAX; n];
        assert_eq!(mean(&values, ScanPolicy::AllFinite), f64::MAX);
        assert_eq!(
            variance_mean(&values, 0, ScanPolicy::AllFinite),
            (0.0, f64::MAX)
        );
    }
}

#[test]
fn overflow_recovery_gives_zero_variance_for_extreme_constants() {
    // These magnitudes make even a one-ULP center error overflow its square.
    for value in [1e200, 1e307, 1e308, f64::MAX] {
        for sign in [-1.0, 1.0] {
            for n in 2..=128 {
                let values = vec![sign * value; n];
                assert_eq!(
                    variance_mean(&values, 0, ScanPolicy::AllFinite),
                    (0.0, sign * value),
                    "value={value}, sign={sign}, n={n}"
                );
            }
        }
    }
}

#[test]
fn axis_overflow_recovery_preserves_policies_and_insufficient_count_means() {
    let data = [1e308, f64::NAN, 1e308, f64::INFINITY];
    for ddof in [0, 2] {
        let (variances, means) = variance_mean_axis0(&data, 2, 2, ddof, ScanPolicy::SkipNonFinite);
        assert_eq!(means[0], 1e308);
        assert!(means[1].is_nan());
        assert!(variances[1].is_nan());
        if ddof == 0 {
            assert_eq!(variances[0], 0.0);
        } else {
            assert!(variances[0].is_nan());
        }
    }

    let last = [1e308, 1e308, f64::NAN, f64::INFINITY];
    for policy in [ScanPolicy::AllValues, ScanPolicy::SkipNan] {
        let (variances, means) = variance_mean_axis_last(&last, 2, 2, 0, policy);
        assert_eq!((variances[0], means[0]), (0.0, 1e308));
        assert!(variances[1].is_nan());
        if matches!(policy, ScanPolicy::SkipNan) {
            assert_eq!(means[1], f64::INFINITY);
        } else {
            assert!(means[1].is_nan());
        }
    }
}

#[test]
fn finite_f32_results_and_axis_narrowing_keep_their_output_contract() {
    let constant = [f32::MAX; 8];
    assert_eq!(
        variance_mean(&constant, 0, ScanPolicy::AllFinite),
        (0.0, f32::MAX as f64)
    );
    let values = [-f32::MAX, f32::MAX, -f32::MAX, f32::MAX];
    let (variance, mean) = variance_mean(&values, 0, ScanPolicy::AllFinite);
    assert_eq!(mean, 0.0);
    assert!(variance.is_finite());
    let (variances, means) = variance_mean_axis0(&values, 4, 1, 0, ScanPolicy::AllFinite);
    assert_eq!(variances, [f32::INFINITY]);
    assert_eq!(means, [0.0_f32]);
}
