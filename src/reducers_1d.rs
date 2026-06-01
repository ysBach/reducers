//! 1-D reducer kernels, parameterized by [`ScanPolicy`].
//!
//! All public functions return `f64` results promoted internally for numerical
//! stability; the PyO3 layer narrows back to the input dtype where appropriate.
//! These are the building blocks reused by the axis adapters and by external
//! Rust-crate consumers.

use rayon::prelude::*;

use crate::finite::{Float, ScanPolicy};
use crate::parallel::minmax_1d_parallel_chunks;

#[inline]
fn cmp_float<T: Float>(a: &T, b: &T) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Non-floating numeric element that can be reduced without NaN handling.
pub trait Number: Copy + Ord + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

pub trait Weight: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

impl Weight for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Weight for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

impl Weight for bool {
    #[inline]
    fn to_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
}

macro_rules! impl_weight {
    ($($t:ty),* $(,)?) => {
        $(
            impl Weight for $t {
                #[inline]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_weight!(i8, u8, i16, u16, i32, u32, i64, u64);

impl Number for bool {
    #[inline]
    fn to_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
}

macro_rules! impl_number {
    ($($t:ty),* $(,)?) => {
        $(
            impl Number for $t {
                #[inline]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_number!(i8, u8, i16, u16, i32, u32, i64, u64);

// --------------------------------------------------------------------------
// sum / count (mean, sum)
// --------------------------------------------------------------------------

#[inline]
fn sum_count_all<T: Float>(values: &[T]) -> (f64, usize) {
    let mut sums = [0.0_f64; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        sums[0] += chunk[0].to_f64();
        sums[1] += chunk[1].to_f64();
        sums[2] += chunk[2].to_f64();
        sums[3] += chunk[3].to_f64();
    }
    for &x in chunks.remainder() {
        sums[0] += x.to_f64();
    }
    (sums.iter().sum::<f64>(), values.len())
}

#[inline]
fn sum_count_pred<T: Float, K: Fn(T) -> bool>(values: &[T], keep: K) -> (f64, usize) {
    let mut sums = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let x = chunk[lane];
            if keep(x) {
                sums[lane] += x.to_f64();
                counts[lane] += 1;
            }
        }
    }
    for &x in chunks.remainder() {
        if keep(x) {
            sums[0] += x.to_f64();
            counts[0] += 1;
        }
    }
    (sums.iter().sum::<f64>(), counts.iter().sum::<usize>())
}

#[inline]
fn sum_count<T: Float>(values: &[T], policy: ScanPolicy) -> (f64, usize) {
    match policy {
        ScanPolicy::AllValues | ScanPolicy::AllFinite => sum_count_all(values),
        ScanPolicy::SkipNan => sum_count_pred(values, |x: T| !x.is_nan()),
        ScanPolicy::SkipNonFinite => sum_count_pred(values, |x: T| x.is_finite()),
    }
}

pub fn mean<T: Float>(values: &[T], policy: ScanPolicy) -> f64 {
    let (sum, count) = sum_count(values, policy);
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

pub fn sum<T: Float>(values: &[T], policy: ScanPolicy) -> f64 {
    let (sum, count) = sum_count(values, policy);
    // numpy sum of an all-skipped vector is 0.0; mean of one is NaN.
    if count == 0 && matches!(policy, ScanPolicy::SkipNan | ScanPolicy::SkipNonFinite) {
        0.0
    } else {
        sum
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WeightedMean {
    pub value: f64,
    pub sum_weight: f64,
    pub count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct WeightedSum {
    pub weighted_sum: f64,
    pub sum_weights: f64,
    pub unweighted_sum: f64,
    pub count: usize,
}

#[inline]
fn finish_weighted(sum: f64, sum_weight: f64, count: usize) -> WeightedMean {
    WeightedMean {
        value: if count == 0 {
            f64::NAN
        } else {
            sum / sum_weight
        },
        sum_weight,
        count,
    }
}

pub fn weighted_average<T: Float, W: Weight>(
    values: &[T],
    weights: &[W],
    policy: ScanPolicy,
) -> WeightedMean {
    let mut sums = [0.0_f64; 4];
    let mut sum_weights = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut v_chunks = values.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (v, w) in (&mut v_chunks).zip(&mut w_chunks) {
        for lane in 0..4 {
            let x = v[lane];
            let keep = match policy {
                ScanPolicy::AllValues | ScanPolicy::AllFinite => true,
                ScanPolicy::SkipNan => !x.is_nan(),
                ScanPolicy::SkipNonFinite => x.is_finite(),
            };
            if keep {
                let w = w[lane].to_f64();
                sums[lane] += x.to_f64() * w;
                sum_weights[lane] += w;
                counts[lane] += 1;
            }
        }
    }
    for (&x, &w) in v_chunks.remainder().iter().zip(w_chunks.remainder()) {
        let keep = match policy {
            ScanPolicy::AllValues | ScanPolicy::AllFinite => true,
            ScanPolicy::SkipNan => !x.is_nan(),
            ScanPolicy::SkipNonFinite => x.is_finite(),
        };
        if keep {
            let w = w.to_f64();
            sums[0] += x.to_f64() * w;
            sum_weights[0] += w;
            counts[0] += 1;
        }
    }
    finish_weighted(
        sums.iter().sum(),
        sum_weights.iter().sum(),
        counts.iter().sum(),
    )
}

pub fn weighted_sum<T: Float, W: Weight>(
    values: &[T],
    weights: &[W],
    policy: ScanPolicy,
) -> WeightedSum {
    let mut weighted_sums = [0.0_f64; 4];
    let mut sum_weights = [0.0_f64; 4];
    let mut unweighted_sums = [0.0_f64; 4];
    let mut counts = [0usize; 4];
    let mut v_chunks = values.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (v, w) in (&mut v_chunks).zip(&mut w_chunks) {
        for lane in 0..4 {
            let x = v[lane];
            let keep = match policy {
                ScanPolicy::AllValues | ScanPolicy::AllFinite => true,
                ScanPolicy::SkipNan => !x.is_nan(),
                ScanPolicy::SkipNonFinite => x.is_finite(),
            };
            if keep {
                let x = x.to_f64();
                let w = w[lane].to_f64();
                weighted_sums[lane] += x * w;
                sum_weights[lane] += w;
                unweighted_sums[lane] += x;
                counts[lane] += 1;
            }
        }
    }
    for (&x, &w) in v_chunks.remainder().iter().zip(w_chunks.remainder()) {
        let keep = match policy {
            ScanPolicy::AllValues | ScanPolicy::AllFinite => true,
            ScanPolicy::SkipNan => !x.is_nan(),
            ScanPolicy::SkipNonFinite => x.is_finite(),
        };
        if keep {
            let x = x.to_f64();
            let w = w.to_f64();
            weighted_sums[0] += x * w;
            sum_weights[0] += w;
            unweighted_sums[0] += x;
            counts[0] += 1;
        }
    }
    WeightedSum {
        weighted_sum: weighted_sums.iter().sum(),
        sum_weights: sum_weights.iter().sum(),
        unweighted_sum: unweighted_sums.iter().sum(),
        count: counts.iter().sum(),
    }
}

fn weighted_sum_parts<T: Float, W: Weight, const WITH_WEIGHTS: bool, const WITH_UNWEIGHTED: bool>(
    values: &[T],
    weights: &[W],
    policy: ScanPolicy,
) -> (f64, f64, f64) {
    let mut weighted_sums = [0.0_f64; 4];
    let mut sum_weights = [0.0_f64; 4];
    let mut unweighted_sums = [0.0_f64; 4];
    let mut v_chunks = values.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (v, w) in (&mut v_chunks).zip(&mut w_chunks) {
        for lane in 0..4 {
            let x = v[lane];
            let keep = match policy {
                ScanPolicy::AllValues | ScanPolicy::AllFinite => true,
                ScanPolicy::SkipNan => !x.is_nan(),
                ScanPolicy::SkipNonFinite => x.is_finite(),
            };
            if keep {
                let x = x.to_f64();
                let w = w[lane].to_f64();
                weighted_sums[lane] += x * w;
                if WITH_WEIGHTS {
                    sum_weights[lane] += w;
                }
                if WITH_UNWEIGHTED {
                    unweighted_sums[lane] += x;
                }
            }
        }
    }
    for (&x, &w) in v_chunks.remainder().iter().zip(w_chunks.remainder()) {
        let keep = match policy {
            ScanPolicy::AllValues | ScanPolicy::AllFinite => true,
            ScanPolicy::SkipNan => !x.is_nan(),
            ScanPolicy::SkipNonFinite => x.is_finite(),
        };
        if keep {
            let x = x.to_f64();
            let w = w.to_f64();
            weighted_sums[0] += x * w;
            if WITH_WEIGHTS {
                sum_weights[0] += w;
            }
            if WITH_UNWEIGHTED {
                unweighted_sums[0] += x;
            }
        }
    }
    (
        weighted_sums.iter().sum(),
        if WITH_WEIGHTS {
            sum_weights.iter().sum()
        } else {
            0.0
        },
        if WITH_UNWEIGHTED {
            unweighted_sums.iter().sum()
        } else {
            0.0
        },
    )
}

pub fn weighted_sum_only<T: Float, W: Weight>(
    values: &[T],
    weights: &[W],
    policy: ScanPolicy,
) -> f64 {
    weighted_sum_parts::<T, W, false, false>(values, weights, policy).0
}

pub fn weighted_sum_and_weights<T: Float, W: Weight>(
    values: &[T],
    weights: &[W],
    policy: ScanPolicy,
) -> (f64, f64) {
    let (weighted_sum, sum_weights, _) =
        weighted_sum_parts::<T, W, true, false>(values, weights, policy);
    (weighted_sum, sum_weights)
}

pub fn weighted_sum_and_unweighted<T: Float, W: Weight>(
    values: &[T],
    weights: &[W],
    policy: ScanPolicy,
) -> (f64, f64) {
    let (weighted_sum, _, unweighted_sum) =
        weighted_sum_parts::<T, W, false, true>(values, weights, policy);
    (weighted_sum, unweighted_sum)
}

// --------------------------------------------------------------------------
// variance / std
// --------------------------------------------------------------------------

// Sum of squared deviations from `mean`, 4-wide. Two-pass variance avoids the
// catastrophic cancellation of the one-pass `sumsq - sum*mean` form (which is
// wrong for large-offset data, e.g. `[1e16, 1e16+2, 1e16+4]`), while keeping
// the SIMD-friendly lane accumulation. NaN/inf propagate through `mean`.
#[inline]
fn ss_all<T: Float>(values: &[T], mean: f64) -> f64 {
    let mut acc = [0.0_f64; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let d = chunk[lane].to_f64() - mean;
            acc[lane] += d * d;
        }
    }
    for &x in chunks.remainder() {
        let d = x.to_f64() - mean;
        acc[0] += d * d;
    }
    acc.iter().sum()
}

#[inline]
fn ss_pred<T: Float, K: Fn(T) -> bool>(values: &[T], mean: f64, keep: K) -> f64 {
    let mut acc = [0.0_f64; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let x = chunk[lane];
            if keep(x) {
                let d = x.to_f64() - mean;
                acc[lane] += d * d;
            }
        }
    }
    for &x in chunks.remainder() {
        if keep(x) {
            let d = x.to_f64() - mean;
            acc[0] += d * d;
        }
    }
    acc.iter().sum()
}

/// Returns `(variance, mean)` using a numerically stable two-pass algorithm.
pub fn variance_mean<T: Float>(values: &[T], ddof: usize, policy: ScanPolicy) -> (f64, f64) {
    let (sum, count) = sum_count(values, policy);
    if count == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = sum / count as f64;
    if count <= ddof {
        return (f64::NAN, mean);
    }
    let ss = match policy {
        ScanPolicy::AllValues | ScanPolicy::AllFinite => ss_all(values, mean),
        ScanPolicy::SkipNan => ss_pred(values, mean, |x: T| !x.is_nan()),
        ScanPolicy::SkipNonFinite => ss_pred(values, mean, |x: T| x.is_finite()),
    };
    (ss / (count - ddof) as f64, mean)
}

pub fn variance<T: Float>(values: &[T], ddof: usize, policy: ScanPolicy) -> f64 {
    variance_mean(values, ddof, policy).0
}

pub fn std<T: Float>(values: &[T], ddof: usize, policy: ScanPolicy) -> f64 {
    variance(values, ddof, policy).sqrt()
}

/// Sum over a caller-validated float slice.
///
/// This assumes the caller has already selected the intended values. It performs
/// no NaN or infinity filtering and returns `0.0` for an empty slice.
#[inline]
pub fn sum_valid<T: Float>(values: &[T]) -> f64 {
    sum(values, ScanPolicy::AllFinite)
}

/// Mean over a caller-validated float slice.
///
/// This assumes the caller has already selected the intended values. It performs
/// no NaN or infinity filtering and returns NaN for an empty slice.
#[inline]
pub fn mean_valid<T: Float>(values: &[T]) -> f64 {
    mean(values, ScanPolicy::AllFinite)
}

/// Minimum over a caller-validated float slice.
///
/// This assumes the caller has already selected the intended values. It performs
/// no NaN or infinity filtering and returns NaN for an empty slice.
#[inline]
pub fn min_valid<T: Float>(values: &[T]) -> T {
    min(values, ScanPolicy::AllFinite)
}

/// Maximum over a caller-validated float slice.
///
/// This assumes the caller has already selected the intended values. It performs
/// no NaN or infinity filtering and returns NaN for an empty slice.
#[inline]
pub fn max_valid<T: Float>(values: &[T]) -> T {
    max(values, ScanPolicy::AllFinite)
}

/// `(min, max)` over a caller-validated float slice in one pass.
///
/// This assumes the caller has already selected the intended values. It performs
/// no NaN or infinity filtering and returns `(NaN, NaN)` for an empty slice.
#[inline]
pub fn minmax_valid<T: Float>(values: &[T]) -> (T, T) {
    minmax(values, ScanPolicy::AllFinite)
}

/// Returns `(variance, mean)` over a caller-validated float slice.
#[inline]
pub fn var_mean_valid<T: Float>(values: &[T], ddof: usize) -> (f64, f64) {
    variance_mean(values, ddof, ScanPolicy::AllFinite)
}

/// Variance over a caller-validated float slice.
#[inline]
pub fn var_valid<T: Float>(values: &[T], ddof: usize) -> f64 {
    var_mean_valid(values, ddof).0
}

/// Standard deviation over a caller-validated float slice.
#[inline]
pub fn std_valid<T: Float>(values: &[T], ddof: usize) -> f64 {
    var_valid(values, ddof).sqrt()
}

/// Returns `(standard_deviation, mean)` over a caller-validated float slice.
#[inline]
pub fn std_mean_valid<T: Float>(values: &[T], ddof: usize) -> (f64, f64) {
    let (var, mean) = var_mean_valid(values, ddof);
    (var.sqrt(), mean)
}

/// Sum while skipping NaN values and retaining infinities.
#[inline]
pub fn sum_skip_nan<T: Float>(values: &[T]) -> f64 {
    sum(values, ScanPolicy::SkipNan)
}

/// Mean while skipping NaN values and retaining infinities.
#[inline]
pub fn mean_skip_nan<T: Float>(values: &[T]) -> f64 {
    mean(values, ScanPolicy::SkipNan)
}

/// Minimum while skipping NaN values and retaining infinities.
#[inline]
pub fn min_skip_nan<T: Float>(values: &[T]) -> T {
    min(values, ScanPolicy::SkipNan)
}

/// Maximum while skipping NaN values and retaining infinities.
#[inline]
pub fn max_skip_nan<T: Float>(values: &[T]) -> T {
    max(values, ScanPolicy::SkipNan)
}

/// `(min, max)` while skipping NaN values and retaining infinities.
#[inline]
pub fn minmax_skip_nan<T: Float>(values: &[T]) -> (T, T) {
    minmax(values, ScanPolicy::SkipNan)
}

/// Returns `(variance, mean)` while skipping NaN values and retaining infinities.
#[inline]
pub fn var_mean_skip_nan<T: Float>(values: &[T], ddof: usize) -> (f64, f64) {
    variance_mean(values, ddof, ScanPolicy::SkipNan)
}

/// Variance while skipping NaN values and retaining infinities.
#[inline]
pub fn var_skip_nan<T: Float>(values: &[T], ddof: usize) -> f64 {
    var_mean_skip_nan(values, ddof).0
}

/// Standard deviation while skipping NaN values and retaining infinities.
#[inline]
pub fn std_skip_nan<T: Float>(values: &[T], ddof: usize) -> f64 {
    var_skip_nan(values, ddof).sqrt()
}

/// Returns `(standard_deviation, mean)` while skipping NaN values and retaining
/// infinities.
#[inline]
pub fn std_mean_skip_nan<T: Float>(values: &[T], ddof: usize) -> (f64, f64) {
    let (var, mean) = var_mean_skip_nan(values, ddof);
    (var.sqrt(), mean)
}

/// Sum while skipping all non-finite values.
#[inline]
pub fn sum_skip_nonfinite<T: Float>(values: &[T]) -> f64 {
    sum(values, ScanPolicy::SkipNonFinite)
}

/// Mean while skipping all non-finite values.
#[inline]
pub fn mean_skip_nonfinite<T: Float>(values: &[T]) -> f64 {
    mean(values, ScanPolicy::SkipNonFinite)
}

/// Minimum while skipping all non-finite values.
#[inline]
pub fn min_skip_nonfinite<T: Float>(values: &[T]) -> T {
    min(values, ScanPolicy::SkipNonFinite)
}

/// Maximum while skipping all non-finite values.
#[inline]
pub fn max_skip_nonfinite<T: Float>(values: &[T]) -> T {
    max(values, ScanPolicy::SkipNonFinite)
}

/// `(min, max)` while skipping all non-finite values.
#[inline]
pub fn minmax_skip_nonfinite<T: Float>(values: &[T]) -> (T, T) {
    minmax(values, ScanPolicy::SkipNonFinite)
}

/// Returns `(variance, mean)` while skipping all non-finite values.
#[inline]
pub fn var_mean_skip_nonfinite<T: Float>(values: &[T], ddof: usize) -> (f64, f64) {
    variance_mean(values, ddof, ScanPolicy::SkipNonFinite)
}

/// Variance while skipping all non-finite values.
#[inline]
pub fn var_skip_nonfinite<T: Float>(values: &[T], ddof: usize) -> f64 {
    var_mean_skip_nonfinite(values, ddof).0
}

/// Standard deviation while skipping all non-finite values.
#[inline]
pub fn std_skip_nonfinite<T: Float>(values: &[T], ddof: usize) -> f64 {
    var_skip_nonfinite(values, ddof).sqrt()
}

/// Returns `(standard_deviation, mean)` while skipping all non-finite values.
#[inline]
pub fn std_mean_skip_nonfinite<T: Float>(values: &[T], ddof: usize) -> (f64, f64) {
    let (var, mean) = var_mean_skip_nonfinite(values, ddof);
    (var.sqrt(), mean)
}

// --------------------------------------------------------------------------
// min / max / minmax
// --------------------------------------------------------------------------

// All min/max slice kernels use 4 independent accumulators (`chunks_exact(4)`)
// to break the reduction dependency chain so the loop autovectorizes - the same
// trick as the sum/variance kernels.

// Plain (numpy `min`/`max`) propagate path: `min_num` ignores NaN; a separate
// branch-light `has_nan` OR records propagation. Returns `(value, has_nan)`.
#[inline]
fn chunk_min<T: Float>(values: &[T]) -> (T, bool) {
    let mut outs = [T::nan(); 4];
    let mut nans = false;
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        outs[0] = outs[0].min_num(chunk[0]);
        outs[1] = outs[1].min_num(chunk[1]);
        outs[2] = outs[2].min_num(chunk[2]);
        outs[3] = outs[3].min_num(chunk[3]);
        nans |= chunk[0].is_nan() | chunk[1].is_nan() | chunk[2].is_nan() | chunk[3].is_nan();
    }
    let mut out = outs.into_iter().reduce(|a, b| a.min_num(b)).unwrap();
    for &x in chunks.remainder() {
        out = out.min_num(x);
        nans |= x.is_nan();
    }
    (out, nans)
}

#[inline]
fn chunk_max<T: Float>(values: &[T]) -> (T, bool) {
    let mut outs = [T::nan(); 4];
    let mut nans = false;
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        outs[0] = outs[0].max_num(chunk[0]);
        outs[1] = outs[1].max_num(chunk[1]);
        outs[2] = outs[2].max_num(chunk[2]);
        outs[3] = outs[3].max_num(chunk[3]);
        nans |= chunk[0].is_nan() | chunk[1].is_nan() | chunk[2].is_nan() | chunk[3].is_nan();
    }
    let mut out = outs.into_iter().reduce(|a, b| a.max_num(b)).unwrap();
    for &x in chunks.remainder() {
        out = out.max_num(x);
        nans |= x.is_nan();
    }
    (out, nans)
}

#[inline]
fn min_plain_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [values[0]; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        if chunk[0] < outs[0] {
            outs[0] = chunk[0];
        }
        if chunk[1] < outs[1] {
            outs[1] = chunk[1];
        }
        if chunk[2] < outs[2] {
            outs[2] = chunk[2];
        }
        if chunk[3] < outs[3] {
            outs[3] = chunk[3];
        }
    }
    let mut out = outs
        .into_iter()
        .reduce(|a, b| if b < a { b } else { a })
        .unwrap();
    for &x in chunks.remainder() {
        if x < out {
            out = x;
        }
    }
    out
}

#[inline]
fn max_plain_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [values[0]; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        if chunk[0] > outs[0] {
            outs[0] = chunk[0];
        }
        if chunk[1] > outs[1] {
            outs[1] = chunk[1];
        }
        if chunk[2] > outs[2] {
            outs[2] = chunk[2];
        }
        if chunk[3] > outs[3] {
            outs[3] = chunk[3];
        }
    }
    let mut out = outs
        .into_iter()
        .reduce(|a, b| if b > a { b } else { a })
        .unwrap();
    for &x in chunks.remainder() {
        if x > out {
            out = x;
        }
    }
    out
}

#[inline]
fn nanmin_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [T::nan(); 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        outs[0] = outs[0].min_num(chunk[0]);
        outs[1] = outs[1].min_num(chunk[1]);
        outs[2] = outs[2].min_num(chunk[2]);
        outs[3] = outs[3].min_num(chunk[3]);
    }
    let mut out = outs.into_iter().reduce(|a, b| a.min_num(b)).unwrap();
    for &x in chunks.remainder() {
        out = out.min_num(x);
    }
    out
}

#[inline]
fn nanmax_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [T::nan(); 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        outs[0] = outs[0].max_num(chunk[0]);
        outs[1] = outs[1].max_num(chunk[1]);
        outs[2] = outs[2].max_num(chunk[2]);
        outs[3] = outs[3].max_num(chunk[3]);
    }
    let mut out = outs.into_iter().reduce(|a, b| a.max_num(b)).unwrap();
    for &x in chunks.remainder() {
        out = out.max_num(x);
    }
    out
}

#[inline]
fn min_finite_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [T::nan(); 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            if chunk[lane].is_finite() {
                outs[lane] = outs[lane].min_num(chunk[lane]);
            }
        }
    }
    let mut out = outs.into_iter().reduce(|a, b| a.min_num(b)).unwrap();
    for &x in chunks.remainder() {
        if x.is_finite() {
            out = out.min_num(x);
        }
    }
    out
}

#[inline]
fn max_finite_slice<T: Float>(values: &[T]) -> T {
    let mut outs = [T::nan(); 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            if chunk[lane].is_finite() {
                outs[lane] = outs[lane].max_num(chunk[lane]);
            }
        }
    }
    let mut out = outs.into_iter().reduce(|a, b| a.max_num(b)).unwrap();
    for &x in chunks.remainder() {
        if x.is_finite() {
            out = out.max_num(x);
        }
    }
    out
}

#[inline]
fn minmax_chunk<T: Float>(values: &[T]) -> (T, T, bool) {
    let mut los = [T::nan(); 4];
    let mut his = [T::nan(); 4];
    let mut has_nan = false;
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let x = chunk[lane];
            los[lane] = los[lane].min_num(x);
            his[lane] = his[lane].max_num(x);
            has_nan |= x.is_nan();
        }
    }
    let mut lo = los.into_iter().reduce(|a, b| a.min_num(b)).unwrap();
    let mut hi = his.into_iter().reduce(|a, b| a.max_num(b)).unwrap();
    for &x in chunks.remainder() {
        lo = lo.min_num(x);
        hi = hi.max_num(x);
        has_nan |= x.is_nan();
    }
    (lo, hi, has_nan)
}

#[inline]
fn minmax_plain_slice<T: Float>(values: &[T]) -> (T, T) {
    let mut los = [values[0]; 4];
    let mut his = [values[0]; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let x = chunk[lane];
            if x < los[lane] {
                los[lane] = x;
            }
            if x > his[lane] {
                his[lane] = x;
            }
        }
    }
    let mut lo = los
        .into_iter()
        .reduce(|a, b| if b < a { b } else { a })
        .unwrap();
    let mut hi = his
        .into_iter()
        .reduce(|a, b| if b > a { b } else { a })
        .unwrap();
    for &x in chunks.remainder() {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    (lo, hi)
}

#[inline]
fn minmax_finite_slice<T: Float>(values: &[T]) -> (T, T) {
    let mut los = [T::nan(); 4];
    let mut his = [T::nan(); 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let x = chunk[lane];
            if x.is_finite() {
                los[lane] = los[lane].min_num(x);
                his[lane] = his[lane].max_num(x);
            }
        }
    }
    let mut lo = los.into_iter().reduce(|a, b| a.min_num(b)).unwrap();
    let mut hi = his.into_iter().reduce(|a, b| a.max_num(b)).unwrap();
    for &x in chunks.remainder() {
        if x.is_finite() {
            lo = lo.min_num(x);
            hi = hi.max_num(x);
        }
    }
    (lo, hi)
}

pub fn min<T: Float>(values: &[T], policy: ScanPolicy) -> T {
    if values.is_empty() {
        return T::nan();
    }
    let chunks = minmax_1d_parallel_chunks(values.len());
    let chunk_len = values.len().div_ceil(chunks);
    match policy {
        ScanPolicy::AllValues => {
            let (m, has_nan) = if chunks > 1 {
                values.par_chunks(chunk_len).map(chunk_min).reduce(
                    || (T::nan(), false),
                    |(a, an), (b, bn)| (a.min_num(b), an | bn),
                )
            } else {
                chunk_min(values)
            };
            if has_nan {
                T::nan()
            } else {
                m
            }
        }
        ScanPolicy::AllFinite => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(min_plain_slice)
                    .reduce(|| values[0], |a, b| if b < a { b } else { a })
            } else {
                min_plain_slice(values)
            }
        }
        ScanPolicy::SkipNan => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(nanmin_slice)
                    .reduce(T::nan, |a, b| a.min_num(b))
            } else {
                nanmin_slice(values)
            }
        }
        ScanPolicy::SkipNonFinite => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(min_finite_slice)
                    .reduce(T::nan, |a, b| a.min_num(b))
            } else {
                min_finite_slice(values)
            }
        }
    }
}

pub fn max<T: Float>(values: &[T], policy: ScanPolicy) -> T {
    if values.is_empty() {
        return T::nan();
    }
    let chunks = minmax_1d_parallel_chunks(values.len());
    let chunk_len = values.len().div_ceil(chunks);
    match policy {
        ScanPolicy::AllValues => {
            let (m, has_nan) = if chunks > 1 {
                values.par_chunks(chunk_len).map(chunk_max).reduce(
                    || (T::nan(), false),
                    |(a, an), (b, bn)| (a.max_num(b), an | bn),
                )
            } else {
                chunk_max(values)
            };
            if has_nan {
                T::nan()
            } else {
                m
            }
        }
        ScanPolicy::AllFinite => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(max_plain_slice)
                    .reduce(|| values[0], |a, b| if b > a { b } else { a })
            } else {
                max_plain_slice(values)
            }
        }
        ScanPolicy::SkipNan => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(nanmax_slice)
                    .reduce(T::nan, |a, b| a.max_num(b))
            } else {
                nanmax_slice(values)
            }
        }
        ScanPolicy::SkipNonFinite => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(max_finite_slice)
                    .reduce(T::nan, |a, b| a.max_num(b))
            } else {
                max_finite_slice(values)
            }
        }
    }
}

/// `(min, max)` in a single pass.
pub fn minmax<T: Float>(values: &[T], policy: ScanPolicy) -> (T, T) {
    if values.is_empty() {
        return (T::nan(), T::nan());
    }
    let chunks = minmax_1d_parallel_chunks(values.len());
    let chunk_len = values.len().div_ceil(chunks);
    match policy {
        ScanPolicy::AllValues => {
            let (lo, hi, has_nan) = if chunks > 1 {
                values.par_chunks(chunk_len).map(minmax_chunk).reduce(
                    || (T::nan(), T::nan(), false),
                    |(alo, ahi, an), (blo, bhi, bn)| (alo.min_num(blo), ahi.max_num(bhi), an | bn),
                )
            } else {
                minmax_chunk(values)
            };
            if has_nan {
                (T::nan(), T::nan())
            } else {
                (lo, hi)
            }
        }
        ScanPolicy::AllFinite => {
            if chunks > 1 {
                values.par_chunks(chunk_len).map(minmax_plain_slice).reduce(
                    || (values[0], values[0]),
                    |(alo, ahi), (blo, bhi)| {
                        (
                            if blo < alo { blo } else { alo },
                            if bhi > ahi { bhi } else { ahi },
                        )
                    },
                )
            } else {
                minmax_plain_slice(values)
            }
        }
        ScanPolicy::SkipNan => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(|chunk| {
                        let (lo, hi, _) = minmax_chunk(chunk);
                        (lo, hi)
                    })
                    .reduce(
                        || (T::nan(), T::nan()),
                        |(alo, ahi), (blo, bhi)| (alo.min_num(blo), ahi.max_num(bhi)),
                    )
            } else {
                let (lo, hi, _) = minmax_chunk(values);
                (lo, hi)
            }
        }
        ScanPolicy::SkipNonFinite => {
            if chunks > 1 {
                values
                    .par_chunks(chunk_len)
                    .map(minmax_finite_slice)
                    .reduce(
                        || (T::nan(), T::nan()),
                        |(alo, ahi), (blo, bhi)| (alo.min_num(blo), ahi.max_num(bhi)),
                    )
            } else {
                minmax_finite_slice(values)
            }
        }
    }
}

// --------------------------------------------------------------------------
// order statistics: median / lmedian / percentiles
// --------------------------------------------------------------------------

/// Reorder `buf` in place so the values to operate on occupy `buf[..count]`,
/// returning `count`. `None` means the result must be NaN (plain `AllValues`
/// policy with a NaN present - numpy propagation).
fn retain_for_order<T: Float>(buf: &mut [T], policy: ScanPolicy) -> Option<usize> {
    match policy {
        ScanPolicy::AllValues => {
            if buf.iter().any(|x| x.is_nan()) {
                None
            } else {
                Some(buf.len())
            }
        }
        ScanPolicy::AllFinite => Some(buf.len()),
        ScanPolicy::SkipNan => {
            let mut w = 0;
            for r in 0..buf.len() {
                if !buf[r].is_nan() {
                    buf[w] = buf[r];
                    w += 1;
                }
            }
            Some(w)
        }
        ScanPolicy::SkipNonFinite => {
            let mut w = 0;
            for r in 0..buf.len() {
                if buf[r].is_finite() {
                    buf[w] = buf[r];
                    w += 1;
                }
            }
            Some(w)
        }
    }
}

/// Median operating in place on an already valid buffer.
///
/// This low-level primitive assumes the caller has already compacted the
/// intended values into `buf` and that `buf` contains no NaN values. It does no
/// validation or filtering, may reorder `buf`, and returns NaN for an empty
/// buffer.
#[inline]
pub fn median_valid_in_place<T: Float>(buf: &mut [T]) -> f64 {
    if buf.is_empty() {
        return f64::NAN;
    }
    let mid = buf.len() / 2;
    if buf.len() % 2 == 1 {
        let (_, value, _) = buf.select_nth_unstable_by(mid, cmp_float);
        value.to_f64()
    } else {
        let (_, upper, _) = buf.select_nth_unstable_by(mid, cmp_float);
        let upper = upper.to_f64();
        let lower = buf[..mid]
            .iter()
            .copied()
            .max_by(cmp_float)
            .expect("even median lower partition is non-empty")
            .to_f64();
        (lower + upper) / 2.0
    }
}

/// Lower median operating in place on an already valid buffer.
///
/// This low-level primitive assumes the caller has already compacted the
/// intended values into `buf` and that `buf` contains no NaN values. It does no
/// validation or filtering, may reorder `buf`, and returns NaN for an empty
/// buffer.
#[inline]
pub fn lmedian_valid_in_place<T: Float>(buf: &mut [T]) -> f64 {
    if buf.is_empty() {
        return f64::NAN;
    }
    let idx = (buf.len() - 1) / 2;
    let (_, value, _) = buf.select_nth_unstable_by(idx, cmp_float);
    value.to_f64()
}

/// Median operating in place on `buf` (reordered). Used by axis adapters.
pub fn median_in_place<T: Float>(buf: &mut [T], policy: ScanPolicy) -> f64 {
    match retain_for_order(buf, policy) {
        None => f64::NAN,
        Some(count) => median_valid_in_place(&mut buf[..count]),
    }
}

pub fn median<T: Float>(values: &[T], policy: ScanPolicy) -> f64 {
    let mut buf = values.to_vec();
    median_in_place(&mut buf, policy)
}

/// Lower value-selecting median operating in place on `buf`.
pub fn lmedian_in_place<T: Float>(buf: &mut [T], policy: ScanPolicy) -> f64 {
    match retain_for_order(buf, policy) {
        None => f64::NAN,
        Some(count) => lmedian_valid_in_place(&mut buf[..count]),
    }
}

pub fn lmedian<T: Float>(values: &[T], policy: ScanPolicy) -> f64 {
    let mut buf = values.to_vec();
    lmedian_in_place(&mut buf, policy)
}

#[inline]
fn percentile_rank(count: usize, q: f64) -> (usize, usize, f64) {
    let rank = (q / 100.0) * (count - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    (lower, upper, rank - lower as f64)
}

#[inline]
fn selection_rank_budget(count: usize) -> usize {
    usize::BITS as usize - count.leading_zeros() as usize - 1
}

/// numpy-`linear` interpolation percentile operating in place on an already
/// valid buffer.
///
/// `q` is in `[0, 100]`. This low-level primitive assumes the caller has
/// already compacted the intended values into `buf` and that `buf` contains no
/// NaN values. It does no validation or filtering, may reorder `buf`, allocates
/// no heap memory, and returns NaN for an empty buffer.
pub fn percentile_valid_in_place<T: Float>(buf: &mut [T], q: f64) -> f64 {
    if buf.is_empty() {
        return f64::NAN;
    }
    let (lower, upper, frac) = percentile_rank(buf.len(), q);
    let (_, lo_value, _) = buf.select_nth_unstable_by(lower, cmp_float);
    let lo = lo_value.to_f64();
    if lower == upper {
        return lo;
    }
    let (_, hi_value, _) = buf[lower + 1..].select_nth_unstable_by(upper - lower - 1, cmp_float);
    let hi = hi_value.to_f64();
    lo + (hi - lo) * frac
}

/// numpy-`linear` interpolation quantile operating in place on an already
/// valid buffer.
///
/// `q` is in `[0, 1]`. This is equivalent to
/// ``percentile_valid_in_place(buf, q * 100.0)`` and has the same caller
/// contract.
#[inline]
pub fn quantile_valid_in_place<T: Float>(buf: &mut [T], q: f64) -> f64 {
    percentile_valid_in_place(buf, q * 100.0)
}

/// numpy-`linear` interpolation percentiles into `out`, operating in place on
/// `buf` after applying `policy`.
///
/// `qs` are in `[0, 100]`. This helper may reorder `buf` and may allocate work
/// vectors proportional to `qs.len()`.
pub fn percentiles_in_place<T: Float>(
    buf: &mut [T],
    qs: &[f64],
    out: &mut [f64],
    policy: ScanPolicy,
) {
    out.fill(f64::NAN);
    let count = match retain_for_order(buf, policy) {
        None => return,
        Some(0) => return,
        Some(count) => count,
    };
    let buf = &mut buf[..count];
    percentiles_valid_in_place(buf, qs, out);
}

/// numpy-`linear` interpolation percentiles into `out`, operating in place on
/// an already valid buffer.
///
/// `qs` are in `[0, 100]`. This low-level primitive assumes the caller has
/// already compacted the intended values into `buf` and that `buf` contains no
/// NaN values. It does no validation or filtering and may reorder `buf`. The
/// scalar [`percentile_valid_in_place`] primitive allocates no heap memory; this
/// multi-rank helper may allocate work vectors proportional to `qs.len()`.
pub fn percentiles_valid_in_place<T: Float>(buf: &mut [T], qs: &[f64], out: &mut [f64]) {
    out.fill(f64::NAN);
    if buf.is_empty() {
        return;
    }
    if qs.len() == 1 && out.len() == 1 {
        out[0] = percentile_valid_in_place(buf, qs[0]);
        return;
    }
    let ranks: Vec<(usize, usize, f64)> =
        qs.iter().map(|&q| percentile_rank(buf.len(), q)).collect();
    let mut needed: Vec<usize> = ranks.iter().flat_map(|&(l, u, _)| [l, u]).collect();
    needed.sort_unstable();
    needed.dedup();

    if needed.len() <= selection_rank_budget(buf.len()) {
        let mut selected = Vec::<f64>::with_capacity(needed.len());
        let mut start = 0usize;
        for &idx in &needed {
            let (_, value, _) = buf[start..].select_nth_unstable_by(idx - start, cmp_float);
            selected.push(value.to_f64());
            start = idx + 1;
        }
        for (dst, &(lower, upper, frac)) in out.iter_mut().zip(ranks.iter()) {
            let lo = selected[needed.binary_search(&lower).unwrap()];
            let hi = selected[needed.binary_search(&upper).unwrap()];
            *dst = lo + (hi - lo) * frac;
        }
    } else {
        buf.sort_unstable_by(cmp_float);
        for (dst, &(lower, upper, frac)) in out.iter_mut().zip(ranks.iter()) {
            let lo = buf[lower].to_f64();
            let hi = buf[upper].to_f64();
            *dst = lo + (hi - lo) * frac;
        }
    }
}

/// numpy-`linear` interpolation percentiles. `qs` in `[0, 100]`.
pub fn percentiles<T: Float>(values: &[T], qs: &[f64], policy: ScanPolicy) -> Vec<f64> {
    let mut buf = values.to_vec();
    let mut out = vec![f64::NAN; qs.len()];
    percentiles_in_place(&mut buf, qs, &mut out, policy);
    out
}

// --------------------------------------------------------------------------
// count_finite
// --------------------------------------------------------------------------

pub fn count_finite<T: Float>(values: &[T]) -> usize {
    values.iter().filter(|x| x.is_finite()).count()
}

// --------------------------------------------------------------------------
// non-floating numeric kernels
// --------------------------------------------------------------------------

#[inline]
fn number_sum_count<T: Number>(values: &[T]) -> (f64, usize) {
    let mut sums = [0.0_f64; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        sums[0] += chunk[0].to_f64();
        sums[1] += chunk[1].to_f64();
        sums[2] += chunk[2].to_f64();
        sums[3] += chunk[3].to_f64();
    }
    for &x in chunks.remainder() {
        sums[0] += x.to_f64();
    }
    (sums.iter().sum::<f64>(), values.len())
}

pub fn number_mean<T: Number>(values: &[T]) -> f64 {
    let (sum, count) = number_sum_count(values);
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

pub fn number_sum<T: Number>(values: &[T]) -> f64 {
    number_sum_count(values).0
}

pub fn number_weighted_average<T: Number, W: Weight>(values: &[T], weights: &[W]) -> WeightedMean {
    let mut sums = [0.0_f64; 4];
    let mut sum_weights = [0.0_f64; 4];
    let mut v_chunks = values.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (v, w) in (&mut v_chunks).zip(&mut w_chunks) {
        sums[0] += v[0].to_f64() * w[0].to_f64();
        sums[1] += v[1].to_f64() * w[1].to_f64();
        sums[2] += v[2].to_f64() * w[2].to_f64();
        sums[3] += v[3].to_f64() * w[3].to_f64();
        sum_weights[0] += w[0].to_f64();
        sum_weights[1] += w[1].to_f64();
        sum_weights[2] += w[2].to_f64();
        sum_weights[3] += w[3].to_f64();
    }
    for (&x, &w) in v_chunks.remainder().iter().zip(w_chunks.remainder()) {
        let w = w.to_f64();
        sums[0] += x.to_f64() * w;
        sum_weights[0] += w;
    }
    finish_weighted(sums.iter().sum(), sum_weights.iter().sum(), values.len())
}

pub fn number_weighted_sum<T: Number, W: Weight>(values: &[T], weights: &[W]) -> WeightedSum {
    let mut weighted_sums = [0.0_f64; 4];
    let mut sum_weights = [0.0_f64; 4];
    let mut unweighted_sums = [0.0_f64; 4];
    let mut v_chunks = values.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (v, w) in (&mut v_chunks).zip(&mut w_chunks) {
        let x0 = v[0].to_f64();
        let x1 = v[1].to_f64();
        let x2 = v[2].to_f64();
        let x3 = v[3].to_f64();
        let w0 = w[0].to_f64();
        let w1 = w[1].to_f64();
        let w2 = w[2].to_f64();
        let w3 = w[3].to_f64();
        weighted_sums[0] += x0 * w0;
        weighted_sums[1] += x1 * w1;
        weighted_sums[2] += x2 * w2;
        weighted_sums[3] += x3 * w3;
        sum_weights[0] += w0;
        sum_weights[1] += w1;
        sum_weights[2] += w2;
        sum_weights[3] += w3;
        unweighted_sums[0] += x0;
        unweighted_sums[1] += x1;
        unweighted_sums[2] += x2;
        unweighted_sums[3] += x3;
    }
    for (&x, &w) in v_chunks.remainder().iter().zip(w_chunks.remainder()) {
        let x = x.to_f64();
        let w = w.to_f64();
        weighted_sums[0] += x * w;
        sum_weights[0] += w;
        unweighted_sums[0] += x;
    }
    WeightedSum {
        weighted_sum: weighted_sums.iter().sum(),
        sum_weights: sum_weights.iter().sum(),
        unweighted_sum: unweighted_sums.iter().sum(),
        count: values.len(),
    }
}

fn number_weighted_sum_parts<
    T: Number,
    W: Weight,
    const WITH_WEIGHTS: bool,
    const WITH_UNWEIGHTED: bool,
>(
    values: &[T],
    weights: &[W],
) -> (f64, f64, f64) {
    let mut weighted_sums = [0.0_f64; 4];
    let mut sum_weights = [0.0_f64; 4];
    let mut unweighted_sums = [0.0_f64; 4];
    let mut v_chunks = values.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (v, w) in (&mut v_chunks).zip(&mut w_chunks) {
        for lane in 0..4 {
            let x = v[lane].to_f64();
            let w = w[lane].to_f64();
            weighted_sums[lane] += x * w;
            if WITH_WEIGHTS {
                sum_weights[lane] += w;
            }
            if WITH_UNWEIGHTED {
                unweighted_sums[lane] += x;
            }
        }
    }
    for (&x, &w) in v_chunks.remainder().iter().zip(w_chunks.remainder()) {
        let x = x.to_f64();
        let w = w.to_f64();
        weighted_sums[0] += x * w;
        if WITH_WEIGHTS {
            sum_weights[0] += w;
        }
        if WITH_UNWEIGHTED {
            unweighted_sums[0] += x;
        }
    }
    (
        weighted_sums.iter().sum(),
        if WITH_WEIGHTS {
            sum_weights.iter().sum()
        } else {
            0.0
        },
        if WITH_UNWEIGHTED {
            unweighted_sums.iter().sum()
        } else {
            0.0
        },
    )
}

pub fn number_weighted_sum_only<T: Number, W: Weight>(values: &[T], weights: &[W]) -> f64 {
    number_weighted_sum_parts::<T, W, false, false>(values, weights).0
}

pub fn number_weighted_sum_and_weights<T: Number, W: Weight>(
    values: &[T],
    weights: &[W],
) -> (f64, f64) {
    let (weighted_sum, sum_weights, _) =
        number_weighted_sum_parts::<T, W, true, false>(values, weights);
    (weighted_sum, sum_weights)
}

pub fn number_weighted_sum_and_unweighted<T: Number, W: Weight>(
    values: &[T],
    weights: &[W],
) -> (f64, f64) {
    let (weighted_sum, _, unweighted_sum) =
        number_weighted_sum_parts::<T, W, false, true>(values, weights);
    (weighted_sum, unweighted_sum)
}

#[inline]
fn number_ss<T: Number>(values: &[T], mean: f64) -> f64 {
    let mut acc = [0.0_f64; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let d = chunk[lane].to_f64() - mean;
            acc[lane] += d * d;
        }
    }
    for &x in chunks.remainder() {
        let d = x.to_f64() - mean;
        acc[0] += d * d;
    }
    acc.iter().sum()
}

pub fn number_variance_mean<T: Number>(values: &[T], ddof: usize) -> (f64, f64) {
    let (sum, count) = number_sum_count(values);
    if count == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = sum / count as f64;
    if count <= ddof {
        return (f64::NAN, mean);
    }
    (number_ss(values, mean) / (count - ddof) as f64, mean)
}

pub fn number_variance<T: Number>(values: &[T], ddof: usize) -> f64 {
    number_variance_mean(values, ddof).0
}

pub fn number_std<T: Number>(values: &[T], ddof: usize) -> f64 {
    number_variance(values, ddof).sqrt()
}

pub fn number_min<T: Number>(values: &[T]) -> f64 {
    number_min_value(values).map_or(f64::NAN, Number::to_f64)
}

pub fn number_min_value<T: Number>(values: &[T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    let mut outs = [values[0]; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            if chunk[lane] < outs[lane] {
                outs[lane] = chunk[lane];
            }
        }
    }
    let mut out = outs
        .into_iter()
        .reduce(|a, b| if b < a { b } else { a })
        .unwrap();
    for &x in chunks.remainder() {
        if x < out {
            out = x;
        }
    }
    Some(out)
}

pub fn number_max<T: Number>(values: &[T]) -> f64 {
    number_max_value(values).map_or(f64::NAN, Number::to_f64)
}

pub fn number_max_value<T: Number>(values: &[T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    let mut outs = [values[0]; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            if chunk[lane] > outs[lane] {
                outs[lane] = chunk[lane];
            }
        }
    }
    let mut out = outs
        .into_iter()
        .reduce(|a, b| if b > a { b } else { a })
        .unwrap();
    for &x in chunks.remainder() {
        if x > out {
            out = x;
        }
    }
    Some(out)
}

pub fn number_minmax<T: Number>(values: &[T]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut los = [values[0]; 4];
    let mut his = [values[0]; 4];
    let mut chunks = values.chunks_exact(4);
    for chunk in &mut chunks {
        for lane in 0..4 {
            let x = chunk[lane];
            if x < los[lane] {
                los[lane] = x;
            }
            if x > his[lane] {
                his[lane] = x;
            }
        }
    }
    let mut lo = los
        .into_iter()
        .reduce(|a, b| if b < a { b } else { a })
        .unwrap();
    let mut hi = his
        .into_iter()
        .reduce(|a, b| if b > a { b } else { a })
        .unwrap();
    for &x in chunks.remainder() {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    (lo.to_f64(), hi.to_f64())
}

#[inline]
fn number_median_of<T: Number>(buf: &mut [T]) -> f64 {
    if buf.is_empty() {
        return f64::NAN;
    }
    let mid = buf.len() / 2;
    if buf.len() % 2 == 1 {
        let (_, value, _) = buf.select_nth_unstable(mid);
        value.to_f64()
    } else {
        let (_, upper, _) = buf.select_nth_unstable(mid);
        let upper = upper.to_f64();
        let lower = buf[..mid]
            .iter()
            .copied()
            .max()
            .expect("even median lower partition is non-empty")
            .to_f64();
        (lower + upper) / 2.0
    }
}

/// Median operating in place on an already valid non-floating buffer.
///
/// This low-level primitive assumes `buf` contains exactly the intended values.
/// It does no validation, may reorder `buf`, and returns NaN for an empty
/// buffer.
#[inline]
pub fn number_median_valid_in_place<T: Number>(buf: &mut [T]) -> f64 {
    number_median_of(buf)
}

pub fn number_median_in_place<T: Number>(buf: &mut [T]) -> f64 {
    number_median_valid_in_place(buf)
}

pub fn number_median<T: Number>(values: &[T]) -> f64 {
    let mut buf = values.to_vec();
    number_median_in_place(&mut buf)
}

/// Lower median operating in place on an already valid non-floating buffer.
///
/// This low-level primitive assumes `buf` contains exactly the intended values.
/// It does no validation, may reorder `buf`, and returns NaN for an empty
/// buffer.
#[inline]
pub fn number_lmedian_valid_in_place<T: Number>(buf: &mut [T]) -> f64 {
    number_lmedian_value_valid_in_place(buf).map_or(f64::NAN, Number::to_f64)
}

pub fn number_lmedian_in_place<T: Number>(buf: &mut [T]) -> f64 {
    number_lmedian_valid_in_place(buf)
}

/// Lower median selected value operating in place on an already valid
/// non-floating buffer.
///
/// This low-level primitive assumes `buf` contains exactly the intended values.
/// It does no validation, may reorder `buf`, and returns `None` for an empty
/// buffer.
#[inline]
pub fn number_lmedian_value_valid_in_place<T: Number>(buf: &mut [T]) -> Option<T> {
    if buf.is_empty() {
        return None;
    }
    let idx = (buf.len() - 1) / 2;
    let (_, value, _) = buf.select_nth_unstable(idx);
    Some(*value)
}

pub fn number_lmedian_value_in_place<T: Number>(buf: &mut [T]) -> Option<T> {
    number_lmedian_value_valid_in_place(buf)
}

pub fn number_lmedian<T: Number>(values: &[T]) -> f64 {
    let mut buf = values.to_vec();
    number_lmedian_in_place(&mut buf)
}

/// numpy-`linear` interpolation percentile operating in place on an already
/// valid non-floating buffer.
///
/// `q` is in `[0, 100]`. This low-level primitive assumes `buf` contains
/// exactly the intended values. It does no validation, may reorder `buf`,
/// allocates no heap memory, and returns NaN for an empty buffer.
pub fn number_percentile_valid_in_place<T: Number>(buf: &mut [T], q: f64) -> f64 {
    if buf.is_empty() {
        return f64::NAN;
    }
    let (lower, upper, frac) = percentile_rank(buf.len(), q);
    let (_, lo_value, _) = buf.select_nth_unstable(lower);
    let lo = lo_value.to_f64();
    if lower == upper {
        return lo;
    }
    let (_, hi_value, _) = buf[lower + 1..].select_nth_unstable(upper - lower - 1);
    let hi = hi_value.to_f64();
    lo + (hi - lo) * frac
}

/// numpy-`linear` interpolation quantile operating in place on an already valid
/// non-floating buffer.
///
/// `q` is in `[0, 1]`. This is equivalent to
/// ``number_percentile_valid_in_place(buf, q * 100.0)`` and has the same caller
/// contract.
#[inline]
pub fn number_quantile_valid_in_place<T: Number>(buf: &mut [T], q: f64) -> f64 {
    number_percentile_valid_in_place(buf, q * 100.0)
}

/// numpy-`linear` interpolation percentiles into `out`, operating in place on
/// an already valid non-floating buffer.
///
/// `qs` are in `[0, 100]`. This low-level primitive assumes `buf` contains
/// exactly the intended values. It does no validation or filtering and may
/// reorder `buf`. The scalar [`number_percentile_valid_in_place`] primitive
/// allocates no heap memory; this multi-rank helper may allocate work vectors
/// proportional to `qs.len()`.
pub fn number_percentiles_valid_in_place<T: Number>(buf: &mut [T], qs: &[f64], out: &mut [f64]) {
    out.fill(f64::NAN);
    if buf.is_empty() {
        return;
    }
    if qs.len() == 1 && out.len() == 1 {
        out[0] = number_percentile_valid_in_place(buf, qs[0]);
        return;
    }
    let ranks: Vec<(usize, usize, f64)> =
        qs.iter().map(|&q| percentile_rank(buf.len(), q)).collect();
    let mut needed: Vec<usize> = ranks.iter().flat_map(|&(l, u, _)| [l, u]).collect();
    needed.sort_unstable();
    needed.dedup();

    if needed.len() <= selection_rank_budget(buf.len()) {
        let mut selected = Vec::<f64>::with_capacity(needed.len());
        let mut start = 0usize;
        for &idx in &needed {
            let (_, value, _) = buf[start..].select_nth_unstable(idx - start);
            selected.push(value.to_f64());
            start = idx + 1;
        }
        for (dst, &(lower, upper, frac)) in out.iter_mut().zip(ranks.iter()) {
            let lo = selected[needed.binary_search(&lower).unwrap()];
            let hi = selected[needed.binary_search(&upper).unwrap()];
            *dst = lo + (hi - lo) * frac;
        }
    } else {
        buf.sort_unstable();
        for (dst, &(lower, upper, frac)) in out.iter_mut().zip(ranks.iter()) {
            let lo = buf[lower].to_f64();
            let hi = buf[upper].to_f64();
            *dst = lo + (hi - lo) * frac;
        }
    }
}

pub fn number_percentiles_in_place<T: Number>(buf: &mut [T], qs: &[f64], out: &mut [f64]) {
    number_percentiles_valid_in_place(buf, qs, out);
}

pub fn number_percentiles<T: Number>(values: &[T], qs: &[f64]) -> Vec<f64> {
    let mut buf = values.to_vec();
    let mut out = vec![f64::NAN; qs.len()];
    number_percentiles_in_place(&mut buf, qs, &mut out);
    out
}

pub fn number_count_finite<T: Number>(values: &[T]) -> usize {
    values.len()
}

#[inline]
pub fn apply_number<T: Number>(kind: Kind, s: &[T], ddof: usize) -> f64 {
    match kind {
        Kind::Mean => number_mean(s),
        Kind::Sum => number_sum(s),
        Kind::Min => number_min(s),
        Kind::Max => number_max(s),
        Kind::Var => number_variance(s, ddof),
        Kind::Std => number_std(s, ddof),
        Kind::CountFinite => number_count_finite(s) as f64,
        Kind::Median => number_median(s),
        Kind::LMedian => number_lmedian(s),
    }
}

#[inline]
pub fn apply_number_mut<T: Number>(kind: Kind, buf: &mut [T], ddof: usize) -> f64 {
    match kind {
        Kind::Median => number_median_in_place(buf),
        Kind::LMedian => number_lmedian_in_place(buf),
        _ => apply_number(kind, buf, ddof),
    }
}

// --------------------------------------------------------------------------
// axis dispatch helpers (used by the `axis` module and the PyO3 layer)
// --------------------------------------------------------------------------

/// Scalar reducer kind, for axis dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mean,
    Sum,
    Min,
    Max,
    Median,
    LMedian,
    Var,
    Std,
    CountFinite,
}

impl Kind {
    /// Code used at the PyO3 boundary.
    #[inline]
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Mean,
            1 => Self::Sum,
            2 => Self::Min,
            3 => Self::Max,
            4 => Self::Median,
            5 => Self::LMedian,
            6 => Self::Var,
            7 => Self::Std,
            8 => Self::CountFinite,
            other => panic!("invalid Kind code: {other}"),
        }
    }

    /// Does this kind reorder its input buffer (order statistic)?
    #[inline]
    pub fn needs_mut(self) -> bool {
        matches!(self, Self::Median | Self::LMedian)
    }
}

/// Apply a read-only scalar reducer to a contiguous slice.
#[inline]
pub fn apply<T: Float>(kind: Kind, s: &[T], ddof: usize, policy: ScanPolicy) -> T {
    match kind {
        Kind::Mean => T::from_f64(mean(s, policy)),
        Kind::Sum => T::from_f64(sum(s, policy)),
        Kind::Min => min(s, policy),
        Kind::Max => max(s, policy),
        Kind::Var => T::from_f64(variance(s, ddof, policy)),
        Kind::Std => T::from_f64(std(s, ddof, policy)),
        Kind::CountFinite => T::from_f64(count_finite(s) as f64),
        Kind::Median => T::from_f64(median(s, policy)),
        Kind::LMedian => T::from_f64(lmedian(s, policy)),
    }
}

/// Apply a scalar reducer to a scratch buffer it may reorder (no allocation).
#[inline]
pub fn apply_mut<T: Float>(kind: Kind, buf: &mut [T], ddof: usize, policy: ScanPolicy) -> T {
    match kind {
        Kind::Median => T::from_f64(median_in_place(buf, policy)),
        Kind::LMedian => T::from_f64(lmedian_in_place(buf, policy)),
        _ => apply(kind, buf, ddof, policy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finite::ScanPolicy::*;

    #[test]
    fn mean_policies() {
        let v = [1.0_f64, 2.0, f64::NAN, f64::INFINITY, 5.0];
        assert!(mean(&v, AllValues).is_nan()); // propagate NaN
        assert_eq!(mean(&v, SkipNan), f64::INFINITY); // keep inf
        assert_eq!(mean(&v, SkipNonFinite), 8.0 / 3.0); // finite only
    }

    #[test]
    fn valid_scan_primitives_use_all_values_without_filtering() {
        let v = [1.0_f64, 2.0, 3.0, f64::INFINITY];
        assert_eq!(sum_valid(&v), f64::INFINITY);
        assert_eq!(mean_valid(&v), f64::INFINITY);
        assert_eq!(min_valid(&v), 1.0);
        assert_eq!(max_valid(&v), f64::INFINITY);
        assert_eq!(minmax_valid(&v), (1.0, f64::INFINITY));

        let finite = [1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(var_mean_valid(&finite, 1), (5.0 / 3.0, 2.5));
        assert_eq!(var_valid(&finite, 0), 1.25);
        assert_eq!(std_valid(&finite, 0), 1.25_f64.sqrt());
        assert_eq!(std_mean_valid(&finite, 0), (1.25_f64.sqrt(), 2.5));
    }

    #[test]
    fn skip_nonfinite_scan_primitives_drop_nan_and_inf() {
        let v = [1.0_f64, f64::NAN, 2.0, f64::INFINITY, 4.0];
        assert_eq!(sum_skip_nonfinite(&v), 7.0);
        assert_eq!(mean_skip_nonfinite(&v), 7.0 / 3.0);
        assert_eq!(min_skip_nonfinite(&v), 1.0);
        assert_eq!(max_skip_nonfinite(&v), 4.0);
        assert_eq!(minmax_skip_nonfinite(&v), (1.0, 4.0));
        assert_eq!(var_mean_skip_nonfinite(&v, 0), (14.0 / 9.0, 7.0 / 3.0));
        assert_eq!(
            std_mean_skip_nonfinite(&v, 0),
            ((14.0_f64 / 9.0).sqrt(), 7.0 / 3.0)
        );
    }

    #[test]
    fn skip_nan_scan_primitives_keep_inf() {
        let v = [1.0_f64, f64::NAN, 2.0, f64::INFINITY];
        assert_eq!(sum_skip_nan(&v), f64::INFINITY);
        assert_eq!(mean_skip_nan(&v), f64::INFINITY);
        assert_eq!(min_skip_nan(&v), 1.0);
        assert_eq!(max_skip_nan(&v), f64::INFINITY);
        assert_eq!(minmax_skip_nan(&v), (1.0, f64::INFINITY));
    }

    #[test]
    fn scan_primitives_document_empty_and_ddof_behavior() {
        let empty: [f64; 0] = [];
        assert_eq!(sum_valid(&empty), 0.0);
        assert!(mean_valid(&empty).is_nan());
        assert!(min_valid(&empty).is_nan());
        assert!(max_valid(&empty).is_nan());
        assert!(var_valid(&empty, 0).is_nan());
        assert_eq!(sum_skip_nonfinite(&[f64::NAN, f64::INFINITY]), 0.0);
        assert!(mean_skip_nonfinite(&[f64::NAN, f64::INFINITY]).is_nan());

        let single = [2.0_f64];
        let (var, mean) = var_mean_valid(&single, 1);
        assert!(var.is_nan());
        assert_eq!(mean, 2.0);
        assert_eq!(std_mean_valid(&single, 1).1, 2.0);
        assert!(std_mean_valid(&single, 1).0.is_nan());
    }

    #[test]
    fn min_max_propagate_and_skip() {
        let v = [3.0_f64, 1.0, f64::NAN, 2.0];
        assert!(min(&v, AllValues).is_nan());
        assert!(max(&v, AllValues).is_nan());
        assert_eq!(min(&v, SkipNan), 1.0);
        assert_eq!(max(&v, SkipNan), 3.0);
        let w = [1.0_f64, f64::INFINITY, -2.0];
        assert_eq!(min(&w, SkipNonFinite), -2.0);
        assert_eq!(max(&w, SkipNonFinite), 1.0);
    }

    #[test]
    fn median_even_odd_and_nan() {
        assert_eq!(median(&[3.0_f64, 1.0, 2.0], AllValues), 2.0);
        assert_eq!(median(&[4.0_f64, 1.0, 3.0, 2.0], AllValues), 2.5);
        assert!(median(&[1.0_f64, f64::NAN, 2.0], AllValues).is_nan());
        assert_eq!(median(&[1.0_f64, f64::NAN, 2.0, 3.0], SkipNan), 2.0);
    }

    #[test]
    fn valid_order_statistics_operate_on_compacted_buffer() {
        let mut median_buf = [9.0_f64, 1.0, 4.0, 2.0];
        assert_eq!(median_valid_in_place(&mut median_buf), 3.0);

        let mut lmedian_buf = [9.0_f64, 1.0, 4.0, 2.0];
        assert_eq!(lmedian_valid_in_place(&mut lmedian_buf), 2.0);

        let mut percentile_buf = [0.0_f64, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile_valid_in_place(&mut percentile_buf, 25.0), 10.0);

        let mut interpolated_buf = [0.0_f64, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(percentile_valid_in_place(&mut interpolated_buf, 12.5), 5.0);

        let mut quantile_buf = [0.0_f64, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(quantile_valid_in_place(&mut quantile_buf, 0.125), 5.0);

        let mut percentiles_buf = [0.0_f64, 10.0, 20.0, 30.0, 40.0];
        let mut out = [f64::NAN; 3];
        percentiles_valid_in_place(&mut percentiles_buf, &[0.0, 12.5, 100.0], &mut out);
        assert_eq!(out, [0.0, 5.0, 40.0]);
    }

    #[test]
    fn number_valid_order_statistics_operate_on_compacted_buffer() {
        let mut median_buf = [9_i16, 1, 4, 2];
        assert_eq!(number_median_valid_in_place(&mut median_buf), 3.0);

        let mut lmedian_buf = [9_i16, 1, 4, 2];
        assert_eq!(number_lmedian_valid_in_place(&mut lmedian_buf), 2.0);

        let mut lmedian_value_buf = [9_i16, 1, 4, 2];
        assert_eq!(
            number_lmedian_value_valid_in_place(&mut lmedian_value_buf),
            Some(2)
        );

        let mut percentile_buf = [0_i16, 10, 20, 30, 40];
        assert_eq!(
            number_percentile_valid_in_place(&mut percentile_buf, 12.5),
            5.0
        );

        let mut quantile_buf = [0_i16, 10, 20, 30, 40];
        assert_eq!(
            number_quantile_valid_in_place(&mut quantile_buf, 0.125),
            5.0
        );

        let mut percentiles_buf = [0_i16, 10, 20, 30, 40];
        let mut out = [f64::NAN; 3];
        number_percentiles_valid_in_place(&mut percentiles_buf, &[0.0, 12.5, 100.0], &mut out);
        assert_eq!(out, [0.0, 5.0, 40.0]);
    }

    #[test]
    fn number_scan_primitives_document_existing_semantics() {
        let v = [1_u16, 2, 3, 4];
        assert_eq!(number_sum(&v), 10.0);
        assert_eq!(number_mean(&v), 2.5);
        assert_eq!(number_min(&v), 1.0);
        assert_eq!(number_max(&v), 4.0);
        assert_eq!(number_minmax(&v), (1.0, 4.0));
        assert_eq!(number_variance_mean(&v, 1), (5.0 / 3.0, 2.5));
        assert_eq!(number_variance(&v, 0), 1.25);
        assert_eq!(number_std(&v, 0), 1.25_f64.sqrt());
    }

    #[test]
    fn valid_order_statistics_keep_infinity_values() {
        let mut median_buf = [1.0_f64, f64::INFINITY, 3.0];
        assert_eq!(median_valid_in_place(&mut median_buf), 3.0);

        let mut percentile_buf = [1.0_f64, 3.0, f64::INFINITY];
        assert_eq!(
            percentile_valid_in_place(&mut percentile_buf, 100.0),
            f64::INFINITY
        );
    }

    #[test]
    fn valid_order_statistics_return_nan_for_empty_buffers() {
        let mut buf: [f64; 0] = [];
        assert!(median_valid_in_place(&mut buf).is_nan());
        assert!(lmedian_valid_in_place(&mut buf).is_nan());
        assert!(percentile_valid_in_place(&mut buf, 50.0).is_nan());
    }

    #[test]
    fn variance_matches_population() {
        let v = [1.0_f64, 2.0, 3.0, 4.0];
        assert!((variance(&v, 0, AllValues) - 1.25).abs() < 1e-12);
        assert!((variance(&v, 1, AllValues) - 5.0 / 3.0).abs() < 1e-12);
        assert!(variance(&[1.0_f64, f64::NAN], 0, AllValues).is_nan());
    }

    #[test]
    fn variance_stable_large_offset() {
        // One-pass sumsq would cancel catastrophically; two-pass stays exact.
        let v = [1e16_f64, 1e16 + 2.0, 1e16 + 4.0];
        assert!((variance(&v, 0, AllValues) - 8.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_linear() {
        let v: Vec<f64> = (0..=10).map(|x| x as f64).collect();
        let out = percentiles(&v, &[0.0, 25.0, 50.0, 100.0], AllValues);
        assert_eq!(out, vec![0.0, 2.5, 5.0, 10.0]);
    }
}
