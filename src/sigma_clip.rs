//! Serial, finite-only sigma clipping with caller-owned reusable storage.
//!
//! Input slices are never changed. The returned mask uses original sample
//! indices: `true` means excluded by the input mask, nonfinite, or rejected.
//! Each worker should own a [`Workspace`]; this module never schedules parallel
//! work. A result borrows the workspace until the caller has consumed it.
//!
//! # Example
//!
//! ```
//! use reducers::sigma_clip::{self, OutputOptions, Params, Reduction, Workspace};
//! let params = Params {
//!     sigma_lower: 2.0, sigma_upper: 2.0, ..Params::default()
//! };
//! let mut workspace = Workspace::with_capacity(5);
//! let result = sigma_clip::clip(&[1.0, 2.0, 3.0, 4.0, 100.0], None,
//!     &params, OutputOptions { reduction: Some(Reduction::Mean),
//!     diagnostics: false }, &mut workspace).unwrap();
//! assert_eq!(result.rejected, &[false, false, false, false, true]);
//! assert_eq!(result.retained, 4);
//! assert_eq!(result.value, Some(2.5));
//! ```
//!
//! # Arithmetic
//!
//! Centers and MAD narrow to the input dtype before threshold comparisons. Mean-based
//! variance uses [`crate::reducers_1d::var_mean_valid`]; variance about other
//! centers and final means use sequential f64 sums. MAD is scaled by exactly
//! `1.4826` and ignores `ddof`. Squared residuals are compared strictly (`>`)
//! against squared asymmetric limits; equality survives. Zero spread rejects
//! unequal samples. Large sums, residual squares, or squared limits can overflow;
//! tiny squares can underflow.
//! NaN centers or nonfinite squared spread stop clipping without changing that
//! iteration's mask. A finite squared spread can still narrow to infinite f32
//! diagnostic standard deviation. Diagnostic spread is from the last valid
//! attempted iteration, not a recomputation over the final retained samples.

// Adapted from imcombiners, Copyright 2026 @ysBach, BSD-3-Clause.
// Full notice: python/reducers/licenses/imcombiners-BSD-3-Clause.txt.

use crate::reducers_1d::{lmedian_valid_in_place, median_valid_in_place, var_mean_valid};
use crate::Float;
use std::fmt;

/// Statistic used as the center of the rejection interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CenFunc {
    Mean,
    Median,
    LowerMedian,
}

/// Center about which dispersion is calculated, independently of rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipCenter {
    Mean,
    Median,
    LowerMedian,
    /// Use the selected rejection center, including its input-dtype rounding.
    ClippingCenter,
}

/// Dispersion estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdFunc {
    /// Sum of squared deviations divided by `retained - ddof`.
    Std,
    /// Median absolute deviation about the dispersion center, times `1.4826`.
    /// `ddof` is ignored.
    Mad,
}

/// Clipping parameters, with survivor rollback opt-in.
///
/// `Default` selects sigma=(3,3), maxiters=5, ddof=0, median center, standard
/// deviation, nkeep=0 without survivor rollback, and unlimited maxrej. `clip_cen=None`
/// follows the estimator selected by `cenfunc`. The additional
/// `final_min_retained` gate defaults to zero (disabled).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Finite, nonnegative multiplier below the rejection center.
    pub sigma_lower: f64,
    /// Finite, nonnegative multiplier above the rejection center.
    pub sigma_upper: f64,
    /// Maximum attempted iterations; zero only applies initial exclusions.
    pub maxiters: usize,
    /// Degrees of freedom subtracted for standard deviation. `n <= ddof`
    /// produces invalid statistics and stops clipping. Ignored for MAD.
    pub ddof: usize,
    pub cenfunc: CenFunc,
    /// `None` uses the estimator selected by `cenfunc`.
    /// For a mean center this selects `ClipCenter::Mean`, NOT `ClippingCenter`;
    /// the two variance paths can differ through dtype rounding/summation.
    pub clip_cen: Option<ClipCenter>,
    pub stdfunc: StdFunc,
    /// Roll back the entire last iteration if it would retain fewer samples.
    /// Earlier rejections remain. Does not resurrect initial exclusions or
    /// enforce a final minimum. Only active when `revert_on_nkeep` is true.
    /// Defaults to zero (no minimum).
    pub nkeep: usize,
    /// Enable whole-iteration rollback when fewer than `nkeep` samples survive.
    /// Defaults to false; clipping may reject every sample.
    pub revert_on_nkeep: bool,
    /// Roll back the entire last iteration if total excluded samples exceed
    /// this limit, INCLUDING input-masked and nonfinite samples. `None` means
    /// unlimited. Already-exceeded limits cannot restore initial exclusions.
    pub maxrej: Option<usize>,
    /// Final count required for a requested reduction. Failure returns NaN for
    /// the value and `minimum_met = false`, without altering the rejection mask.
    pub final_min_retained: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            sigma_lower: 3.0,
            sigma_upper: 3.0,
            maxiters: 5,
            ddof: 0,
            cenfunc: CenFunc::Median,
            clip_cen: None,
            stdfunc: StdFunc::Std,
            nkeep: 0,
            revert_on_nkeep: false,
            maxrej: None,
            final_min_retained: 0,
        }
    }
}

/// Optional final reduction of retained original input values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Mean,
    Median,
}

/// Request only the optional work needed by the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutputOptions {
    /// `None` requests only the mask, retained count, and completion metadata.
    pub reduction: Option<Reduction>,
    /// Collect final extrema, last valid spread, and restoration information.
    pub diagnostics: bool,
}

/// Why the loop stopped. A final-count failure is separate (`minimum_met`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Converged,
    MaxIterations,
    InvalidStatistics,
    /// Both constraints may be violated in the same attempted iteration.
    RejectionLimit {
        min_retained: bool,
        max_rejected: bool,
    },
}

/// Optional diagnostics, with explicit compatibility metadata.
#[derive(Debug)]
pub struct Diagnostics<'a, T> {
    /// Minimum retained value, or NaN when none survive; not a clipping bound.
    pub low: T,
    /// Maximum retained value, or NaN when none survive; not a clipping bound.
    pub upp: T,
    /// Last valid attempted spread, including an iteration that was rolled back.
    /// NaN if no valid iteration ran. Not recomputed after the last rejection.
    pub std: T,
    /// Counter: starts at one, increments after each accepted
    /// changing iteration, and saturates at 255. Prefer `ClipResult::iterations`
    /// for the actual number of attempted iterations.
    pub legacy_nit: u8,
    /// Status bits: 1=input mask contains true, 2=iteration limit,
    /// 4=minimum-survivor rollback, 8=maximum-rejection rollback. A nonfinite
    /// input alone does not set bit 1. Invalid statistics preserve the
    /// iteration-limit bit (set only when the final allowed iteration began).
    pub legacy_flags: u8,
    /// Original-order sample bits for tentative rejections restored by the
    /// final rollback: 64=minimum survivors, 128=maximum rejection count.
    pub restored_flags: &'a [u8],
}

/// Result borrowing the caller's workspace. Inputs remain unchanged.
#[derive(Debug)]
pub struct ClipResult<'a, T> {
    pub rejected: &'a [bool],
    pub retained: usize,
    /// `None` if no reduction was requested; NaN if empty or below final minimum.
    pub value: Option<T>,
    pub minimum_met: bool,
    /// Number of attempted iterations, including invalid or rolled-back ones.
    pub iterations: usize,
    pub stop_reason: StopReason,
    pub diagnostics: Option<Diagnostics<'a, T>>,
}

/// Invalid input; checked before changing any workspace contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipError {
    MaskLength { samples: usize, mask: usize },
    InvalidSigma,
}

impl fmt::Display for ClipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaskLength { samples, mask } => {
                write!(f, "mask length {mask} differs from sample length {samples}")
            }
            Self::InvalidSigma => write!(f, "sigma thresholds must be finite and nonnegative"),
        }
    }
}

impl std::error::Error for ClipError {}

/// Reusable storage independent of input dtype. No copies of the input slice
/// are owned; retained values are gathered into f64 scratch for statistics.
///
/// [`Workspace::with_capacity`] preallocates all buffers, including diagnostic
/// flags. Calls up to that capacity perform no heap allocations. Larger calls
/// grow buffers as needed, and subsequent calls reuse them. Scratch costs about
/// `size_of::<bool>() + size_of::<u8>() + size_of::<usize>() + size_of::<f64>()`
/// bytes per sample, plus vector headers/capacity rounding. No shrink occurs.
#[derive(Debug, Default)]
pub struct Workspace {
    mask: Vec<bool>,
    newly_rejected: Vec<usize>,
    statistics: Vec<f64>,
    restored_flags: Vec<u8>,
}

impl Workspace {
    /// Create empty storage; buffers grow on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve storage for `capacity` samples in all output modes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            mask: Vec::with_capacity(capacity),
            newly_rejected: Vec::with_capacity(capacity),
            statistics: Vec::with_capacity(capacity),
            restored_flags: Vec::with_capacity(capacity),
        }
    }
}

/// Clip a sample slice cumulatively, optionally reducing the retained samples.
///
/// No initial exclusion or earlier rejection is undone. Limits are tested
/// before convergence, even if the current iteration rejects nothing. Only
/// rejections from a violating iteration are restored. Nonfinite inputs are
/// always excluded. For a mask-only call, the final minimum is still reported
/// via `minimum_met` but never alters the mask.
///
/// # Errors
/// Returns [`ClipError`] for a mask length mismatch or a negative/nonfinite
/// sigma multiplier. Workspace contents are unchanged on error.
pub fn clip<'a, T: Float>(
    samples: &[T],
    input_mask: Option<&[bool]>,
    params: &Params,
    output: OutputOptions,
    workspace: &'a mut Workspace,
) -> Result<ClipResult<'a, T>, ClipError> {
    if let Some(mask) = input_mask {
        if mask.len() != samples.len() {
            return Err(ClipError::MaskLength {
                samples: samples.len(),
                mask: mask.len(),
            });
        }
    }
    if !params.sigma_lower.is_finite()
        || !params.sigma_upper.is_finite()
        || params.sigma_lower < 0.0
        || params.sigma_upper < 0.0
    {
        return Err(ClipError::InvalidSigma);
    }
    let n = samples.len();
    let Workspace {
        mask,
        newly_rejected,
        statistics,
        restored_flags,
    } = workspace;
    mask.resize(n, false);
    if output.diagnostics {
        restored_flags.resize(n, 0);
        restored_flags.fill(0);
    }
    let mut retained = 0;
    for (i, &value) in samples.iter().enumerate() {
        mask[i] = input_mask.is_some_and(|m| m[i]) || !value.is_finite();
        retained += usize::from(!mask[i]);
    }
    let mut legacy_nit = 1_u8;
    let mut std_out = T::nan();
    let mut stopped_min = false;
    let mut stopped_max = false;
    let mut stopped_iterations = params.maxiters == 0;
    let mut stop_reason = StopReason::MaxIterations;
    let mut iterations = 0;
    let mut final_median = None;
    let mut final_statistics_current = false;

    for k in 0..params.maxiters {
        iterations += 1;
        stopped_iterations = k + 1 == params.maxiters;
        let (center, std, spread_sq) =
            center_and_spread(samples, mask, params, output.diagnostics, statistics);
        final_statistics_current = true;
        // The rejection center is also the final median while this mask is
        // unchanged. Reuse it after convergence, rollback or invalid spread;
        // an accepted rejection below invalidates it.
        final_median = (params.cenfunc == CenFunc::Median).then_some(center);
        if center.is_nan() || !spread_sq.is_finite() {
            stop_reason = StopReason::InvalidStatistics;
            break;
        }
        std_out = std;
        let center = center.to_f64();
        let lower_limit_sq = params.sigma_lower * params.sigma_lower * spread_sq;
        let upper_limit_sq = params.sigma_upper * params.sigma_upper * spread_sq;
        newly_rejected.clear();
        let mut kept = 0;
        for (i, (&value, rejected)) in samples.iter().zip(mask.iter_mut()).enumerate() {
            if !*rejected {
                let delta = value.to_f64() - center;
                let delta_sq = delta * delta;
                if (delta < 0.0 && delta_sq > lower_limit_sq)
                    || (delta > 0.0 && delta_sq > upper_limit_sq)
                {
                    *rejected = true;
                    newly_rejected.push(i);
                } else {
                    kept += 1;
                }
            }
        }
        let min_retained = params.revert_on_nkeep && kept < params.nkeep;
        let max_rejected = params.maxrej.is_some_and(|max| n - kept > max);
        if min_retained || max_rejected {
            stopped_min = min_retained;
            stopped_max = max_rejected;
            stopped_iterations = false;
            for &i in newly_rejected.iter() {
                mask[i] = false;
                if output.diagnostics {
                    restored_flags[i] =
                        (u8::from(min_retained) * 64) | (u8::from(max_rejected) * 128);
                }
            }
            stop_reason = StopReason::RejectionLimit {
                min_retained,
                max_rejected,
            };
            break;
        }
        if retained == kept {
            stopped_iterations = false;
            stop_reason = StopReason::Converged;
            break;
        }
        retained = kept;
        final_median = None;
        final_statistics_current = false;
        legacy_nit = legacy_nit.saturating_add(1);
    }

    let minimum_met = retained >= params.final_min_retained;
    let value = output.reduction.map(|reduction| {
        if !minimum_met {
            T::nan()
        } else {
            let ordered_statistics_current = final_statistics_current
                && params.cenfunc == CenFunc::Mean
                && params.stdfunc == StdFunc::Std
                && params.clip_cen.unwrap_or(ClipCenter::Mean) == ClipCenter::Mean;
            match reduction {
                Reduction::Mean if ordered_statistics_current => {
                    // This path leaves retained samples in their original order
                    // in scratch. Sum them once without another mask scan; the
                    // variance helper's differently ordered mean is not reused.
                    let mut sum = 0.0_f64;
                    for &value in statistics.iter() {
                        sum += value;
                    }
                    if statistics.is_empty() {
                        T::nan()
                    } else {
                        T::from_f64(sum / statistics.len() as f64)
                    }
                }
                Reduction::Mean => mean(samples, mask),
                Reduction::Median => final_median.unwrap_or_else(|| {
                    if ordered_statistics_current {
                        T::from_f64(median_valid_in_place(statistics))
                    } else {
                        median(samples, mask, statistics, false)
                    }
                }),
            }
        }
    });
    let diagnostics = if output.diagnostics {
        let (low, upp) = minmax(samples, mask);
        let pre_masked = input_mask.is_some_and(|m| m.iter().any(|&m| m));
        Some(Diagnostics {
            low,
            upp,
            std: std_out,
            legacy_nit,
            legacy_flags: u8::from(pre_masked)
                | (u8::from(stopped_iterations) << 1)
                | (u8::from(stopped_min) << 2)
                | (u8::from(stopped_max) << 3),
            restored_flags,
        })
    } else {
        None
    };
    Ok(ClipResult {
        rejected: mask,
        retained,
        value,
        minimum_met,
        iterations,
        stop_reason,
        diagnostics,
    })
}

fn center_and_spread<T: Float>(
    values: &[T],
    mask: &[bool],
    p: &Params,
    diagnostics: bool,
    buf: &mut Vec<f64>,
) -> (T, T, f64) {
    let clip_cen = p.clip_cen.unwrap_or(match p.cenfunc {
        CenFunc::Mean => ClipCenter::Mean,
        CenFunc::Median => ClipCenter::Median,
        CenFunc::LowerMedian => ClipCenter::LowerMedian,
    });
    // Preserve the baseline's distinct mean paths, even for MAD: mean-centered
    // dispersion uses var_mean_valid's summation order, other means are serial.
    let mean_stats = if clip_cen == ClipCenter::Mean {
        gather(values, mask, buf);
        let (var, mean) = var_mean_valid(buf, p.ddof);
        Some((T::from_f64(mean), var))
    } else {
        None
    };
    let center = match p.cenfunc {
        CenFunc::Mean => mean_stats.map_or_else(|| mean(values, mask), |stats| stats.0),
        CenFunc::Median => median(values, mask, buf, false),
        CenFunc::LowerMedian => median(values, mask, buf, true),
    };
    let spread_center = match clip_cen {
        ClipCenter::Mean => mean_stats.expect("mean statistics were computed").0,
        ClipCenter::Median if p.cenfunc != CenFunc::Median => median(values, mask, buf, false),
        ClipCenter::LowerMedian if p.cenfunc != CenFunc::LowerMedian => {
            median(values, mask, buf, true)
        }
        _ => center,
    };
    match p.stdfunc {
        StdFunc::Std => {
            let var = if let Some(stats) = mean_stats {
                stats.1
            } else {
                variance_about(values, mask, spread_center, p.ddof)
            };
            let std = if diagnostics && var.is_finite() {
                T::from_f64(var.sqrt())
            } else {
                T::nan()
            };
            (center, std, var)
        }
        StdFunc::Mad => {
            let spread = if spread_center.is_nan() {
                T::nan()
            } else {
                // Mean dispersion and order statistics already gathered the
                // retained samples. Selection may permute them, which cannot
                // change the median of their absolute deviations.
                if mean_stats.is_none()
                    && p.cenfunc == CenFunc::Mean
                    && clip_cen == ClipCenter::ClippingCenter
                {
                    gather(values, mask, buf);
                }
                for value in buf.iter_mut() {
                    *value = (*value - spread_center.to_f64()).abs();
                }
                T::from_f64(1.4826 * median_valid_in_place(buf))
            };
            (center, spread, spread.to_f64() * spread.to_f64())
        }
    }
}

fn gather<T: Float>(values: &[T], mask: &[bool], buf: &mut Vec<f64>) {
    buf.clear();
    for (&value, &rejected) in values.iter().zip(mask) {
        if !rejected {
            buf.push(value.to_f64());
        }
    }
}

fn mean<T: Float>(values: &[T], mask: &[bool]) -> T {
    let mut sum = 0.0_f64;
    let mut count = 0;
    for (&value, &rejected) in values.iter().zip(mask) {
        if !rejected {
            sum += value.to_f64();
            count += 1;
        }
    }
    if count == 0 {
        T::nan()
    } else {
        T::from_f64(sum / count as f64)
    }
}

fn median<T: Float>(values: &[T], mask: &[bool], buf: &mut Vec<f64>, lower: bool) -> T {
    gather(values, mask, buf);
    T::from_f64(if lower {
        lmedian_valid_in_place(buf)
    } else {
        median_valid_in_place(buf)
    })
}

fn variance_about<T: Float>(values: &[T], mask: &[bool], center: T, ddof: usize) -> f64 {
    if center.is_nan() {
        return f64::NAN;
    }
    let mut sum = 0.0_f64;
    let mut count = 0;
    for (&value, &rejected) in values.iter().zip(mask) {
        if !rejected {
            let delta = value.to_f64() - center.to_f64();
            sum += delta * delta;
            count += 1;
        }
    }
    if count <= ddof {
        f64::NAN
    } else {
        sum / (count - ddof) as f64
    }
}

fn minmax<T: Float>(values: &[T], mask: &[bool]) -> (T, T) {
    let mut low = T::nan();
    let mut upp = T::nan();
    for (&value, &rejected) in values.iter().zip(mask) {
        if !rejected {
            if low.is_nan() || value < low {
                low = value;
            }
            if upp.is_nan() || value > upp {
                upp = value;
            }
        }
    }
    (low, upp)
}
