//! IRAF `imcombine`-style per-sample rejection: CCD-noise clipping, linear
//! thresholds, min/max rank rejection, and percentile clipping.
//!
//! Each function processes one 1-D sample set (for example, one output pixel's
//! image stack, or one moving window) and never schedules parallel work. Input
//! slices are never changed. The returned [`RejectResult::rejected`] mask uses
//! original sample indices: `true` means excluded by the input mask, nonfinite
//! (NaN or ±inf), or rejected. Each worker should own a [`Workspace`]; a result
//! borrows the workspace until the caller has consumed it. For generic
//! sigma clipping (standard deviation or MAD spread), use
//! [`crate::sigma_clip`].
//!
//! All arithmetic is in `f64`. Centers (mean, median, lower median) are rounded
//! to the input dtype `T` before they are used, so `f32` inputs see an
//! `f32`-rounded center. Final means use [`crate::reducers_1d::mean_valid`] and
//! medians use [`crate::reducers_1d::median_valid_in_place`] over the retained
//! samples, in the order documented on each function.
//!
//! [`minmax_iter`] and [`pclip_iter`] accept `(value, input_masked)` items
//! instead of slices, so strided data (such as an image-stack column) can be
//! processed without a separate gather pass. [`ccdclip`] and [`linearclip`]
//! revisit samples every iteration and therefore take slices only.
//!
//! # Example
//!
//! ```
//! use reducers::reject::{self, MinMaxParams, Workspace};
//! use reducers::sigma_clip::{OutputOptions, Reduction};
//! let mut workspace = Workspace::with_capacity(5);
//! let params = MinMaxParams { q_low: 0.2, q_upp: 0.2 };
//! let result = reject::minmax(&[3.0_f32, 1.0, 2.0, 100.0, 4.0], None, &params,
//!     OutputOptions { reduction: Some(Reduction::Mean), diagnostics: false },
//!     &mut workspace).unwrap();
//! assert_eq!(result.rejected, &[false, true, false, true, false]);
//! assert_eq!(result.retained, 3);
//! assert_eq!(result.value, Some(3.0));
//! ```

// Adapted from imcombiners, Copyright 2026 @ysBach, BSD-3-Clause.
// Full notice: python/reducers/licenses/imcombiners-BSD-3-Clause.txt.

use crate::reducers_1d::{lmedian_valid_in_place, mean_valid, median_valid_in_place, minmax_valid};
use crate::sigma_clip::{CenFunc, ClipCenter, OutputOptions, Reduction};
use crate::Float;
use std::fmt;

/// Per-sample restored flag: the sample was tentatively rejected in the final
/// iteration, and that iteration was rolled back because fewer than `nkeep`
/// samples would survive.
pub const RESTORED_NKEEP: u8 = 64;
/// Per-sample restored flag: the sample was tentatively rejected in the final
/// iteration, and that iteration was rolled back because more than `maxrej`
/// samples would be excluded.
pub const RESTORED_MAXREJ: u8 = 128;

/// Minimum number of retained samples for which [`pclip`] rejects anything.
const PCLIP_MINCLIP: usize = 3;
/// IRAF's residual tie tolerance when [`pclip`] restores samples to `nkeep`.
const IRAF_RESID_TOL: f64 = 0.001;
/// IRAF's floor on the read-noise variance, `1e4 / FLT_MAX`.
const NM1_FLOOR: f64 = 1e4 / 3.402_823_466e38;

/// CCD noise-model clipping parameters for [`ccdclip`] (IRAF `ccdclip`).
///
/// The noise sigma of sample `i` about center `c` (both in normalized data
/// units) is
///
/// - without scaling (`scales` and `zeros` empty):
///   ``sqrt(v_r + s/gain + (s*snoise)**2)`` with ``s = max(c, 0)``;
/// - with scaling: ``sqrt(v_r + s/gain + (s*snoise)**2) / scales[i]`` with
///   ``s = max(scales[i] * (c + zeros[i]), 0)``,
///
/// where ``v_r = max((rdnoise/gain)**2, 1e4/3.402823466e38)``. Inputs satisfy
/// ``input = scales[i] * (normalized + zeros[i])``, so the model is evaluated
/// in pre-normalization input units and divided back by the scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CcdClipParams<'a> {
    /// Rejection threshold below the center, in units of the per-sample noise
    /// sigma. A sample is rejected only when its residual is strictly greater.
    pub sigma_lower: f64,
    /// Rejection threshold above the center, in units of the per-sample noise
    /// sigma. A sample is rejected only when its residual is strictly greater.
    pub sigma_upper: f64,
    /// Maximum number of clipping iterations; zero applies only the initial
    /// exclusions (input mask and nonfinite values).
    pub maxiters: usize,
    /// Minimum number of survivors. Only used when `revert_on_nkeep` is true.
    pub nkeep: usize,
    /// Maximum number of excluded samples, counting input-masked and nonfinite
    /// samples. Use `samples.len()` (or larger) for no limit.
    pub maxrej: usize,
    /// Roll back the whole last iteration if fewer than `nkeep` samples would
    /// survive. Earlier rejections remain.
    pub revert_on_nkeep: bool,
    /// Rejection center statistic.
    pub cenfunc: CenFunc,
    /// Center at which the noise model is evaluated. `ClippingCenter` reuses
    /// the rejection center.
    pub clip_cen: ClipCenter,
    /// Read noise, in electrons.
    pub rdnoise: f64,
    /// Sensitivity noise, as a fraction of the signal.
    pub snoise: f64,
    /// Gain, in electrons per data unit.
    pub gain: f64,
    /// Per-sample scale factors. Empty (together with `zeros`) selects the
    /// unscaled model; otherwise both must have `samples.len()` elements.
    pub scales: &'a [f64],
    /// Per-sample zero offsets, in normalized data units. See `scales`.
    pub zeros: &'a [f64],
}

/// Linear-threshold clipping parameters for [`linearclip`].
///
/// With center `c`, a sample `x` is rejected when
/// ``x < low_scale*c - low`` or ``x > upp_scale*c + upp`` (strict).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearClipParams {
    /// Multiplier of the center for the lower threshold.
    pub low_scale: f64,
    /// Offset subtracted from ``low_scale*c`` for the lower threshold.
    pub low: f64,
    /// Multiplier of the center for the upper threshold.
    pub upp_scale: f64,
    /// Offset added to ``upp_scale*c`` for the upper threshold.
    pub upp: f64,
    /// Maximum number of clipping iterations; zero applies only the initial
    /// exclusions.
    pub maxiters: usize,
    /// Minimum number of survivors. Only used when `revert_on_nkeep` is true.
    pub nkeep: usize,
    /// Maximum number of samples rejected by clipping, NOT counting
    /// input-masked or nonfinite samples. Use `samples.len()` for no limit.
    pub maxrej: usize,
    /// Center statistic.
    pub cenfunc: CenFunc,
    /// Roll back the whole last iteration if fewer than `nkeep` samples would
    /// survive. Earlier rejections remain.
    pub revert_on_nkeep: bool,
}

/// Rank-rejection parameters for [`minmax`].
///
/// With `m` finite, unmasked samples, the ``int(m*q_low + 0.001)`` lowest and
/// ``int(m*q_upp + 0.001)`` highest are rejected (IRAF `minmax`), so the
/// fractions usually come from ``nlow / N`` and ``nhigh / N``.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxParams {
    /// Fraction of finite, unmasked samples rejected from the low end, `>= 0`.
    pub q_low: f64,
    /// Fraction of finite, unmasked samples rejected from the high end, `>= 0`.
    pub q_upp: f64,
}

/// Percentile-clipping parameters for [`pclip`] (IRAF `pclip`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PclipParams {
    /// Signed rank offset from the median that defines sigma; usually from
    /// [`pclip_offset`]. Negative values use a lower rank.
    pub offset: isize,
    /// Rejection threshold below the median, in units of the pclip sigma.
    /// A sample is rejected when its residual is `>= sigma_lower`.
    pub sigma_lower: f64,
    /// Rejection threshold above the median, in units of the pclip sigma.
    /// A sample is rejected when its residual is `>= sigma_upper`.
    pub sigma_upper: f64,
    /// Minimum number of survivors; the lowest-residual rejected samples are
    /// restored until at least `min(nkeep, m)` remain, where `m` is the number of
    /// finite, unmasked samples.
    pub nkeep: usize,
}

/// Optional diagnostics. Meanings of `low` and `upp` depend on the method.
#[derive(Debug)]
pub struct Diagnostics<'a, T> {
    /// [`ccdclip`], [`minmax`], [`pclip`]: minimum retained value.
    /// [`linearclip`]: lower threshold of the last accepted iteration.
    /// NaN when undefined (no retained sample or no accepted iteration).
    pub low: T,
    /// Upper counterpart of `low`.
    pub upp: T,
    /// [`ccdclip`]: noise sigma at the noise-model center in the last valid
    /// iteration (including a rolled-back one); NaN otherwise.
    pub std: T,
    /// IRAF-compatible iteration counter, saturating at 255. [`ccdclip`]:
    /// starts at one and counts accepted changing iterations. [`linearclip`]:
    /// counts attempted iterations that had a valid center. [`minmax`] and
    /// [`pclip`]: always one.
    pub nit: u8,
    /// Status bits: 1 = the input mask contains `true`, 2 = iteration limit,
    /// 4 = minimum-survivor rollback, 8 = maximum-rejection rollback.
    /// [`minmax`] and [`pclip`] only set bit 1.
    pub flags: u8,
    /// Original-order bits for tentative rejections undone by the final
    /// rollback: [`RESTORED_NKEEP`] and/or [`RESTORED_MAXREJ`].
    /// Empty for [`minmax`] and [`pclip`], which never roll back.
    pub restored_flags: &'a [u8],
}

/// Result borrowing the caller's workspace. Inputs remain unchanged.
#[derive(Debug)]
pub struct RejectResult<'a, T> {
    /// `true` = input-masked, nonfinite, or rejected; one entry per sample.
    pub rejected: &'a [bool],
    /// Number of `false` entries in `rejected`.
    pub retained: usize,
    /// Requested reduction of the retained samples; `None` if none was
    /// requested, NaN if no sample is retained.
    pub value: Option<T>,
    /// Present only when diagnostics were requested.
    pub diagnostics: Option<Diagnostics<'a, T>>,
}

/// Invalid input; checked before changing any workspace contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectError {
    MaskLength {
        samples: usize,
        mask: usize,
    },
    /// [`CcdClipParams::scales`] and `zeros` must both be empty or both have
    /// one element per sample.
    ScaleLength {
        samples: usize,
        scales: usize,
        zeros: usize,
    },
}

impl fmt::Display for RejectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaskLength { samples, mask } => {
                write!(f, "mask length {mask} differs from sample length {samples}")
            }
            Self::ScaleLength {
                samples,
                scales,
                zeros,
            } => write!(
                f,
                "scales ({scales}) and zeros ({zeros}) must both be empty or match sample length {samples}"
            ),
        }
    }
}

impl std::error::Error for RejectError {}

/// Reusable storage independent of input dtype; no copy of the input slice is
/// owned.
///
/// [`Workspace::with_capacity`] preallocates all buffers. Calls up to that
/// capacity perform no heap allocations. Larger calls grow buffers as needed,
/// and subsequent calls reuse them. Scratch costs about
/// `size_of::<bool>() + size_of::<u8>() + 2 * size_of::<usize>() + 2 * size_of::<f64>()`
/// bytes per sample, plus vector headers and capacity rounding. No shrink occurs.
#[derive(Debug, Default)]
pub struct Workspace {
    mask: Vec<bool>,
    newly_rejected: Vec<usize>,
    pairs: Vec<(f64, usize)>,
    statistics: Vec<f64>,
    /// All zero unless `restored_dirty`; cleared lazily so calls without a
    /// rollback do not rewrite it.
    restored_flags: Vec<u8>,
    restored_dirty: bool,
}

impl Workspace {
    /// Create empty storage without allocating; buffers grow on first use.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserve storage for `capacity` samples in all methods and output modes.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            mask: Vec::with_capacity(capacity),
            newly_rejected: Vec::with_capacity(capacity),
            pairs: Vec::with_capacity(capacity),
            statistics: Vec::with_capacity(capacity),
            restored_flags: Vec::with_capacity(capacity),
            restored_dirty: false,
        }
    }
}

/// Prepare all-zero restored flags of length `n`.
#[inline]
fn reset_restored(flags: &mut Vec<u8>, dirty: &mut bool, n: usize) {
    if *dirty {
        flags.fill(0);
        *dirty = false;
    }
    flags.resize(n, 0);
}

/// IRAF `pclip` rank offset for a stack of `n` samples.
///
/// `n` is the total sample count, including masked and nonfinite samples.
/// With ``half = max(n / 2, 1)`` (integer division), ``|pclip| < 1`` is a
/// fraction of `half`; otherwise `pclip` is a rank count. The result is
/// truncated toward zero and clamped to `[-half, -1]` for negative `pclip`
/// and to `[1, half]` otherwise (including `pclip = 0`).
#[inline]
pub fn pclip_offset(pclip: f64, n: usize) -> isize {
    let half = (n / 2).max(1);
    let mut offset = pclip;
    if offset.abs() < 1.0 {
        offset *= half as f64;
    }
    let offset = offset.trunc() as isize;
    if pclip < 0.0 {
        offset.max(-(half as isize)).min(-1)
    } else {
        offset.max(1).min(half as isize)
    }
}

#[inline]
fn iraf_nint(x: f64) -> isize {
    if x >= 0.0 {
        (x + 0.5).floor() as isize
    } else {
        (x - 0.5).ceil() as isize
    }
}

#[inline]
fn check_mask<T>(samples: &[T], input_mask: Option<&[bool]>) -> Result<(), RejectError> {
    match input_mask {
        Some(mask) if mask.len() != samples.len() => Err(RejectError::MaskLength {
            samples: samples.len(),
            mask: mask.len(),
        }),
        _ => Ok(()),
    }
}

/// Mark input-masked and nonfinite samples in `mask` (same length as
/// `samples`); returns the number of unmarked samples.
#[inline]
fn init_mask<T: Float>(samples: &[T], input_mask: Option<&[bool]>, mask: &mut [bool]) -> usize {
    let mut retained = 0usize;
    match input_mask {
        Some(input) => {
            for ((m, &x), &masked) in mask.iter_mut().zip(samples).zip(input) {
                *m = masked || !x.is_finite();
                retained += usize::from(!*m);
            }
        }
        None => {
            for (m, &x) in mask.iter_mut().zip(samples) {
                *m = !x.is_finite();
                retained += usize::from(!*m);
            }
        }
    }
    retained
}

#[inline]
fn debug_assert_mask_covers_nonfinite<T: Float>(vals: &[T], mask: &[bool]) {
    debug_assert_eq!(vals.len(), mask.len());
    debug_assert!(
        vals.iter().zip(mask).all(|(&x, &m)| m || x.is_finite()),
        "mask must mark every non-finite value"
    );
}

// ---------- retained-sample statistics in original order ----------
// The caller has already folded input masks and non-finite values into `mask`.

#[inline]
fn col_nanmean<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push(x.to_f64());
        }
    }
    T::from_f64(mean_valid(buf))
}

#[inline]
fn col_nanmedian<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push(x.to_f64());
        }
    }
    let n = buf.len();
    if n == 0 {
        return T::nan();
    }
    T::from_f64(median_valid_in_place(buf))
}

#[inline]
fn col_nanlmedian<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
    debug_assert_mask_covers_nonfinite(vals, mask);
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push(x.to_f64());
        }
    }
    let n = buf.len();
    if n == 0 {
        return T::nan();
    }
    T::from_f64(lmedian_valid_in_place(buf))
}

#[inline]
fn col_minmax<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> (T, T) {
    debug_assert_mask_covers_nonfinite(vals, mask);
    buf.clear();
    for (i, &x) in vals.iter().enumerate() {
        if !mask[i] {
            buf.push(x.to_f64());
        }
    }
    let (lo, upp) = minmax_valid(buf);
    (T::from_f64(lo), T::from_f64(upp))
}

#[inline]
fn final_combine<T: Float>(vals: &[T], mask: &[bool], kind: Reduction, buf: &mut Vec<f64>) -> T {
    match kind {
        Reduction::Mean => col_nanmean(vals, mask, buf),
        Reduction::Median => col_nanmedian(vals, mask, buf),
    }
}

// ---------- retained-sample statistics in pair order ----------

// Forced for the same reason as `fill_pairs`: small, called once per sample set.
#[inline(always)]
fn final_combine_from_sorted_pairs<T: Float>(
    pairs: &[(f64, usize)],
    mask: &[bool],
    kind: Reduction,
    buf: &mut Vec<f64>,
) -> T {
    buf.clear();
    for &(value, k) in pairs {
        if !mask[k] {
            buf.push(value);
        }
    }
    T::from_f64(match kind {
        Reduction::Mean => mean_valid(buf),
        Reduction::Median => median_valid_in_place(buf),
    })
}

#[inline]
fn col_minmax_from_masked_pairs<T: Float>(
    pairs: &[(f64, usize)],
    mask: &[bool],
    buf: &mut Vec<f64>,
) -> (T, T) {
    buf.clear();
    for &(value, k) in pairs {
        if !mask[k] {
            buf.push(value);
        }
    }
    let (lo, upp) = minmax_valid(buf);
    (T::from_f64(lo), T::from_f64(upp))
}

// ---------- ccdclip ----------

/// IRAF ccdclip noise sigma for sample `i` about `center`.
///
/// `iccclip.x` evaluates the model in pre-normalization input DN and divides
/// the result by the plane scale to compare with normalized data. `icomb.x`
/// floors the read-noise variance at a positive f32-scale value.
#[inline]
fn ccdclip_sample_sigma(p: &CcdClipParams<'_>, center: f64, i: usize) -> f64 {
    let read_variance = (p.rdnoise / p.gain).powi(2).max(NM1_FLOOR);
    if p.scales.is_empty() {
        let signal = center.max(0.0);
        (read_variance + signal / p.gain + (signal * p.snoise).powi(2)).sqrt()
    } else {
        let scale = p.scales[i];
        let signal = (scale * (center + p.zeros[i])).max(0.0);
        (read_variance + signal / p.gain + (signal * p.snoise).powi(2)).sqrt() / scale
    }
}

/// Returns `(center, noise-model center, diagnostic std, squared spread)`.
fn ccd_center_and_spread<T: Float>(
    col: &[T],
    mask: &[bool],
    p: &CcdClipParams<'_>,
    diagnostics: bool,
    buf: &mut Vec<f64>,
) -> (T, T, T, f64) {
    // Preserve the mean-first ordering when either the clipping center or
    // center function requests the mean.
    let (cen, mean_value) = if p.cenfunc == CenFunc::Mean || p.clip_cen == ClipCenter::Mean {
        let mean = col_nanmean(col, mask, buf);
        let cen = match p.cenfunc {
            CenFunc::Mean => mean,
            CenFunc::Median => col_nanmedian(col, mask, buf),
            CenFunc::LowerMedian => col_nanlmedian(col, mask, buf),
        };
        (cen, Some(mean))
    } else {
        let cen = match p.cenfunc {
            CenFunc::Mean => col_nanmean(col, mask, buf),
            CenFunc::Median => col_nanmedian(col, mask, buf),
            CenFunc::LowerMedian => col_nanlmedian(col, mask, buf),
        };
        (cen, None)
    };
    let std_center = match p.clip_cen {
        ClipCenter::Mean => mean_value.unwrap_or_else(|| col_nanmean(col, mask, buf)),
        ClipCenter::Median => {
            if p.cenfunc == CenFunc::Median {
                cen
            } else {
                col_nanmedian(col, mask, buf)
            }
        }
        ClipCenter::LowerMedian => {
            if p.cenfunc == CenFunc::LowerMedian {
                cen
            } else {
                col_nanlmedian(col, mask, buf)
            }
        }
        ClipCenter::ClippingCenter => cen,
    };
    // Variance from the CCD noise model. The square root is only needed for
    // diagnostics; the validity test uses this value.
    let (std, spread_sq) = if std_center.is_nan() {
        (T::nan(), f64::NAN)
    } else {
        let c = std_center.to_f64();
        let signal = c.max(0.0);
        let v = (p.rdnoise / p.gain).powi(2).max(NM1_FLOOR)
            + signal / p.gain
            + (signal * p.snoise).powi(2);
        let std = if diagnostics {
            T::from_f64(v.sqrt())
        } else {
            T::nan()
        };
        (std, v)
    };
    (cen, std_center, std, spread_sq)
}

/// Clip one sample set with the IRAF CCD noise model, cumulatively.
///
/// Each iteration computes the rejection center ([`CcdClipParams::cenfunc`])
/// over the retained samples and stops if it is NaN (no retained sample) or
/// if the unscaled noise variance at [`CcdClipParams::clip_cen`] is not finite.
/// The retained samples are then sorted by value (ties by index). From the low
/// end, each sample whose residual ``(center - x) / sigma_i`` is
/// `> sigma_lower` is rejected; the scan stops at the first sample within its
/// own threshold. The high end is scanned the same way with
/// ``(x - center) / sigma_i > sigma_upper``. `sigma_i` is the per-sample noise
/// sigma documented on [`CcdClipParams`].
///
/// After each iteration, if `revert_on_nkeep` and fewer than `nkeep` samples
/// survive, or if more than `maxrej` samples (including initial exclusions)
/// are excluded, that iteration's rejections are undone and clipping stops.
/// Clipping also stops when an iteration rejects nothing or after `maxiters`
/// iterations. Earlier rejections are never undone.
///
/// `output.reduction` reduces the retained samples in original order.
/// `output.diagnostics` fills [`Diagnostics`]: minimum/maximum retained value,
/// noise sigma, iteration counter, status bits, and restored flags.
///
/// # Errors
/// [`RejectError::MaskLength`] if `input_mask` length differs from `samples`;
/// [`RejectError::ScaleLength`] if `scales`/`zeros` are inconsistent.
/// Workspace contents are unchanged on error.
pub fn ccdclip<'a, T: Float>(
    samples: &[T],
    input_mask: Option<&[bool]>,
    params: &CcdClipParams<'_>,
    output: OutputOptions,
    workspace: &'a mut Workspace,
) -> Result<RejectResult<'a, T>, RejectError> {
    check_mask(samples, input_mask)?;
    let n = samples.len();
    let p = params;
    if !(p.scales.is_empty() && p.zeros.is_empty()) && (p.scales.len() != n || p.zeros.len() != n) {
        return Err(RejectError::ScaleLength {
            samples: n,
            scales: p.scales.len(),
            zeros: p.zeros.len(),
        });
    }
    let diagnostics = output.diagnostics;
    let Workspace {
        mask,
        newly_rejected,
        pairs,
        statistics: buf,
        restored_flags,
        restored_dirty,
    } = workspace;
    let pre_masked = input_mask.is_some_and(|m| m.iter().any(|&m| m));
    if diagnostics {
        reset_restored(restored_flags, restored_dirty, n);
    }
    // Working mask: true = currently rejected / non-finite.
    mask.resize(n, false);
    let mask = mask.as_mut_slice();
    let n_finite = init_mask(samples, input_mask, mask);

    let nkeep = p.nkeep;
    let maxrej = p.maxrej;

    let mut nit: u8 = 1;
    let mut std_out = T::nan();
    let mut n_finite_old = n_finite;
    let mut stopped_nkeep = false;
    let mut stopped_maxrej = false;
    let mut stopped_maxiters = p.maxiters == 0;
    // Shortcut only: with these limits the rollback tests below cannot trip.
    let unbounded = (!p.revert_on_nkeep || nkeep == 0) && maxrej == n;

    for k in 0..p.maxiters {
        stopped_maxiters = k + 1 == p.maxiters;
        let (cen, _std_center, std, spread_sq) =
            ccd_center_and_spread(samples, mask, p, diagnostics, buf);

        if cen.is_nan() || !spread_sq.is_finite() {
            break;
        }
        std_out = std;

        let center = cen.to_f64();
        newly_rejected.clear();
        // IRAF scans the value-sorted retained window from each end and stops
        // that side at the first sample within its own noise sigma.
        pairs.clear();
        pairs.extend(
            (0..n)
                .filter(|&i| !mask[i])
                .map(|i| (samples[i].to_f64(), i)),
        );
        pairs.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("unmasked ccdclip samples must be finite")
                .then(a.1.cmp(&b.1))
        });
        let mut lo = 0usize;
        let mut hi = pairs.len();
        while lo < hi {
            let (value, i) = pairs[lo];
            let residual = (center - value) / ccdclip_sample_sigma(p, center, i);
            if residual <= p.sigma_lower {
                break;
            }
            mask[i] = true;
            newly_rejected.push(i);
            lo += 1;
        }
        while hi > lo {
            let (value, i) = pairs[hi - 1];
            let residual = (value - center) / ccdclip_sample_sigma(p, center, i);
            if residual <= p.sigma_upper {
                break;
            }
            mask[i] = true;
            newly_rejected.push(i);
            hi -= 1;
        }
        let n_finite_new = n_finite_old - newly_rejected.len();

        if unbounded {
            let n_change = n_finite_old - n_finite_new;
            if n_change == 0 {
                stopped_maxiters = false;
                break;
            }
            n_finite_old = n_finite_new;
        } else {
            let nrej = n - n_finite_new;
            let mask_nkeep = p.revert_on_nkeep && n_finite_new < nkeep;
            let mask_maxrej = nrej > maxrej;
            let mask_pix = mask_nkeep || mask_maxrej;

            if mask_pix {
                stopped_nkeep = mask_nkeep;
                stopped_maxrej = mask_maxrej;
                stopped_maxiters = false;
                if diagnostics {
                    let restored_flag = ((mask_nkeep as u8) * RESTORED_NKEEP)
                        | ((mask_maxrej as u8) * RESTORED_MAXREJ);
                    for &i in newly_rejected.iter() {
                        restored_flags[i] |= restored_flag;
                    }
                    *restored_dirty = true;
                }
                // Revert this iteration's rejections; keep previous bounds.
                for &i in newly_rejected.iter() {
                    mask[i] = false;
                }
                break;
            } else {
                let n_change = n_finite_old - n_finite_new;
                if n_change == 0 {
                    stopped_maxiters = false;
                    break;
                }
                n_finite_old = n_finite_new;
            }
        }

        nit = nit.saturating_add(1);
    }

    let value = output
        .reduction
        .map(|kind| final_combine(samples, mask, kind, buf));
    let diagnostics = if diagnostics {
        let (low, upp) = col_minmax(samples, mask, buf);
        Some(Diagnostics {
            low,
            upp,
            std: std_out,
            nit,
            flags: (pre_masked as u8)
                | ((stopped_maxiters as u8) << 1)
                | ((stopped_nkeep as u8) << 2)
                | ((stopped_maxrej as u8) << 3),
            restored_flags,
        })
    } else {
        None
    };
    Ok(RejectResult {
        rejected: mask,
        retained: n_finite_old,
        value,
        diagnostics,
    })
}

// ---------- linearclip ----------

#[inline]
fn linear_center<T: Float>(vals: &[T], mask: &[bool], cenfunc: CenFunc, buf: &mut Vec<f64>) -> T {
    match cenfunc {
        CenFunc::Mean => col_nanmean(vals, mask, buf),
        CenFunc::Median => col_nanmedian(vals, mask, buf),
        CenFunc::LowerMedian => col_nanlmedian(vals, mask, buf),
    }
}

/// Clip one sample set against thresholds linear in the center, cumulatively.
///
/// Each iteration computes the center over the retained samples (stopping if
/// it is NaN, i.e. no sample is retained) and rejects every retained sample
/// outside the thresholds documented on [`LinearClipParams`]. Clipping stops
/// when an iteration rejects nothing (its thresholds are accepted), after
/// `maxiters` iterations, or when a rollback triggers: if `revert_on_nkeep`
/// and fewer than `nkeep` samples survive, or if more than `maxrej` samples
/// have been rejected by clipping (input-masked and nonfinite samples are not
/// counted), that iteration's rejections are undone. Earlier rejections are
/// never undone.
///
/// `output.reduction` reduces the retained samples in original order.
/// `output.diagnostics` fills [`Diagnostics`] with the last accepted
/// thresholds as `low`/`upp`, NaN `std`, the iteration counter, status bits
/// (the iteration-limit bit, value 2, is set only when `maxiters > 1` and the
/// last allowed iteration both changed the mask and was accepted), and
/// restored flags.
///
/// # Errors
/// [`RejectError::MaskLength`] if `input_mask` length differs from `samples`.
/// Workspace contents are unchanged on error.
pub fn linearclip<'a, T: Float>(
    samples: &[T],
    input_mask: Option<&[bool]>,
    params: &LinearClipParams,
    output: OutputOptions,
    workspace: &'a mut Workspace,
) -> Result<RejectResult<'a, T>, RejectError> {
    check_mask(samples, input_mask)?;
    let n = samples.len();
    let p = params;
    let diagnostics = output.diagnostics;
    let Workspace {
        mask,
        newly_rejected,
        statistics: buf,
        restored_flags,
        restored_dirty,
        ..
    } = workspace;
    let pre_masked = input_mask.is_some_and(|m| m.iter().any(|&m| m));
    if diagnostics {
        reset_restored(restored_flags, restored_dirty, n);
    }

    mask.resize(n, false);
    let mask = mask.as_mut_slice();
    let n_finite_initial = init_mask(samples, input_mask, mask);

    let mut retained = n_finite_initial;
    let mut low_out = T::nan();
    let mut upp_out = T::nan();
    let mut nit = 0u8;
    let mut stopped_nkeep = false;
    let mut stopped_maxrej = false;
    let mut stopped_maxiters = false;

    for iter in 0..p.maxiters {
        let center = linear_center(samples, mask, p.cenfunc, buf);
        if center.is_nan() {
            break;
        }
        let center = center.to_f64();
        let lo_try = p.low_scale * center - p.low;
        let up_try = p.upp_scale * center + p.upp;

        newly_rejected.clear();
        let mut n_finite_new = 0usize;
        for i in 0..n {
            if !mask[i] {
                let x = samples[i].to_f64();
                if x < lo_try || x > up_try {
                    mask[i] = true;
                    newly_rejected.push(i);
                } else {
                    n_finite_new += 1;
                }
            }
        }

        nit = nit.saturating_add(1);
        if newly_rejected.is_empty() {
            low_out = T::from_f64(lo_try);
            upp_out = T::from_f64(up_try);
            break;
        }

        let nrej = n_finite_initial.saturating_sub(n_finite_new);
        let mask_nkeep = p.revert_on_nkeep && n_finite_new < p.nkeep;
        let mask_maxrej = nrej > p.maxrej;
        if mask_nkeep || mask_maxrej {
            stopped_nkeep = mask_nkeep;
            stopped_maxrej = mask_maxrej;
            if diagnostics {
                let restored_flag =
                    ((mask_nkeep as u8) * RESTORED_NKEEP) | ((mask_maxrej as u8) * RESTORED_MAXREJ);
                for &i in newly_rejected.iter() {
                    restored_flags[i] |= restored_flag;
                }
                *restored_dirty = true;
            }
            for &i in newly_rejected.iter() {
                mask[i] = false;
            }
            break;
        }

        retained = n_finite_new;
        low_out = T::from_f64(lo_try);
        upp_out = T::from_f64(up_try);
        if iter + 1 == p.maxiters {
            stopped_maxiters = p.maxiters > 1;
        }
    }

    let value = output
        .reduction
        .map(|kind| final_combine(samples, mask, kind, buf));
    let diagnostics = diagnostics.then(|| Diagnostics {
        low: low_out,
        upp: upp_out,
        std: T::nan(),
        nit,
        flags: (pre_masked as u8)
            | ((stopped_maxiters as u8) << 1)
            | ((stopped_nkeep as u8) << 2)
            | ((stopped_maxrej as u8) << 3),
        restored_flags,
    });
    Ok(RejectResult {
        rejected: mask,
        retained,
        value,
        diagnostics,
    })
}

// ---------- minmax ----------

/// Fill `mask` (input mask or nonfinite) and `pairs` (finite, unmasked values
/// with their indices, in input order) from `(value, input_masked)` items.
/// Returns whether any item was input-masked.
// Forced: without it LLVM may keep this loop out of line, which costs a few
// ns per call relative to the fused gather loop imcombiners used before.
#[inline(always)]
fn fill_pairs<T: Float, I: IntoIterator<Item = (T, bool)>>(
    samples: I,
    mask: &mut Vec<bool>,
    pairs: &mut Vec<(f64, usize)>,
) -> bool
where
    I::IntoIter: ExactSizeIterator,
{
    let samples = samples.into_iter();
    let n = samples.len();
    mask.resize(n, false);
    pairs.clear();
    let mut pre_masked = false;
    let mut filled = 0;
    for ((x, masked_by_input), m) in samples.zip(mask.iter_mut()) {
        pre_masked |= masked_by_input;
        let masked = masked_by_input || !x.is_finite();
        *m = masked;
        if !masked {
            pairs.push((x.to_f64(), filled));
        }
        filled += 1;
    }
    // An iterator that reports a wrong length is a bug; never keep stale
    // entries from an earlier call.
    debug_assert_eq!(filled, n, "ExactSizeIterator reported a wrong length");
    mask.truncate(filled);
    pre_masked
}

/// Returns `(low, upp, retained)`.
fn apply_minmax_pairs<T: Float>(
    finite_pairs: &mut [(f64, usize)],
    mask: &mut [bool],
    q_low: f64,
    q_upp: f64,
    diagnostics: bool,
) -> (T, T, usize) {
    let n_finite = finite_pairs.len();
    let n_rej_low = (n_finite as f64 * q_low + 0.001) as usize;
    let n_rej_upp = (n_finite as f64 * q_upp + 0.001) as usize;

    // Stable sort: ties keep input order, which decides the rejected duplicate.
    finite_pairs.sort_by(|a, b| f64::total_cmp(&a.0, &b.0));
    let low_count = n_rej_low.min(n_finite);
    for &(_, k) in finite_pairs.iter().take(low_count) {
        mask[k] = true;
    }
    let high_count = n_rej_upp.min(n_finite.saturating_sub(n_rej_low));
    for &(_, k) in finite_pairs.iter().rev().take(high_count) {
        mask[k] = true;
    }
    let retained = n_finite - low_count - high_count;

    if diagnostics && n_rej_low + high_count < n_finite {
        (
            T::from_f64(finite_pairs[n_rej_low].0),
            T::from_f64(finite_pairs[n_finite - high_count - 1].0),
            retained,
        )
    } else {
        (T::nan(), T::nan(), retained)
    }
}

/// Reject the lowest and highest finite, unmasked samples by rank (IRAF
/// `minmax`).
///
/// The rejection counts are documented on [`MinMaxParams`]. Samples are ranked
/// by value with a stable sort (``f64::total_cmp``, so `-0.0 < +0.0`), and
/// equal values keep input order: among ties, the earliest are rejected from
/// the low end and the latest from the high end. If the counts cover every
/// finite sample, all are rejected.
///
/// `output.reduction` reduces the retained samples in ascending value order.
/// `output.diagnostics` fills [`Diagnostics`] with the minimum/maximum
/// retained values, NaN `std`, `nit = 1`, and status bit 1.
///
/// # Errors
/// [`RejectError::MaskLength`] if `input_mask` length differs from `samples`.
/// Workspace contents are unchanged on error.
pub fn minmax<'a, T: Float>(
    samples: &[T],
    input_mask: Option<&[bool]>,
    params: &MinMaxParams,
    output: OutputOptions,
    workspace: &'a mut Workspace,
) -> Result<RejectResult<'a, T>, RejectError> {
    check_mask(samples, input_mask)?;
    Ok(match input_mask {
        Some(mask) => minmax_iter(
            samples.iter().copied().zip(mask.iter().copied()),
            params,
            output,
            workspace,
        ),
        None => minmax_iter(
            samples.iter().map(|&x| (x, false)),
            params,
            output,
            workspace,
        ),
    })
}

/// [`minmax`] with samples given as `(value, input_masked)` items in sample
/// order, for example a strided image-stack column, without first gathering
/// them into slices. Item `k` is reported at index `k` of
/// [`RejectResult::rejected`]. The iterator must report its exact length.
#[inline]
pub fn minmax_iter<'a, T: Float, I>(
    samples: I,
    params: &MinMaxParams,
    output: OutputOptions,
    workspace: &'a mut Workspace,
) -> RejectResult<'a, T>
where
    I: IntoIterator<Item = (T, bool)>,
    I::IntoIter: ExactSizeIterator,
{
    let Workspace {
        mask,
        pairs,
        statistics,
        ..
    } = workspace;
    let pre_masked = fill_pairs(samples, mask, pairs);
    let (low, upp, retained) =
        apply_minmax_pairs::<T>(pairs, mask, params.q_low, params.q_upp, output.diagnostics);
    let value = output
        .reduction
        .map(|kind| final_combine_from_sorted_pairs(pairs, mask, kind, statistics));
    RejectResult {
        rejected: mask,
        retained,
        value,
        diagnostics: output.diagnostics.then(|| Diagnostics {
            low,
            upp,
            std: T::nan(),
            nit: 1,
            flags: pre_masked as u8,
            restored_flags: &[],
        }),
    }
}

// ---------- pclip ----------

/// Returns `(low, upp, retained)`.
// Kept out of line so the small fill and reduction loops inline into the
// caller, as in the imcombiners 0.1.2 layout; measured faster for short stacks.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn apply_pclip_pairs<T: Float>(
    pairs: &mut [(f64, usize)],
    mask: &mut [bool],
    resid: &mut Vec<f64>,
    pclip_offset: isize,
    sigma_lower: f64,
    sigma_upper: f64,
    nkeep: usize,
    diagnostics: bool,
) -> (T, T, usize) {
    let n_total = pairs.len();
    if n_total == 0 || n_total < PCLIP_MINCLIP || n_total < nkeep.saturating_add(1) {
        let (low, upp) = if diagnostics {
            col_minmax_from_masked_pairs::<T>(pairs, mask, resid)
        } else {
            (T::nan(), T::nan())
        };
        return (low, upp, n_total);
    }

    pairs.sort_by(|a, b| f64::total_cmp(&a.0, &b.0));

    let n2_1based = 1 + n_total / 2;
    let even = n_total % 2 == 0;
    let med = if even {
        // Preserve the sorted sample IDs; the median helper handles the finite
        // midpoint without overflowing when the two central values share a sign.
        median_valid_in_place(&mut [pairs[n2_1based - 2].0, pairs[n2_1based - 1].0])
    } else {
        pairs[n2_1based - 1].0
    };

    let n3_1based = if pclip_offset < 0 {
        let base = if even { n2_1based - 1 } else { n2_1based };
        iraf_nint(base as f64 + pclip_offset as f64)
            .max(1)
            .min(n_total as isize) as usize
    } else {
        iraf_nint(n2_1based as f64 + pclip_offset as f64)
            .max(1)
            .min(n_total as isize) as usize
    };
    let sign = if pclip_offset < 0 { -1.0 } else { 1.0 };
    let sigma = sign * (pairs[n3_1based - 1].0 - med);
    if sigma == 0.0 {
        let (low, upp) = if diagnostics {
            col_minmax_from_masked_pairs::<T>(pairs, mask, resid)
        } else {
            (T::nan(), T::nan())
        };
        return (low, upp, n_total);
    }

    resid.clear();
    resid.resize(n_total, 0.0);
    let mut nl = 0usize;
    while nl < n_total {
        let r = (med - pairs[nl].0) / sigma;
        if r < sigma_lower {
            break;
        }
        resid[nl] = r;
        nl += 1;
    }
    let mut nh = n_total - 1;
    loop {
        if nh < nl {
            break;
        }
        let r = (pairs[nh].0 - med) / sigma;
        if r < sigma_upper {
            break;
        }
        resid[nh] = r;
        if nh == 0 {
            break;
        }
        nh -= 1;
    }

    let maxkeep = nkeep.min(n_total);
    let mut n_kept = if nh >= nl { nh - nl + 1 } else { 0 };
    while n_kept < maxkeep {
        if nl == 0 {
            nh += 1;
        } else if nh + 1 == n_total {
            nl -= 1;
        } else {
            let r = resid[nl - 1];
            let s = resid[nh + 1];
            if r < s {
                nl -= 1;
                let r_tol = r + IRAF_RESID_TOL;
                if s <= r_tol {
                    nh += 1;
                }
                if nl > 0 && resid[nl - 1] <= r_tol {
                    nl -= 1;
                }
            } else {
                nh += 1;
                let s_tol = s + IRAF_RESID_TOL;
                if r <= s_tol {
                    nl -= 1;
                }
                if nh + 1 < n_total && resid[nh + 1] <= s_tol {
                    nh += 1;
                }
            }
        }
        n_kept = nh - nl + 1;
    }

    let mut retained = 0usize;
    for (rank, &(_, k)) in pairs.iter().enumerate() {
        if rank < nl || rank > nh {
            mask[k] = true;
        } else {
            retained += 1;
        }
    }

    let (low, upp) = if diagnostics {
        col_minmax_from_masked_pairs::<T>(pairs, mask, resid)
    } else {
        (T::nan(), T::nan())
    };
    (low, upp, retained)
}

/// Percentile clipping about the median (IRAF `pclip`), in one pass.
///
/// Let `m` be the number of finite, unmasked samples. Nothing is rejected if
/// `m < 3` or `m <= nkeep`. Otherwise the samples are sorted (stable,
/// ``f64::total_cmp``), the median `med` is computed (midpoint of the two
/// central values for even `m`), and ``sigma = ±(x[r] - med)`` uses the sample
/// at 1-based rank ``r = nint(base + offset)`` clamped to `[1, m]`, where
/// ``base = m/2`` when `offset < 0` and `m` is even, and ``base = 1 + m/2``
/// otherwise (integer division); the sign makes `sigma >= 0`. If
/// `sigma == 0`, nothing is rejected. Otherwise samples are rejected from each
/// sorted end while ``(med - x)/sigma >= sigma_lower`` (low end) or
/// ``(x - med)/sigma >= sigma_upper`` (high end). Finally, while fewer than
/// ``min(nkeep, m)`` samples remain, the rejected neighbor with the smaller
/// residual is restored, together with any neighbor within 0.001 of it.
///
/// `output.reduction` reduces the retained samples in ascending value order
/// (input order when nothing could be rejected because of `m` or `nkeep`).
/// `output.diagnostics` fills [`Diagnostics`] with the minimum/maximum
/// retained values, NaN `std`, `nit = 1`, and status bit 1.
///
/// # Errors
/// [`RejectError::MaskLength`] if `input_mask` length differs from `samples`.
/// Workspace contents are unchanged on error.
pub fn pclip<'a, T: Float>(
    samples: &[T],
    input_mask: Option<&[bool]>,
    params: &PclipParams,
    output: OutputOptions,
    workspace: &'a mut Workspace,
) -> Result<RejectResult<'a, T>, RejectError> {
    check_mask(samples, input_mask)?;
    Ok(match input_mask {
        Some(mask) => pclip_iter(
            samples.iter().copied().zip(mask.iter().copied()),
            params,
            output,
            workspace,
        ),
        None => pclip_iter(
            samples.iter().map(|&x| (x, false)),
            params,
            output,
            workspace,
        ),
    })
}

/// [`pclip`] with samples given as `(value, input_masked)` items in sample
/// order, for example a strided image-stack column, without first gathering
/// them into slices. Item `k` is reported at index `k` of
/// [`RejectResult::rejected`]. The iterator must report its exact length.
// Forced so the fill and reduction loops inline into each caller while the
// rejection core (`apply_pclip_pairs`) stays one out-of-line call.
#[inline(always)]
pub fn pclip_iter<'a, T: Float, I>(
    samples: I,
    params: &PclipParams,
    output: OutputOptions,
    workspace: &'a mut Workspace,
) -> RejectResult<'a, T>
where
    I: IntoIterator<Item = (T, bool)>,
    I::IntoIter: ExactSizeIterator,
{
    let Workspace {
        mask,
        pairs,
        statistics,
        ..
    } = workspace;
    let pre_masked = fill_pairs(samples, mask, pairs);
    let (low, upp, retained) = apply_pclip_pairs::<T>(
        pairs,
        mask,
        statistics,
        params.offset,
        params.sigma_lower,
        params.sigma_upper,
        params.nkeep,
        output.diagnostics,
    );
    let value = output
        .reduction
        .map(|kind| final_combine_from_sorted_pairs(pairs, mask, kind, statistics));
    RejectResult {
        rejected: mask,
        retained,
        value,
        diagnostics: output.diagnostics.then(|| Diagnostics {
            low,
            upp,
            std: T::nan(),
            nit: 1,
            flags: pre_masked as u8,
            restored_flags: &[],
        }),
    }
}
