//! Tests for `reducers::reject`.
//!
//! The first part checks hand-derived behavior. The second part compares the
//! public functions bit for bit against a frozen copy of the per-pixel code
//! that `imcombiners` 0.1.2 used before the move (module `legacy`), over
//! hand-picked edge cases and random stacks. One workspace is reused across
//! all calls to catch stale scratch state.

use reducers::reducers_1d::{
    lmedian_valid_in_place, mean_valid, median_valid_in_place, minmax_valid,
};
use reducers::reject::{
    self, CcdClipParams, LinearClipParams, MinMaxParams, PclipParams, RejectError, Workspace,
    RESTORED_MAXREJ, RESTORED_NKEEP,
};
use reducers::sigma_clip::{CenFunc, ClipCenter, OutputOptions, Reduction};
use reducers::Float;

const DIAG: OutputOptions = OutputOptions {
    reduction: None,
    diagnostics: true,
};
const MASK_ONLY: OutputOptions = OutputOptions {
    reduction: None,
    diagnostics: false,
};

fn reduce(kind: Reduction) -> OutputOptions {
    OutputOptions {
        reduction: Some(kind),
        diagnostics: false,
    }
}

fn ccd_params(scales: &'static [f64], zeros: &'static [f64]) -> CcdClipParams<'static> {
    CcdClipParams {
        sigma_lower: 3.0,
        sigma_upper: 3.0,
        maxiters: 5,
        nkeep: 1,
        maxrej: usize::MAX,
        revert_on_nkeep: true,
        cenfunc: CenFunc::Median,
        clip_cen: ClipCenter::ClippingCenter,
        rdnoise: 0.0,
        snoise: 0.0,
        gain: 1.0,
        scales,
        zeros,
    }
}

// ---------- hand-derived behavior ----------

#[test]
fn minmax_rejects_ranks_and_breaks_ties_by_input_order() {
    let mut ws = Workspace::new();
    // Two lowest ties (1.0 at indices 0 and 2) and two highest ties (9.0 at 1, 4).
    let x = [1.0_f64, 9.0, 1.0, 5.0, 9.0];
    let p = MinMaxParams {
        q_low: 0.2,
        q_upp: 0.2,
    };
    let r = reject::minmax(&x, None, &p, DIAG, &mut ws).unwrap();
    // Low end rejects the earliest tie; high end rejects the latest tie.
    assert_eq!(r.rejected, &[true, false, false, false, true]);
    assert_eq!(r.retained, 3);
    let d = r.diagnostics.unwrap();
    assert_eq!((d.low, d.upp, d.nit, d.flags), (1.0, 9.0, 1, 0));
    assert!(d.std.is_nan());
    assert!(d.restored_flags.is_empty());
}

#[test]
fn minmax_orders_signed_zeros_and_skips_nonfinite() {
    let mut ws = Workspace::new();
    let x = [0.0_f32, -0.0, f32::NAN, f32::INFINITY, 3.0];
    let p = MinMaxParams {
        q_low: 0.34,
        q_upp: 0.0,
    };
    // Three finite samples: int(3 * 0.34 + 0.001) = 1 low rejection, -0.0 first.
    let r = reject::minmax(
        &x,
        Some(&[false, false, false, false, false]),
        &p,
        DIAG,
        &mut ws,
    )
    .unwrap();
    assert_eq!(r.rejected, &[false, true, true, true, false]);
    assert_eq!(r.retained, 2);
    let d = r.diagnostics.unwrap();
    assert_eq!(d.low.to_bits(), 0.0_f32.to_bits());
    assert_eq!(d.upp, 3.0);
}

#[test]
fn minmax_all_masked_and_overlapping_counts_reject_everything() {
    let mut ws = Workspace::new();
    let x = [1.0_f64, 2.0, 3.0];
    let r = reject::minmax(
        &x,
        Some(&[true, true, true]),
        &MinMaxParams {
            q_low: 0.5,
            q_upp: 0.5,
        },
        OutputOptions {
            reduction: Some(Reduction::Mean),
            diagnostics: true,
        },
        &mut ws,
    )
    .unwrap();
    assert_eq!(r.rejected, &[true; 3]);
    assert_eq!(r.retained, 0);
    assert!(r.value.unwrap().is_nan());
    let d = r.diagnostics.unwrap();
    assert!(d.low.is_nan() && d.upp.is_nan());
    assert_eq!(d.flags, 1);

    let r = reject::minmax(
        &x,
        None,
        &MinMaxParams {
            q_low: 1.0,
            q_upp: 1.0,
        },
        MASK_ONLY,
        &mut ws,
    )
    .unwrap();
    assert_eq!(r.rejected, &[true; 3]);
    assert_eq!(r.retained, 0);
}

#[test]
fn pclip_rejects_tails_and_restores_to_nkeep() {
    let mut ws = Workspace::new();
    let x = [10.0_f64, 11.0, 9.0, 10.0, 100.0, 10.5, -80.0];
    let p = PclipParams {
        offset: reject::pclip_offset(-0.5, x.len()),
        sigma_lower: 3.0,
        sigma_upper: 3.0,
        nkeep: 0,
    };
    assert_eq!(p.offset, -1);
    // Sorted: -80, 9, 10, 10, 10.5, 11, 100; median 10; rank 3 gives sigma 0.
    let r = reject::pclip(&x, None, &p, DIAG, &mut ws).unwrap();
    assert_eq!(r.rejected, &[false; 7]);
    // Rank offset -2 gives sigma = 10 - 9 = 1: both outliers are rejected.
    let p = PclipParams { offset: -2, ..p };
    let r = reject::pclip(&x, None, &p, DIAG, &mut ws).unwrap();
    assert_eq!(r.rejected, &[false, false, false, false, true, false, true]);
    assert_eq!(r.retained, 5);
    let d = r.diagnostics.unwrap();
    assert_eq!((d.low, d.upp), (9.0, 11.0));
    // Fewer than nkeep + 1 retained samples: nothing is rejected.
    let p = PclipParams { nkeep: 7, ..p };
    assert_eq!(
        reject::pclip(&x, None, &p, MASK_ONLY, &mut ws)
            .unwrap()
            .retained,
        7
    );
}

#[test]
fn pclip_offset_matches_iraf_clamping() {
    assert_eq!(reject::pclip_offset(-0.5, 10), -2);
    assert_eq!(reject::pclip_offset(0.5, 10), 2);
    assert_eq!(reject::pclip_offset(0.0, 10), 1);
    assert_eq!(reject::pclip_offset(-0.01, 10), -1);
    assert_eq!(reject::pclip_offset(-9.0, 10), -5);
    assert_eq!(reject::pclip_offset(9.0, 10), 5);
    assert_eq!(reject::pclip_offset(-0.5, 1), -1);
    assert_eq!(reject::pclip_offset(2.7, 0), 1);
}

#[test]
fn ccdclip_rejects_outlier_with_poisson_noise() {
    let mut ws = Workspace::with_capacity(6);
    // Center 100, gain 1, no read noise: sigma = 10, so 3 sigma = 30.
    let x = [100.0_f64, 95.0, 105.0, 100.0, 160.0, 100.0];
    let r = reject::ccdclip(&x, None, &ccd_params(&[], &[]), DIAG, &mut ws).unwrap();
    assert_eq!(r.rejected, &[false, false, false, false, true, false]);
    assert_eq!(r.retained, 5);
    let d = r.diagnostics.unwrap();
    assert_eq!(
        (d.low, d.upp, d.std, d.nit, d.flags),
        (95.0, 105.0, 10.0, 2, 0)
    );
    assert_eq!(d.restored_flags, &[0; 6]);
}

#[test]
fn ccdclip_rollback_marks_restored_samples() {
    let mut ws = Workspace::new();
    let x = [100.0_f64, 100.0, 100.0, 160.0, 40.0];
    let p = CcdClipParams {
        nkeep: 4,
        ..ccd_params(&[], &[])
    };
    let r = reject::ccdclip(&x, None, &p, DIAG, &mut ws).unwrap();
    assert_eq!(r.rejected, &[false; 5]);
    assert_eq!(r.retained, 5);
    let d = r.diagnostics.unwrap();
    assert_eq!(d.flags, 4);
    assert_eq!(d.restored_flags, &[0, 0, 0, RESTORED_NKEEP, RESTORED_NKEEP]);

    // maxrej counts initial exclusions: one NaN plus two tentative rejections.
    let x = [100.0_f64, 100.0, f64::NAN, 160.0, 40.0, 100.0];
    let p = CcdClipParams {
        maxrej: 2,
        ..ccd_params(&[], &[])
    };
    let r = reject::ccdclip(&x, None, &p, DIAG, &mut ws).unwrap();
    assert_eq!(r.rejected, &[false, false, true, false, false, false]);
    let d = r.diagnostics.unwrap();
    assert_eq!(d.flags, 8);
    assert_eq!(
        d.restored_flags,
        &[0, 0, 0, RESTORED_MAXREJ, RESTORED_MAXREJ, 0]
    );
}

#[test]
fn ccdclip_scaled_noise_uses_input_units() {
    let mut ws = Workspace::new();
    // Normalized center 100 and scale 4 give input signal 400 and input sigma
    // 20, i.e. normalized sigma 5. A residual of 16 (> 3 * 5) is rejected.
    let x = [100.0_f64, 100.0, 116.0, 100.0];
    let p = ccd_params(&[4.0; 4], &[0.0; 4]);
    let r = reject::ccdclip(&x, None, &p, MASK_ONLY, &mut ws).unwrap();
    assert_eq!(r.rejected, &[false, false, true, false]);
    // Unscaled: sigma 10, residual 16 survives.
    let r = reject::ccdclip(&x, None, &ccd_params(&[], &[]), MASK_ONLY, &mut ws).unwrap();
    assert_eq!(r.rejected, &[false; 4]);
}

#[test]
fn linearclip_maxrej_ignores_initial_exclusions_and_mask_includes_them() {
    let mut ws = Workspace::new();
    let x = [10.0_f64, 10.0, 50.0, f64::NAN, 10.0, -30.0];
    let p = LinearClipParams {
        low_scale: 1.0,
        low: 5.0,
        upp_scale: 1.0,
        upp: 5.0,
        maxiters: 3,
        nkeep: 0,
        maxrej: 2,
        cenfunc: CenFunc::Median,
        revert_on_nkeep: false,
    };
    let mask = [true, false, false, false, false, false];
    let r = reject::linearclip(&x, Some(&mask), &p, DIAG, &mut ws).unwrap();
    assert_eq!(r.rejected, &[true, false, true, true, false, true]);
    assert_eq!(r.retained, 2);
    let d = r.diagnostics.unwrap();
    // Iteration 2 rejects nothing, so its thresholds are accepted.
    assert_eq!((d.low, d.upp, d.nit, d.flags), (5.0, 15.0, 2, 1));
    let p = LinearClipParams { maxrej: 1, ..p };
    let r = reject::linearclip(&x, Some(&mask), &p, DIAG, &mut ws).unwrap();
    assert_eq!(r.rejected, &[true, false, false, true, false, false]);
    let d = r.diagnostics.unwrap();
    assert_eq!(d.flags, 1 | 8);
    assert_eq!(
        d.restored_flags,
        &[0, 0, RESTORED_MAXREJ, 0, 0, RESTORED_MAXREJ]
    );
    assert!(d.low.is_nan() && d.upp.is_nan());
}

#[test]
fn reductions_follow_retained_samples() {
    let mut ws = Workspace::new();
    let x = [1.0_f64, 2.0, 3.0, 4.0, 100.0];
    let mm = MinMaxParams {
        q_low: 0.0,
        q_upp: 0.2,
    };
    let mean = reject::minmax(&x, None, &mm, reduce(Reduction::Mean), &mut ws).unwrap();
    assert_eq!(mean.value, Some(2.5));
    let median = reject::minmax(&x, None, &mm, reduce(Reduction::Median), &mut ws).unwrap();
    assert_eq!(median.value, Some(2.5));
    let ccd = reject::ccdclip(
        &[100.0_f32, 160.0, 100.0, 100.0],
        None,
        &ccd_params(&[], &[]),
        reduce(Reduction::Mean),
        &mut ws,
    )
    .unwrap();
    assert_eq!(ccd.value, Some(100.0));
}

#[test]
fn invalid_lengths_are_reported_without_touching_workspace() {
    let mut ws = Workspace::new();
    let x = [1.0_f64, 2.0, 3.0];
    let mm = MinMaxParams {
        q_low: 0.0,
        q_upp: 0.0,
    };
    assert_eq!(
        reject::minmax(&x, Some(&[false]), &mm, MASK_ONLY, &mut ws).unwrap_err(),
        RejectError::MaskLength {
            samples: 3,
            mask: 1
        }
    );
    let p = ccd_params(&[1.0, 1.0, 1.0], &[]);
    assert_eq!(
        reject::ccdclip(&x, None, &p, MASK_ONLY, &mut ws).unwrap_err(),
        RejectError::ScaleLength {
            samples: 3,
            scales: 3,
            zeros: 0
        }
    );
    assert!(!RejectError::ScaleLength {
        samples: 3,
        scales: 3,
        zeros: 0
    }
    .to_string()
    .is_empty());
}

#[test]
fn empty_input_is_valid() {
    let mut ws = Workspace::new();
    let x: [f64; 0] = [];
    let opts = OutputOptions {
        reduction: Some(Reduction::Median),
        diagnostics: true,
    };
    let r = reject::ccdclip(&x, None, &ccd_params(&[], &[]), opts, &mut ws).unwrap();
    assert!(r.rejected.is_empty() && r.retained == 0 && r.value.unwrap().is_nan());
    let mm = MinMaxParams {
        q_low: 0.5,
        q_upp: 0.5,
    };
    assert_eq!(
        reject::minmax(&x, None, &mm, opts, &mut ws)
            .unwrap()
            .retained,
        0
    );
}

// ---------- frozen imcombiners 0.1.2 per-pixel code ----------

/// Verbatim per-pixel logic from imcombiners 0.1.2 `src/kernel/reject.rs`,
/// with parameter structs renamed to the reducers types. Do not "fix" it:
/// it is the reference for bit-identical behavior.
mod legacy {
    use super::*;
    use std::cmp::Ordering;

    pub struct Out<T> {
        pub mask: Vec<bool>,
        pub low: T,
        pub upp: T,
        pub std: T,
        pub nit: u8,
        pub flags: u8,
        pub restored: Vec<u8>,
        pub value: T,
    }

    fn col_nanmean<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
        buf.clear();
        for (i, &x) in vals.iter().enumerate() {
            if !mask[i] {
                buf.push(x.to_f64());
            }
        }
        T::from_f64(mean_valid(buf))
    }

    fn col_nanmedian<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
        buf.clear();
        for (i, &x) in vals.iter().enumerate() {
            if !mask[i] {
                buf.push(x.to_f64());
            }
        }
        if buf.is_empty() {
            return T::nan();
        }
        T::from_f64(median_valid_in_place(buf))
    }

    fn col_nanlmedian<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> T {
        buf.clear();
        for (i, &x) in vals.iter().enumerate() {
            if !mask[i] {
                buf.push(x.to_f64());
            }
        }
        if buf.is_empty() {
            return T::nan();
        }
        T::from_f64(lmedian_valid_in_place(buf))
    }

    fn col_minmax<T: Float>(vals: &[T], mask: &[bool], buf: &mut Vec<f64>) -> (T, T) {
        buf.clear();
        for (i, &x) in vals.iter().enumerate() {
            if !mask[i] {
                buf.push(x.to_f64());
            }
        }
        let (lo, upp) = minmax_valid(buf);
        (T::from_f64(lo), T::from_f64(upp))
    }

    fn final_combine<T: Float>(
        vals: &[T],
        mask: &[bool],
        kind: Reduction,
        buf: &mut Vec<f64>,
    ) -> T {
        match kind {
            Reduction::Mean => col_nanmean(vals, mask, buf),
            Reduction::Median => col_nanmedian(vals, mask, buf),
        }
    }

    fn final_combine_from_sorted_pairs<T: Float, V: Float>(
        pairs: &[(V, usize)],
        mask: &[bool],
        kind: Reduction,
        buf: &mut Vec<f64>,
    ) -> T {
        buf.clear();
        for &(value, k) in pairs {
            if !mask[k] {
                buf.push(value.to_f64());
            }
        }
        T::from_f64(match kind {
            Reduction::Mean => mean_valid(buf),
            Reduction::Median => median_valid_in_place(buf),
        })
    }

    fn ccdclip_sample_sigma(p: &CcdClipParams, center: f64, i: usize) -> f64 {
        const NM1_FLOOR: f64 = 1e4 / 3.402_823_466e38;
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

    fn ccd_center_and_spread<T: Float>(
        col: &[T],
        mask: &[bool],
        p: &CcdClipParams,
        diagnostics: bool,
        buf: &mut Vec<f64>,
    ) -> (T, T, f64) {
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
        let (std, spread_sq) = if std_center.is_nan() {
            (T::nan(), f64::NAN)
        } else {
            let c = std_center.to_f64();
            const NM1_FLOOR: f64 = 1e4 / 3.402_823_466e38;
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
        (cen, std, spread_sq)
    }

    pub fn ccdclip<T: Float>(
        col: &[T],
        mask_in: &[bool],
        p: &CcdClipParams,
        kind: Reduction,
    ) -> Out<T> {
        let n = col.len();
        let diagnostics = true;
        let pre_masked = mask_in.iter().any(|&m| m);
        let mut flags = vec![0u8; n];
        let mut buf = Vec::new();
        let mut newly_masked = Vec::new();
        let mut sorted: Vec<(f64, usize)> = Vec::new();
        let mut mask = vec![false; n];
        let mut n_finite: usize = 0;
        for i in 0..n {
            if mask_in[i] || !col[i].is_finite() {
                mask[i] = true;
            } else {
                mask[i] = false;
                n_finite += 1;
            }
        }
        let nkeep = p.nkeep;
        let maxrej = p.maxrej;
        let mut nit: u8 = 1;
        let mut std_out = T::nan();
        let mut n_finite_old = n_finite;
        let mut stopped_nkeep = false;
        let mut stopped_maxrej = false;
        let mut stopped_maxiters = p.maxiters == 0;
        let unbounded = (!p.revert_on_nkeep || nkeep == 0) && maxrej == n;
        for k in 0..p.maxiters {
            stopped_maxiters = k + 1 == p.maxiters;
            let (cen, std, spread_sq) = ccd_center_and_spread(col, &mask, p, diagnostics, &mut buf);
            if cen.is_nan() || !spread_sq.is_finite() {
                break;
            }
            std_out = std;
            let center = cen.to_f64();
            newly_masked.clear();
            sorted.clear();
            sorted.extend((0..n).filter(|&i| !mask[i]).map(|i| (col[i].to_f64(), i)));
            sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            let mut lo = 0usize;
            let mut hi = sorted.len();
            while lo < hi {
                let (value, i) = sorted[lo];
                let residual = (center - value) / ccdclip_sample_sigma(p, center, i);
                if residual <= p.sigma_lower {
                    break;
                }
                mask[i] = true;
                newly_masked.push(i);
                lo += 1;
            }
            while hi > lo {
                let (value, i) = sorted[hi - 1];
                let residual = (value - center) / ccdclip_sample_sigma(p, center, i);
                if residual <= p.sigma_upper {
                    break;
                }
                mask[i] = true;
                newly_masked.push(i);
                hi -= 1;
            }
            let n_finite_new = n_finite_old - newly_masked.len();
            if unbounded {
                if n_finite_old == n_finite_new {
                    stopped_maxiters = false;
                    break;
                }
                n_finite_old = n_finite_new;
            } else {
                let nrej = n - n_finite_new;
                let mask_nkeep = p.revert_on_nkeep && n_finite_new < nkeep;
                let mask_maxrej = nrej > maxrej;
                if mask_nkeep || mask_maxrej {
                    stopped_nkeep = mask_nkeep;
                    stopped_maxrej = mask_maxrej;
                    stopped_maxiters = false;
                    let restored_flag = ((mask_nkeep as u8) * 64) | ((mask_maxrej as u8) * 128);
                    for &i in &newly_masked {
                        flags[i] |= restored_flag;
                    }
                    for &i in &newly_masked {
                        mask[i] = false;
                    }
                    break;
                } else {
                    if n_finite_old == n_finite_new {
                        stopped_maxiters = false;
                        break;
                    }
                    n_finite_old = n_finite_new;
                }
            }
            nit = nit.saturating_add(1);
        }
        let (low, upp) = col_minmax(col, &mask, &mut buf);
        let value = final_combine(col, &mask, kind, &mut buf);
        Out {
            mask,
            low,
            upp,
            std: std_out,
            nit,
            flags: (pre_masked as u8)
                | ((stopped_maxiters as u8) << 1)
                | ((stopped_nkeep as u8) << 2)
                | ((stopped_maxrej as u8) << 3),
            restored: flags,
            value,
        }
    }

    /// Returns the working mask (input mask included), as before the
    /// imcombiners-only `&& !mask_in` output step.
    pub fn linearclip<T: Float>(
        col: &[T],
        mask_in: &[bool],
        p: &LinearClipParams,
        kind: Reduction,
    ) -> Out<T> {
        let n = col.len();
        let pre_masked = mask_in.iter().any(|&m| m);
        let mut flags = vec![0u8; n];
        let mut buf = Vec::new();
        let mut newly_masked = Vec::new();
        let mut mask = vec![false; n];
        let mut n_finite_initial = 0usize;
        for i in 0..n {
            if mask_in[i] || !col[i].is_finite() {
                mask[i] = true;
            } else {
                mask[i] = false;
                n_finite_initial += 1;
            }
        }
        let mut low_out = T::nan();
        let mut upp_out = T::nan();
        let mut nit = 0u8;
        let mut stopped_nkeep = false;
        let mut stopped_maxrej = false;
        let mut stopped_maxiters = false;
        for iter in 0..p.maxiters {
            let center = match p.cenfunc {
                CenFunc::Mean => col_nanmean(col, &mask, &mut buf),
                CenFunc::Median => col_nanmedian(col, &mask, &mut buf),
                CenFunc::LowerMedian => col_nanlmedian(col, &mask, &mut buf),
            };
            if center.is_nan() {
                break;
            }
            let center = center.to_f64();
            let lo_try = p.low_scale * center - p.low;
            let up_try = p.upp_scale * center + p.upp;
            newly_masked.clear();
            let mut n_finite_new = 0usize;
            for i in 0..n {
                if !mask[i] {
                    let x = col[i].to_f64();
                    if x < lo_try || x > up_try {
                        mask[i] = true;
                        newly_masked.push(i);
                    } else {
                        n_finite_new += 1;
                    }
                }
            }
            nit = nit.saturating_add(1);
            if newly_masked.is_empty() {
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
                let restored_flag = ((mask_nkeep as u8) * 64) | ((mask_maxrej as u8) * 128);
                for &i in &newly_masked {
                    flags[i] |= restored_flag;
                }
                for &i in &newly_masked {
                    mask[i] = false;
                }
                break;
            }
            low_out = T::from_f64(lo_try);
            upp_out = T::from_f64(up_try);
            if iter + 1 == p.maxiters {
                stopped_maxiters = p.maxiters > 1;
            }
        }
        let value = final_combine(col, &mask, kind, &mut buf);
        Out {
            mask,
            low: low_out,
            upp: upp_out,
            std: T::nan(),
            nit,
            flags: (pre_masked as u8)
                | ((stopped_maxiters as u8) << 1)
                | ((stopped_nkeep as u8) << 2)
                | ((stopped_maxrej as u8) << 3),
            restored: flags,
            value,
        }
    }

    fn cmp_pair_t<T: Float>(a: &(T, usize), b: &(T, usize)) -> Ordering {
        f64::total_cmp(&a.0.to_f64(), &b.0.to_f64())
    }

    /// The legacy minmax sorted `(T, usize)` pairs; reducers sorts `(f64, usize)`.
    pub fn minmax<T: Float>(
        values: &[T],
        mask_in: Option<&[bool]>,
        q_low: f64,
        q_upp: f64,
        kind: Reduction,
    ) -> Out<T> {
        let n = values.len();
        let mut mask = vec![false; n];
        let mut finite_pairs: Vec<(T, usize)> = Vec::with_capacity(n);
        let mut pre_masked = false;
        for (k, &x) in values.iter().enumerate() {
            let masked_by_input = mask_in.is_some_and(|m| m[k]);
            pre_masked |= masked_by_input;
            let masked = masked_by_input || !x.is_finite();
            if masked {
                mask[k] = true;
            } else {
                finite_pairs.push((x, k));
            }
        }
        let n_finite = finite_pairs.len();
        let n_rej_low = (n_finite as f64 * q_low + 0.001) as usize;
        let n_rej_upp = (n_finite as f64 * q_upp + 0.001) as usize;
        finite_pairs.sort_by(cmp_pair_t);
        for &(_, k) in finite_pairs.iter().take(n_rej_low.min(n_finite)) {
            mask[k] = true;
        }
        let high_count = n_rej_upp.min(n_finite.saturating_sub(n_rej_low));
        for &(_, k) in finite_pairs.iter().rev().take(high_count) {
            mask[k] = true;
        }
        let (low, upp) = if n_rej_low + high_count < n_finite {
            (
                finite_pairs[n_rej_low].0,
                finite_pairs[n_finite - high_count - 1].0,
            )
        } else {
            (T::nan(), T::nan())
        };
        let mut buf = Vec::new();
        let value = final_combine_from_sorted_pairs(&finite_pairs, &mask, kind, &mut buf);
        Out {
            mask,
            low,
            upp,
            std: T::nan(),
            nit: 1,
            flags: pre_masked as u8,
            restored: Vec::new(),
            value,
        }
    }

    fn iraf_nint(x: f64) -> isize {
        if x >= 0.0 {
            (x + 0.5).floor() as isize
        } else {
            (x - 0.5).ceil() as isize
        }
    }

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

    #[allow(clippy::too_many_arguments)]
    pub fn pclip<T: Float>(
        values: &[T],
        mask_in: Option<&[bool]>,
        pclip_offset: isize,
        sigma_lower: f64,
        sigma_upper: f64,
        nkeep: usize,
        kind: Reduction,
    ) -> Out<T> {
        let nimages = values.len();
        let mut mask = vec![false; nimages];
        let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(nimages);
        let mut resid = Vec::<f64>::with_capacity(nimages);
        let mut pre_masked = false;
        for (k, &x) in values.iter().enumerate() {
            let masked_by_input = mask_in.is_some_and(|m| m[k]);
            pre_masked |= masked_by_input;
            if masked_by_input || !x.is_finite() {
                mask[k] = true;
            } else {
                pairs.push((x.to_f64(), k));
            }
        }
        let (low, upp) = apply_pclip_pairs::<T>(
            &mut pairs,
            &mut mask,
            &mut resid,
            pclip_offset,
            sigma_lower,
            sigma_upper,
            nkeep,
        );
        let value = final_combine_from_sorted_pairs(&pairs, &mask, kind, &mut resid);
        Out {
            mask,
            low,
            upp,
            std: T::nan(),
            nit: 1,
            flags: pre_masked as u8,
            restored: Vec::new(),
            value,
        }
    }

    fn apply_pclip_pairs<T: Float>(
        pairs: &mut [(f64, usize)],
        mask: &mut [bool],
        resid: &mut Vec<f64>,
        pclip_offset: isize,
        sigma_lower: f64,
        sigma_upper: f64,
        nkeep: usize,
    ) -> (T, T) {
        let n_total = pairs.len();
        if n_total == 0 || n_total < 3 || n_total < nkeep.saturating_add(1) {
            return col_minmax_from_masked_pairs::<T>(pairs, mask, resid);
        }
        pairs.sort_by(|a, b| f64::total_cmp(&a.0, &b.0));
        let n2_1based = 1 + n_total / 2;
        let even = n_total % 2 == 0;
        let med = if even {
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
            return col_minmax_from_masked_pairs::<T>(pairs, mask, resid);
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
                    let r_tol = r + 0.001;
                    if s <= r_tol {
                        nh += 1;
                    }
                    if nl > 0 && resid[nl - 1] <= r_tol {
                        nl -= 1;
                    }
                } else {
                    nh += 1;
                    let s_tol = s + 0.001;
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
        for (rank, &(_, k)) in pairs.iter().enumerate() {
            if rank < nl || rank > nh {
                mask[k] = true;
            }
        }
        col_minmax_from_masked_pairs::<T>(pairs, mask, resid)
    }
}

// ---------- differential comparison ----------

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

/// Hand-picked edge cases plus random small stacks with ties, outliers,
/// NaN/inf, and optional input masks.
fn stacks(rng: &mut Rng) -> Vec<(Vec<f64>, Option<Vec<bool>>)> {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let mut v: Vec<(Vec<f64>, Option<Vec<bool>>)> = vec![
        (vec![], None),
        (vec![5.0], None),
        (vec![1.0, 2.0], None),
        (vec![1.0; 5], None),
        (vec![1.0, 1.0, 2.0, 2.0, 100.0, 100.0], None),
        (vec![0.0, -0.0, 0.0, -0.0, 5.0, -0.0], None),
        (vec![3.0, -0.0, 0.0, 0.0, -0.0, -5.0, 0.0], None),
        (vec![nan; 4], None),
        (vec![1.0, 2.0, 3.0, 4.0], Some(vec![true; 4])),
        (vec![1.0, nan, inf, -inf, 2.0, 3.0, 50.0], None),
        (
            vec![10.0, 11.0, 9.0, 10.5, 1000.0, 10.2, -500.0, 9.8],
            Some(vec![false, true, false, false, true, false, false, false]),
        ),
        (vec![1e30, -1e30, 1e30, 3e30, 2e30], None),
        (vec![1e308, 1e308, 1e308, -1e308, 1.7e308], None),
        (
            vec![100.0, 101.0, 99.0, 100.0, 100.0, 300.0, 100.5, 99.5, 100.0],
            None,
        ),
        (vec![2.0, 2.0, 2.0, 9.0, 9.0, 9.0], None),
        (vec![1.0, 1.0 + 1e-7, 1.0 - 1e-7, 1.0, 1.0 + 2e-7], None),
    ];
    for _ in 0..120 {
        let n = 1 + rng.below(16);
        let quant = rng.below(3) as f64;
        let vals = (0..n)
            .map(|_| {
                let u = rng.uniform();
                let mut x = 100.0 + 15.0 * (rng.uniform() - 0.5);
                if quant > 0.0 {
                    x = (x / (5.0 * quant)).round() * 5.0 * quant;
                }
                match u {
                    u if u < 0.1 => x + 200.0,
                    u if u < 0.15 => x - 200.0,
                    u if u < 0.18 => f64::NAN,
                    u if u < 0.2 => f64::INFINITY,
                    _ => x,
                }
            })
            .collect();
        let mask = (rng.below(2) == 0).then(|| (0..n).map(|_| rng.uniform() < 0.2).collect());
        v.push((vals, mask));
    }
    v
}

fn bits<T: Float>(x: T) -> u64 {
    x.to_f64().to_bits()
}

fn assert_same<T: Float>(
    ctx: &str,
    new: &reject::RejectResult<'_, T>,
    old: &legacy::Out<T>,
    restored: bool,
) {
    assert_eq!(new.rejected, &old.mask[..], "{ctx}: mask");
    assert_eq!(
        new.retained,
        old.mask.iter().filter(|&&m| !m).count(),
        "{ctx}: retained"
    );
    let d = new.diagnostics.as_ref().expect("diagnostics requested");
    assert_eq!(bits(d.low), bits(old.low), "{ctx}: low");
    assert_eq!(bits(d.upp), bits(old.upp), "{ctx}: upp");
    assert_eq!(bits(d.std), bits(old.std), "{ctx}: std");
    assert_eq!((d.nit, d.flags), (old.nit, old.flags), "{ctx}: nit/flags");
    if restored {
        assert_eq!(d.restored_flags, &old.restored[..], "{ctx}: restored flags");
    } else {
        assert!(d.restored_flags.is_empty(), "{ctx}: restored flags");
    }
}

fn check_dtype<T: Float>() {
    let mut rng = Rng(0x5EED_0001);
    let mut ws = Workspace::new();
    let cen = [CenFunc::Mean, CenFunc::Median, CenFunc::LowerMedian];
    let cc = [
        ClipCenter::Mean,
        ClipCenter::Median,
        ClipCenter::LowerMedian,
        ClipCenter::ClippingCenter,
    ];
    let kinds = [Reduction::Mean, Reduction::Median];
    // Coverage counters: the comparison must exercise rollbacks and rejections.
    let (mut ccd_rollbacks, mut lin_rollbacks, mut mm_rejects, mut pc_rejects) = (0, 0, 0, 0);
    for (si, (vals64, mask)) in stacks(&mut rng).into_iter().enumerate() {
        let vals: Vec<T> = vals64.iter().map(|&x| T::from_f64(x)).collect();
        let n = vals.len();
        let mask_full = mask.clone().unwrap_or_else(|| vec![false; n]);
        let m = mask.as_deref();
        for pi in 0..24 {
            let kind = kinds[pi % 2];
            let full = OutputOptions {
                reduction: Some(kind),
                diagnostics: true,
            };
            let scales: Vec<f64> = (0..n).map(|k| 0.5 + 0.1 * k as f64).collect();
            let zeros: Vec<f64> = (0..n).map(|k| -2.0 * k as f64).collect();
            let scaled = pi % 3 == 0;
            let sig = rng.pick(&[(3.0, 3.0), (1.0, 1.0), (0.0, 0.0), (2.5, 0.5)]);
            let cp = CcdClipParams {
                sigma_lower: sig.0,
                sigma_upper: sig.1,
                maxiters: rng.pick(&[0, 1, 3, 10]),
                nkeep: rng.pick(&[0, 1, 3, n, n + 1]),
                maxrej: rng.pick(&[0, 1, n / 2, n, n + 3]),
                revert_on_nkeep: rng.below(2) == 0,
                cenfunc: rng.pick(&cen),
                clip_cen: rng.pick(&cc),
                rdnoise: rng.pick(&[0.0, 5.0, 1e-30]),
                snoise: rng.pick(&[0.0, 0.05]),
                gain: rng.pick(&[1.0, 2.5]),
                scales: if scaled { &scales } else { &[] },
                zeros: if scaled { &zeros } else { &[] },
            };
            let ctx = format!("ccdclip stack {si} params {pi}");
            let old = legacy::ccdclip(&vals, &mask_full, &cp, kind);
            let new = reject::ccdclip(&vals, Some(&mask_full), &cp, full, &mut ws).unwrap();
            assert_same(&ctx, &new, &old, true);
            ccd_rollbacks += usize::from(old.restored.iter().any(|&f| f != 0));
            assert_eq!(bits(new.value.unwrap()), bits(old.value), "{ctx}: value");
            // Mask-only and no-input-mask calls give the same mask.
            if m.is_none() {
                let r = reject::ccdclip(&vals, None, &cp, MASK_ONLY, &mut ws).unwrap();
                assert_eq!(r.rejected, &old.mask[..], "{ctx}: mask-only");
            }

            let lp = LinearClipParams {
                low_scale: rng.pick(&[1.0, 0.9, 0.0]),
                low: rng.pick(&[0.0, 5.0, 50.0]),
                upp_scale: rng.pick(&[1.0, 1.1, 0.0]),
                upp: rng.pick(&[0.0, 5.0, 50.0, 200.0]),
                maxiters: rng.pick(&[0, 1, 3, 10]),
                nkeep: rng.pick(&[0, 1, 3, n, n + 1]),
                maxrej: rng.pick(&[0, 1, n / 2, n, n + 3]),
                cenfunc: rng.pick(&cen),
                revert_on_nkeep: rng.below(2) == 0,
            };
            let ctx = format!("linearclip stack {si} params {pi}");
            let old = legacy::linearclip(&vals, &mask_full, &lp, kind);
            let new = reject::linearclip(&vals, m, &lp, full, &mut ws).unwrap();
            assert_same(&ctx, &new, &old, true);
            lin_rollbacks += usize::from(old.restored.iter().any(|&f| f != 0));
            assert_eq!(bits(new.value.unwrap()), bits(old.value), "{ctx}: value");

            let (ql, qu) = rng.pick(&[
                (0.0, 0.0),
                (0.2, 0.2),
                (0.5, 0.5),
                (0.4, 0.0),
                (1.0, 1.0),
                (0.0, 0.34),
            ]);
            let ctx = format!("minmax stack {si} params {pi}");
            let old = legacy::minmax(&vals, m, ql, qu, kind);
            let p = MinMaxParams {
                q_low: ql,
                q_upp: qu,
            };
            let new = reject::minmax(&vals, m, &p, full, &mut ws).unwrap();
            assert_same(&ctx, &new, &old, false);
            mm_rejects += usize::from(new.retained + mask_full.iter().filter(|&&x| x).count() < n);
            assert_eq!(bits(new.value.unwrap()), bits(old.value), "{ctx}: value");

            let offset = reject::pclip_offset(rng.pick(&[-0.5, 0.5, -1.0, 1.0, -3.0, 2.0, 0.3]), n);
            let (sl, su) = rng.pick(&[(3.0, 3.0), (0.5, 0.5), (0.0, 0.0), (1.0, 2.0)]);
            let nkeep = rng.pick(&[0, 1, 3, 10]);
            let ctx = format!("pclip stack {si} params {pi}");
            let old = legacy::pclip(&vals, m, offset, sl, su, nkeep, kind);
            let p = PclipParams {
                offset,
                sigma_lower: sl,
                sigma_upper: su,
                nkeep,
            };
            let new = reject::pclip(&vals, m, &p, full, &mut ws).unwrap();
            assert_same(&ctx, &new, &old, false);
            assert_eq!(bits(new.value.unwrap()), bits(old.value), "{ctx}: value");
            pc_rejects += usize::from(new.retained + mask_full.iter().filter(|&&x| x).count() < n);
        }
    }
    assert!(
        ccd_rollbacks > 50 && lin_rollbacks > 50,
        "{ccd_rollbacks} {lin_rollbacks}"
    );
    assert!(
        mm_rejects > 200 && pc_rejects > 200,
        "{mm_rejects} {pc_rejects}"
    );
}

#[test]
fn matches_legacy_imcombiners_f64() {
    check_dtype::<f64>();
}

#[test]
fn matches_legacy_imcombiners_f32() {
    check_dtype::<f32>();
}
