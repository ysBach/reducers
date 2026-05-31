//! Float trait and scan policy shared by all reducer kernels.
//!
//! The core trait deliberately does **not** require `numpy::Element`, so the
//! kernels compile with no PyO3/NumPy dependency for pure-Rust consumers.

/// Floating element supported by the reducer kernels.
pub trait Float: Copy + PartialOrd + Send + Sync + 'static {
    fn nan() -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn min_num(self, other: Self) -> Self;
    fn max_num(self, other: Self) -> Self;
    fn zero() -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Float for f32 {
    #[inline]
    fn nan() -> Self {
        f32::NAN
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn min_num(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max_num(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Float for f64 {
    #[inline]
    fn nan() -> Self {
        f64::NAN
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn min_num(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max_num(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// How a kernel treats non-finite values during a scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanPolicy {
    /// Include every value; NaN/inf follow the plain NumPy-style reducers.
    AllValues,
    /// Caller guarantees all values are finite.
    AllFinite,
    /// Skip NaN, keep inf - `np.nanmean` parity.
    SkipNan,
    /// Skip NaN and inf - finite-only (`ignore_inf=True`).
    SkipNonFinite,
}

impl ScanPolicy {
    /// Policy code used at the PyO3 boundary (`reducers._core`).
    #[inline]
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::AllValues,
            1 => Self::AllFinite,
            2 => Self::SkipNan,
            3 => Self::SkipNonFinite,
            other => panic!("invalid ScanPolicy code: {other}"),
        }
    }
}
