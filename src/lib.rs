//! Rust-backed reductions for NumPy arrays (plain + NaN-aware).
//!
//! The kernel modules (`finite`, `parallel`, `reducers_1d`, `axis`) are pure
//! Rust with no PyO3/NumPy dependency. The PyO3 extension (`reducers._core`) is
//! compiled only with the `python` / `extension-module` feature.
//!
//! # Rust usage
//!
//! ```
//! use reducers::{reducers_1d, ScanPolicy};
//!
//! let v = [1.0_f64, 2.0, f64::NAN, 4.0];
//! // NaN-aware mean (np.nanmean parity: skip NaN, keep inf):
//! assert_eq!(reducers_1d::mean(&v, ScanPolicy::SkipNan), 7.0 / 3.0);
//! // Plain mean propagates NaN (np.mean):
//! assert!(reducers_1d::mean(&v, ScanPolicy::AllValues).is_nan());
//! // Finite-only (ignore inf too):
//! let w = [1.0_f64, f64::INFINITY, 3.0];
//! assert_eq!(reducers_1d::mean(&w, ScanPolicy::SkipNonFinite), 2.0);
//! ```

pub mod axis;
pub mod finite;
pub mod parallel;
pub mod reducers_1d;

pub use finite::{Float, ScanPolicy};

#[cfg(feature = "python")]
mod pyapi;

#[cfg(feature = "extension-module")]
#[pymodule]
fn _core(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    pyapi::register(m)
}

#[cfg(feature = "extension-module")]
use pyo3::pymodule;
