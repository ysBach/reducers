//! PyO3 wrappers. Each public function dispatches in one call to the matching
//! `reducers_1d` kernel with a [`ScanPolicy`] code (no extra Python round-trips).
//!
//! Inputs are contiguous real numeric arrays prepared by the Python layer. Float
//! arrays use the NaN-aware kernels; integer and bool arrays use finite numeric
//! kernels and return the same public result types as the validated Python path.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;

use crate::axis;
use crate::finite::{Float, ScanPolicy};
use crate::parallel::{
    axis_order_grain, axis_scan_grain, default_parallel_grains, minmax_1d_grain, parallel_grain,
    parallel_grains, set_axis_order_grain, set_axis_scan_grain, set_minmax_1d_grain,
    set_parallel_grain,
};
use crate::reducers_1d::{self, Kind};

macro_rules! dispatch_numeric_slice {
    ($arr:expr, $s:ident => $float_body:expr, $n:ident => $number_body:expr) => {{
        if let Ok(a) = $arr.extract::<PyReadonlyArray1<f64>>() {
            let $s = a.as_slice()?;
            Ok($float_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<f32>>() {
            let $s = a.as_slice()?;
            Ok($float_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<bool>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<i8>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<u8>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<i16>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<u16>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<i32>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<u32>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<i64>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else if let Ok(a) = $arr.extract::<PyReadonlyArray1<u64>>() {
            let $n = a.as_slice()?;
            Ok($number_body)
        } else {
            Err(PyTypeError::new_err(
                "reducers: expected a contiguous 1-D array with a supported real dtype",
            ))
        }
    }};
}

macro_rules! dispatch_numeric_matrix {
    ($arr:expr, $a:ident => $float_body:expr, $n:ident => $number_body:expr) => {{
        if let Ok($a) = $arr.extract::<PyReadonlyArray2<f64>>() {
            Ok($float_body)
        } else if let Ok($a) = $arr.extract::<PyReadonlyArray2<f32>>() {
            Ok($float_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<bool>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<i8>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<u8>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<i16>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<u16>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<i32>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<u32>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<i64>>() {
            Ok($number_body)
        } else if let Ok($n) = $arr.extract::<PyReadonlyArray2<u64>>() {
            Ok($number_body)
        } else {
            Err(PyTypeError::new_err(
                "reducers: expected a contiguous 2-D array with a supported real dtype",
            ))
        }
    }};
}

macro_rules! dispatch_weighted_slice {
    ($arr:expr, $weights:expr, $s:ident, $w:ident => $float_body:expr, $n:ident => $number_body:expr) => {{
        if let Ok(weights) = $weights.extract::<PyReadonlyArray1<f64>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<f32>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<bool>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i8>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u8>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i16>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u16>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i32>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u32>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i64>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u64>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_slice!($arr, $s => $float_body, $n => $number_body)
        } else {
            Err(PyTypeError::new_err(
                "reducers: expected a contiguous weights array with a supported real dtype",
            ))
        }
    }};
}

macro_rules! dispatch_weighted_matrix {
    ($arr:expr, $weights:expr, $a:ident, $w:ident => $float_body:expr, $n:ident => $number_body:expr) => {{
        if let Ok(weights) = $weights.extract::<PyReadonlyArray1<f64>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<f32>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<bool>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i8>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u8>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i16>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u16>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i32>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u32>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<i64>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else if let Ok(weights) = $weights.extract::<PyReadonlyArray1<u64>>() {
            let $w = weights.as_slice()?;
            dispatch_numeric_matrix!($arr, $a => $float_body, $n => $number_body)
        } else {
            Err(PyTypeError::new_err(
                "reducers: expected a contiguous weights array with a supported real dtype",
            ))
        }
    }};
}

macro_rules! exact_or_nan {
    ($py:expr, $opt:expr) => {
        match $opt {
            Some(v) => v.into_bound_py_any($py)?,
            None => f64::NAN.into_bound_py_any($py)?,
        }
    };
}

#[inline]
fn weighted_value(result: reducers_1d::WeightedMean, policy: ScanPolicy) -> PyResult<f64> {
    let zero_weight = result.count > 0 && result.sum_weight == 0.0;
    let empty_strict =
        result.count == 0 && matches!(policy, ScanPolicy::AllValues | ScanPolicy::AllFinite);
    if zero_weight || empty_strict {
        Err(PyZeroDivisionError::new_err(
            "weights sum to zero, can't be normalized",
        ))
    } else {
        Ok(result.value)
    }
}

#[inline]
fn weighted_axis_values(result: axis::WeightedAxis, policy: ScanPolicy) -> PyResult<Vec<f64>> {
    if result.zero_weight
        || (result.empty && matches!(policy, ScanPolicy::AllValues | ScanPolicy::AllFinite))
    {
        Err(PyZeroDivisionError::new_err(
            "weights sum to zero, can't be normalized",
        ))
    } else {
        Ok(result.values)
    }
}

macro_rules! scalar_op {
    ($name:ident, $float_kernel:path, $number_kernel:path) => {
        #[pyfunction]
        fn $name(arr: &Bound<'_, PyAny>, policy: u8) -> PyResult<f64> {
            let p = ScanPolicy::from_code(policy);
            dispatch_numeric_slice!(arr, s => $float_kernel(s, p), n => $number_kernel(n))
        }
    };
}

scalar_op!(mean_1d, reducers_1d::mean, reducers_1d::number_mean);
scalar_op!(sum_1d, reducers_1d::sum, reducers_1d::number_sum);
scalar_op!(median_1d, reducers_1d::median, reducers_1d::number_median);

#[pyfunction]
fn min_1d<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    policy: u8,
) -> PyResult<Bound<'py, PyAny>> {
    let p = ScanPolicy::from_code(policy);
    dispatch_numeric_slice!(
        arr,
        s => reducers_1d::min(s, p).to_f64().into_bound_py_any(py)?,
        n => exact_or_nan!(py, reducers_1d::number_min_value(n))
    )
}

#[pyfunction]
fn max_1d<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    policy: u8,
) -> PyResult<Bound<'py, PyAny>> {
    let p = ScanPolicy::from_code(policy);
    dispatch_numeric_slice!(
        arr,
        s => reducers_1d::max(s, p).to_f64().into_bound_py_any(py)?,
        n => exact_or_nan!(py, reducers_1d::number_max_value(n))
    )
}

#[pyfunction]
fn lmedian_1d<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    policy: u8,
) -> PyResult<Bound<'py, PyAny>> {
    let p = ScanPolicy::from_code(policy);
    dispatch_numeric_slice!(
        arr,
        s => reducers_1d::lmedian(s, p).into_bound_py_any(py)?,
        n => {
            let mut buf = n.to_vec();
            exact_or_nan!(py, reducers_1d::number_lmedian_value_in_place(&mut buf))
        }
    )
}

#[pyfunction]
fn minmax_1d(arr: &Bound<'_, PyAny>, policy: u8) -> PyResult<(f64, f64)> {
    let p = ScanPolicy::from_code(policy);
    dispatch_numeric_slice!(
        arr,
        s => {
            let (lo, hi) = reducers_1d::minmax(s, p);
            (lo.to_f64(), hi.to_f64())
        },
        n => reducers_1d::number_minmax(n)
    )
}

/// Always returns `(variance, mean)`; the Python layer selects.
#[pyfunction]
fn var_1d(arr: &Bound<'_, PyAny>, ddof: usize, policy: u8) -> PyResult<(f64, f64)> {
    let p = ScanPolicy::from_code(policy);
    dispatch_numeric_slice!(
        arr,
        s => reducers_1d::variance_mean(s, ddof, p),
        n => reducers_1d::number_variance_mean(n, ddof)
    )
}

#[pyfunction]
fn count_finite_1d(arr: &Bound<'_, PyAny>) -> PyResult<usize> {
    dispatch_numeric_slice!(
        arr,
        s => reducers_1d::count_finite(s),
        n => reducers_1d::number_count_finite(n)
    )
}

#[pyfunction]
fn average_1d(arr: &Bound<'_, PyAny>, weights: &Bound<'_, PyAny>, policy: u8) -> PyResult<f64> {
    let p = ScanPolicy::from_code(policy);
    let result = dispatch_weighted_slice!(
        arr,
        weights,
        s,
        w => reducers_1d::weighted_average(s, w, p),
        n => reducers_1d::number_weighted_average(n, w)
    )?;
    weighted_value(result, p)
}

#[pyfunction]
fn percentile_1d<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    q: PyReadonlyArray1<'py, f64>,
    policy: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = ScanPolicy::from_code(policy);
    let qs = q.as_slice()?;
    let out: Vec<f64> = dispatch_numeric_slice!(
        arr,
        s => reducers_1d::percentiles(s, qs, p),
        n => reducers_1d::number_percentiles(n, qs)
    )?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn average_axis<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    weights_1d: bool,
    axis_last: bool,
    policy: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = ScanPolicy::from_code(policy);
    let out = dispatch_weighted_matrix!(
        arr,
        weights,
        a,
        w => {
            let (d0, d1) = a.as_array().dim();
            let s = a.as_slice()?;
            let result = if axis_last {
                axis::weighted_axis_last(s, w, weights_1d, d0, d1, p)
            } else {
                axis::weighted_axis0(s, w, weights_1d, d0, d1, p)
            };
            weighted_axis_values(result, p)?
        },
        n => {
            let (d0, d1) = n.as_array().dim();
            let s = n.as_slice()?;
            let result = if axis_last {
                axis::weighted_axis_last_number(s, w, weights_1d, d0, d1)
            } else {
                axis::weighted_axis0_number(s, w, weights_1d, d0, d1)
            };
            weighted_axis_values(result, p)?
        }
    )?;
    Ok(PyArray1::from_vec(py, out))
}

/// Axis reduction over a normalized 2-D contiguous array. `axis_last=true`
/// expects shape `(outer, n)`; `axis_last=false` expects `(n, outer)`. Returns
/// a flat array of length `outer` (the Python layer reshapes to `out_shape`).
#[pyfunction]
fn reduce_axis<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    kind: u8,
    axis_last: bool,
    ddof: usize,
    policy: u8,
) -> PyResult<Bound<'py, PyAny>> {
    let k = Kind::from_code(kind);
    let p = ScanPolicy::from_code(policy);
    dispatch_numeric_matrix!(
        arr,
        a => {
            let (d0, d1) = a.as_array().dim();
            let s = a.as_slice()?;
            let out = if axis_last {
                axis::reduce_axis_last(s, d0, d1, k, ddof, p)
            } else {
                axis::reduce_axis0(s, d0, d1, k, ddof, p)
            };
            PyArray1::from_vec(py, out).into_any()
        },
        n => {
            let (d0, d1) = n.as_array().dim();
            let s = n.as_slice()?;
            if matches!(k, Kind::Min | Kind::Max | Kind::LMedian) && if axis_last { d1 > 0 } else { d0 > 0 } {
                let out = if axis_last {
                    axis::reduce_axis_last_number_exact(s, d0, d1, k)
                } else {
                    axis::reduce_axis0_number_exact(s, d0, d1, k)
                };
                PyArray1::from_vec(py, out).into_any()
            } else {
                let out = if axis_last {
                    axis::reduce_axis_last_number(s, d0, d1, k, ddof)
                } else {
                    axis::reduce_axis0_number(s, d0, d1, k, ddof)
                };
                PyArray1::from_vec(py, out).into_any()
            }
        }
    )
}

/// Axis percentiles. Returns a flat `(nq * outer)` f64 array laid out as `nq`
/// rows of `outer` (the Python layer reshapes to `(nq, *out_shape)`).
#[pyfunction]
fn percentile_axis<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    q: PyReadonlyArray1<'py, f64>,
    axis_last: bool,
    policy: u8,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = ScanPolicy::from_code(policy);
    let qs = q.as_slice()?;
    let out = dispatch_numeric_matrix!(
        arr,
        a => {
            let (d0, d1) = a.as_array().dim();
            let s = a.as_slice()?;
            if axis_last {
                axis::percentiles_axis_last(s, d0, d1, qs, p)
            } else {
                axis::percentiles_axis0(s, d0, d1, qs, p)
            }
        },
        n => {
            let (d0, d1) = n.as_array().dim();
            let s = n.as_slice()?;
            if axis_last {
                axis::percentiles_axis_last_number(s, d0, d1, qs)
            } else {
                axis::percentiles_axis0_number(s, d0, d1, qs)
            }
        }
    )?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Return all parallel work grains as a dict.
#[pyfunction]
fn get_parallel_grains(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in parallel_grains() {
        out.set_item(key, value)?;
    }
    Ok(out)
}

/// Return the built-in parallel work grains as a dict.
#[pyfunction]
fn get_default_parallel_grains(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in default_parallel_grains() {
        out.set_item(key, value)?;
    }
    Ok(out)
}

/// Set one named parallel work grain.
#[pyfunction(name = "set_parallel_grain")]
fn set_parallel_grain_py(key: &str, value: usize) -> PyResult<usize> {
    set_parallel_grain(key, value).map_err(PyValueError::new_err)
}

/// Set multiple named parallel work grains.
#[pyfunction(name = "set_parallel_grains")]
fn set_parallel_grains_py(py: Python<'_>, mapping: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let mut pairs = Vec::with_capacity(mapping.len());
    for (key, value) in mapping {
        let key = key.extract::<String>()?;
        let value = value.extract::<usize>()?;
        parallel_grain(&key).map_err(PyValueError::new_err)?;
        if value == 0 {
            return Err(PyValueError::new_err(
                "parallel grains must be positive integers",
            ));
        }
        pairs.push((key, value));
    }
    for (key, value) in pairs {
        set_parallel_grain(&key, value).map_err(PyValueError::new_err)?;
    }
    Ok(get_parallel_grains(py)?.unbind())
}

/// Set the scan-reducer axis work grain.
#[pyfunction(name = "set_axis_scan_grain")]
fn set_axis_scan_grain_py(value: usize) -> PyResult<usize> {
    set_axis_scan_grain(value).map_err(PyValueError::new_err)?;
    Ok(axis_scan_grain())
}

/// Set the order-statistic axis work grain.
#[pyfunction(name = "set_axis_order_grain")]
fn set_axis_order_grain_py(value: usize) -> PyResult<usize> {
    set_axis_order_grain(value).map_err(PyValueError::new_err)?;
    Ok(axis_order_grain())
}

/// Set the 1-D min/max work grain.
#[pyfunction(name = "set_minmax_1d_grain")]
fn set_minmax_1d_grain_py(value: usize) -> PyResult<usize> {
    set_minmax_1d_grain(value).map_err(PyValueError::new_err)?;
    Ok(minmax_1d_grain())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sum_1d, m)?)?;
    m.add_function(wrap_pyfunction!(median_1d, m)?)?;
    m.add_function(wrap_pyfunction!(lmedian_1d, m)?)?;
    m.add_function(wrap_pyfunction!(min_1d, m)?)?;
    m.add_function(wrap_pyfunction!(max_1d, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_1d, m)?)?;
    m.add_function(wrap_pyfunction!(var_1d, m)?)?;
    m.add_function(wrap_pyfunction!(count_finite_1d, m)?)?;
    m.add_function(wrap_pyfunction!(average_1d, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_1d, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_axis, m)?)?;
    m.add_function(wrap_pyfunction!(average_axis, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_axis, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_parallel_grains, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_parallel_grains, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_grain_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_grains_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_axis_scan_grain_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_axis_order_grain_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_minmax_1d_grain_py, m)?)?;
    Ok(())
}
