//! NumPy bindings for the generic sigma-clipping engine.
//!
//! The Python boundary validates and borrows contiguous arrays while holding
//! the GIL, then clips each axis-0 column through a reusable Rust
//! [`crate::sigma_clip::Workspace`] per Rayon job. The generic engine has
//! no knowledge of NumPy or Python and is therefore also usable from the
//! public Rust API.

use numpy::{
    Element, IxDyn, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

use crate::parallel::{axis_order_grain, chunks_for};
use crate::sigma_clip::{
    CenFunc, ClipCenter, ClipError, OutputOptions, Params, Reduction, StdFunc, Workspace,
};
use crate::Float;

#[derive(Clone, Copy)]
enum BatchKind {
    Full,
    Mask,
    Mean,
    Median,
    RestoredFlags,
}

impl BatchKind {
    fn output_options(self) -> OutputOptions {
        match self {
            Self::Full | Self::RestoredFlags => OutputOptions {
                reduction: None,
                diagnostics: true,
            },
            Self::Mask => OutputOptions {
                reduction: None,
                diagnostics: false,
            },
            Self::Mean => OutputOptions {
                reduction: Some(Reduction::Mean),
                diagnostics: false,
            },
            Self::Median => OutputOptions {
                reduction: Some(Reduction::Median),
                diagnostics: false,
            },
        }
    }

    fn wants_mask(self) -> bool {
        matches!(self, Self::Full | Self::Mask)
    }

    fn wants_diagnostics(self) -> bool {
        matches!(self, Self::Full)
    }

    fn wants_value(self) -> bool {
        matches!(self, Self::Mean | Self::Median)
    }
}

struct BatchOutput<T> {
    shape: Vec<usize>,
    rejected: Vec<bool>,
    std: Vec<T>,
    low: Vec<T>,
    upp: Vec<T>,
    nit: Vec<u8>,
    flags: Vec<u8>,
    values: Vec<T>,
    restored_flags: Vec<u8>,
}

enum BatchOutputEither {
    F32(BatchOutput<f32>),
    F64(BatchOutput<f64>),
}

struct OneOutput<T> {
    rejected: Vec<bool>,
    std: T,
    low: T,
    upp: T,
    nit: u8,
    flags: u8,
    value: Option<T>,
}

enum OneOutputEither {
    F32(OneOutput<f32>),
    F64(OneOutput<f64>),
}

fn parse_cenfunc(value: &str) -> PyResult<CenFunc> {
    match value.to_ascii_lowercase().as_str() {
        "mean" | "average" | "avg" => Ok(CenFunc::Mean),
        "median" | "med" => Ok(CenFunc::Median),
        "lmedian" | "lmed" | "lower_median" | "lower-median" | "lower median" => {
            Ok(CenFunc::LowerMedian)
        }
        _ => Err(PyValueError::new_err(format!("unknown cenfunc: {value}"))),
    }
}

fn parse_clip_center(value: Option<&str>) -> PyResult<Option<ClipCenter>> {
    value
        .map(|value| match value.to_ascii_lowercase().as_str() {
            "mean" | "average" | "avg" => Ok(ClipCenter::Mean),
            "median" | "med" => Ok(ClipCenter::Median),
            "lmedian" | "lmed" | "lower_median" | "lower-median" | "lower median" => {
                Ok(ClipCenter::LowerMedian)
            }
            "center" | "cenfunc" | "clipping_center" | "clipcenter" => {
                Ok(ClipCenter::ClippingCenter)
            }
            _ => Err(PyValueError::new_err(format!("unknown clip_cen: {value}"))),
        })
        .transpose()
}

fn parse_stdfunc(value: &str) -> PyResult<StdFunc> {
    match value.to_ascii_lowercase().as_str() {
        "std" | "standard_deviation" | "standard-deviation" => Ok(StdFunc::Std),
        "mad" | "median_absolute_deviation" | "median-absolute-deviation" => Ok(StdFunc::Mad),
        _ => Err(PyValueError::new_err(format!("unknown stdfunc: {value}"))),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_params(
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
) -> PyResult<Params> {
    if !sigma_lower.is_finite()
        || !sigma_upper.is_finite()
        || sigma_lower < 0.0
        || sigma_upper < 0.0
    {
        return Err(PyValueError::new_err(
            "sigma thresholds must be finite and non-negative",
        ));
    }
    Ok(Params {
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        cenfunc: parse_cenfunc(cenfunc)?,
        clip_cen: parse_clip_center(clip_cen)?,
        stdfunc: parse_stdfunc(stdfunc)?,
        nkeep,
        revert_on_nkeep,
        maxrej,
        final_min_retained: 0,
    })
}

fn clip_error(error: ClipError) -> PyErr {
    match error {
        ClipError::MaskLength { samples, mask } => PyValueError::new_err(format!(
            "mask length {mask} differs from sample length {samples}"
        )),
        ClipError::InvalidSigma => {
            PyValueError::new_err("sigma thresholds must be finite and non-negative")
        }
    }
}

fn checked_product(shape: &[usize]) -> PyResult<usize> {
    shape.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            PyOverflowError::new_err("array dimensions are too large for sigma clipping")
        })
    })
}

fn validate_batch_shape(shape: &[usize]) -> PyResult<usize> {
    if shape.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "arr must have at least 2 dimensions (N, *spatial); got shape {shape:?}"
        )));
    }
    checked_product(&shape[1..])
}

fn validate_1d_shape(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "values must be 1-D; got shape {shape:?}"
        )));
    }
    Ok(())
}

fn validate_mask<'a, 'py>(
    mask: Option<&'a PyReadonlyArrayDyn<'py, bool>>,
    shape: &[usize],
) -> PyResult<Option<&'a [bool]>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    if !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("mask must be C-contiguous"));
    }
    if mask.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "mask shape {:?} does not match arr shape {:?}",
            mask.shape(),
            shape
        )));
    }
    Ok(Some(mask.as_slice().map_err(|_| {
        PyValueError::new_err("mask must be C-contiguous")
    })?))
}

fn validate_1d_mask<'a, 'py>(
    mask: Option<&'a PyReadonlyArrayDyn<'py, bool>>,
    length: usize,
) -> PyResult<Option<&'a [bool]>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    if !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("mask must be C-contiguous"));
    }
    if mask.ndim() != 1 || mask.shape()[0] != length {
        return Err(PyValueError::new_err(format!(
            "mask length {} does not match values length {length}",
            mask.len()
        )));
    }
    Ok(Some(mask.as_slice().map_err(|_| {
        PyValueError::new_err("mask must be C-contiguous")
    })?))
}

struct ColumnScratch<T> {
    workspace: Workspace,
    values: Vec<T>,
    input_mask: Option<Vec<bool>>,
}

impl<T: Float> ColumnScratch<T> {
    fn new(samples: usize, has_mask: bool) -> Self {
        Self {
            workspace: Workspace::with_capacity(samples),
            values: vec![T::nan(); samples],
            input_mask: has_mask.then(|| vec![false; samples]),
        }
    }

    fn load(
        &mut self,
        data: &[T],
        input_mask: Option<&[bool]>,
        samples: usize,
        columns: usize,
        column: usize,
    ) {
        for sample in 0..samples {
            let offset = sample * columns + column;
            self.values[sample] = data[offset];
            if let Some(column_mask) = self.input_mask.as_mut() {
                column_mask[sample] =
                    input_mask.expect("mask scratch requires an input mask")[offset];
            }
        }
    }
}

fn run_column<'a, T: Float>(
    scratch: &'a mut ColumnScratch<T>,
    input_mask: Option<&[bool]>,
    params: &Params,
    kind: BatchKind,
) -> Result<crate::sigma_clip::ClipResult<'a, T>, ClipError> {
    let mask = scratch.input_mask.as_deref();
    debug_assert!(mask.is_none() == input_mask.is_none());
    crate::sigma_clip::clip(
        &scratch.values,
        mask,
        params,
        kind.output_options(),
        &mut scratch.workspace,
    )
}

fn output_chunks(columns: usize, samples: usize) -> (usize, usize) {
    let grain = axis_order_grain();
    let chunks = chunks_for(
        columns.saturating_mul(samples),
        grain,
        rayon::current_num_threads(),
    );
    // Retain the configured serial crossover, but leave enough independent
    // blocks for Rayon to balance uneven clipping work and worker speeds.
    // Previously, one block per worker prevented work stealing after workers
    // started their blocks. Round up to whole sample columns.
    (chunks, grain.div_ceil(samples).max(1))
}

fn process_mask_range<T: Float>(
    data: &[T],
    samples: usize,
    columns: usize,
    input_mask: Option<&[bool]>,
    params: &Params,
    output: &mut [bool],
) -> Result<(), ClipError> {
    let (chunks, chunk_columns) = output_chunks(columns, samples);
    if chunks == 1 {
        let mut scratch = ColumnScratch::<T>::new(samples, input_mask.is_some());
        for column in 0..columns {
            scratch.load(data, input_mask, samples, columns, column);
            let result = run_column(&mut scratch, input_mask, params, BatchKind::Mask)?;
            output[column * samples..(column + 1) * samples].copy_from_slice(result.rejected);
        }
        return Ok(());
    }
    output
        .par_chunks_mut(samples * chunk_columns)
        .enumerate()
        .try_for_each_init(
            || ColumnScratch::<T>::new(samples, input_mask.is_some()),
            |scratch, (chunk_index, output_chunk)| {
                let start = chunk_index * chunk_columns;
                let count = output_chunk.len() / samples;
                for local in 0..count {
                    let column = start + local;
                    scratch.load(data, input_mask, samples, columns, column);
                    let result = run_column(scratch, input_mask, params, BatchKind::Mask)?;
                    output_chunk[local * samples..(local + 1) * samples]
                        .copy_from_slice(result.rejected);
                }
                Ok(())
            },
        )
}

fn process_value_range<T: Float>(
    data: &[T],
    samples: usize,
    columns: usize,
    input_mask: Option<&[bool]>,
    params: &Params,
    kind: BatchKind,
    output: &mut [T],
) -> Result<(), ClipError> {
    let (chunks, chunk_columns) = output_chunks(columns, samples);
    if chunks == 1 {
        let mut scratch = ColumnScratch::<T>::new(samples, input_mask.is_some());
        for (column, destination) in output.iter_mut().enumerate() {
            scratch.load(data, input_mask, samples, columns, column);
            let result = run_column(&mut scratch, input_mask, params, kind)?;
            *destination = result
                .value
                .expect("reduction requested for fused batch output");
        }
        return Ok(());
    }
    output
        .par_chunks_mut(chunk_columns)
        .enumerate()
        .try_for_each_init(
            || ColumnScratch::<T>::new(samples, input_mask.is_some()),
            |scratch, (chunk_index, output_chunk)| {
                let start = chunk_index * chunk_columns;
                for (local, destination) in output_chunk.iter_mut().enumerate() {
                    scratch.load(data, input_mask, samples, columns, start + local);
                    let result = run_column(scratch, input_mask, params, kind)?;
                    *destination = result
                        .value
                        .expect("reduction requested for fused batch output");
                }
                Ok(())
            },
        )
}

#[allow(clippy::too_many_arguments)]
fn process_full_range<T: Float>(
    data: &[T],
    samples: usize,
    columns: usize,
    input_mask: Option<&[bool]>,
    params: &Params,
    rejected: &mut [bool],
    std: &mut [T],
    low: &mut [T],
    upp: &mut [T],
    nit: &mut [u8],
    flags: &mut [u8],
) -> Result<(), ClipError> {
    let (chunks, chunk_columns) = output_chunks(columns, samples);
    if chunks == 1 {
        let mut scratch = ColumnScratch::<T>::new(samples, input_mask.is_some());
        for column in 0..columns {
            scratch.load(data, input_mask, samples, columns, column);
            let result = run_column(&mut scratch, input_mask, params, BatchKind::Full)?;
            rejected[column * samples..(column + 1) * samples].copy_from_slice(result.rejected);
            let diagnostics = result
                .diagnostics
                .as_ref()
                .expect("diagnostics requested for full batch output");
            std[column] = diagnostics.std;
            low[column] = diagnostics.low;
            upp[column] = diagnostics.upp;
            nit[column] = diagnostics.legacy_nit;
            flags[column] = diagnostics.legacy_flags;
        }
        return Ok(());
    }
    rejected
        .par_chunks_mut(samples * chunk_columns)
        .zip(std.par_chunks_mut(chunk_columns))
        .zip(low.par_chunks_mut(chunk_columns))
        .zip(upp.par_chunks_mut(chunk_columns))
        .zip(nit.par_chunks_mut(chunk_columns))
        .zip(flags.par_chunks_mut(chunk_columns))
        .enumerate()
        .try_for_each_init(
            || ColumnScratch::<T>::new(samples, input_mask.is_some()),
            |scratch,
             (
                chunk_index,
                (((((rejected_chunk, std_chunk), low_chunk), upp_chunk), nit_chunk), flags_chunk),
            )| {
                let start = chunk_index * chunk_columns;
                for local in 0..std_chunk.len() {
                    scratch.load(data, input_mask, samples, columns, start + local);
                    let result = run_column(scratch, input_mask, params, BatchKind::Full)?;
                    rejected_chunk[local * samples..(local + 1) * samples]
                        .copy_from_slice(result.rejected);
                    let diagnostics = result
                        .diagnostics
                        .as_ref()
                        .expect("diagnostics requested for full batch output");
                    std_chunk[local] = diagnostics.std;
                    low_chunk[local] = diagnostics.low;
                    upp_chunk[local] = diagnostics.upp;
                    nit_chunk[local] = diagnostics.legacy_nit;
                    flags_chunk[local] = diagnostics.legacy_flags;
                }
                Ok(())
            },
        )
}

fn process_restored_range<T: Float>(
    data: &[T],
    samples: usize,
    columns: usize,
    input_mask: Option<&[bool]>,
    params: &Params,
    restored: &mut [u8],
) -> Result<(), ClipError> {
    let (chunks, chunk_columns) = output_chunks(columns, samples);
    if chunks == 1 {
        let mut scratch = ColumnScratch::<T>::new(samples, input_mask.is_some());
        for column in 0..columns {
            scratch.load(data, input_mask, samples, columns, column);
            let result = run_column(&mut scratch, input_mask, params, BatchKind::RestoredFlags)?;
            let diagnostics = result
                .diagnostics
                .as_ref()
                .expect("diagnostics requested for restored flags");
            restored[column * samples..(column + 1) * samples]
                .copy_from_slice(diagnostics.restored_flags);
        }
        return Ok(());
    }
    restored
        .par_chunks_mut(samples * chunk_columns)
        .enumerate()
        .try_for_each_init(
            || ColumnScratch::<T>::new(samples, input_mask.is_some()),
            |scratch, (chunk_index, restored_chunk)| {
                let start = chunk_index * chunk_columns;
                let count = restored_chunk.len() / samples;
                for local in 0..count {
                    scratch.load(data, input_mask, samples, columns, start + local);
                    let result = run_column(scratch, input_mask, params, BatchKind::RestoredFlags)?;
                    let diagnostics = result
                        .diagnostics
                        .as_ref()
                        .expect("diagnostics requested for restored flags");
                    restored_chunk[local * samples..(local + 1) * samples]
                        .copy_from_slice(diagnostics.restored_flags);
                }
                Ok(())
            },
        )
}

fn process_empty_columns<T: Float>(
    columns: usize,
    input_mask: Option<&[bool]>,
    params: &Params,
    kind: BatchKind,
    output: &mut BatchOutput<T>,
) -> Result<(), ClipError> {
    let empty: [T; 0] = [];
    let mut workspace = Workspace::with_capacity(0);
    for column in 0..columns {
        let result = crate::sigma_clip::clip(
            &empty,
            input_mask,
            params,
            kind.output_options(),
            &mut workspace,
        )?;
        if kind.wants_diagnostics() {
            let diagnostics = result
                .diagnostics
                .as_ref()
                .expect("diagnostics requested for empty output");
            output.std[column] = diagnostics.std;
            output.low[column] = diagnostics.low;
            output.upp[column] = diagnostics.upp;
            output.nit[column] = diagnostics.legacy_nit;
            output.flags[column] = diagnostics.legacy_flags;
        }
        if kind.wants_value() {
            output.values[column] = result.value.expect("reduction requested for empty output");
        }
    }
    Ok(())
}

fn transpose_columns<T: Copy>(column_major: &[T], samples: usize, columns: usize) -> Vec<T> {
    let mut row_major = if let Some(&first) = column_major.first() {
        vec![first; column_major.len()]
    } else {
        Vec::new()
    };
    for column in 0..columns {
        for sample in 0..samples {
            row_major[sample * columns + column] = column_major[column * samples + sample];
        }
    }
    row_major
}

fn run_batch<T: Float>(
    values: &[T],
    shape: &[usize],
    input_mask: Option<&[bool]>,
    params: &Params,
    kind: BatchKind,
) -> Result<BatchOutput<T>, ClipError> {
    let samples = shape[0];
    let columns = shape[1..].iter().product::<usize>();
    let mut output = BatchOutput {
        shape: shape.to_vec(),
        // Mask and restored flags are kept column-major while workers run so
        // each chunk has contiguous output. They are transposed once below.
        rejected: if kind.wants_mask() {
            vec![false; values.len()]
        } else {
            Vec::new()
        },
        std: if kind.wants_diagnostics() {
            vec![T::nan(); columns]
        } else {
            Vec::new()
        },
        low: if kind.wants_diagnostics() {
            vec![T::nan(); columns]
        } else {
            Vec::new()
        },
        upp: if kind.wants_diagnostics() {
            vec![T::nan(); columns]
        } else {
            Vec::new()
        },
        nit: if kind.wants_diagnostics() {
            vec![0; columns]
        } else {
            Vec::new()
        },
        flags: if kind.wants_diagnostics() {
            vec![0; columns]
        } else {
            Vec::new()
        },
        values: if kind.wants_value() {
            vec![T::nan(); columns]
        } else {
            Vec::new()
        },
        restored_flags: if matches!(kind, BatchKind::RestoredFlags) {
            vec![0; values.len()]
        } else {
            Vec::new()
        },
    };
    if columns == 0 {
        return Ok(output);
    }
    if samples == 0 {
        process_empty_columns(columns, input_mask, params, kind, &mut output)?;
    } else {
        match kind {
            BatchKind::Full => process_full_range(
                values,
                samples,
                columns,
                input_mask,
                params,
                &mut output.rejected,
                &mut output.std,
                &mut output.low,
                &mut output.upp,
                &mut output.nit,
                &mut output.flags,
            )?,
            BatchKind::Mask => process_mask_range(
                values,
                samples,
                columns,
                input_mask,
                params,
                &mut output.rejected,
            )?,
            BatchKind::Mean | BatchKind::Median => process_value_range(
                values,
                samples,
                columns,
                input_mask,
                params,
                kind,
                &mut output.values,
            )?,
            BatchKind::RestoredFlags => process_restored_range(
                values,
                samples,
                columns,
                input_mask,
                params,
                &mut output.restored_flags,
            )?,
        }
    }
    if kind.wants_mask() {
        output.rejected = transpose_columns(&output.rejected, samples, columns);
    }
    if matches!(kind, BatchKind::RestoredFlags) {
        output.restored_flags = transpose_columns(&output.restored_flags, samples, columns);
    }
    Ok(output)
}

fn run_one<T: Float>(
    values: &[T],
    input_mask: Option<&[bool]>,
    params: &Params,
    kind: BatchKind,
) -> Result<OneOutput<T>, ClipError> {
    let mut workspace = Workspace::with_capacity(values.len());
    let result = crate::sigma_clip::clip(
        values,
        input_mask,
        params,
        kind.output_options(),
        &mut workspace,
    )?;
    let (std, low, upp, nit, flags) = if let Some(diagnostics) = result.diagnostics.as_ref() {
        (
            diagnostics.std,
            diagnostics.low,
            diagnostics.upp,
            diagnostics.legacy_nit,
            diagnostics.legacy_flags,
        )
    } else {
        (T::nan(), T::nan(), T::nan(), 0, 0)
    };
    Ok(OneOutput {
        rejected: if kind.wants_mask() {
            result.rejected.to_vec()
        } else {
            Vec::new()
        },
        std,
        low,
        upp,
        nit,
        flags,
        value: result.value,
    })
}

fn dispatch_batch<'py>(
    _py: Python<'py>,
    array: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    params: Params,
    kind: BatchKind,
) -> PyResult<BatchOutputEither> {
    if let Ok(values) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        if !values.is_c_contiguous() {
            return Err(PyValueError::new_err("arr must be C-contiguous"));
        }
        let shape = values.shape().to_vec();
        validate_batch_shape(&shape)?;
        let data = values
            .as_slice()
            .map_err(|_| PyValueError::new_err("arr must be C-contiguous"))?;
        if data.len() != checked_product(&shape)? {
            return Err(PyValueError::new_err("arr has inconsistent dimensions"));
        }
        let input_mask = validate_mask(mask.as_ref(), &shape)?;
        return run_batch(data, &shape, input_mask, &params, kind)
            .map(BatchOutputEither::F32)
            .map_err(clip_error);
    }
    if let Ok(values) = array.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        if !values.is_c_contiguous() {
            return Err(PyValueError::new_err("arr must be C-contiguous"));
        }
        let shape = values.shape().to_vec();
        validate_batch_shape(&shape)?;
        let data = values
            .as_slice()
            .map_err(|_| PyValueError::new_err("arr must be C-contiguous"))?;
        if data.len() != checked_product(&shape)? {
            return Err(PyValueError::new_err("arr has inconsistent dimensions"));
        }
        let input_mask = validate_mask(mask.as_ref(), &shape)?;
        return run_batch(data, &shape, input_mask, &params, kind)
            .map(BatchOutputEither::F64)
            .map_err(clip_error);
    }
    Err(PyTypeError::new_err(
        "arr must be a contiguous float32 or float64 NumPy array",
    ))
}

fn dispatch_one<'py>(
    _py: Python<'py>,
    array: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    params: Params,
    kind: BatchKind,
) -> PyResult<OneOutputEither> {
    if let Ok(values) = array.extract::<PyReadonlyArrayDyn<'py, f32>>() {
        if !values.is_c_contiguous() {
            return Err(PyValueError::new_err("values must be C-contiguous"));
        }
        validate_1d_shape(values.shape())?;
        let data = values
            .as_slice()
            .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
        let input_mask = validate_1d_mask(mask.as_ref(), data.len())?;
        return run_one(data, input_mask, &params, kind)
            .map(OneOutputEither::F32)
            .map_err(clip_error);
    }
    if let Ok(values) = array.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        if !values.is_c_contiguous() {
            return Err(PyValueError::new_err("values must be C-contiguous"));
        }
        validate_1d_shape(values.shape())?;
        let data = values
            .as_slice()
            .map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
        let input_mask = validate_1d_mask(mask.as_ref(), data.len())?;
        return run_one(data, input_mask, &params, kind)
            .map(OneOutputEither::F64)
            .map_err(clip_error);
    }
    Err(PyTypeError::new_err(
        "values must be a contiguous float32 or float64 NumPy array",
    ))
}

fn array_from_vec<'py, T: Element>(
    py: Python<'py>,
    values: Vec<T>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    PyArray1::from_vec(py, values)
        .reshape(IxDyn(shape))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn batch_tuple<'py, T: Float + Element>(
    py: Python<'py>,
    output: BatchOutput<T>,
) -> PyResult<Bound<'py, PyTuple>> {
    let shape = output.shape;
    let trailing = shape[1..].to_vec();
    let mask = array_from_vec(py, output.rejected, &shape)?.into_any();
    let std = array_from_vec(py, output.std, &trailing)?.into_any();
    let low = array_from_vec(py, output.low, &trailing)?.into_any();
    let upp = array_from_vec(py, output.upp, &trailing)?.into_any();
    let nit = array_from_vec(py, output.nit, &trailing)?.into_any();
    let flags = array_from_vec(py, output.flags, &trailing)?.into_any();
    PyTuple::new(py, [mask, std, low, upp, nit, flags])
}

fn one_tuple<'py, T: Float + Element>(
    py: Python<'py>,
    output: OneOutput<T>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mask = PyArray1::from_vec(py, output.rejected).into_any();
    let std = PyArray1::from_vec(py, vec![output.std]).into_any();
    let low = PyArray1::from_vec(py, vec![output.low]).into_any();
    let upp = PyArray1::from_vec(py, vec![output.upp]).into_any();
    let nit = PyArray1::from_vec(py, vec![output.nit]).into_any();
    let flags = PyArray1::from_vec(py, vec![output.flags]).into_any();
    PyTuple::new(py, [mask, std, low, upp, nit, flags])
}

fn batch_mask<'py>(py: Python<'py>, output: BatchOutputEither) -> PyResult<Bound<'py, PyAny>> {
    match output {
        BatchOutputEither::F32(output) => {
            let shape = output.shape;
            Ok(array_from_vec(py, output.rejected, &shape)?.into_any())
        }
        BatchOutputEither::F64(output) => {
            let shape = output.shape;
            Ok(array_from_vec(py, output.rejected, &shape)?.into_any())
        }
    }
}

fn batch_values<'py>(py: Python<'py>, output: BatchOutputEither) -> PyResult<Bound<'py, PyAny>> {
    match output {
        BatchOutputEither::F32(output) => {
            let shape = output.shape[1..].to_vec();
            Ok(array_from_vec(py, output.values, &shape)?.into_any())
        }
        BatchOutputEither::F64(output) => {
            let shape = output.shape[1..].to_vec();
            Ok(array_from_vec(py, output.values, &shape)?.into_any())
        }
    }
}

fn batch_restored_flags<'py>(
    py: Python<'py>,
    output: BatchOutputEither,
) -> PyResult<Bound<'py, PyAny>> {
    match output {
        BatchOutputEither::F32(output) => {
            let shape = output.shape;
            Ok(array_from_vec(py, output.restored_flags, &shape)?.into_any())
        }
        BatchOutputEither::F64(output) => {
            let shape = output.shape;
            Ok(array_from_vec(py, output.restored_flags, &shape)?.into_any())
        }
    }
}

fn one_mask<'py>(py: Python<'py>, output: OneOutputEither) -> PyResult<Bound<'py, PyAny>> {
    match output {
        OneOutputEither::F32(output) => Ok(PyArray1::from_vec(py, output.rejected).into_any()),
        OneOutputEither::F64(output) => Ok(PyArray1::from_vec(py, output.rejected).into_any()),
    }
}

fn one_value(output: OneOutputEither) -> f64 {
    match output {
        OneOutputEither::F32(output) => output.value.unwrap_or(f32::NAN) as f64,
        OneOutputEither::F64(output) => output.value.unwrap_or(f64::NAN),
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    match dispatch_batch(py, arr, mask, params, BatchKind::Full)? {
        BatchOutputEither::F32(output) => batch_tuple(py, output),
        BatchOutputEither::F64(output) => batch_tuple(py, output),
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    match dispatch_one(py, values, mask, params, BatchKind::Full)? {
        OneOutputEither::F32(output) => one_tuple(py, output),
        OneOutputEither::F64(output) => one_tuple(py, output),
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_mask<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    batch_mask(py, dispatch_batch(py, arr, mask, params, BatchKind::Mask)?)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_mask_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    one_mask(py, dispatch_one(py, values, mask, params, BatchKind::Mask)?)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_mean<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    batch_values(py, dispatch_batch(py, arr, mask, params, BatchKind::Mean)?)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_mean_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<f64> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    Ok(one_value(dispatch_one(
        py,
        values,
        mask,
        params,
        BatchKind::Mean,
    )?))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_median<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    batch_values(
        py,
        dispatch_batch(py, arr, mask, params, BatchKind::Median)?,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    values,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_median_1d<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<f64> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    Ok(one_value(dispatch_one(
        py,
        values,
        mask,
        params,
        BatchKind::Median,
    )?))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    arr,
    *,
    mask = None,
    sigma_lower = 3.0,
    sigma_upper = 3.0,
    maxiters = 5,
    ddof = 0,
    nkeep = 0,
    maxrej = None,
    cenfunc = "median",
    clip_cen = None,
    stdfunc = "std",
    revert_on_nkeep = false,
    validate = true,
))]
fn sigclip_restored_flags<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: Option<PyReadonlyArrayDyn<'py, bool>>,
    sigma_lower: f64,
    sigma_upper: f64,
    maxiters: usize,
    ddof: usize,
    nkeep: usize,
    maxrej: Option<usize>,
    cenfunc: &str,
    clip_cen: Option<&str>,
    stdfunc: &str,
    revert_on_nkeep: bool,
    validate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = validate;
    let params = make_params(
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        cenfunc,
        clip_cen,
        stdfunc,
        revert_on_nkeep,
    )?;
    batch_restored_flags(
        py,
        dispatch_batch(py, arr, mask, params, BatchKind::RestoredFlags)?,
    )
}

#[pyfunction]
#[pyo3(signature = (mask, grow))]
fn grow_mask<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, bool>,
    grow: f64,
) -> PyResult<Bound<'py, PyAny>> {
    if !mask.is_c_contiguous() {
        return Err(PyValueError::new_err("grow mask must be C-contiguous"));
    }
    let shape = mask.shape().to_vec();
    let data = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("grow mask must be C-contiguous"))?;
    let output = crate::mask::grow_mask(data, &shape, grow).map_err(PyValueError::new_err)?;
    array_from_vec(py, output, &shape).map(|array| array.into_any())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sigclip, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mask, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mask_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mean, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_mean_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_median, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_median_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sigclip_restored_flags, m)?)?;
    m.add_function(wrap_pyfunction!(grow_mask, m)?)?;
    Ok(())
}
