//! Axis reductions for normalized 2-D inputs.
//!
//! Last-axis reductions operate on contiguous rows. Axis-0 order statistics use
//! scratch buffers because they reorder values; cheap min/max scans stream the
//! input rows directly into the output buffer.

use rayon::prelude::*;

use crate::finite::{Float, ScanPolicy};
use crate::parallel::{axis_parallel_chunks, AxisParallelClass};
use crate::reducers_1d::{
    apply, apply_mut, apply_number, apply_number_mut, number_lmedian_value_in_place,
    number_max_value, number_min_value, number_percentiles_in_place, number_weighted_average,
    percentiles_in_place, weighted_average, Kind, Number, Weight, WeightedMean,
};

#[inline]
fn scan_class(kind: Kind, policy: ScanPolicy) -> AxisParallelClass {
    match kind {
        Kind::Var | Kind::Std => AxisParallelClass::ScanVar,
        _ if matches!(policy, ScanPolicy::SkipNan | ScanPolicy::SkipNonFinite) => {
            AxisParallelClass::ScanNan
        }
        _ => AxisParallelClass::ScanPlain,
    }
}

#[inline]
fn number_scan_class(kind: Kind) -> AxisParallelClass {
    match kind {
        Kind::Var | Kind::Std => AxisParallelClass::ScanVar,
        _ => AxisParallelClass::ScanPlain,
    }
}

// --------------------------------------------------------------------------
// scalar reducers
// --------------------------------------------------------------------------

/// Reduce each contiguous row of `data` shaped `(outer, n)` (row-major).
pub fn reduce_axis_last<T: Float>(
    data: &[T],
    outer: usize,
    n: usize,
    kind: Kind,
    ddof: usize,
    policy: ScanPolicy,
) -> Vec<T> {
    let row = |i: usize| &data[i * n..(i + 1) * n];
    if kind.needs_mut() {
        // Order statistics: copy the row into a reused scratch buffer, reorder.
        let work = |buf: &mut Vec<T>, i: usize| {
            buf.clear();
            buf.extend_from_slice(row(i));
            apply_mut(kind, buf, ddof, policy)
        };
        let chunks = axis_parallel_chunks(AxisParallelClass::OrderMedian, outer, n);
        if chunks > 1 {
            (0..outer)
                .into_par_iter()
                .with_min_len(outer.div_ceil(chunks))
                .map_init(|| Vec::<T>::with_capacity(n), work)
                .collect()
        } else {
            let mut buf = Vec::<T>::with_capacity(n);
            (0..outer).map(|i| work(&mut buf, i)).collect()
        }
    } else {
        // Read-only scan reducers: operate on the contiguous row directly.
        let chunks = axis_parallel_chunks(scan_class(kind, policy), outer, n);
        if chunks > 1 {
            (0..outer)
                .into_par_iter()
                .with_min_len(outer.div_ceil(chunks))
                .map(|i| apply(kind, row(i), ddof, policy))
                .collect()
        } else {
            (0..outer)
                .map(|i| apply(kind, row(i), ddof, policy))
                .collect()
        }
    }
}

/// Reduce each strided reducing-axis slice of `data` shaped `(n, outer)`
/// (row-major), i.e. element `k` of output element `j` is
/// `data[k * outer + j]`.
pub fn reduce_axis0<T: Float>(
    data: &[T],
    n: usize,
    outer: usize,
    kind: Kind,
    ddof: usize,
    policy: ScanPolicy,
) -> Vec<T> {
    if matches!(kind, Kind::Min | Kind::Max) {
        let mut out = vec![T::nan(); outer];
        if n == 0 {
            return out;
        }

        let reduce_chunk = |start: usize, out_chunk: &mut [T]| {
            let len = out_chunk.len();
            let first = &data[start..start + len];
            match (kind, policy) {
                (Kind::Min, ScanPolicy::AllValues) => {
                    out_chunk.copy_from_slice(first);
                    let mut has_nan = vec![0_u8; len];
                    for (flag, &x) in has_nan.iter_mut().zip(first) {
                        *flag = u8::from(x.is_nan());
                    }
                    for k in 1..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((dst, flag), &x) in
                            out_chunk.iter_mut().zip(has_nan.iter_mut()).zip(row)
                        {
                            *dst = (*dst).min_num(x);
                            *flag |= u8::from(x.is_nan());
                        }
                    }
                    for (dst, &flag) in out_chunk.iter_mut().zip(&has_nan) {
                        if flag != 0 {
                            *dst = T::nan();
                        }
                    }
                }
                (Kind::Max, ScanPolicy::AllValues) => {
                    out_chunk.copy_from_slice(first);
                    let mut has_nan = vec![0_u8; len];
                    for (flag, &x) in has_nan.iter_mut().zip(first) {
                        *flag = u8::from(x.is_nan());
                    }
                    for k in 1..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((dst, flag), &x) in
                            out_chunk.iter_mut().zip(has_nan.iter_mut()).zip(row)
                        {
                            *dst = (*dst).max_num(x);
                            *flag |= u8::from(x.is_nan());
                        }
                    }
                    for (dst, &flag) in out_chunk.iter_mut().zip(&has_nan) {
                        if flag != 0 {
                            *dst = T::nan();
                        }
                    }
                }
                (Kind::Min, ScanPolicy::AllFinite) => {
                    out_chunk.copy_from_slice(first);
                    for k in 1..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            if x < *dst {
                                *dst = x;
                            }
                        }
                    }
                }
                (Kind::Max, ScanPolicy::AllFinite) => {
                    out_chunk.copy_from_slice(first);
                    for k in 1..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            if x > *dst {
                                *dst = x;
                            }
                        }
                    }
                }
                (Kind::Min, ScanPolicy::SkipNan) => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            *dst = (*dst).min_num(x);
                        }
                    }
                }
                (Kind::Max, ScanPolicy::SkipNan) => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            *dst = (*dst).max_num(x);
                        }
                    }
                }
                (Kind::Min, ScanPolicy::SkipNonFinite) => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            if x.is_finite() {
                                *dst = (*dst).min_num(x);
                            }
                        }
                    }
                }
                (Kind::Max, ScanPolicy::SkipNonFinite) => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            if x.is_finite() {
                                *dst = (*dst).max_num(x);
                            }
                        }
                    }
                }
                _ => unreachable!("axis-0 direct scan is only used for min/max"),
            }
        };

        let chunks = axis_parallel_chunks(scan_class(kind, policy), outer, n);
        if chunks > 1 {
            let chunk_len = outer.div_ceil(chunks);
            out.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    reduce_chunk(chunk_idx * chunk_len, out_chunk);
                });
        } else {
            reduce_chunk(0, &mut out);
        }
        return out;
    }

    if matches!(kind, Kind::Mean | Kind::Sum) {
        let mut out = vec![T::nan(); outer];
        if n == 0 {
            if matches!(kind, Kind::Sum) {
                out.fill(T::zero());
            }
            return out;
        }

        let reduce_chunk = |start: usize, out_chunk: &mut [T]| {
            let len = out_chunk.len();
            let mut sums = vec![0.0_f64; len];
            let mut counts = vec![0usize; len];
            for k in 0..n {
                let row = &data[k * outer + start..k * outer + start + len];
                match policy {
                    ScanPolicy::AllValues | ScanPolicy::AllFinite => {
                        for (sum, &x) in sums.iter_mut().zip(row) {
                            *sum += x.to_f64();
                        }
                    }
                    ScanPolicy::SkipNan => {
                        for ((sum, count), &x) in sums.iter_mut().zip(&mut counts).zip(row) {
                            if !x.is_nan() {
                                *sum += x.to_f64();
                                *count += 1;
                            }
                        }
                    }
                    ScanPolicy::SkipNonFinite => {
                        for ((sum, count), &x) in sums.iter_mut().zip(&mut counts).zip(row) {
                            if x.is_finite() {
                                *sum += x.to_f64();
                                *count += 1;
                            }
                        }
                    }
                }
            }

            match policy {
                ScanPolicy::AllValues | ScanPolicy::AllFinite => match kind {
                    Kind::Mean => {
                        for (dst, &sum) in out_chunk.iter_mut().zip(&sums) {
                            *dst = T::from_f64(sum / n as f64);
                        }
                    }
                    Kind::Sum => {
                        for (dst, &sum) in out_chunk.iter_mut().zip(&sums) {
                            *dst = T::from_f64(sum);
                        }
                    }
                    _ => unreachable!("axis-0 direct sum scan is only used for mean/sum"),
                },
                ScanPolicy::SkipNan | ScanPolicy::SkipNonFinite => match kind {
                    Kind::Mean => {
                        for ((dst, &sum), &count) in out_chunk.iter_mut().zip(&sums).zip(&counts) {
                            *dst = if count == 0 {
                                T::nan()
                            } else {
                                T::from_f64(sum / count as f64)
                            };
                        }
                    }
                    Kind::Sum => {
                        for (dst, &sum) in out_chunk.iter_mut().zip(&sums) {
                            *dst = T::from_f64(sum);
                        }
                    }
                    _ => unreachable!("axis-0 direct sum scan is only used for mean/sum"),
                },
            }
        };

        let chunks = axis_parallel_chunks(scan_class(kind, policy), outer, n);
        if chunks > 1 {
            let chunk_len = outer.div_ceil(chunks);
            out.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    reduce_chunk(chunk_idx * chunk_len, out_chunk);
                });
        } else {
            reduce_chunk(0, &mut out);
        }
        return out;
    }

    let gather = |buf: &mut Vec<T>, j: usize| {
        buf.clear();
        for k in 0..n {
            buf.push(data[k * outer + j]);
        }
        apply_mut(kind, buf, ddof, policy)
    };
    let class = if kind.needs_mut() {
        AxisParallelClass::OrderMedian
    } else {
        scan_class(kind, policy)
    };
    let chunks = axis_parallel_chunks(class, outer, n);
    if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map_init(|| Vec::<T>::with_capacity(n), gather)
            .collect()
    } else {
        let mut buf = Vec::<T>::with_capacity(n);
        (0..outer).map(|j| gather(&mut buf, j)).collect()
    }
}

/// Reduce each contiguous row of non-floating numeric `data` shaped
/// `(outer, n)` (row-major).
pub fn reduce_axis_last_number<T: Number>(
    data: &[T],
    outer: usize,
    n: usize,
    kind: Kind,
    ddof: usize,
) -> Vec<f64> {
    let row = |i: usize| &data[i * n..(i + 1) * n];
    if kind.needs_mut() {
        let work = |buf: &mut Vec<T>, i: usize| {
            buf.clear();
            buf.extend_from_slice(row(i));
            apply_number_mut(kind, buf, ddof)
        };
        let chunks = axis_parallel_chunks(AxisParallelClass::OrderMedian, outer, n);
        if chunks > 1 {
            (0..outer)
                .into_par_iter()
                .with_min_len(outer.div_ceil(chunks))
                .map_init(|| Vec::<T>::with_capacity(n), work)
                .collect()
        } else {
            let mut buf = Vec::<T>::with_capacity(n);
            (0..outer).map(|i| work(&mut buf, i)).collect()
        }
    } else {
        let chunks = axis_parallel_chunks(number_scan_class(kind), outer, n);
        if chunks > 1 {
            (0..outer)
                .into_par_iter()
                .with_min_len(outer.div_ceil(chunks))
                .map(|i| apply_number(kind, row(i), ddof))
                .collect()
        } else {
            (0..outer)
                .map(|i| apply_number(kind, row(i), ddof))
                .collect()
        }
    }
}

/// Reduce each strided reducing-axis slice of non-floating numeric `data`
/// shaped `(n, outer)` (row-major).
pub fn reduce_axis0_number<T: Number>(
    data: &[T],
    n: usize,
    outer: usize,
    kind: Kind,
    ddof: usize,
) -> Vec<f64> {
    if matches!(kind, Kind::Min | Kind::Max) {
        let mut out = vec![f64::NAN; outer];
        if n == 0 {
            return out;
        }

        let reduce_chunk = |start: usize, out_chunk: &mut [f64]| {
            let len = out_chunk.len();
            let first = &data[start..start + len];
            for (dst, &x) in out_chunk.iter_mut().zip(first) {
                *dst = x.to_f64();
            }
            for k in 1..n {
                let row = &data[k * outer + start..k * outer + start + len];
                match kind {
                    Kind::Min => {
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            let x = x.to_f64();
                            if x < *dst {
                                *dst = x;
                            }
                        }
                    }
                    Kind::Max => {
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            let x = x.to_f64();
                            if x > *dst {
                                *dst = x;
                            }
                        }
                    }
                    _ => unreachable!("axis-0 direct scan is only used for min/max"),
                }
            }
        };

        let chunks = axis_parallel_chunks(number_scan_class(kind), outer, n);
        if chunks > 1 {
            let chunk_len = outer.div_ceil(chunks);
            out.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    reduce_chunk(chunk_idx * chunk_len, out_chunk);
                });
        } else {
            reduce_chunk(0, &mut out);
        }
        return out;
    }

    if matches!(kind, Kind::Mean | Kind::Sum) {
        let mut out = vec![
            if matches!(kind, Kind::Sum) {
                0.0
            } else {
                f64::NAN
            };
            outer
        ];
        if n == 0 {
            return out;
        }

        let reduce_chunk = |start: usize, out_chunk: &mut [f64]| {
            let len = out_chunk.len();
            out_chunk.fill(0.0);
            for k in 0..n {
                let row = &data[k * outer + start..k * outer + start + len];
                for (dst, &x) in out_chunk.iter_mut().zip(row) {
                    *dst += x.to_f64();
                }
            }
            if matches!(kind, Kind::Mean) {
                for dst in out_chunk {
                    *dst /= n as f64;
                }
            }
        };

        let chunks = axis_parallel_chunks(number_scan_class(kind), outer, n);
        if chunks > 1 {
            let chunk_len = outer.div_ceil(chunks);
            out.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    reduce_chunk(chunk_idx * chunk_len, out_chunk);
                });
        } else {
            reduce_chunk(0, &mut out);
        }
        return out;
    }

    let gather = |buf: &mut Vec<T>, j: usize| {
        buf.clear();
        for k in 0..n {
            buf.push(data[k * outer + j]);
        }
        apply_number_mut(kind, buf, ddof)
    };
    let class = if kind.needs_mut() {
        AxisParallelClass::OrderMedian
    } else {
        number_scan_class(kind)
    };
    let chunks = axis_parallel_chunks(class, outer, n);
    if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map_init(|| Vec::<T>::with_capacity(n), gather)
            .collect()
    } else {
        let mut buf = Vec::<T>::with_capacity(n);
        (0..outer).map(|j| gather(&mut buf, j)).collect()
    }
}

pub fn reduce_axis_last_number_exact<T: Number>(
    data: &[T],
    outer: usize,
    n: usize,
    kind: Kind,
) -> Vec<T> {
    let row = |i: usize| &data[i * n..(i + 1) * n];
    match kind {
        Kind::Min => map_axis_last_exact(outer, n, |i| {
            number_min_value(row(i)).expect("non-empty reducing-axis slice")
        }),
        Kind::Max => map_axis_last_exact(outer, n, |i| {
            number_max_value(row(i)).expect("non-empty reducing-axis slice")
        }),
        Kind::LMedian => {
            let work = |buf: &mut Vec<T>, i: usize| {
                buf.clear();
                buf.extend_from_slice(row(i));
                number_lmedian_value_in_place(buf).expect("non-empty reducing-axis slice")
            };
            let chunks = axis_parallel_chunks(AxisParallelClass::OrderMedian, outer, n);
            if chunks > 1 {
                (0..outer)
                    .into_par_iter()
                    .with_min_len(outer.div_ceil(chunks))
                    .map_init(|| Vec::<T>::with_capacity(n), work)
                    .collect()
            } else {
                let mut buf = Vec::<T>::with_capacity(n);
                (0..outer).map(|i| work(&mut buf, i)).collect()
            }
        }
        _ => unreachable!("exact numeric output is only used for min/max/lmedian"),
    }
}

#[inline]
fn map_axis_last_exact<T, F>(outer: usize, n: usize, f: F) -> Vec<T>
where
    T: Number,
    F: Fn(usize) -> T + Send + Sync,
{
    let chunks = axis_parallel_chunks(AxisParallelClass::ScanPlain, outer, n);
    if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map(f)
            .collect()
    } else {
        (0..outer).map(f).collect()
    }
}

pub fn reduce_axis0_number_exact<T: Number>(
    data: &[T],
    n: usize,
    outer: usize,
    kind: Kind,
) -> Vec<T> {
    if matches!(kind, Kind::Min | Kind::Max) {
        let mut out = data[..outer].to_vec();
        let reduce_chunk = |start: usize, out_chunk: &mut [T]| {
            let len = out_chunk.len();
            for k in 1..n {
                let row = &data[k * outer + start..k * outer + start + len];
                match kind {
                    Kind::Min => {
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            if x < *dst {
                                *dst = x;
                            }
                        }
                    }
                    Kind::Max => {
                        for (dst, &x) in out_chunk.iter_mut().zip(row) {
                            if x > *dst {
                                *dst = x;
                            }
                        }
                    }
                    _ => unreachable!("axis-0 direct exact scan is only used for min/max"),
                }
            }
        };

        let chunks = axis_parallel_chunks(AxisParallelClass::ScanPlain, outer, n);
        if chunks > 1 {
            let chunk_len = outer.div_ceil(chunks);
            out.par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    reduce_chunk(chunk_idx * chunk_len, out_chunk);
                });
        } else {
            reduce_chunk(0, &mut out);
        }
        return out;
    }

    let gather = |buf: &mut Vec<T>, j: usize| {
        buf.clear();
        for k in 0..n {
            buf.push(data[k * outer + j]);
        }
        number_lmedian_value_in_place(buf).expect("non-empty reducing-axis slice")
    };
    let chunks = axis_parallel_chunks(AxisParallelClass::OrderMedian, outer, n);
    if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map_init(|| Vec::<T>::with_capacity(n), gather)
            .collect()
    } else {
        let mut buf = Vec::<T>::with_capacity(n);
        (0..outer).map(|j| gather(&mut buf, j)).collect()
    }
}

#[derive(Debug)]
pub struct WeightedAxis {
    pub values: Vec<f64>,
    pub zero_weight: bool,
    pub empty: bool,
}

#[inline]
fn weighted_axis_finish(items: Vec<WeightedMean>) -> WeightedAxis {
    let mut zero_weight = false;
    let mut empty = false;
    let values = items
        .into_iter()
        .map(|item| {
            empty |= item.count == 0;
            zero_weight |= item.count > 0 && item.sum_weight == 0.0;
            item.value
        })
        .collect();
    WeightedAxis {
        values,
        zero_weight,
        empty,
    }
}

pub fn weighted_axis_last<T: Float, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    outer: usize,
    n: usize,
    policy: ScanPolicy,
) -> WeightedAxis {
    let compute = |i: usize| {
        let row = &data[i * n..(i + 1) * n];
        let w = if weights_1d {
            weights
        } else {
            &weights[i * n..(i + 1) * n]
        };
        weighted_average(row, w, policy)
    };
    let chunks = axis_parallel_chunks(AxisParallelClass::Weighted, outer, n);
    let items: Vec<WeightedMean> = if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map(compute)
            .collect()
    } else {
        (0..outer).map(compute).collect()
    };
    weighted_axis_finish(items)
}

pub fn weighted_axis0<T: Float, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    n: usize,
    outer: usize,
    policy: ScanPolicy,
) -> WeightedAxis {
    let compute_direct = |start: usize, len: usize| {
        let mut sums = vec![0.0_f64; len];
        let mut sum_weights = vec![0.0_f64; len];
        let mut counts = vec![0usize; len];
        for k in 0..n {
            let row = &data[k * outer + start..k * outer + start + len];
            if weights_1d {
                let w = weights[k].to_f64();
                match policy {
                    ScanPolicy::AllValues | ScanPolicy::AllFinite => {
                        for ((sum, sum_weight), &x) in
                            sums.iter_mut().zip(&mut sum_weights).zip(row)
                        {
                            *sum += x.to_f64() * w;
                            *sum_weight += w;
                        }
                        for count in &mut counts {
                            *count += 1;
                        }
                    }
                    ScanPolicy::SkipNan => {
                        for (((sum, sum_weight), count), &x) in sums
                            .iter_mut()
                            .zip(&mut sum_weights)
                            .zip(&mut counts)
                            .zip(row)
                        {
                            if !x.is_nan() {
                                *sum += x.to_f64() * w;
                                *sum_weight += w;
                                *count += 1;
                            }
                        }
                    }
                    ScanPolicy::SkipNonFinite => {
                        for (((sum, sum_weight), count), &x) in sums
                            .iter_mut()
                            .zip(&mut sum_weights)
                            .zip(&mut counts)
                            .zip(row)
                        {
                            if x.is_finite() {
                                *sum += x.to_f64() * w;
                                *sum_weight += w;
                                *count += 1;
                            }
                        }
                    }
                }
            } else {
                let wrow = &weights[k * outer + start..k * outer + start + len];
                match policy {
                    ScanPolicy::AllValues | ScanPolicy::AllFinite => {
                        for (((sum, sum_weight), count), (&x, &w)) in sums
                            .iter_mut()
                            .zip(&mut sum_weights)
                            .zip(&mut counts)
                            .zip(row.iter().zip(wrow))
                        {
                            let w = w.to_f64();
                            *sum += x.to_f64() * w;
                            *sum_weight += w;
                            *count += 1;
                        }
                    }
                    ScanPolicy::SkipNan => {
                        for (((sum, sum_weight), count), (&x, &w)) in sums
                            .iter_mut()
                            .zip(&mut sum_weights)
                            .zip(&mut counts)
                            .zip(row.iter().zip(wrow))
                        {
                            if !x.is_nan() {
                                let w = w.to_f64();
                                *sum += x.to_f64() * w;
                                *sum_weight += w;
                                *count += 1;
                            }
                        }
                    }
                    ScanPolicy::SkipNonFinite => {
                        for (((sum, sum_weight), count), (&x, &w)) in sums
                            .iter_mut()
                            .zip(&mut sum_weights)
                            .zip(&mut counts)
                            .zip(row.iter().zip(wrow))
                        {
                            if x.is_finite() {
                                let w = w.to_f64();
                                *sum += x.to_f64() * w;
                                *sum_weight += w;
                                *count += 1;
                            }
                        }
                    }
                }
            }
        }
        sums.into_iter()
            .zip(sum_weights)
            .zip(counts)
            .map(|((sum, sum_weight), count)| WeightedMean {
                value: if count == 0 {
                    f64::NAN
                } else {
                    sum / sum_weight
                },
                sum_weight,
                count,
            })
            .collect::<Vec<_>>()
    };

    let chunks = axis_parallel_chunks(AxisParallelClass::Weighted, outer, n);
    let items = if chunks > 1 {
        let chunk_len = outer.div_ceil(chunks);
        (0..outer)
            .step_by(chunk_len)
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|start| compute_direct(start, chunk_len.min(outer - start)))
            .collect()
    } else {
        compute_direct(0, outer)
    };
    weighted_axis_finish(items)
}

pub fn weighted_axis_last_number<T: Number, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    outer: usize,
    n: usize,
) -> WeightedAxis {
    let compute = |i: usize| {
        let row = &data[i * n..(i + 1) * n];
        let w = if weights_1d {
            weights
        } else {
            &weights[i * n..(i + 1) * n]
        };
        number_weighted_average(row, w)
    };
    let chunks = axis_parallel_chunks(AxisParallelClass::Weighted, outer, n);
    let items: Vec<WeightedMean> = if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map(compute)
            .collect()
    } else {
        (0..outer).map(compute).collect()
    };
    weighted_axis_finish(items)
}

pub fn weighted_axis0_number<T: Number, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    n: usize,
    outer: usize,
) -> WeightedAxis {
    let compute_direct = |start: usize, len: usize| {
        let mut sums = vec![0.0_f64; len];
        let mut sum_weights = vec![0.0_f64; len];
        let mut counts = vec![0usize; len];
        for k in 0..n {
            let row = &data[k * outer + start..k * outer + start + len];
            if weights_1d {
                let w = weights[k].to_f64();
                for ((sum, sum_weight), &x) in sums.iter_mut().zip(&mut sum_weights).zip(row) {
                    *sum += x.to_f64() * w;
                    *sum_weight += w;
                }
                for count in &mut counts {
                    *count += 1;
                }
            } else {
                let wrow = &weights[k * outer + start..k * outer + start + len];
                for (((sum, sum_weight), count), (&x, &w)) in sums
                    .iter_mut()
                    .zip(&mut sum_weights)
                    .zip(&mut counts)
                    .zip(row.iter().zip(wrow))
                {
                    let w = w.to_f64();
                    *sum += x.to_f64() * w;
                    *sum_weight += w;
                    *count += 1;
                }
            }
        }
        sums.into_iter()
            .zip(sum_weights)
            .zip(counts)
            .map(|((sum, sum_weight), count)| WeightedMean {
                value: if count == 0 {
                    f64::NAN
                } else {
                    sum / sum_weight
                },
                sum_weight,
                count,
            })
            .collect::<Vec<_>>()
    };

    let chunks = axis_parallel_chunks(AxisParallelClass::Weighted, outer, n);
    let items = if chunks > 1 {
        let chunk_len = outer.div_ceil(chunks);
        (0..outer)
            .step_by(chunk_len)
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|start| compute_direct(start, chunk_len.min(outer - start)))
            .collect()
    } else {
        compute_direct(0, outer)
    };
    weighted_axis_finish(items)
}

// --------------------------------------------------------------------------
// percentiles (output laid out as `nq` rows of `outer`: out[qi * outer + j])
// --------------------------------------------------------------------------

pub fn percentiles_axis_last<T: Float>(
    data: &[T],
    outer: usize,
    n: usize,
    qs: &[f64],
    policy: ScanPolicy,
) -> Vec<f64> {
    let gather = |buf: &mut Vec<T>, i: usize| {
        buf.clear();
        buf.extend_from_slice(&data[i * n..(i + 1) * n]);
    };
    percentile_outputs(outer, n, qs, gather, policy)
}

pub fn percentiles_axis0<T: Float>(
    data: &[T],
    n: usize,
    outer: usize,
    qs: &[f64],
    policy: ScanPolicy,
) -> Vec<f64> {
    let gather = |buf: &mut Vec<T>, j: usize| {
        buf.clear();
        for k in 0..n {
            buf.push(data[k * outer + j]);
        }
    };
    percentile_outputs(outer, n, qs, gather, policy)
}

/// Compute `nq` percentiles for each of `outer` output elements, returning a
/// `(nq, outer)` row-major result. `gather(buf, j)` fills `buf` with output
/// element `j`'s reducing-axis values (reused per worker thread).
#[inline]
fn percentile_outputs<T, G>(
    outer: usize,
    n: usize,
    qs: &[f64],
    gather: G,
    policy: ScanPolicy,
) -> Vec<f64>
where
    T: Float,
    G: Fn(&mut Vec<T>, usize) + Sync,
{
    let nq = qs.len();
    let compute = |buf: &mut Vec<T>, j: usize| {
        gather(buf, j);
        let mut local = vec![f64::NAN; nq];
        percentiles_in_place(buf, qs, &mut local, policy);
        local
    };
    let chunks = axis_parallel_chunks(AxisParallelClass::OrderPercentile, outer, n);
    let per_output: Vec<Vec<f64>> = if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map_init(|| Vec::<T>::with_capacity(n), &compute)
            .collect()
    } else {
        let mut buf = Vec::<T>::with_capacity(n);
        (0..outer).map(|j| compute(&mut buf, j)).collect()
    };
    let mut out = vec![f64::NAN; nq * outer];
    for (j, values) in per_output.into_iter().enumerate() {
        for (qi, v) in values.into_iter().enumerate() {
            out[qi * outer + j] = v;
        }
    }
    out
}

pub fn percentiles_axis_last_number<T: Number>(
    data: &[T],
    outer: usize,
    n: usize,
    qs: &[f64],
) -> Vec<f64> {
    let gather = |buf: &mut Vec<T>, i: usize| {
        buf.clear();
        buf.extend_from_slice(&data[i * n..(i + 1) * n]);
    };
    percentile_outputs_number(outer, n, qs, gather)
}

pub fn percentiles_axis0_number<T: Number>(
    data: &[T],
    n: usize,
    outer: usize,
    qs: &[f64],
) -> Vec<f64> {
    let gather = |buf: &mut Vec<T>, j: usize| {
        buf.clear();
        for k in 0..n {
            buf.push(data[k * outer + j]);
        }
    };
    percentile_outputs_number(outer, n, qs, gather)
}

#[inline]
fn percentile_outputs_number<T, G>(outer: usize, n: usize, qs: &[f64], gather: G) -> Vec<f64>
where
    T: Number,
    G: Fn(&mut Vec<T>, usize) + Sync,
{
    let nq = qs.len();
    let compute = |buf: &mut Vec<T>, j: usize| {
        gather(buf, j);
        let mut local = vec![f64::NAN; nq];
        number_percentiles_in_place(buf, qs, &mut local);
        local
    };
    let chunks = axis_parallel_chunks(AxisParallelClass::OrderPercentile, outer, n);
    let per_output: Vec<Vec<f64>> = if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map_init(|| Vec::<T>::with_capacity(n), &compute)
            .collect()
    } else {
        let mut buf = Vec::<T>::with_capacity(n);
        (0..outer).map(|j| compute(&mut buf, j)).collect()
    };
    let mut out = vec![f64::NAN; nq * outer];
    for (j, values) in per_output.into_iter().enumerate() {
        for (qi, v) in values.into_iter().enumerate() {
            out[qi * outer + j] = v;
        }
    }
    out
}
