//! Axis reductions for normalized 2-D inputs.
//!
//! Last-axis reductions operate on contiguous rows. Axis-0 order statistics use
//! scratch buffers because they reorder values; cheap min/max scans stream the
//! input rows directly into the output buffer.

use rayon::prelude::*;

use crate::finite::{Float, ScanPolicy};
use crate::parallel::{axis_parallel_chunks, AxisParallelClass};
use crate::reducers_1d::{
    apply, apply_mut, apply_number, apply_number_mut, lmedian_valid_in_place,
    median_valid_in_place, number_lmedian_value_in_place, number_max_value, number_min_value,
    number_percentiles_in_place, number_weighted_average, number_weighted_sum,
    percentiles_in_place, weighted_average, weighted_sum, Kind, Number, Weight, WeightedMean,
    WeightedSum,
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

#[inline]
fn gather_axis0_all_values<T: Float>(
    data: &[T],
    n: usize,
    outer: usize,
    j: usize,
    buf: &mut [T],
) -> Option<usize> {
    let mut idx = j;
    for slot in &mut buf[..n] {
        let x = data[idx];
        if x.is_nan() {
            return None;
        }
        *slot = x;
        idx += outer;
    }
    Some(n)
}

#[inline]
fn gather_axis0_all_finite<T: Float>(
    data: &[T],
    n: usize,
    outer: usize,
    j: usize,
    buf: &mut [T],
) -> usize {
    let mut idx = j;
    for slot in &mut buf[..n] {
        *slot = data[idx];
        idx += outer;
    }
    n
}

#[inline]
fn gather_axis0_skip_nan<T: Float>(
    data: &[T],
    n: usize,
    outer: usize,
    j: usize,
    buf: &mut [T],
) -> usize {
    let mut idx = j;
    let mut count = 0;
    for _ in 0..n {
        let x = data[idx];
        if !x.is_nan() {
            buf[count] = x;
            count += 1;
        }
        idx += outer;
    }
    count
}

#[inline]
fn gather_axis0_skip_nonfinite<T: Float>(
    data: &[T],
    n: usize,
    outer: usize,
    j: usize,
    buf: &mut [T],
) -> usize {
    let mut idx = j;
    let mut count = 0;
    for _ in 0..n {
        let x = data[idx];
        if x.is_finite() {
            buf[count] = x;
            count += 1;
        }
        idx += outer;
    }
    count
}

#[inline]
fn axis_lmedian_number_value<T: Number>(buf: &mut [T]) -> T {
    let idx = (buf.len() - 1) / 2;
    let (_, value, _) = buf.select_nth_unstable(idx);
    *value
}

#[inline]
fn axis_median_number_value<T: Number>(buf: &mut [T]) -> f64 {
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

fn reduce_axis0_gathered_float<T: Float>(
    data: &[T],
    n: usize,
    outer: usize,
    kind: Kind,
    ddof: usize,
    policy: ScanPolicy,
) -> Vec<T> {
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

    if matches!(kind, Kind::CountFinite) {
        let mut out = vec![T::zero(); outer];
        if n == 0 {
            return out;
        }

        let reduce_chunk = |start: usize, out_chunk: &mut [T]| {
            let len = out_chunk.len();
            let mut counts = vec![0usize; len];
            for k in 0..n {
                let row = &data[k * outer + start..k * outer + start + len];
                for (count, &x) in counts.iter_mut().zip(row) {
                    if x.is_finite() {
                        *count += 1;
                    }
                }
            }
            for (dst, &count) in out_chunk.iter_mut().zip(&counts) {
                *dst = T::from_f64(count as f64);
            }
        };

        let chunks = axis_parallel_chunks(AxisParallelClass::ScanNan, outer, n);
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

    if matches!(kind, Kind::Var | Kind::Std) {
        let mut out = vec![T::nan(); outer];
        if n == 0 {
            return out;
        }

        let reduce_chunk = |start: usize, out_chunk: &mut [T]| {
            let len = out_chunk.len();
            let mut sums = vec![0.0_f64; len];
            let mut counts = vec![0usize; len];
            match policy {
                ScanPolicy::AllValues | ScanPolicy::AllFinite => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((sum, count), &x) in sums.iter_mut().zip(&mut counts).zip(row) {
                            *sum += x.to_f64();
                            *count += 1;
                        }
                    }
                }
                ScanPolicy::SkipNan => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((sum, count), &x) in sums.iter_mut().zip(&mut counts).zip(row) {
                            if !x.is_nan() {
                                *sum += x.to_f64();
                                *count += 1;
                            }
                        }
                    }
                }
                ScanPolicy::SkipNonFinite => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((sum, count), &x) in sums.iter_mut().zip(&mut counts).zip(row) {
                            if x.is_finite() {
                                *sum += x.to_f64();
                                *count += 1;
                            }
                        }
                    }
                }
            }

            let mut means = vec![f64::NAN; len];
            let mut ss = vec![0.0_f64; len];
            for ((mean, &sum), &count) in means.iter_mut().zip(&sums).zip(&counts) {
                if count > 0 {
                    *mean = sum / count as f64;
                }
            }

            match policy {
                ScanPolicy::AllValues | ScanPolicy::AllFinite => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((acc, &mean), &x) in ss.iter_mut().zip(&means).zip(row) {
                            let d = x.to_f64() - mean;
                            *acc += d * d;
                        }
                    }
                }
                ScanPolicy::SkipNan => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((acc, &mean), &x) in ss.iter_mut().zip(&means).zip(row) {
                            if !x.is_nan() {
                                let d = x.to_f64() - mean;
                                *acc += d * d;
                            }
                        }
                    }
                }
                ScanPolicy::SkipNonFinite => {
                    for k in 0..n {
                        let row = &data[k * outer + start..k * outer + start + len];
                        for ((acc, &mean), &x) in ss.iter_mut().zip(&means).zip(row) {
                            if x.is_finite() {
                                let d = x.to_f64() - mean;
                                *acc += d * d;
                            }
                        }
                    }
                }
            }

            for ((dst, &acc), &count) in out_chunk.iter_mut().zip(&ss).zip(&counts) {
                if count <= ddof {
                    *dst = T::nan();
                } else {
                    let variance = acc / (count - ddof) as f64;
                    *dst = if matches!(kind, Kind::Std) {
                        T::from_f64(variance.sqrt())
                    } else {
                        T::from_f64(variance)
                    };
                }
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

    if matches!(kind, Kind::Median | Kind::LMedian) {
        let chunks = axis_parallel_chunks(AxisParallelClass::OrderMedian, outer, n);
        let mut out = vec![T::nan(); outer];
        let reduce_chunk = |start: usize, out_chunk: &mut [T]| {
            let mut buf = vec![T::zero(); n];
            match (kind, policy) {
                (Kind::Median, ScanPolicy::AllValues) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count =
                            gather_axis0_all_values(data, n, outer, start + offset, &mut buf);
                        *dst = count.map_or_else(T::nan, |count| {
                            T::from_f64(median_valid_in_place(&mut buf[..count]))
                        });
                    }
                }
                (Kind::Median, ScanPolicy::AllFinite) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count =
                            gather_axis0_all_finite(data, n, outer, start + offset, &mut buf);
                        *dst = T::from_f64(median_valid_in_place(&mut buf[..count]));
                    }
                }
                (Kind::Median, ScanPolicy::SkipNan) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count = gather_axis0_skip_nan(data, n, outer, start + offset, &mut buf);
                        *dst = T::from_f64(median_valid_in_place(&mut buf[..count]));
                    }
                }
                (Kind::Median, ScanPolicy::SkipNonFinite) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count =
                            gather_axis0_skip_nonfinite(data, n, outer, start + offset, &mut buf);
                        *dst = T::from_f64(median_valid_in_place(&mut buf[..count]));
                    }
                }
                (Kind::LMedian, ScanPolicy::AllValues) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count =
                            gather_axis0_all_values(data, n, outer, start + offset, &mut buf);
                        *dst = count.map_or_else(T::nan, |count| {
                            T::from_f64(lmedian_valid_in_place(&mut buf[..count]))
                        });
                    }
                }
                (Kind::LMedian, ScanPolicy::AllFinite) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count =
                            gather_axis0_all_finite(data, n, outer, start + offset, &mut buf);
                        *dst = T::from_f64(lmedian_valid_in_place(&mut buf[..count]));
                    }
                }
                (Kind::LMedian, ScanPolicy::SkipNan) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count = gather_axis0_skip_nan(data, n, outer, start + offset, &mut buf);
                        *dst = T::from_f64(lmedian_valid_in_place(&mut buf[..count]));
                    }
                }
                (Kind::LMedian, ScanPolicy::SkipNonFinite) => {
                    for (offset, dst) in out_chunk.iter_mut().enumerate() {
                        let count =
                            gather_axis0_skip_nonfinite(data, n, outer, start + offset, &mut buf);
                        *dst = T::from_f64(lmedian_valid_in_place(&mut buf[..count]));
                    }
                }
                _ => unreachable!("axis-0 order path is only used for median/lmedian"),
            }
        };
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

    reduce_axis0_gathered_float(data, n, outer, kind, ddof, policy)
}

fn reduce_axis0_number_order<T: Number>(
    data: &[T],
    n: usize,
    outer: usize,
    kind: Kind,
    ddof: usize,
) -> Vec<f64> {
    let chunks = axis_parallel_chunks(AxisParallelClass::OrderMedian, outer, n);
    let mut out = vec![f64::NAN; outer];
    if n == 0 {
        return out;
    }
    let reduce_chunk = |start: usize, out_chunk: &mut [f64]| {
        let mut buf = vec![data[0]; n];
        match kind {
            Kind::Median => {
                for (offset, dst) in out_chunk.iter_mut().enumerate() {
                    let mut idx = start + offset;
                    for slot in &mut buf {
                        *slot = data[idx];
                        idx += outer;
                    }
                    *dst = axis_median_number_value(&mut buf);
                }
            }
            Kind::LMedian => {
                for (offset, dst) in out_chunk.iter_mut().enumerate() {
                    let mut idx = start + offset;
                    for slot in &mut buf {
                        *slot = data[idx];
                        idx += outer;
                    }
                    *dst = axis_lmedian_number_value(&mut buf).to_f64();
                }
            }
            _ => {
                for (offset, dst) in out_chunk.iter_mut().enumerate() {
                    let mut idx = start + offset;
                    for slot in &mut buf {
                        *slot = data[idx];
                        idx += outer;
                    }
                    *dst = apply_number_mut(kind, &mut buf[..n], ddof);
                }
            }
        }
    };
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
    out
}

fn reduce_axis0_number_order_exact<T: Number>(data: &[T], n: usize, outer: usize) -> Vec<T> {
    let chunks = axis_parallel_chunks(AxisParallelClass::OrderMedian, outer, n);
    if outer == 0 {
        return Vec::new();
    }
    let mut out = vec![data[0]; outer];
    let reduce_chunk = |start: usize, out_chunk: &mut [T]| {
        let mut buf = vec![data[0]; n];
        for (offset, dst) in out_chunk.iter_mut().enumerate() {
            let mut idx = start + offset;
            for slot in &mut buf {
                *slot = data[idx];
                idx += outer;
            }
            *dst = axis_lmedian_number_value(&mut buf[..n]);
        }
    };
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
    out
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

    let class = if kind.needs_mut() {
        AxisParallelClass::OrderMedian
    } else {
        number_scan_class(kind)
    };
    let chunks = axis_parallel_chunks(class, outer, n);

    if kind.needs_mut() {
        reduce_axis0_number_order(data, n, outer, kind, ddof)
    } else {
        let gather = |buf: &mut Vec<T>, j: usize| {
            let mut idx = j;
            for item in buf.iter_mut().take(n) {
                *item = data[idx];
                idx += outer;
            }
            apply_number_mut(kind, &mut buf[..n], ddof)
        };
        let init_scratch = || {
            let mut buf = Vec::<T>::with_capacity(n);
            if n > 0 {
                buf.resize(n, data[0]);
            }
            buf
        };
        if chunks > 1 {
            (0..outer)
                .into_par_iter()
                .with_min_len(outer.div_ceil(chunks))
                .map_init(init_scratch, gather)
                .collect()
        } else {
            let mut buf = init_scratch();
            (0..outer).map(|j| gather(&mut buf, j)).collect()
        }
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

    if matches!(kind, Kind::LMedian) {
        return reduce_axis0_number_order_exact(data, n, outer);
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

#[derive(Debug)]
pub struct WeightedSumAxis {
    pub weighted_sums: Vec<f64>,
    pub sum_weights: Vec<f64>,
    pub unweighted_sums: Vec<f64>,
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

#[inline]
fn weighted_sum_axis_finish(items: Vec<WeightedSum>) -> WeightedSumAxis {
    let mut weighted_sums = Vec::with_capacity(items.len());
    let mut sum_weights = Vec::with_capacity(items.len());
    let mut unweighted_sums = Vec::with_capacity(items.len());
    for item in items {
        weighted_sums.push(item.weighted_sum);
        sum_weights.push(item.sum_weights);
        unweighted_sums.push(item.unweighted_sum);
    }
    WeightedSumAxis {
        weighted_sums,
        sum_weights,
        unweighted_sums,
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

pub fn weighted_sum_axis_last<T: Float, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    outer: usize,
    n: usize,
    policy: ScanPolicy,
) -> WeightedSumAxis {
    let compute = |i: usize| {
        let row = &data[i * n..(i + 1) * n];
        let w = if weights_1d {
            weights
        } else {
            &weights[i * n..(i + 1) * n]
        };
        weighted_sum(row, w, policy)
    };
    let chunks = axis_parallel_chunks(AxisParallelClass::Weighted, outer, n);
    let items: Vec<WeightedSum> = if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map(compute)
            .collect()
    } else {
        (0..outer).map(compute).collect()
    };
    weighted_sum_axis_finish(items)
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

pub fn weighted_sum_axis0<T: Float, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    n: usize,
    outer: usize,
    policy: ScanPolicy,
) -> WeightedSumAxis {
    let compute_direct = |start: usize, len: usize| {
        let mut weighted_sums = vec![0.0_f64; len];
        let mut sum_weights = vec![0.0_f64; len];
        let mut unweighted_sums = vec![0.0_f64; len];
        for k in 0..n {
            let row = &data[k * outer + start..k * outer + start + len];
            if weights_1d {
                let w = weights[k].to_f64();
                match policy {
                    ScanPolicy::AllValues | ScanPolicy::AllFinite => {
                        for ((weighted_sum, unweighted_sum), &x) in
                            weighted_sums.iter_mut().zip(&mut unweighted_sums).zip(row)
                        {
                            let x = x.to_f64();
                            *weighted_sum += x * w;
                            *unweighted_sum += x;
                        }
                        for sum_weight in &mut sum_weights {
                            *sum_weight += w;
                        }
                    }
                    ScanPolicy::SkipNan => {
                        for (((weighted_sum, sum_weight), unweighted_sum), &x) in weighted_sums
                            .iter_mut()
                            .zip(&mut sum_weights)
                            .zip(&mut unweighted_sums)
                            .zip(row)
                        {
                            if !x.is_nan() {
                                let x = x.to_f64();
                                *weighted_sum += x * w;
                                *sum_weight += w;
                                *unweighted_sum += x;
                            }
                        }
                    }
                    ScanPolicy::SkipNonFinite => {
                        for (((weighted_sum, sum_weight), unweighted_sum), &x) in weighted_sums
                            .iter_mut()
                            .zip(&mut sum_weights)
                            .zip(&mut unweighted_sums)
                            .zip(row)
                        {
                            if x.is_finite() {
                                let x = x.to_f64();
                                *weighted_sum += x * w;
                                *sum_weight += w;
                                *unweighted_sum += x;
                            }
                        }
                    }
                }
            } else {
                let wrow = &weights[k * outer + start..k * outer + start + len];
                match policy {
                    ScanPolicy::AllValues | ScanPolicy::AllFinite => {
                        for (((weighted_sum, sum_weight), unweighted_sum), (&x, &w)) in
                            weighted_sums
                                .iter_mut()
                                .zip(&mut sum_weights)
                                .zip(&mut unweighted_sums)
                                .zip(row.iter().zip(wrow))
                        {
                            let x = x.to_f64();
                            let w = w.to_f64();
                            *weighted_sum += x * w;
                            *sum_weight += w;
                            *unweighted_sum += x;
                        }
                    }
                    ScanPolicy::SkipNan => {
                        for (((weighted_sum, sum_weight), unweighted_sum), (&x, &w)) in
                            weighted_sums
                                .iter_mut()
                                .zip(&mut sum_weights)
                                .zip(&mut unweighted_sums)
                                .zip(row.iter().zip(wrow))
                        {
                            if !x.is_nan() {
                                let x = x.to_f64();
                                let w = w.to_f64();
                                *weighted_sum += x * w;
                                *sum_weight += w;
                                *unweighted_sum += x;
                            }
                        }
                    }
                    ScanPolicy::SkipNonFinite => {
                        for (((weighted_sum, sum_weight), unweighted_sum), (&x, &w)) in
                            weighted_sums
                                .iter_mut()
                                .zip(&mut sum_weights)
                                .zip(&mut unweighted_sums)
                                .zip(row.iter().zip(wrow))
                        {
                            if x.is_finite() {
                                let x = x.to_f64();
                                let w = w.to_f64();
                                *weighted_sum += x * w;
                                *sum_weight += w;
                                *unweighted_sum += x;
                            }
                        }
                    }
                }
            }
        }
        weighted_sums
            .into_iter()
            .zip(sum_weights)
            .zip(unweighted_sums)
            .map(
                |((weighted_sum, sum_weights), unweighted_sum)| WeightedSum {
                    weighted_sum,
                    sum_weights,
                    unweighted_sum,
                    count: 0,
                },
            )
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
    weighted_sum_axis_finish(items)
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

pub fn weighted_sum_axis_last_number<T: Number, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    outer: usize,
    n: usize,
) -> WeightedSumAxis {
    let compute = |i: usize| {
        let row = &data[i * n..(i + 1) * n];
        let w = if weights_1d {
            weights
        } else {
            &weights[i * n..(i + 1) * n]
        };
        number_weighted_sum(row, w)
    };
    let chunks = axis_parallel_chunks(AxisParallelClass::Weighted, outer, n);
    let items: Vec<WeightedSum> = if chunks > 1 {
        (0..outer)
            .into_par_iter()
            .with_min_len(outer.div_ceil(chunks))
            .map(compute)
            .collect()
    } else {
        (0..outer).map(compute).collect()
    };
    weighted_sum_axis_finish(items)
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

pub fn weighted_sum_axis0_number<T: Number, W: Weight>(
    data: &[T],
    weights: &[W],
    weights_1d: bool,
    n: usize,
    outer: usize,
) -> WeightedSumAxis {
    let compute_direct = |start: usize, len: usize| {
        let mut weighted_sums = vec![0.0_f64; len];
        let mut sum_weights = vec![0.0_f64; len];
        let mut unweighted_sums = vec![0.0_f64; len];
        for k in 0..n {
            let row = &data[k * outer + start..k * outer + start + len];
            if weights_1d {
                let w = weights[k].to_f64();
                for ((weighted_sum, unweighted_sum), &x) in
                    weighted_sums.iter_mut().zip(&mut unweighted_sums).zip(row)
                {
                    let x = x.to_f64();
                    *weighted_sum += x * w;
                    *unweighted_sum += x;
                }
                for sum_weight in &mut sum_weights {
                    *sum_weight += w;
                }
            } else {
                let wrow = &weights[k * outer + start..k * outer + start + len];
                for (((weighted_sum, sum_weight), unweighted_sum), (&x, &w)) in weighted_sums
                    .iter_mut()
                    .zip(&mut sum_weights)
                    .zip(&mut unweighted_sums)
                    .zip(row.iter().zip(wrow))
                {
                    let x = x.to_f64();
                    let w = w.to_f64();
                    *weighted_sum += x * w;
                    *sum_weight += w;
                    *unweighted_sum += x;
                }
            }
        }
        weighted_sums
            .into_iter()
            .zip(sum_weights)
            .zip(unweighted_sums)
            .map(
                |((weighted_sum, sum_weights), unweighted_sum)| WeightedSum {
                    weighted_sum,
                    sum_weights,
                    unweighted_sum,
                    count: 0,
                },
            )
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
    weighted_sum_axis_finish(items)
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
