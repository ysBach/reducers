//! Mask operations shared by rejection pipelines.
//!
//! The public kernel uses a flat row-major buffer and an explicit shape so it
//! remains independent of ndarray and PyO3.  Axis 0 is the stack axis; growth
//! is applied independently inside each spatial frame.

// Adapted from imcombiners, Copyright 2026 @ysBach, BSD-3-Clause.
// Full notice: python/reducers/licenses/imcombiners-BSD-3-Clause.txt.

use rayon::prelude::*;

fn checked_product(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        Some(0)
    } else {
        shape
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
    }
}

fn spatial_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn unravel_index(mut index: usize, strides: &[usize], shape: &[usize], coords: &mut [usize]) {
    for ((coord, &stride), &dimension) in coords.iter_mut().zip(strides).zip(shape) {
        *coord = index / stride;
        index %= stride;
        debug_assert!(*coord < dimension);
    }
}

fn build_offsets(spatial_shape: &[usize], radius: f64) -> Vec<Vec<isize>> {
    let radius2 = radius * radius;
    let radius_floor = radius.floor();
    // Offsets outside the spatial extent can never contribute to the output.
    // Capping each axis is mathematically equivalent to the uncapped
    // enumeration and avoids enormous allocations for a radius larger than a
    // frame while retaining exact Euclidean inclusion at the boundary.
    let max_deltas: Vec<isize> = spatial_shape
        .iter()
        .map(|&dimension| {
            let extent = dimension.saturating_sub(1);
            if radius_floor >= extent as f64 {
                extent as isize
            } else {
                radius_floor as isize
            }
        })
        .collect();
    let mut offsets = Vec::new();
    let mut current = vec![0isize; spatial_shape.len()];

    fn visit(
        axis: usize,
        max_deltas: &[isize],
        radius2: f64,
        current: &mut [isize],
        offsets: &mut Vec<Vec<isize>>,
    ) {
        if axis == current.len() {
            let distance2 = current
                .iter()
                .map(|&value| {
                    let value = value as f64;
                    value * value
                })
                .sum::<f64>();
            if distance2 <= radius2 {
                offsets.push(current.to_vec());
            }
            return;
        }
        for delta in -max_deltas[axis]..=max_deltas[axis] {
            current[axis] = delta;
            visit(axis + 1, max_deltas, radius2, current, offsets);
        }
    }

    visit(0, &max_deltas, radius2, &mut current, &mut offsets);
    offsets
}

/// Dilate a stack mask over its spatial axes with an exact Euclidean radius.
///
/// `mask` is a flat row-major buffer whose dimensions are given by `shape`.
/// The shape must have at least two dimensions, with axis 0 being the stack
/// axis. Every `true` sample marks spatial samples at integer-grid Euclidean
/// distance less than or equal to `grow`; samples are never propagated between
/// stack frames. The returned buffer has the same flat shape as the input.
///
/// A radius of zero returns a copy. A radius below one therefore adds no
/// samples, while preserving all existing `true` values.
pub fn grow_mask(mask: &[bool], shape: &[usize], grow: f64) -> Result<Vec<bool>, String> {
    if shape.len() < 2 {
        return Err(format!(
            "grow mask must have shape (N, *spatial); got {shape:?}"
        ));
    }
    if !grow.is_finite() || grow < 0.0 {
        return Err("grow must be a finite non-negative radius".to_string());
    }

    let expected_len =
        checked_product(shape).ok_or_else(|| "grow mask dimensions are too large".to_string())?;
    if mask.len() != expected_len {
        return Err(format!(
            "mask length {} does not match shape {shape:?} (expected {expected_len})",
            mask.len()
        ));
    }

    let spatial_shape = &shape[1..];
    let spatial_size = checked_product(spatial_shape)
        .ok_or_else(|| "grow mask dimensions are too large".to_string())?;
    if spatial_size == 0 || grow == 0.0 || !mask.iter().any(|&value| value) {
        return Ok(mask.to_vec());
    }

    let max_dist2 = spatial_shape
        .iter()
        .map(|&dimension| {
            let delta = dimension.saturating_sub(1) as f64;
            delta * delta
        })
        .sum::<f64>();
    let frame_count = shape[0];
    let mut output = vec![false; mask.len()];

    // If a single source sample reaches every point in a frame, preserve the
    // input's per-frame behavior while avoiding offset construction.
    if grow * grow >= max_dist2 {
        output
            .par_chunks_mut(spatial_size)
            .enumerate()
            .for_each(|(frame, output_frame)| {
                let start = frame * spatial_size;
                let input_frame = &mask[start..start + spatial_size];
                if input_frame.iter().any(|&value| value) {
                    output_frame.fill(true);
                }
            });
        return Ok(output);
    }

    let strides = spatial_strides(spatial_shape);
    let offsets = build_offsets(spatial_shape, grow);

    output
        .par_chunks_mut(spatial_size)
        .enumerate()
        .take(frame_count)
        .for_each(|(frame, output_frame)| {
            let start = frame * spatial_size;
            let input_frame = &mask[start..start + spatial_size];
            let mut coords = vec![0usize; spatial_shape.len()];

            for (source_index, &is_rejected) in input_frame.iter().enumerate() {
                if !is_rejected {
                    continue;
                }
                unravel_index(source_index, &strides, spatial_shape, &mut coords);
                for offset in &offsets {
                    let mut destination_index = 0usize;
                    let mut in_bounds = true;
                    for axis in 0..spatial_shape.len() {
                        let coordinate = coords[axis] as isize + offset[axis];
                        if coordinate < 0 || coordinate >= spatial_shape[axis] as isize {
                            in_bounds = false;
                            break;
                        }
                        destination_index += coordinate as usize * strides[axis];
                    }
                    if in_bounds {
                        output_frame[destination_index] = true;
                    }
                }
            }
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::grow_mask;

    #[test]
    fn grows_exact_euclidean_neighborhood_without_crossing_stack_axis() {
        let input = vec![false, false, false, false, true, false, false, false, false];
        let cross = vec![false, true, false, true, true, true, false, true, false];
        assert_eq!(grow_mask(&input, &[1, 3, 3], 1.0).unwrap(), cross);

        let diagonal = vec![true; 9];
        assert_eq!(grow_mask(&input, &[1, 3, 3], 1.5).unwrap(), diagonal);

        let two_frames = vec![false, true, false, false, false, false];
        assert_eq!(
            grow_mask(&two_frames, &[2, 1, 3], 1.0).unwrap(),
            vec![true, true, true, false, false, false]
        );
    }

    #[test]
    fn zero_radius_and_empty_masks_are_copies() {
        let input = vec![false, true, false, false];
        assert_eq!(grow_mask(&input, &[1, 2, 2], 0.0).unwrap(), input);
        assert_eq!(grow_mask(&input, &[1, 2, 2], 0.5).unwrap(), input);
        assert_eq!(grow_mask(&[], &[0, 3, 3], 2.0).unwrap(), Vec::<bool>::new());
    }

    #[test]
    fn rejects_invalid_radius_shape_and_length() {
        let input = vec![false, true, false, false];
        assert!(grow_mask(&input, &[4], 1.0).is_err());
        assert!(grow_mask(&input, &[1, 2, 2], -1.0).is_err());
        assert!(grow_mask(&input, &[1, 2, 2], f64::NAN).is_err());
        assert!(grow_mask(&input[..3], &[1, 2, 2], 1.0).is_err());
    }
}
