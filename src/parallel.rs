//! Rayon grain controls, overridable via environment variables.

use std::collections::BTreeMap;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_AXIS_SCAN_PLAIN_GRAIN: usize = 262_144;
const DEFAULT_AXIS_SCAN_NAN_GRAIN: usize = 262_144;
const DEFAULT_AXIS_SCAN_VAR_GRAIN: usize = 262_144;
const DEFAULT_AXIS_WEIGHTED_GRAIN: usize = 262_144;
const DEFAULT_AXIS_ORDER_MEDIAN_GRAIN: usize = 1_024;
const DEFAULT_AXIS_ORDER_PERCENTILE_GRAIN: usize = 1_024;
const DEFAULT_MINMAX_1D_GRAIN: usize = 65_536;

static AXIS_SCAN_PLAIN_GRAIN: AtomicUsize = AtomicUsize::new(0);
static AXIS_SCAN_NAN_GRAIN: AtomicUsize = AtomicUsize::new(0);
static AXIS_SCAN_VAR_GRAIN: AtomicUsize = AtomicUsize::new(0);
static AXIS_WEIGHTED_GRAIN: AtomicUsize = AtomicUsize::new(0);
static AXIS_ORDER_MEDIAN_GRAIN: AtomicUsize = AtomicUsize::new(0);
static AXIS_ORDER_PERCENTILE_GRAIN: AtomicUsize = AtomicUsize::new(0);
static MINMAX_1D_GRAIN: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisParallelClass {
    ScanPlain,
    ScanNan,
    ScanVar,
    Weighted,
    OrderMedian,
    OrderPercentile,
}

struct GrainSpec {
    env_name: &'static str,
    default: usize,
    cache: &'static AtomicUsize,
}

const GRAIN_KEYS: &[&str] = &[
    "axis_scan_plain",
    "axis_scan_nan",
    "axis_scan_var",
    "axis_weighted",
    "axis_order_median",
    "axis_order_percentile",
    "minmax_1d",
];

fn grain_spec(key: &str) -> Option<GrainSpec> {
    Some(match key {
        "axis_scan_plain" => GrainSpec {
            env_name: "REDUCERS_AXIS_SCAN_PLAIN_GRAIN",
            default: DEFAULT_AXIS_SCAN_PLAIN_GRAIN,
            cache: &AXIS_SCAN_PLAIN_GRAIN,
        },
        "axis_scan_nan" => GrainSpec {
            env_name: "REDUCERS_AXIS_SCAN_NAN_GRAIN",
            default: DEFAULT_AXIS_SCAN_NAN_GRAIN,
            cache: &AXIS_SCAN_NAN_GRAIN,
        },
        "axis_scan_var" => GrainSpec {
            env_name: "REDUCERS_AXIS_SCAN_VAR_GRAIN",
            default: DEFAULT_AXIS_SCAN_VAR_GRAIN,
            cache: &AXIS_SCAN_VAR_GRAIN,
        },
        "axis_weighted" => GrainSpec {
            env_name: "REDUCERS_AXIS_WEIGHTED_GRAIN",
            default: DEFAULT_AXIS_WEIGHTED_GRAIN,
            cache: &AXIS_WEIGHTED_GRAIN,
        },
        "axis_order_median" => GrainSpec {
            env_name: "REDUCERS_AXIS_ORDER_MEDIAN_GRAIN",
            default: DEFAULT_AXIS_ORDER_MEDIAN_GRAIN,
            cache: &AXIS_ORDER_MEDIAN_GRAIN,
        },
        "axis_order_percentile" => GrainSpec {
            env_name: "REDUCERS_AXIS_ORDER_PERCENTILE_GRAIN",
            default: DEFAULT_AXIS_ORDER_PERCENTILE_GRAIN,
            cache: &AXIS_ORDER_PERCENTILE_GRAIN,
        },
        "minmax_1d" => GrainSpec {
            env_name: "REDUCERS_MINMAX_1D_GRAIN",
            default: DEFAULT_MINMAX_1D_GRAIN,
            cache: &MINMAX_1D_GRAIN,
        },
        _ => return None,
    })
}

pub fn axis_scan_grain() -> usize {
    parallel_grain("axis_scan_plain").expect("known grain")
}

pub fn axis_order_grain() -> usize {
    parallel_grain("axis_order_median").expect("known grain")
}

pub fn minmax_1d_grain() -> usize {
    parallel_grain("minmax_1d").expect("known grain")
}

pub fn set_axis_scan_grain(value: usize) -> Result<(), &'static str> {
    set_parallel_grain("axis_scan_plain", value).map(|_| ())
}

pub fn set_axis_order_grain(value: usize) -> Result<(), &'static str> {
    set_parallel_grain("axis_order_median", value).map(|_| ())
}

pub fn set_minmax_1d_grain(value: usize) -> Result<(), &'static str> {
    set_parallel_grain("minmax_1d", value).map(|_| ())
}

pub fn parallel_grain(key: &str) -> Result<usize, &'static str> {
    let Some(spec) = grain_spec(key) else {
        return Err("unknown parallel grain");
    };
    Ok(cached_value(spec.cache, spec.env_name, spec.default))
}

pub fn set_parallel_grain(key: &str, value: usize) -> Result<usize, &'static str> {
    let Some(spec) = grain_spec(key) else {
        return Err("unknown parallel grain");
    };
    store_positive(spec.cache, value)?;
    Ok(cached_value(spec.cache, spec.env_name, spec.default))
}

pub fn parallel_grains() -> BTreeMap<&'static str, usize> {
    GRAIN_KEYS
        .iter()
        .map(|&key| (key, parallel_grain(key).expect("known grain")))
        .collect()
}

pub fn default_parallel_grains() -> BTreeMap<&'static str, usize> {
    GRAIN_KEYS
        .iter()
        .map(|&key| {
            let spec = grain_spec(key).expect("known grain");
            (key, spec.default)
        })
        .collect()
}

pub fn axis_parallel_chunks(class: AxisParallelClass, outer: usize, n: usize) -> usize {
    let grain = match class {
        AxisParallelClass::ScanPlain => parallel_grain("axis_scan_plain").expect("known grain"),
        AxisParallelClass::ScanNan => parallel_grain("axis_scan_nan").expect("known grain"),
        AxisParallelClass::ScanVar => parallel_grain("axis_scan_var").expect("known grain"),
        AxisParallelClass::Weighted => parallel_grain("axis_weighted").expect("known grain"),
        AxisParallelClass::OrderMedian => parallel_grain("axis_order_median").expect("known grain"),
        AxisParallelClass::OrderPercentile => {
            parallel_grain("axis_order_percentile").expect("known grain")
        }
    };
    chunks_for(outer.saturating_mul(n), grain, rayon::current_num_threads())
}

pub fn minmax_1d_parallel_chunks(len: usize) -> usize {
    chunks_for(len, minmax_1d_grain(), rayon::current_num_threads())
}

pub fn chunks_for(work: usize, grain: usize, num_threads: usize) -> usize {
    let num_threads = num_threads.max(1);
    let grain = grain.max(1);
    let chunks = work / grain;
    chunks.clamp(1, num_threads)
}

fn cached_value(cache: &AtomicUsize, env_name: &str, default: usize) -> usize {
    let current = cache.load(Ordering::Relaxed);
    if current != 0 {
        return current;
    }
    let value = env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default);
    let _ = cache.compare_exchange(0, value, Ordering::Relaxed, Ordering::Relaxed);
    value
}

fn store_positive(cache: &AtomicUsize, value: usize) -> Result<(), &'static str> {
    if value == 0 {
        return Err("parallel grains must be positive integers");
    }
    cache.store(value, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{chunks_for, AxisParallelClass};

    #[test]
    fn chunks_for_ramps_with_work_and_caps_at_threads() {
        assert_eq!(chunks_for(0, 16, 8), 1);
        assert_eq!(chunks_for(15, 16, 8), 1);
        assert_eq!(chunks_for(16, 16, 8), 1);
        assert_eq!(chunks_for(32, 16, 8), 2);
        assert_eq!(chunks_for(64, 16, 8), 4);
        assert_eq!(chunks_for(1024, 16, 8), 8);
    }

    #[test]
    fn chunks_for_handles_zero_thread_input() {
        assert_eq!(chunks_for(1024, 16, 0), 1);
    }

    #[test]
    fn axis_parallel_classes_are_explicit() {
        assert_eq!(AxisParallelClass::ScanPlain, AxisParallelClass::ScanPlain);
        assert_ne!(AxisParallelClass::ScanPlain, AxisParallelClass::OrderMedian);
    }
}
