# reducers

Shortname `rd`.

Rust-backed reduction functions for NumPy arrays - plain (numpy-like) and NaN-aware. The functions I implemented are those listed in the [numba](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#reductions) documentation.

The target is

1. much faster than numpy in many use cases,
2. much faster than bottleneck in many use cases, and
3. especially maximum performance for median and variance calculations, which are often bottlenecks in data processing pipelines.

`reducers` might be slower than numpy or bottleneck for small arrays. However, the most time-consuming reductions like large arrays or deep stacks, `median`, `percentile` or `quantile`, `var` and `std` are frequently several times (>100 times for nanpercentiles) faster than numpy and bottleneck.

## After installation

Run the autotuner once on the machine where `reducers` will run:

```bash
python -m reducers.autotuner
```

It saves parallel-grain settings for that CPU and workload profile. Future
`import reducers` calls apply those settings automatically. The built-in
defaults are still valid; use `python -m reducers.autotuner --reset` to remove
the saved tuning file and return to them.

```python
import numpy as np
import reducers as rd

a = np.array([1.0, 2.0, np.nan, np.inf, 5.0])

rd.mean(a)                      # nan: plain reducers propagate NaN/inf
rd.nanmean(a)                   # inf: skip NaN, keep inf
rd.nanmean(a, ignore_inf=True)  # finite-only
rd.nanminmax(a)                 # one fused 1-D scan for nanmin + nanmax
rd.nanpercentile(a, [16, 50, 84])
```

Axis reductions cover the layouts this package optimizes:

```python
rng = np.random.default_rng(20250311)
stack = rng.normal(size=(31, 256, 256)).astype("f4")
rows = rng.normal(size=(256, 256, 31)).astype("f4")

rd.nanmedian(stack, axis=0)      # stack reduction -> shape (256, 256)
rd.nanmean(rows, axis=-1)        # contiguous trailing-axis reduction
```

For spread reducers, `return_mean=True` returns the already-computed mean with
the variance or standard deviation:

```python
std, mean = rd.nanstd(a, ddof=1, return_mean=True)
```

Dual use: the kernel modules are pure Rust (no PyO3/NumPy) and usable as a crate;
the `reducers._core` Python extension is built with the `extension-module`
feature.

## Current limits

- `axis` may be `None` (default, whole-array), `0` or `-1` (identical to `a.ndim - 1`); other axes raise `NotImplementedError`. This keeps hidden transpose/copy costs out of the API and lets the Rust kernels specialize for the supported layouts.

- NumPy-like **subset**: There are many unsupported parameters like `out`, `keepdims`, `where`, `dtype`, or percentile `method` (linear only). Adding them will not likely be considered unless there is a strong use case, as they add complexity and maintenance burden. The main focus is on the core reduction logic and, more importantly, performance.
