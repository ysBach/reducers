"""Rust-backed reductions for NumPy arrays (plain + NaN-aware).

``mean``/``median``/... follow numpy plain semantics (NaN/inf propagate).
``nanmean``/``nanmedian``/... follow ``np.nan*`` (skip NaN, keep inf), with an
``ignore_inf=True`` option to also drop ``+/-inf``.

Each function's docstring covers call-site behavior only. Kernel design,
parallelization, and benchmarks are documented in the project docs (the
"How It Gets Fast" page).
"""

from ._array import (
    average,
    count_finite,
    lmedian,
    max,
    mean,
    median,
    min,
    minmax,
    nanaverage,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanminmax,
    nanpercentile,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
    percentile,
    quantile,
    std,
    sum,
    var,
)
from ._config import (
    apply_parallel_grains_config,
    apply_saved_parallel_grains_on_import,
    clear_parallel_grains_config,
    load_parallel_grains_config,
    parallel_grains_config_path,
    save_parallel_grains_config,
    use_default_parallel_grains,
)
from ._core import (
    get_default_parallel_grains,
    get_num_threads,
    get_parallel_grains,
    set_axis_order_grain,
    set_axis_scan_grain,
    set_minmax_1d_grain,
    set_parallel_grain,
    set_parallel_grains,
)

apply_saved_parallel_grains_on_import()

try:
    from ._core import __version__
except ImportError:  # pragma: no cover - extension not yet built
    __version__ = "0.0.0+unbuilt"

__all__ = [
    "average",
    "count_finite",
    "apply_parallel_grains_config",
    "clear_parallel_grains_config",
    "get_default_parallel_grains",
    "get_num_threads",
    "get_parallel_grains",
    "lmedian",
    "load_parallel_grains_config",
    "max",
    "mean",
    "median",
    "min",
    "minmax",
    "nanaverage",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanminmax",
    "nanpercentile",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    "parallel_grains_config_path",
    "percentile",
    "quantile",
    "save_parallel_grains_config",
    "set_axis_order_grain",
    "set_axis_scan_grain",
    "set_minmax_1d_grain",
    "set_parallel_grain",
    "set_parallel_grains",
    "std",
    "sum",
    "use_default_parallel_grains",
    "var",
]
