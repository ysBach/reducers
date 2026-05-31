"""Shared helpers for reducers benchmark scripts."""

from __future__ import annotations

import gc
import platform
import statistics
import sys
import time
from collections.abc import Callable
from importlib import metadata

import numpy as np
import reducers as rd

GRAIN_ENV_NAMES = {
    "axis_scan_plain": "REDUCERS_AXIS_SCAN_PLAIN_GRAIN",
    "axis_scan_nan": "REDUCERS_AXIS_SCAN_NAN_GRAIN",
    "axis_scan_var": "REDUCERS_AXIS_SCAN_VAR_GRAIN",
    "axis_weighted": "REDUCERS_AXIS_WEIGHTED_GRAIN",
    "axis_order_median": "REDUCERS_AXIS_ORDER_MEDIAN_GRAIN",
    "axis_order_percentile": "REDUCERS_AXIS_ORDER_PERCENTILE_GRAIN",
    "minmax_1d": "REDUCERS_MINMAX_1D_GRAIN",
}


def package_version(name: str) -> str:
    """Return an installed package version, or ``not installed``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment_lines(*, bottleneck_available: bool) -> list[str]:
    """Return reproducibility metadata lines for benchmark output."""
    uname = platform.uname()
    lines = [
        f"python: {sys.version.split()[0]} ({platform.python_implementation()})",
        f"reducers: {package_version('reducers')}",
        f"numpy: {package_version('numpy')}",
        "bottleneck: "
        + (package_version("bottleneck") if bottleneck_available else "not installed"),
        f"os: {platform.platform()}",
        f"kernel: {uname.system} {uname.release} {uname.version}",
        f"machine: {uname.machine}",
    ]
    processor = uname.processor or platform.processor()
    if processor:
        lines.append(f"processor: {processor}")
    lines.extend(
        [
            f"reducers threads: {rd.get_num_threads()}",
        ]
    )
    grains = rd.get_parallel_grains()
    for key, env_name in GRAIN_ENV_NAMES.items():
        lines.append(f"{env_name}: {grains[key]}")
    return lines


def print_environment(*, title: str, bottleneck_available: bool) -> None:
    """Print reproducibility metadata above a benchmark table."""
    print(f"# {title}")
    print("# environment")
    for line in environment_lines(bottleneck_available=bottleneck_available):
        print(f"# - {line}")
    print()


def ratio_cell(comp: float | None, rd: float) -> str:
    """Format a ``competitor / reducers`` speedup for a Markdown table cell.

    Returns ``"-"`` when the competitor is absent. Otherwise the ratio is shown
    as ``{:.2f}x`` and bolded when it exceeds ``0.95`` - i.e. when `reducers` is
    the fastest within a 5% margin.
    """
    if comp is None:
        return "-"
    r = comp / rd
    cell = f"{r:.2f}x"
    return f"**{cell}**" if r > 0.95 else cell


def assert_equivalent(got: object, expected: object, *, dtype: str, label: str) -> None:
    """Assert benchmark outputs agree before timing a workload."""
    rtol = 1e-6 if np.dtype(dtype) == np.float32 else 1e-12
    atol = 1e-5 if np.dtype(dtype) == np.float32 else 1e-10
    if isinstance(got, tuple) and isinstance(expected, tuple):
        for idx, (got_item, expected_item) in enumerate(
            zip(got, expected, strict=True)
        ):
            assert_equivalent(
                got_item,
                expected_item,
                dtype=dtype,
                label=f"{label}[{idx}]",
            )
        return
    np.testing.assert_allclose(
        got,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=label,
    )


def trimmed_median(samples: list[float]) -> float:
    """Return a median after dropping one low and one high sample when possible."""
    if len(samples) >= 3:
        samples.remove(min(samples))
        samples.remove(max(samples))
    return statistics.median(samples)


def timeit(
    func: Callable[[], object], *, repeats: int, warmups: int, inner: int = 1
) -> float:
    """Return trimmed median milliseconds per logical function call."""
    for _ in range(warmups):
        for _ in range(inner):
            func()
    samples = []
    was = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(inner):
                func()
            samples.append((time.perf_counter() - t0) * 1e3 / inner)
    finally:
        if was:
            gc.enable()
    return trimmed_median(samples)
