"""Check an installed distribution outside the source checkout (Python 3.10+)."""

from __future__ import annotations

import logging
import os
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import reducers
from reducers import _core


def main() -> None:
    """Verify installed paths, version, license, and representative reductions.

    Raises
    ------
    AssertionError
        The distribution is incomplete, imported from outside the environment,
        has the wrong version, or produces incorrect reduction results.
    KeyError
        The caller has not set `EXPECTED_VERSION`.
    """
    expected = os.environ["EXPECTED_VERSION"]
    assert version("reducers") == reducers.__version__ == _core.__version__ == expected
    for module in (reducers, _core):
        assert (
            Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        )
    license_file = files("reducers").joinpath("licenses/imcombiners-BSD-3-Clause.txt")
    assert "Redistribution and use" in license_file.read_text(encoding="utf-8")
    for dtype in (np.float32, np.float64):
        values = np.array([1, 2, 9], dtype=dtype)
        assert reducers.mean(values) == 4
        assert reducers.median(values) == 2
        assert reducers.lowlevel.mean_valid(values) == 4
        assert reducers.nanmean(np.array([1, np.nan, 3], dtype=dtype)) == 2
        samples = np.array([1, 2, 3, 4, 100], dtype=dtype)
        np.testing.assert_array_equal(
            reducers.sigclip_mask_1d(samples, sigma=2),
            [False, False, False, False, True],
        )
        assert reducers.sigclip_combine_1d(samples, combine="mean", sigma=2) == 2.5
        stack = samples.reshape(5, 1, 1)
        assert reducers.sigclip_combine(stack, combine="median", sigma=2)[0, 0] == 2.5
        assert reducers.SigClip(sigma=2).apply(stack)[0][-1, 0, 0]
    logging.info("Installed reducers %s passed smoke checks", expected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
