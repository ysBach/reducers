"""Persisted configuration for runtime-tuned parallel grains."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._core import (
    get_default_parallel_grains,
    get_parallel_grains,
    set_parallel_grains,
)

CONFIG_ENV = "REDUCERS_PARALLEL_GRAINS_FILE"
IGNORE_ENV = "REDUCERS_IGNORE_TUNED_GRAINS"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "reducers" / "parallel_grains.json"


def parallel_grains_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the parallel-grain config path.

    Parameters
    ----------
    path : path-like, optional
        Explicit config path. When omitted, `REDUCERS_PARALLEL_GRAINS_FILE` is
        used if set; otherwise the per-user default path is returned.
    """
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get(CONFIG_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_CONFIG_PATH


def _coerce_grains(data: Any) -> dict[str, int]:
    if not isinstance(data, Mapping):
        msg = "parallel grain config must contain a JSON object"
        raise ValueError(msg)

    defaults = get_default_parallel_grains()
    unknown = sorted(set(data) - set(defaults))
    if unknown:
        msg = f"unknown parallel grain: {unknown[0]}"
        raise ValueError(msg)

    grains = dict(defaults)
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            msg = f"parallel grain `{key}` must be a positive integer"
            raise ValueError(msg)
        grains[key] = value
    return grains


def load_parallel_grains_config(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, int] | None:
    """Load saved parallel grains.

    Parameters
    ----------
    path : path-like, optional
        Config path. Defaults to `parallel_grains_config_path()`.

    Returns
    -------
    dict or None
        Full grain mapping, or `None` when no config file exists.
    """
    config_path = parallel_grains_config_path(path)
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        msg = f"invalid parallel grain config: {config_path}"
        raise ValueError(msg) from exc
    return _coerce_grains(data)


def apply_parallel_grains_config(path: str | os.PathLike[str] | None = None) -> bool:
    """Apply saved parallel grains if a config file exists."""
    if os.environ.get(IGNORE_ENV):
        return False
    grains = load_parallel_grains_config(path)
    if grains is None:
        return False
    set_parallel_grains(grains)
    return True


def save_parallel_grains_config(
    grains: Mapping[str, int] | None = None,
    path: str | os.PathLike[str] | None = None,
) -> Path:
    """Save parallel grains for automatic use on future imports."""
    config_path = parallel_grains_config_path(path)
    values = get_parallel_grains() if grains is None else _coerce_grains(grains)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    return config_path


def use_default_parallel_grains() -> dict[str, int]:
    """Restore the built-in parallel grains for this Python process."""
    return set_parallel_grains(get_default_parallel_grains())


def clear_parallel_grains_config(
    path: str | os.PathLike[str] | None = None,
    *,
    reset: bool = True,
) -> bool:
    """Remove saved grains and optionally restore built-in defaults."""
    config_path = parallel_grains_config_path(path)
    removed = False
    if config_path.exists():
        config_path.unlink()
        removed = True
    if reset:
        use_default_parallel_grains()
    return removed


def apply_saved_parallel_grains_on_import() -> None:
    """Apply saved grains during package import, warning on bad config."""
    try:
        apply_parallel_grains_config()
    except ValueError as exc:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
