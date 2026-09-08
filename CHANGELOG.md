# Changelog

## 0.4.0 - 2026-09-08

- Add reusable Rust and Python sigma-clipping APIs from
  [imcombiners](https://github.com/ysBach/imcombiners), with defaults
  changed to `nkeep=0` and `revert_on_nkeep=False` (more general).
- Speed up sigma clipping through statistics/scratch reuse, reduced Python
  overhead, and better parallel scheduling; see [performance notes](docs/quarto/performance/sigma-clipping.qmd).
- Standardize CI and tag-only trusted publishing to PyPI and crates.io.

## 0.3.2 - 2026-09-05

- Fix floating-point `[nan]median` overflow and make order statistics deterministic.

## 0.3.1 - 2026-07-14

- Speed up `[nan]var` and `[nan]std` when `return_mean=True` (by returning the
  variance (or standard deviation) and mean from one fused reduction).
- Speed up `[nan]minmax` (by computing both outputs in one fused reduction).
