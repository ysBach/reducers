# Changelog

## Unreleased

- CI, tag-only trusted publishing to PyPI and crates.io.

- Add a Python-independent Rust `sigma_clip` API with reusable workspace,
  cumulative finite-only rejection, separate rejection and dispersion centers,
  standard deviation/MAD, rollback limits, final-count requirements, and
  optional clipped mean/median and diagnostics. Defaults disable survivor
  rollback (`nkeep=0`, `revert_on_nkeep=false`).
- Add Python sigma-clipping tools: vector and axis-0
  stack diagnostics, mask-only and fused mean/median calls, restoration flags,
  Euclidean mask growth, and reusable `SigClip` settings. All clipping calls use
  the shared Rust engine, with generic defaults `nkeep=0` and
  `revert_on_nkeep=False`. Support empty sample inputs.
- Remove redundant sigma-clipping scans and final reductions, reuse statistics
  scratch, and balance parallel stack columns using the existing grain setting.
  Preserve clipping arithmetic, input order, defaults, and output layouts.

## 0.3.2 - 2026-09-05

- Fix floating-point `[nan]median` overflow and make order statistics deterministic.

## 0.3.1 - 2026-07-14

- Speed up `[nan]var` and `[nan]std` when `return_mean=True` (by returning the
  variance (or standard deviation) and mean from one fused reduction).
- Speed up `[nan]minmax` (by computing both outputs in one fused reduction).
