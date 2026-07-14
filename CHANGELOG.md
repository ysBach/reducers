# Changelog

## 0.3.1 - 2026-07-14

- Speed up `[nan]var` and `[nan]std` when `return_mean=True` (by returning the
  variance (or standard deviation) and mean from one fused reduction).
- Speed up `[nan]minmax` (by computing both outputs in one fused reduction).

