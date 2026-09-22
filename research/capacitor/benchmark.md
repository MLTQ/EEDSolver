# benchmark.py

## Purpose

Turn the versioned `geometry.json` into reproducible estimates, candidate voltage
curves, conditional amplitude limits and stress-quadrature convergence results.

## Components

`run` reads the adjacent geometry, evaluates `model.py`, and independently
checks closed-source force cancellation with `stress.py`. Writes `results.json`
beside the script and prints a compact result. No network or solver/GPU access.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| plot_results.py | Endpoint, voltage_curve and convergence structures | JSON schema |
| README.md | Named SI quantities and source/config hashes | Normalization or configuration |

## Notes

Run from any directory with a Python environment containing NumPy. Input source
PDF is not required at runtime; its recorded hash identifies the reading copy.
The experimental datum is at +10 kV only; other curve points are model outputs.
Quadrature uses synthetic charges solely to verify momentum/force accounting.
Outputs have no timestamps, allowing deterministic regeneration.
