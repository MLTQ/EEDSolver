# test_benchmark.py

## Purpose

Check independent physical identities and prevent confusing internal force,
subsystem force and complete-source force. Tests use standard unittest plus
NumPy and do not require the GPU application or plotting libraries.

## Components

- `CapacitorChecks`: finite-difference virtual work with correct fixed-Q and
  fixed-V thermodynamic potentials; parity/scaling; reproduction of a published
  rounded prediction; independent cgs conversion; signed bounds.
- `StressChecks`: a surface enclosing one charge must recover the independent
  Coulomb pair force; enclosing all charges must cancel under refinement and
  surface movement; an external uniform field must yield total charge times E.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Research validation | Run via unittest discovery in this directory | Test invocation |

## Notes

Passing tests verifies calculations and stated limits, not the existence or
absence of new physics. The material model and experimental uncertainties
are not independently measured here.
