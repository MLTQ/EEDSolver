# stress.py

## Purpose

Independently check the vacuum stress-integral force accounting. Point charges
are analytic verification problems, not a discretized PTFE capacitor.

## Components

`sphere_force` samples exact Coulomb fields and integrates
`epsilon0 * (E (E.n) - n E^2/2)` over a spherical vacuum surface using
Gauss–Legendre polar and periodic azimuthal quadrature. It returns the force
and the enclosed charge inferred from Gauss's law.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| test_benchmark.py | Cartesian SI vectors; signed outward traction | Sign or units |
| benchmark.py | Force and flux at selectable quadrature order | Output keys |

## Notes

Charges can be inside/outside the surface. None may lie on it. The external
field callback must be source-free in the relevant vacuum region. Static E
only: no magnetic stress or field-momentum derivative. Increasing `order`
resolves steep fields; a small residual at one resolution alone is insufficient.
