# model.py

## Purpose

Reproduce a published capacitor-force formula beside a transparent Maxwell
baseline. All inputs/outputs use SI. This is an analytic benchmark, not FEM or
a simulation of the Dresden apparatus.

## Components

- `Capacitor`: validates finite positive dimensions/material parameters and
  estimates capacitance, dielectric mass, charge, energy and internal load.
- `at_voltage`: evaluates ordinary estimates and two separately labeled
  candidate normalizations. The isolated static self-force is analytically
  zero by translation invariance; it is not a fitted or numerically solved value.
- `signed_amplitude_interval`: computes the interval for a signed multiplier
  from a reported result and uncertainty. It adds no distributional assumption.
- `stray_capacitance_force`: exposes a conventional external-boundary force;
  it is not present for translation of an isolated complete system.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| benchmark.py | SI keys and explicit candidate names | Changed normalization, units or labels |
| test_benchmark.py | Independent energy/scaling/unit checks | Formula or convention changes |

## Notes

Internal pressure assumes constant permittivity under a virtual gap change;
electrostriction and detailed dielectric stress partition are excluded. Radius
to gap ratio is finite, so capacitance/load are leading parallel-plate estimates.
The weak-field junction sheet scale diagnoses the proposed plane metric; it is
not a measured electrode mass or a finished finite-body relativistic solution.
The energy-import weight scale applies to energy added from outside; onboard
battery-to-capacitor transfer does not add that energy to the total apparatus.
