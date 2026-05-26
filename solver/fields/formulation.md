# solver/fields/formulation.py

## Purpose
Defines the UFL variational problems for all three solver formulations. This is the core research code — the governing equations for EED field simulation.

## Components
- `build_problem(...)` — dispatch to correct formulation, returns `(problem, W, J_func)`
- `_build_scalar_only` — CG1 scalar field φ, Helmholtz-like with α² mass term
- `_build_maxwell_only` — N1curl vector potential A, standard magnetostatics (control case)
- `_build_eed_coupled` — mixed CG1×N1curl (φ,A) saddle-point system with β,γ coupling
- `_build_current_density` — DG0 vector J from coil geometry + winding type
- `_compute_current_direction` — per-coil-type azimuthal/poloidal/figure-8 current vectors

## Decisions
- CG1 for φ: standard H¹ scalar FEM — correct for a Helmholtz-type equation
- N1curl (Nédélec) for A: required for H(curl); CG elements for A produce spurious curl-free modes
- DG0 for J: piecewise constant per cell — appropriate for wire subdomain current
- Source term S_φ integrated by parts: `inv_mu0_eps0 * inner(J, grad(ψ)) dx` — avoids computing ∇·J explicitly
- MUMPS direct solver for eed_coupled saddle-point — iterative methods need careful preconditioning for mixed systems; MUMPS is reliable at coarse/medium scale

## Contracts
- `wire_tag` must match the Gmsh physical group tag for "coil_wire" (see coil.py)
- `boundary_tag` must match "boundary_sphere" tag
- `build_problem` returns `problem` compatible with `LinearProblem.solve()` → `Function`
- For eed_coupled, the returned `W` is a MixedFunctionSpace; caller must `split()` the solution to get φ and A separately

## Physics note (FIELD_THEORY.md reference)
S_φ source: computed via IBP from J (not ∇·A) per 2025-05-25 decision. Concentrates at solenoid end caps.
eed_coupled β/γ terms: TODO VERIFY AGAINST DDOF PAPER before trusting quantitative results.
