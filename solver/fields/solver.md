# solver/fields/solver.py

## Purpose
Orchestrates the full solve pipeline: geometry → mesh → dolfinx → solve → SolveOutput. Entry point for the server's `/solve` route.

## Components
- `run_solve(request)` — main entry point, returns `SolveOutput`
- `SolveOutput` — dataclass holding φ, A, J, mesh, timing, and warnings
- `_extract_fields` — unpacks solution depending on formulation (scalar returns φ, maxwell returns A, coupled splits mixed function)

## Decisions
- Physical group tag constants (WIRE_TAG=1, AIR_TAG=2, BOUNDARY_TAG=10) defined here — must match coil.py tag assignments
- Total wall-clock time (including mesh gen) reported in solve_time_s, not just FEniCSx solve
- B = curl(A) computed lazily on SolveOutput.B property access to avoid overhead when not needed
- Warnings collected (not raised) for missing wire cells — allows server to return a result with explanation

## Contracts
- Must run inside the Docker container (`docker compose up solver`) — dolfinx/gmsh are not in the UV venv
- `SolveOutput.phi` is None for maxwell_only
- `SolveOutput.A` is None for scalar_only
- WIRE_TAG / BOUNDARY_TAG must stay in sync with coil.py physical group naming
