"""
FEniCSx solve loop for Oracle.
Wraps LinearProblem into a timed, logged solve with result extraction.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path

import numpy as np
import ufl
import dolfinx
import dolfinx.fem as fem
import basix.ufl as bufl
from ufl import split as ufl_split

from solver.models.params import (
    CoilParams,
    EEDParams,
    FormulationType,
    MeshResolution,
    SolveRequest,
)
from solver.geometry.coil import build_coil_geometry
from solver.geometry.mesh import load_mesh_from_file, get_mesh_stats, apply_resolution_preset
from solver.fields.formulation import build_problem

log = logging.getLogger(__name__)


# Physical group tag values — must match coil.py _tag_physical_groups
WIRE_TAG = 1
AIR_TAG = 2
BOUNDARY_TAG = 10


class SolveOutput:
    """
    Container for solve outputs passed to postproc.
    Not a Pydantic model — stays in numpy land until postproc serializes.
    """
    def __init__(
        self,
        phi: dolfinx.fem.Function | None,
        A: dolfinx.fem.Function | None,
        J: dolfinx.fem.Function,
        mesh: dolfinx.mesh.Mesh,
        formulation: FormulationType,
        solve_time_s: float,
        mesh_stats: dict,
        warnings: list[str],
    ):
        self.phi = phi
        self.A = A
        self.J = J
        self.mesh = mesh
        self.formulation = formulation
        self.solve_time_s = solve_time_s
        self.mesh_stats = mesh_stats
        self.warnings = warnings

    @property
    def B(self) -> dolfinx.fem.Function | None:
        """B = curl(A) — computed lazily on first access."""
        if self.A is None:
            return None
        if not hasattr(self, "_B"):
            # curl of N1curl ∈ DG0 vector space
            V_B = fem.functionspace(
                self.mesh,
                bufl.element("DG", self.mesh.basix_cell(), 0, shape=(3,)),
            )
            B_expr = fem.Expression(
                ufl.curl(self.A),
                V_B.element.interpolation_points(),
            )
            self._B = fem.Function(V_B)
            self._B.interpolate(B_expr)
        return self._B


def run_solve(request: SolveRequest) -> SolveOutput:
    """
    Full solve pipeline:
      1. Build Gmsh geometry
      2. Mesh
      3. Load into dolfinx
      4. Build variational problem
      5. Solve
      6. Return SolveOutput

    Raises on mesh or solver failure. Warnings (non-fatal issues) are
    collected in SolveOutput.warnings.
    """
    warnings: list[str] = []
    t0 = time.perf_counter()

    # 1. Build mesh
    log.info(f"Building {request.coil.coil_type} geometry...")
    msh_path = build_coil_geometry(request.coil, request.domain_radius_m)
    log.info(f"Mesh file: {msh_path}")

    # 2. Load into dolfinx
    log.info("Loading mesh into dolfinx...")
    mesh, cell_tags, facet_tags = load_mesh_from_file(msh_path)
    stats = get_mesh_stats(mesh)
    stats["resolution"] = request.mesh_resolution
    log.info(f"Mesh: {stats['num_cells']} cells, {stats['num_nodes']} nodes")

    if stats["num_cells"] == 0:
        raise RuntimeError("Empty mesh — geometry build failed.")

    # Warn if wire cells are missing
    wire_cells = cell_tags.find(WIRE_TAG)
    if len(wire_cells) == 0:
        warnings.append(
            "No cells tagged as coil_wire. J=0 everywhere. "
            "Check coil geometry — wire_radius_m may be too small relative to mesh size."
        )

    # 3. Build and solve
    log.info(f"Building {request.formulation} problem...")
    problem, W, J_func = build_problem(
        mesh=mesh,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        coil_params=request.coil,
        eed_params=request.eed,
        formulation=request.formulation,
        wire_tag=WIRE_TAG,
        boundary_tag=BOUNDARY_TAG,
    )

    log.info("Solving...")
    t_solve_start = time.perf_counter()
    solution = problem.solve()
    solve_time = time.perf_counter() - t_solve_start
    log.info(f"Solve complete in {solve_time:.2f}s")

    # 4. Extract φ and A from solution
    phi_func, A_func = _extract_fields(solution, request.formulation, W)

    total_time = time.perf_counter() - t0
    log.info(f"Total pipeline time: {total_time:.2f}s")

    return SolveOutput(
        phi=phi_func,
        A=A_func,
        J=J_func,
        mesh=mesh,
        formulation=request.formulation,
        solve_time_s=total_time,
        mesh_stats=stats,
        warnings=warnings,
    )


def _extract_fields(
    solution,
    formulation: FormulationType,
    W,
) -> tuple[dolfinx.fem.Function | None, dolfinx.fem.Function | None]:
    """
    Extract (φ, A) from solve result depending on formulation.
    Returns (phi, A) with None for fields not present in the formulation.
    """
    if formulation == "scalar_only":
        # solution is φ directly
        return solution, None

    elif formulation == "maxwell_only":
        # solution is A directly
        return None, solution

    elif formulation == "eed_coupled":
        # solution is in mixed space W = CG1 × N1curl
        # split into sub-functions
        phi_sub, A_sub = solution.split()
        # Collapse to their own function spaces for postproc
        phi_func = phi_sub.collapse()
        A_func = A_sub.collapse()
        return phi_func, A_func

    else:
        raise ValueError(f"Unknown formulation: {formulation}")
