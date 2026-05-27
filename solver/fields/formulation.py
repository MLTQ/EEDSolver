"""
UFL weak form definitions for Oracle solver.
THIS IS THE CORE RESEARCH CODE. See FIELD_THEORY.md for equation derivations.

Rules:
- Do not change weak forms without updating FIELD_THEORY.md decision log.
- Nédélec N1curl for A, CG1 for φ. Do not change element types.
- EED scalar φ is NOT the standard EM scalar potential. Do not conflate.

Three formulations:
  scalar_only  — EED scalar φ alone (cheapest, start here)
  maxwell_only — Standard magnetostatics A (control/baseline)
  eed_coupled  — Full (φ, A) coupled system (primary research formulation)
"""

from __future__ import annotations

import logging
import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import basix.ufl as bufl
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    FacetNormal,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    curl,
    div,
    dx,
    grad,
    inner,
    split,
)

from solver.models.params import CoilParams, EEDParams, FormulationType

log = logging.getLogger(__name__)


MU0 = 4.0 * np.pi * 1e-7  # Permeability of free space [H/m]
EPS0 = 8.854187817e-12     # Permittivity of free space [F/m]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_problem(
    mesh,
    cell_tags,
    facet_tags,
    coil_params: CoilParams,
    eed_params: EEDParams,
    formulation: FormulationType,
    wire_tag: int,
    boundary_tag: int,
) -> tuple:
    """
    Build the FEniCSx variational problem for the requested formulation.

    Returns (problem, W, J_func) where:
      problem — LinearProblem ready to call .solve()
      W       — FunctionSpace (or MixedFunctionSpace for eed_coupled)
      J_func  — Current density Function on mesh (for postproc)
    """
    dispatch = {
        "scalar_only": _build_scalar_only,
        "maxwell_only": _build_maxwell_only,
        "eed_coupled": _build_eed_coupled,
    }
    return dispatch[formulation](
        mesh, cell_tags, facet_tags, coil_params, eed_params, wire_tag, boundary_tag
    )


# ---------------------------------------------------------------------------
# scalar_only
# ---------------------------------------------------------------------------

def _build_scalar_only(mesh, cell_tags, facet_tags, coil_params, eed_params, wire_tag, boundary_tag):
    """
    EED scalar field φ in isolation.

    Weak form:
      ∫ ∇φ·∇ψ dx + α² ∫ φψ dx = ∫ S_φ ψ dx   ∀ψ ∈ H¹₀(Ω)

    Source term (magnetostatic limit):
      S_φ = -(1/μ₀ε₀) ∇·J

    For a solenoid, ∇·J concentrates at the end caps → φ peaks there.
    This is the key EED prediction distinguishing it from B (which peaks at center).

    Function space: CG1 (standard H¹ scalar FEM)
    """
    V = fem.functionspace(mesh, ("CG", 1))

    phi = TrialFunction(V)
    psi = TestFunction(V)

    alpha = eed_params.alpha
    alpha2 = Constant(mesh, dolfinx.default_scalar_type(alpha ** 2))
    inv_mu0_eps0 = Constant(mesh, dolfinx.default_scalar_type(1.0 / (MU0 * EPS0)))

    # Current density on coil wire subdomain
    J_func = _build_current_density(mesh, cell_tags, coil_params, wire_tag, space_dim=3)

    # Source: S_φ = -(1/μ₀ε₀) ∇·J
    # Weak form of S_φ: -(1/μ₀ε₀) ∫ ∇·J · ψ dx
    # Integrate by parts: = (1/μ₀ε₀) ∫ J·∇ψ dx  (boundary term vanishes with Dirichlet φ=0)
    a = inner(grad(phi), grad(psi)) * dx + alpha2 * inner(phi, psi) * dx
    L = inv_mu0_eps0 * inner(J_func, grad(psi)) * dx

    # Dirichlet BC: φ = 0 on outer boundary
    bc_dofs = _get_boundary_dofs(V, facet_tags, boundary_tag)
    bc = dirichletbc(
        dolfinx.default_scalar_type(0.0),
        bc_dofs,
        V,
    )

    # PCG + Hypre BoomerAMG: optimal for symmetric positive-definite Laplacian-like
    # operators (CG1). Scales as O(n) per iteration vs MUMPS O(n^1.5–2) fill-in.
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="oracle_phi_",
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": 1e-10,
            "ksp_max_it": 500,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_agg_nl": 2,
        },
    )

    return problem, V, J_func


# ---------------------------------------------------------------------------
# maxwell_only
# ---------------------------------------------------------------------------

def _build_maxwell_only(mesh, cell_tags, facet_tags, coil_params, eed_params, wire_tag, boundary_tag):
    """
    Standard magnetostatics — vector potential A.

    Weak form:
      (1/μ₀) ∫ curl(A)·curl(v) dx = ∫ J·v dx   ∀v ∈ H₀(curl)

    Gauge: Coulomb gauge enforced implicitly via Nédélec elements + Dirichlet BCs.
    No scalar DOF — this is the control case.

    Function space: N1curl (Nédélec first kind, degree 1) — required for H(curl).
    Do NOT use CG vector elements for A (produces spurious solutions).

    Solver: MUMPS direct solve with Metis ordering.
    ─────────────────────────────────────────────────────────────────────────
    HYPRE AMS (the theoretically correct iterative solver for H(curl)) was
    investigated but does NOT converge on our highly non-uniform coil meshes
    (100:1 element-size ratio between wire and domain).  Root cause: the
    dolfinx BC application replaces Dirichlet rows with identity rows
    (diag=1), creating a diagonal contrast of ~10¹⁰ vs. interior entries
    that breaks the AMS hierarchy.  Scaling the BC rows to match interior
    entries (zeroRowsColumns with diag=interior_mean) reduces the contrast
    to ~120× but AMS still stagnates at ~1.7% relative residual.

    MUMPS is fast for the mesh sizes used interactively:
      coarse  2 k DOFs →  0.03 s
      medium 28 k DOFs →  1.67 s
      fine  ~300 k DOFs → ~15–20 s  (geometry cached; only fresh on first solve)

    AMS could be revisited if petsc4py exposes PCHYPRESetAMSCoordinateVectors
    (geometric node coordinates, not edge constant vectors), which avoids the
    null-space stagnation observed with edge vectors on non-uniform meshes.
    """
    V = fem.functionspace(mesh, ("N1curl", 1))

    A = TrialFunction(V)
    v = TestFunction(V)

    mu0_inv = Constant(mesh, dolfinx.default_scalar_type(1.0 / MU0))

    J_func = _build_current_density(mesh, cell_tags, coil_params, wire_tag, space_dim=3)

    a = mu0_inv * inner(curl(A), curl(v)) * dx
    L = inner(J_func, v) * dx

    # Tangential BC: A × n̂ = 0 on ∂Ω
    # NOTE: dirichletbc with a numpy constant fails for N1curl because the
    # block size per DOF is not 3. Use a zero Function instead.
    bc_dofs = _get_boundary_dofs(V, facet_tags, boundary_tag)
    zero_A = Function(V)
    bc = dirichletbc(zero_A, bc_dofs)

    # MUMPS direct solve with Metis reordering.
    # ICNTL(7)=5 → Metis nested dissection (better fill-in than default AMD for 3D).
    # ICNTL(14)=50 → 50% extra working memory to avoid OOC on fine meshes.
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="oracle_A_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_7": 5,
            "mat_mumps_icntl_14": 50,
        },
    )

    return problem, V, J_func


# ---------------------------------------------------------------------------
# eed_coupled
# ---------------------------------------------------------------------------

def _build_eed_coupled(mesh, cell_tags, facet_tags, coil_params, eed_params, wire_tag, boundary_tag):
    """
    Full EED coupled (φ, A) system.

    Weak form (block 2×2 saddle-point):
      ∫ ∇φ·∇ψ dx + α² ∫ φψ dx + β ∫ div(A)·ψ dx = ∫ S_φ ψ dx
      (1/μ₀) ∫ curl(A)·curl(v) dx + γ ∫ ∇φ·v dx  = ∫ J·v dx

    Mixed function space: W = CG1 × N1curl

    TODO: VERIFY AGAINST DDOF PAPER — coupling term structure (β div(A) in φ equation,
    γ ∇φ in A equation) is per FIELD_THEORY.md. Confirm sign conventions and
    coefficient normalization against the source paper once citation is provided.
    """
    # Mixed function space: (CG1 for φ) × (N1curl for A)
    el_phi = bufl.element("CG", mesh.basix_cell(), 1)
    el_A   = bufl.element("N1curl", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, bufl.mixed_element([el_phi, el_A]))

    (phi, A) = split(TrialFunction(W))
    (psi, v) = split(TestFunction(W))

    alpha2 = Constant(mesh, dolfinx.default_scalar_type(eed_params.alpha ** 2))
    beta_ = Constant(mesh, dolfinx.default_scalar_type(eed_params.beta))
    gamma_ = Constant(mesh, dolfinx.default_scalar_type(eed_params.gamma))
    mu0_inv = Constant(mesh, dolfinx.default_scalar_type(1.0 / MU0))
    inv_mu0_eps0 = Constant(mesh, dolfinx.default_scalar_type(1.0 / (MU0 * EPS0)))

    J_func = _build_current_density(mesh, cell_tags, coil_params, wire_tag, space_dim=3)

    # φ equation
    a_phi = (
        inner(grad(phi), grad(psi)) * dx
        + alpha2 * inner(phi, psi) * dx
        + beta_ * inner(div(A), psi) * dx
    )
    L_phi = inv_mu0_eps0 * inner(J_func, grad(psi)) * dx

    # A equation
    a_A = (
        mu0_inv * inner(curl(A), curl(v)) * dx
        + gamma_ * inner(grad(phi), v) * dx
    )
    L_A = inner(J_func, v) * dx

    a = a_phi + a_A
    L = L_phi + L_A

    # BCs: φ=0 and A×n̂=0 on ∂Ω
    W0, W1 = W.sub(0), W.sub(1)
    bc_phi_dofs = _get_boundary_dofs(W0.collapse()[0], facet_tags, boundary_tag)
    # Collapsed subspace needed for BC application
    V0, dofs0 = W.sub(0).collapse()
    V1, dofs1 = W.sub(1).collapse()

    bc_phi_dofs_collapsed = locate_dofs_topological(
        (W.sub(0), V0), mesh.topology.dim - 1, _get_boundary_facets(facet_tags, boundary_tag)
    )
    bc_A_dofs_collapsed = locate_dofs_topological(
        (W.sub(1), V1), mesh.topology.dim - 1, _get_boundary_facets(facet_tags, boundary_tag)
    )

    zero_scalar = Function(V0)
    zero_scalar.x.array[:] = 0.0
    zero_vector = Function(V1)
    zero_vector.x.array[:] = 0.0

    bcs = [
        dirichletbc(zero_scalar, bc_phi_dofs_collapsed, W.sub(0)),
        dirichletbc(zero_vector, bc_A_dofs_collapsed, W.sub(1)),
    ]

    # Mixed saddle-point system: MUMPS with Metis ordering.
    # Block preconditioners for (φ,A) mixed systems are complex; MUMPS
    # with Metis reordering gives reasonable fill-in reduction.
    # ICNTL(7)=5 → Metis nested dissection (better than default AMD for 3D).
    # ICNTL(14)=50 → 50% extra working memory to avoid OOC on fine meshes.
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="oracle_eed_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_7": 5,
            "mat_mumps_icntl_14": 50,
        },
    )

    return problem, W, J_func


# ---------------------------------------------------------------------------
# Current density construction
# ---------------------------------------------------------------------------

def _build_current_density(mesh, cell_tags, coil_params: CoilParams, wire_tag: int, space_dim: int = 3):
    """
    Build the current density vector J on the mesh.

    J = (I / wire_cross_section_area) * t̂   inside the coil wire
    J = 0                                     elsewhere

    t̂ is approximated from the coil geometry:
    - solenoid/flat_spiral: azimuthal direction at each point
    - toroid variants: depends on winding direction
    - rodin: alternating azimuthal with z-component

    Uses a DG0 (discontinuous piecewise constant) function space for J,
    which is correct since J is constant within each wire element and zero
    in air elements.
    """
    coil_type = coil_params.coil_type
    wire_area = np.pi * coil_params.wire_radius_m ** 2
    I = coil_params.current_A
    J_magnitude = I / wire_area

    # Scalar DG0 scratch spaces for per-component assignment
    V_J3 = fem.functionspace(mesh, ("DG", 0))
    Jx = Function(V_J3)
    Jy = Function(V_J3)
    Jz = Function(V_J3)

    # Get cell indices for wire subdomain
    wire_cells = cell_tags.find(wire_tag)

    if len(wire_cells) == 0:
        # No wire cells tagged — return zero J (will produce zero field)
        J_func = dolfinx.fem.Function(
            fem.functionspace(mesh, bufl.element("DG", mesh.basix_cell(), 0, shape=(3,)))
        )
        return J_func

    # Evaluate cell midpoints to determine current direction
    midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, wire_cells)

    # Current direction: depends on coil type
    jx_vals, jy_vals, jz_vals = _compute_current_direction(
        midpoints, coil_type, J_magnitude
    )

    # Assign to DG0 cells
    # dolfinx DG0: one DOF per cell, indexed by cell order
    for i, cell_idx in enumerate(wire_cells):
        Jx.x.array[cell_idx] = jx_vals[i]
        Jy.x.array[cell_idx] = jy_vals[i]
        Jz.x.array[cell_idx] = jz_vals[i]

    # Pack into a vector-valued Function via UFL
    V_vec = fem.functionspace(mesh, bufl.element("DG", mesh.basix_cell(), 0, shape=(3,)))
    J_vec = Function(V_vec)
    # Interleave x,y,z components
    J_vec.x.array[0::3] = Jx.x.array
    J_vec.x.array[1::3] = Jy.x.array
    J_vec.x.array[2::3] = Jz.x.array

    return J_vec


def _compute_current_direction(
    midpoints: np.ndarray,
    coil_type: str,
    J_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute current direction unit vector at each midpoint based on coil type.
    Returns (Jx, Jy, Jz) arrays of shape (N,).
    """
    x, y, z = midpoints[:, 0], midpoints[:, 1], midpoints[:, 2]
    r_xy = np.sqrt(x**2 + y**2) + 1e-30  # avoid division by zero

    if coil_type in ("solenoid", "flat_spiral"):
        # Azimuthal direction: t̂ = (-y/r, x/r, 0)
        tx = -y / r_xy
        ty = x / r_xy
        tz = np.zeros_like(x)

    elif coil_type == "toroid":
        # Azimuthal winding: current flows azimuthally around the minor circle
        # t̂ = (-sin(φ_minor), cos(φ_minor), 0) approximated as azimuthal
        tx = -y / r_xy
        ty = x / r_xy
        tz = np.zeros_like(x)

    elif coil_type == "toroid_poloidal":
        # Poloidal winding: current flows in the poloidal direction
        # t̂_poloidal = (-z/|r_minor| * r̂ + r_minor/|r_minor| * ẑ)
        # where r̂ = (x/r_xy, y/r_xy, 0)
        # Approximate: poloidal = ẑ × azimuthal × ẑ
        # t̂ = (-z*x/r_xy/r_total, -z*y/r_xy/r_total, r_xy/r_total)
        r_total = np.sqrt(r_xy**2 + z**2) + 1e-30
        tx = -z * x / (r_xy * r_total)
        ty = -z * y / (r_xy * r_total)
        tz = r_xy / r_total

    elif coil_type == "rodin":
        # Rodin: figure-8, alternating azimuthal + z component
        # Primary azimuthal + z-component that alternates sign with azimuthal angle
        phi_angle = np.arctan2(y, x)
        sign = np.sign(np.cos(2 * phi_angle))  # alternates sign twice per revolution
        az_x = -y / r_xy
        az_y = x / r_xy
        # Mix azimuthal and z with the alternating sign
        tx = az_x * np.cos(np.pi / 4) + sign * np.sin(np.pi / 4) * 0.0
        ty = az_y * np.cos(np.pi / 4)
        tz = sign * np.sin(np.pi / 4) * np.ones_like(x)
        # Renormalize
        norm = np.sqrt(tx**2 + ty**2 + tz**2) + 1e-30
        tx /= norm
        ty /= norm
        tz /= norm

    else:
        # Fallback: azimuthal
        tx = -y / r_xy
        ty = x / r_xy
        tz = np.zeros_like(x)

    return J_magnitude * tx, J_magnitude * ty, J_magnitude * tz


# ---------------------------------------------------------------------------
# BC helpers
# ---------------------------------------------------------------------------

def _get_boundary_facets(facet_tags, boundary_tag: int) -> np.ndarray:
    return facet_tags.find(boundary_tag)


def _get_boundary_dofs(V, facet_tags, boundary_tag: int) -> np.ndarray:
    boundary_facets = _get_boundary_facets(facet_tags, boundary_tag)
    return locate_dofs_topological(V, V.mesh.topology.dim - 1, boundary_facets)
