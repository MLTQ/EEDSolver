"""
Post-processing: extract 2D slices and field maxima from FEniCSx solve output.
Converts dolfinx Functions → flat float arrays for JSON transport.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
import dolfinx
import dolfinx.fem as fem
import dolfinx.geometry as geom
from ufl import sqrt, inner

from solver.models.params import (
    FieldName,
    FieldMaximum,
    MeshResolution,
    SliceData,
    SliceRequest,
    SolveResult,
    VolumeData,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VOLUME_RESOLUTION: dict[MeshResolution, int] = {
    "coarse": 32,
    "medium": 48,
    "fine": 64,
}

# Primary field to extract for 3D volume per formulation
VOLUME_FIELD_FOR_FORMULATION = {
    "scalar_only": "phi",
    "maxwell_only": "B_magnitude",
    "eed_coupled": "phi",
}

# Which fields each formulation actually computes (used for volume_field fallback)
FIELDS_BY_FORMULATION: dict[str, set[str]] = {
    "scalar_only": {"phi", "J_magnitude"},
    "maxwell_only": {"A_magnitude", "B_magnitude", "J_magnitude"},
    "eed_coupled": {"phi", "A_magnitude", "B_magnitude", "J_magnitude"},
}


def extract_results(
    solve_output,   # SolveOutput from solver.py
    slice_requests: list[SliceRequest],
    request_volume: bool = True,
    volume_field: FieldName | None = None,
) -> SolveResult:
    """
    Extract all requested slices + field maxima from solve output.
    Returns a fully populated SolveResult ready for JSON serialization.

    volume_field: which field to render in the 3D viewer. Falls back to the
    formulation's primary field if the requested field is not available.
    """
    from solver.fields.solver import SolveOutput  # avoid circular at module level
    assert isinstance(solve_output, SolveOutput)

    t_post = time.perf_counter()

    slices = []
    for req in slice_requests:
        try:
            s = extract_slice(solve_output, req)
            slices.append(s)
        except Exception as exc:
            log.warning(f"Slice extraction failed for {req}: {exc}")
            slices.append(_zero_slice(req))

    maxima = extract_maxima(solve_output)
    log.debug(f"Maxima extracted in {time.perf_counter() - t_post:.3f}s")

    volume = None
    if request_volume:
        # Honour the caller's field request; fall back to the formulation's
        # primary field if that field is not computed by this formulation.
        primary = VOLUME_FIELD_FOR_FORMULATION.get(solve_output.formulation, "phi")
        available = FIELDS_BY_FORMULATION.get(solve_output.formulation, set())
        if volume_field and volume_field in available:
            vol_field = volume_field
        else:
            if volume_field and volume_field != primary:
                log.info(
                    f"Requested volume field '{volume_field}' not available for "
                    f"'{solve_output.formulation}' (available: {sorted(available)}); "
                    f"using '{primary}' instead."
                )
            vol_field = primary
        resolution = VOLUME_RESOLUTION.get(
            solve_output.mesh_stats.get("resolution", "coarse"), 32
        )
        try:
            volume = extract_volume(solve_output, vol_field, resolution)
        except Exception as exc:
            log.warning(f"Volume extraction failed: {exc}")

    return SolveResult(
        solve_time_s=solve_output.solve_time_s,
        mesh_nodes=solve_output.mesh_stats["num_nodes"],
        slices=slices,
        volume=volume,
        maxima=maxima,
        warnings=solve_output.warnings,
    )


def extract_slice(solve_output, request: SliceRequest) -> SliceData:
    """
    Sample the requested field on a regular grid in the requested plane.

    The slice plane is defined by:
      axis: the normal axis ("x", "y", or "z")
      position: normalized 0–1 along that axis

    Returns a SliceData with the field values as a flat float list.
    """
    mesh = solve_output.mesh

    # Get bounding box of mesh
    coords = mesh.geometry.x
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    zmin, zmax = coords[:, 2].min(), coords[:, 2].max()

    bounds = {"x": (xmin, xmax), "y": (ymin, ymax), "z": (zmin, zmax)}
    axis = request.axis
    pos_phys = bounds[axis][0] + request.position * (bounds[axis][1] - bounds[axis][0])
    n = request.resolution

    # Build grid in the 2 non-axis directions
    ax_idx = {"x": 0, "y": 1, "z": 2}[axis]
    others = [i for i in range(3) if i != ax_idx]
    bound_u = bounds[{0: "x", 1: "y", 2: "z"}[others[0]]]
    bound_v = bounds[{0: "x", 1: "y", 2: "z"}[others[1]]]

    u_vals = np.linspace(bound_u[0], bound_u[1], n)
    v_vals = np.linspace(bound_v[0], bound_v[1], n)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")  # shape (n, n)

    # Build query points array [n*n, 3]
    pts = np.zeros((n * n, 3))
    pts[:, ax_idx] = pos_phys
    pts[:, others[0]] = uu.ravel()
    pts[:, others[1]] = vv.ravel()

    # Get the field function
    field_func = _get_field_function(solve_output, request.field)
    if field_func is None:
        log.warning(f"Field {request.field} not available for formulation {solve_output.formulation}")
        values = np.zeros(n * n)
    else:
        bb_tree = geom.bb_tree(mesh, mesh.topology.dim)
        values = _sample_field_at_points(field_func, mesh, pts, request.field, bb_tree)

    grid = values.reshape(n, n)

    return SliceData(
        axis=axis,
        position=request.position,
        field=request.field,
        shape=[n, n],
        data=grid.ravel().tolist(),
        x_range=[float(bound_u[0]), float(bound_u[1])],
        y_range=[float(bound_v[0]), float(bound_v[1])],
        field_min=float(np.nanmin(values)),
        field_max=float(np.nanmax(values)),
    )


def extract_volume(
    solve_output,
    field_name: FieldName,
    resolution: int,
) -> VolumeData:
    """
    Sample a field on a regular 3D grid and return normalized volume data
    for the Three.js ray-marching viewer.

    Data is normalized to [0, 1] before return.
    Resolution: 32 (coarse) / 48 (medium) / 64 (fine).
    """
    mesh = solve_output.mesh
    coords = mesh.geometry.x
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    zmin, zmax = coords[:, 2].min(), coords[:, 2].max()

    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    zs = np.linspace(zmin, zmax, resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")  # shape (n, n, n)
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    field_func = _get_field_function(solve_output, field_name)
    if field_func is None:
        values = np.zeros(resolution ** 3)
    else:
        bb_tree = geom.bb_tree(mesh, mesh.topology.dim)   # build once, reuse below
        values = _sample_field_at_points(field_func, mesh, pts, field_name, bb_tree)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    fmin = float(values.min())
    fmax = float(values.max())

    # Normalize absolute value to [0, 1].
    # Using abs() means the 3-D viewer always shows *field strength*, not signed
    # scalar — critical for EED φ which is naturally bipolar around the coil.
    # fmin/fmax (signed) are still reported for the info overlay.
    values_abs = np.abs(values)

    # Clip at the 99th percentile of non-zero values before normalising.
    #
    # Why: near the coil wire the FEM field can be geometrically singular
    # (especially on coarse meshes). A single sample voxel at the wire
    # surface can be 100-1000× stronger than the interior field, compressing
    # everything else to < 1 % of the display range → invisible.
    # Clipping the top 1 % of non-zero values restores the interior structure.
    pos_vals = values_abs[values_abs > 0]
    if len(pos_vals) > 10:
        clip_val = float(np.percentile(pos_vals, 99))
        values_abs = np.clip(values_abs, 0.0, clip_val)

    abs_max = float(values_abs.max())
    if abs_max > 0:
        normalized = values_abs / abs_max
    else:
        normalized = np.zeros_like(values_abs)

    # Reorder from Python meshgrid layout (x-major, indexing="ij") to WebGL
    # Data3DTexture layout (z-major / depth-first).
    #
    # Python produces flat index:  ix*ny*nz + iy*nz + iz  → field at (xs[ix], ys[iy], zs[iz])
    # WebGL expects flat index:    iz*nx*ny + iy*nx + ix   → texel at (u=ix/nx, v=iy/ny, w=iz/nz)
    #
    # Reshape (nx,ny,nz), transpose to (nz,ny,nx), then flatten C-order.
    n = resolution
    normalized = normalized.reshape(n, n, n).transpose(2, 1, 0).ravel()

    log.info(
        f"Volume extracted: {resolution}³ = {len(values)} pts, "
        f"{field_name} range [{fmin:.3e}, {fmax:.3e}]"
    )

    return VolumeData(
        field=field_name,
        shape=[resolution, resolution, resolution],
        data=normalized.tolist(),
        x_range=[float(xmin), float(xmax)],
        y_range=[float(ymin), float(ymax)],
        z_range=[float(zmin), float(zmax)],
        field_min=fmin,
        field_max=fmax,
    )


def extract_maxima(solve_output) -> list[FieldMaximum]:
    """
    Find the global maximum for each available field directly from DoF arrays.

    CG1 and DG0 functions store their values at nodes/cell-centres respectively,
    so np.max(abs(func.x.array)) gives the exact FEM maximum with O(n_dofs)
    cost — no point sampling or BoundingBoxTree needed.  This replaces the old
    32³-grid approach (163 K BoundingBoxTree look-ups per solve → ~5 s) with a
    sub-millisecond array reduction.
    """
    maxima = []

    for field_name in ("phi", "A_magnitude", "B_magnitude", "J_magnitude"):
        func = _get_field_function(solve_output, field_name)
        if func is None:
            continue
        try:
            arr = func.x.array
            if not len(arr):
                continue
            abs_arr = np.abs(arr)
            valid = np.isfinite(abs_arr)
            if not valid.any():
                continue

            idx = int(np.argmax(np.where(valid, abs_arr, -np.inf)))
            max_val = float(abs_arr[idx])

            # Spatial location: CG1 DoFs are at mesh nodes
            try:
                coords = func.function_space.tabulate_dof_coordinates()
                max_loc = coords[idx % len(coords)].tolist()
            except Exception:
                max_loc = [0.0, 0.0, 0.0]

            maxima.append(FieldMaximum(
                field=field_name,
                max_value=max_val,
                max_location=max_loc,
            ))
        except Exception as exc:
            log.warning(f"Maxima extraction failed for {field_name}: {exc}")

    return maxima


# ---------------------------------------------------------------------------
# Field function resolution
# ---------------------------------------------------------------------------

def _get_field_function(solve_output, field_name: FieldName):
    """
    Return the dolfinx Function (or derived expression) for the requested field.
    Returns None if the field is not available for the current formulation.

    Magnitude functions (A_magnitude, B_magnitude, J_magnitude) are cached on
    the solve_output object after the first call so that extract_maxima and
    extract_volume don't each trigger a separate L2 projection solve.
    """
    if field_name == "phi":
        return solve_output.phi  # None for maxwell_only

    # For vector-magnitude fields: compute once, cache on solve_output
    cache_attr = f"_cached_{field_name}"
    if hasattr(solve_output, cache_attr):
        return getattr(solve_output, cache_attr)

    result = None
    if field_name == "A_magnitude":
        if solve_output.A is not None:
            result = _make_magnitude_function(solve_output.A, solve_output.mesh)

    elif field_name == "B_magnitude":
        B = solve_output.B  # lazy property, None if A=None
        if B is not None:
            result = _make_magnitude_function(B, solve_output.mesh)

    elif field_name == "J_magnitude":
        if solve_output.J is not None:
            result = _make_magnitude_function(solve_output.J, solve_output.mesh)

    else:
        raise ValueError(f"Unknown field: {field_name}")

    # Cache (even if None — so we don't retry a field that's genuinely absent)
    setattr(solve_output, cache_attr, result)
    return result


def _make_magnitude_function(vec_func: dolfinx.fem.Function, mesh) -> dolfinx.fem.Function:
    """
    Create a scalar CG1 function holding |v| = sqrt(v·v) for a vector function v.

    Uses L2 projection (Galerkin weak form) instead of pointwise interpolation.
    Pointwise interpolation from N1curl (Nedelec) elements is noisy because
    Nedelec elements only enforce tangential continuity — the normal component
    is discontinuous at element boundaries.  The L2 projection:

        ∫ u·w dx = ∫ |v|·w dx   ∀ w ∈ CG1

    produces a globally-smooth field that correctly averages the inter-element
    discontinuities, giving a physically interpretable magnitude map.
    """
    import ufl as _ufl
    from dolfinx.fem.petsc import LinearProblem

    V_scalar = fem.functionspace(mesh, ("CG", 1))
    u = _ufl.TrialFunction(V_scalar)
    w = _ufl.TestFunction(V_scalar)

    mag_expr = sqrt(inner(vec_func, vec_func))
    a_form = _ufl.inner(u, w) * _ufl.dx
    L_form = _ufl.inner(mag_expr, w) * _ufl.dx

    problem = LinearProblem(
        a_form, L_form,
        bcs=[],
        petsc_options_prefix="oracle_mag_",
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-8},
    )
    return problem.solve()


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def _sample_field_at_points(
    field_func: dolfinx.fem.Function,
    mesh: dolfinx.mesh.Mesh,
    points: np.ndarray,
    field_name: str,
    bb_tree=None,
) -> np.ndarray:
    """
    Sample a scalar dolfinx Function at an array of 3D points.
    Points outside the mesh return 0.0.

    Accepts a pre-built bb_tree so callers that evaluate multiple fields on the
    same grid (extract_volume, extract_slice) can build the tree once and reuse.

    The inner loop over colliding_cells.links(i) was the dominant cost for
    large grids (~5 s on 32³ with a coarse mesh).  This version replaces it
    with vectorised numpy operations on the AdjacencyList's flat .array /
    .offsets buffers — O(N) numpy instead of O(N) Python, ~100× faster.
    """
    if bb_tree is None:
        bb_tree = geom.bb_tree(mesh, mesh.topology.dim)

    cell_candidates = geom.compute_collisions_points(bb_tree, points)
    colliding_cells = geom.compute_colliding_cells(mesh, cell_candidates, points)

    # --- vectorised valid-point extraction -----------------------------------
    # AdjacencyList layout:
    #   offsets[i], offsets[i+1]  → slice in .array for point i
    #   array[offsets[i]]         → first colliding cell for point i
    offsets   = colliding_cells.offsets          # shape (N+1,)
    cell_arr  = colliding_cells.array            # flat, dtype int32

    n_links      = np.diff(offsets)              # number of colliding cells per point
    valid_mask   = n_links > 0                   # True for in-mesh points
    valid_indices = np.where(valid_mask)[0]

    values = np.zeros(len(points), dtype=np.float64)

    if len(valid_indices):
        first_cells = cell_arr[offsets[valid_indices]].astype(np.int32)
        valid_pts   = points[valid_indices]      # already float64

        eval_values = field_func.eval(valid_pts, first_cells)
        if eval_values.ndim == 2:
            eval_values = eval_values[:, 0]

        values[valid_indices] = eval_values

    return values


def _zero_slice(request: SliceRequest) -> SliceData:
    """Return a zero-filled SliceData for error recovery."""
    n = request.resolution
    return SliceData(
        axis=request.axis,
        position=request.position,
        field=request.field,
        shape=[n, n],
        data=[0.0] * (n * n),
        x_range=[-1.0, 1.0],
        y_range=[-1.0, 1.0],
        field_min=0.0,
        field_max=0.0,
    )
