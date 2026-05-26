"""
Post-processing: extract 2D slices and field maxima from FEniCSx solve output.
Converts dolfinx Functions → flat float arrays for JSON transport.
"""

from __future__ import annotations

import logging
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


def extract_results(
    solve_output,   # SolveOutput from solver.py
    slice_requests: list[SliceRequest],
    request_volume: bool = True,
) -> SolveResult:
    """
    Extract all requested slices + field maxima from solve output.
    Returns a fully populated SolveResult ready for JSON serialization.
    """
    from solver.fields.solver import SolveOutput  # avoid circular at module level
    assert isinstance(solve_output, SolveOutput)

    slices = []
    for req in slice_requests:
        try:
            s = extract_slice(solve_output, req)
            slices.append(s)
        except Exception as exc:
            log.warning(f"Slice extraction failed for {req}: {exc}")
            slices.append(_zero_slice(req))

    maxima = extract_maxima(solve_output)

    volume = None
    if request_volume:
        vol_field = VOLUME_FIELD_FOR_FORMULATION.get(solve_output.formulation, "phi")
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
        values = _sample_field_at_points(field_func, mesh, pts, request.field)

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
        values = _sample_field_at_points(field_func, mesh, pts, field_name)

    values = np.nan_to_num(values, nan=0.0)
    fmin = float(values.min())
    fmax = float(values.max())

    # Normalize to [0, 1]
    if fmax > fmin:
        normalized = (values - fmin) / (fmax - fmin)
    else:
        normalized = np.zeros_like(values)

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
    Find and return the global maximum for each available field.
    Samples on a coarse volume grid for speed.
    """
    mesh = solve_output.mesh
    coords = mesh.geometry.x
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    zmin, zmax = coords[:, 2].min(), coords[:, 2].max()

    n_sample = 32  # coarse grid for maxima search
    xs = np.linspace(xmin, xmax, n_sample)
    ys = np.linspace(ymin, ymax, n_sample)
    zs = np.linspace(zmin, zmax, n_sample)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    maxima = []
    for field_name in ("phi", "A_magnitude", "B_magnitude", "J_magnitude"):
        field_func = _get_field_function(solve_output, field_name)
        if field_func is None:
            continue
        try:
            values = _sample_field_at_points(field_func, mesh, pts, field_name)
            valid = np.isfinite(values)
            if not valid.any():
                continue
            idx = np.nanargmax(np.where(valid, values, -np.inf))
            loc = pts[idx].tolist()
            maxima.append(FieldMaximum(
                field=field_name,
                max_value=float(values[idx]),
                max_location=loc,
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
    """
    if field_name == "phi":
        return solve_output.phi  # None for maxwell_only

    elif field_name == "A_magnitude":
        if solve_output.A is None:
            return None
        # |A| = sqrt(A·A), scalar function
        return _make_magnitude_function(solve_output.A, solve_output.mesh)

    elif field_name == "B_magnitude":
        B = solve_output.B  # lazy property, None if A=None
        if B is None:
            return None
        return _make_magnitude_function(B, solve_output.mesh)

    elif field_name == "J_magnitude":
        if solve_output.J is None:
            return None
        return _make_magnitude_function(solve_output.J, solve_output.mesh)

    else:
        raise ValueError(f"Unknown field: {field_name}")


def _make_magnitude_function(vec_func: dolfinx.fem.Function, mesh) -> dolfinx.fem.Function:
    """
    Create a scalar CG1 function holding |v| = sqrt(v·v) for a vector function v.
    """
    V_scalar = fem.functionspace(mesh, ("CG", 1))
    mag_expr = fem.Expression(
        sqrt(inner(vec_func, vec_func)),
        V_scalar.element.interpolation_points,   # property in dolfinx 0.10, not a method
    )
    mag_func = fem.Function(V_scalar)
    mag_func.interpolate(mag_expr)
    return mag_func


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

def _sample_field_at_points(
    field_func: dolfinx.fem.Function,
    mesh: dolfinx.mesh.Mesh,
    points: np.ndarray,
    field_name: str,
) -> np.ndarray:
    """
    Sample a scalar dolfinx Function at an array of 3D points.
    Points outside the mesh get NaN (replaced with 0.0 in output).

    Uses dolfinx BoundingBoxTree for efficient point location.
    """
    bb_tree = geom.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = geom.compute_collisions_points(bb_tree, points)
    colliding_cells = geom.compute_colliding_cells(mesh, cell_candidates, points)

    values = np.full(len(points), 0.0, dtype=np.float64)

    # Collect points with valid cells
    valid_pts = []
    valid_cells = []
    valid_indices = []

    for i, pt in enumerate(points):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            valid_pts.append(pt)
            valid_cells.append(cells[0])
            valid_indices.append(i)

    if valid_pts:
        valid_pts_arr = np.array(valid_pts, dtype=np.float64)
        valid_cells_arr = np.array(valid_cells, dtype=np.int32)

        # Evaluate field at valid points
        eval_values = field_func.eval(valid_pts_arr, valid_cells_arr)

        # eval returns shape (N, value_size) — take first component for scalar
        if eval_values.ndim == 2:
            eval_values = eval_values[:, 0]

        for idx, val in zip(valid_indices, eval_values):
            values[idx] = float(val)

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
