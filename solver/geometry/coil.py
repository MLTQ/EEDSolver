"""
Parametric coil geometry builder using Gmsh.
Produces a .msh file with tagged physical groups for the solver.

Supported coil types:
  solenoid         — helical winding (approximated as stacked current loops)
  toroid           — azimuthal toroid winding (standard torus)
  toroid_poloidal  — poloidal toroid winding (key EED discriminator: confines B, not φ)
  flat_spiral      — Archimedean flat spiral
  rodin            — Rodin/Marko coil (figure-8 toroid winding)

Physical groups tagged in output mesh:
  "coil_wire"      — wire volume (current source region)
  "air_domain"     — surrounding air/vacuum
  "boundary_sphere"— outer boundary (ABC/Dirichlet BCs applied here)
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import gmsh

from solver.models.params import CoilParams


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_coil_geometry(params: CoilParams, domain_radius_m: float) -> Path:
    """
    Build the Gmsh geometry for the requested coil type and domain.
    Returns path to the generated .msh file (in a temp directory).
    Caller is responsible for cleanup if needed.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("oracle_coil")

    try:
        _dispatch_coil(params, domain_radius_m)
        msh_path = _finalize_and_mesh(params)
    except Exception:
        gmsh.finalize()
        raise

    gmsh.finalize()
    return msh_path


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch_coil(params: CoilParams, domain_radius_m: float) -> None:
    builder = {
        "solenoid": _build_solenoid,
        "toroid": _build_toroid,
        "toroid_poloidal": _build_toroid_poloidal,
        "flat_spiral": _build_flat_spiral,
        "rodin": _build_rodin,
    }[params.coil_type]
    builder(params, domain_radius_m)


# ---------------------------------------------------------------------------
# Solenoid
# ---------------------------------------------------------------------------

def _build_solenoid(params: CoilParams, domain_radius_m: float) -> None:
    """
    Approximate helical solenoid as N stacked current loops.
    True helix is topologically complex for volume meshing; stacked loops
    are standard FEM practice and produce correct field topology.

    The coil axis is aligned with z. Loops are spaced by pitch_m.
    Total coil height = turns * pitch_m.
    """
    r = params.radius_m
    rw = params.wire_radius_m
    n = params.turns
    pitch = params.pitch_m
    total_height = n * pitch
    z0 = -total_height / 2.0

    wire_tags = []
    for i in range(n):
        z_center = z0 + (i + 0.5) * pitch
        tag = _add_torus(r, rw, z_center, axis="z")
        wire_tags.append(tag)

    _add_domain_sphere(domain_radius_m)
    _tag_physical_groups(wire_tags)


# ---------------------------------------------------------------------------
# Toroid (azimuthal winding — standard)
# ---------------------------------------------------------------------------

def _build_toroid(params: CoilParams, domain_radius_m: float) -> None:
    """
    Standard toroidal winding: wire loops wound azimuthally around a torus.
    N loops evenly spaced in angle around the torus axis (z-axis).
    Each loop is a small torus (the wire cross-section swept around a circle
    of radius r centered at radius params.radius_m from origin).

    This confines B inside the torus but, per EED, should NOT confine φ.
    """
    R = params.radius_m       # Major radius (torus center to loop center)
    r = params.wire_radius_m  # Wire cross-section radius
    n = params.turns

    # Each winding turn is a small tube (minor torus) swept along an arc
    wire_tags = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        cx = R * math.cos(angle)
        cy = R * math.sin(angle)
        # Small torus representing one turn of wire around the minor circle
        tag = _add_torus(
            major_r=params.pitch_m * n / (2 * math.pi),  # minor torus radius
            minor_r=r,
            z_center=0.0,
            axis="z",
            center=(cx, cy, 0.0),
        )
        wire_tags.append(tag)

    _add_domain_sphere(domain_radius_m)
    _tag_physical_groups(wire_tags)


# ---------------------------------------------------------------------------
# Toroid (poloidal winding)
# ---------------------------------------------------------------------------

def _build_toroid_poloidal(params: CoilParams, domain_radius_m: float) -> None:
    """
    Poloidal toroid winding: wire wound poloidally (the short way around the torus).
    N turns spaced along the azimuthal angle.

    In standard EM, a poloidal winding produces a toroidal magnetic field
    entirely within the torus (no external B). In EED, φ should still leak out
    because ∇·J is nonzero at wire endpoints/transitions.

    Approximated as N tilted loops, each in a radial plane of the torus.
    """
    R = params.radius_m
    rw = params.wire_radius_m
    n = params.turns
    # Poloidal loop radius = minor radius of the torus
    r_minor = params.pitch_m * n / (2 * math.pi)

    wire_tags = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        # Loop center is on the torus at this azimuthal angle
        cx = R * math.cos(angle)
        cy = R * math.sin(angle)
        # This loop lives in the radial plane at this angle
        tag = _add_torus(
            major_r=r_minor,
            minor_r=rw,
            z_center=0.0,
            axis="radial",
            center=(cx, cy, 0.0),
            radial_angle=angle,
        )
        wire_tags.append(tag)

    _add_domain_sphere(domain_radius_m)
    _tag_physical_groups(wire_tags)


# ---------------------------------------------------------------------------
# Flat spiral
# ---------------------------------------------------------------------------

def _build_flat_spiral(params: CoilParams, domain_radius_m: float) -> None:
    """
    Archimedean flat spiral coil in the z=0 plane.
    Approximated as N concentric rings with radii spaced by pitch_m.
    """
    rw = params.wire_radius_m
    n = params.turns
    r0 = params.radius_m - (n - 1) * params.pitch_m / 2.0

    wire_tags = []
    for i in range(n):
        r_loop = r0 + i * params.pitch_m
        if r_loop <= rw:
            continue
        tag = _add_torus(r_loop, rw, z_center=0.0, axis="z")
        wire_tags.append(tag)

    _add_domain_sphere(domain_radius_m)
    _tag_physical_groups(wire_tags)


# ---------------------------------------------------------------------------
# Rodin coil
# ---------------------------------------------------------------------------

def _build_rodin(params: CoilParams, domain_radius_m: float) -> None:
    """
    Rodin/Marko coil: a figure-8 winding pattern on a torus.
    The wire crosses the torus axis on each revolution, producing a winding
    that alternates above/below the torus midplane.

    Approximated as N loops with alternating ±tilt angle around the torus.
    Full Rodin geometry is complex; this captures the essential field topology.

    For exact Rodin geometry (36-point star pattern), implement as a future
    improvement using Gmsh spline curves for the wire path.
    """
    R = params.radius_m
    rw = params.wire_radius_m
    n = params.turns
    r_minor = max(params.pitch_m * n / (2 * math.pi), rw * 3)
    tilt = math.pi / 4  # 45-degree figure-8 tilt

    wire_tags = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        sign = 1 if i % 2 == 0 else -1
        cx = R * math.cos(angle)
        cy = R * math.sin(angle)
        # Tilted loop — alternating tilt creates figure-8 crossing pattern
        tag = _add_torus(
            major_r=r_minor,
            minor_r=rw,
            z_center=sign * r_minor * math.sin(tilt) * 0.5,
            axis="radial",
            center=(cx, cy, 0.0),
            radial_angle=angle,
            extra_tilt=sign * tilt,
        )
        wire_tags.append(tag)

    _add_domain_sphere(domain_radius_m)
    _tag_physical_groups(wire_tags)


# ---------------------------------------------------------------------------
# Gmsh primitives
# ---------------------------------------------------------------------------

def _add_torus(
    major_r: float,
    minor_r: float,
    z_center: float,
    axis: str = "z",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radial_angle: float = 0.0,
    extra_tilt: float = 0.0,
) -> int:
    """
    Add a torus (tube) to the Gmsh model. Returns the volume tag.
    axis="z"      — torus axis aligned with z
    axis="radial" — torus axis in the radial direction at radial_angle
    """
    cx, cy, _ = center
    cz = z_center

    if axis == "z":
        tag = gmsh.model.occ.addTorus(cx, cy, cz, major_r, minor_r)
    else:
        # Radial axis torus — create z-axis torus then rotate
        tag = gmsh.model.occ.addTorus(cx, cy, cz, major_r, minor_r)
        # Rotate around the azimuthal tangent direction at this angle
        ax = -math.sin(radial_angle)
        ay = math.cos(radial_angle)
        gmsh.model.occ.rotate(
            [(3, tag)], cx, cy, cz, ax, ay, 0.0, math.pi / 2 + extra_tilt
        )

    return tag


def _add_domain_sphere(radius: float) -> int:
    """Add the bounding sphere (air domain). Returns the volume tag."""
    return gmsh.model.occ.addSphere(0, 0, 0, radius)


def _tag_physical_groups(wire_tags: list[int]) -> None:
    """
    Fragment overlapping volumes (boolean fuse for wire, cut from domain),
    tag physical groups for solver.
    """
    gmsh.model.occ.synchronize()

    # Get all volumes
    all_vols = gmsh.model.getEntities(3)
    vol_tags = [v[1] for v in all_vols]

    # The last added volume is the domain sphere
    domain_tag = vol_tags[-1]
    wire_vol_tags = [t for t in vol_tags if t != domain_tag]

    if not wire_vol_tags:
        raise RuntimeError("No wire volumes found after geometry build")

    # Fragment: cuts wire out of domain, keeping both
    wire_dimtags = [(3, t) for t in wire_vol_tags]
    domain_dimtag = [(3, domain_tag)]

    result, _ = gmsh.model.occ.fragment(domain_dimtag, wire_dimtags)
    gmsh.model.occ.synchronize()

    # Re-query volumes after fragment
    all_vols_after = gmsh.model.getEntities(3)
    final_vol_tags = [v[1] for v in all_vols_after]

    # The fragment result: first entry is the domain remainder, rest are wire fragments
    # We use bounding box heuristic: wire vols are small
    domain_vols = []
    wire_vols = []
    for tag in final_vol_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, tag)
        vol_extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if vol_extent > 0.9 * 2 * gmsh.model.getBoundingBox(3, final_vol_tags[0])[3]:
            domain_vols.append(tag)
        else:
            wire_vols.append(tag)

    # Fallback: first fragment result is the domain
    if not domain_vols:
        domain_vols = [result[0][1]] if result else [final_vol_tags[0]]
        wire_vols = [t for t in final_vol_tags if t not in domain_vols]

    # Physical groups
    if wire_vols:
        gmsh.model.addPhysicalGroup(3, wire_vols, name="coil_wire")
    if domain_vols:
        gmsh.model.addPhysicalGroup(3, domain_vols, name="air_domain")

    # Outer boundary surfaces
    outer_surfs = _get_outer_boundary_surfaces()
    if outer_surfs:
        gmsh.model.addPhysicalGroup(2, outer_surfs, name="boundary_sphere")


def _get_outer_boundary_surfaces() -> list[int]:
    """Find the outermost boundary surfaces (those belonging to only 1 volume)."""
    surfaces = gmsh.model.getEntities(2)
    outer = []
    for _, stag in surfaces:
        up, _ = gmsh.model.getAdjacencies(2, stag)
        if len(up) == 1:  # Only one adjacent volume → boundary
            outer.append(stag)
    return outer


def _finalize_and_mesh(params: CoilParams) -> Path:
    """Set mesh options and generate. Returns path to .msh file."""
    gmsh.model.occ.synchronize()

    # Characteristic length: scale with coil size
    lc_coil = params.wire_radius_m * 2.0
    lc_domain = params.radius_m * 0.5

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_coil * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_domain)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    tmp = tempfile.mkdtemp(prefix="oracle_mesh_")
    msh_path = Path(tmp) / "coil.msh"
    gmsh.write(str(msh_path))
    return msh_path
