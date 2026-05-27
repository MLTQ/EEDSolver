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

Wire geometry:
  For axis="z" coils (solenoid, flat_spiral), each loop is modelled as an
  annular cylinder (cylindrical ring with square cross-section) via
  addCylinder+cut rather than addTorus.  The OCC torus primitive has a
  parametric seam singularity that causes the Gmsh 2D mesher to produce
  degenerate "equivalent triangles" when minor_r << major_r (thin wire),
  which then crashes 3D mesh generation.  The cylinder Boolean approach
  avoids this entirely.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import gmsh

from solver.models.params import CoilParams

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_coil_geometry(
    params: CoilParams,
    domain_radius_m: float,
    mesh_resolution: str = "medium",
) -> Path:
    """
    Build the Gmsh geometry for the requested coil type and domain.
    Returns path to the generated .msh file (in a temp directory).
    Caller is responsible for cleanup if needed.
    """
    # Guard: if a previous solve crashed (SEGV / hard kill) without calling
    # gmsh.finalize(), Gmsh's internal singleton is already initialized.
    # Calling gmsh.initialize() a second time is a no-op in newer Gmsh but
    # was a hard crash in some builds.  Calling finalize + re-initialize is safe.
    if gmsh.isInitialized():
        log.warning("Gmsh already initialized (leftover from a prior crash) — finalizing first")
        gmsh.finalize()

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)   # 1=silent, 3=warnings+errors, 5=debug
    gmsh.model.add("oracle_coil")
    log.info(
        f"Building {params.coil_type} geometry: "
        f"r={params.radius_m}m, rw={params.wire_radius_m}m, "
        f"n={params.turns}, pitch={params.pitch_m}m, domain={domain_radius_m}m"
    )

    try:
        _dispatch_coil(params, domain_radius_m)
        msh_path = _finalize_and_mesh(params, domain_radius_m, mesh_resolution)
    except Exception:
        log.exception("EXCEPTION during geometry/mesh build")
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
    Add a wire ring to the Gmsh model. Returns the volume tag.
    axis="z"      — ring axis aligned with z
    axis="radial" — ring axis in the radial direction at radial_angle

    For axis="z": uses a SQUARE cross-section revolved 360° instead of
    gmsh.model.occ.addTorus.  The OCC addTorus creates a B-spline surface
    with a parametric "seam" singularity at φ=0.  When minor_r << major_r
    (e.g. 1mm wire inside a 50mm coil), the 2D surface mesher encounters
    degenerate "equivalent triangles" near the seam → "Invalid boundary mesh"
    → 3D meshing crash.

    The rectangular revolve creates 4 clean surfaces (top annulus, bottom
    annulus, inner cylinder, outer cylinder) with no seam topology, which the
    2D mesher handles correctly.
    """
    cx, cy, _ = center
    cz = z_center

    if axis == "z":
        # --- cylindrical ring (annular washer) via addCylinder + cut ------
        #
        # We model each wire loop as a cylindrical ring (square cross-section)
        # rather than a circular-cross-section torus.  This avoids two
        # well-known Gmsh/OCC failure modes for thin tori:
        #
        #  1. addTorus seam singularity → "equivalent triangles" in 2D mesher
        #     → "Invalid boundary mesh (overlapping facets)" → 3D SEGV.
        #
        #  2. Full-2π revolve of a planar rectangle → leftover seam-edge nodes
        #     that are NOT referenced by any tet element → dolfinx
        #     extract_geometry AssertionError (non-contiguous node indices).
        #
        # Cylinder Boolean cut has simple, well-tested OCC topology:
        #   outer cylinder radius = major_r + minor_r
        #   inner cylinder radius = major_r - minor_r
        #   height = 2 * minor_r   (centered at z_center)
        #   result: 4 clean surfaces (outer cyl, inner cyl, top annulus, bottom annulus)
        z_base = cz - minor_r
        height = 2.0 * minor_r
        r_out  = major_r + minor_r
        r_in   = major_r - minor_r

        outer = gmsh.model.occ.addCylinder(cx, cy, z_base, 0, 0, height, r_out)
        inner = gmsh.model.occ.addCylinder(cx, cy, z_base, 0, 0, height, r_in)

        # cut(objectDimTags, toolDimTags) → removes tool from object
        result, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)])
        if not result:
            raise RuntimeError(
                f"Boolean cut returned empty result "
                f"(major_r={major_r}, minor_r={minor_r}, z={cz})"
            )
        return result[0][1]

    else:
        # --- axis="radial": fall back to OCC addTorus + rotation ----------
        # Used by toroid, toroid_poloidal, rodin (where the torus is small
        # and rotated; the meshing issues are less severe for those shapes).
        tag = gmsh.model.occ.addTorus(cx, cy, cz, major_r, minor_r)
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
    Fragment overlapping volumes (boolean cut of wire from domain),
    then assign physical groups with EXPLICIT tags that match the constants
    in solver/fields/solver.py:
      coil_wire      = 1   (WIRE_TAG)
      air_domain     = 2   (AIR_TAG)
      boundary_sphere = 10  (BOUNDARY_TAG)

    Classification: the domain volume is the one with the largest bounding-box
    extent (the sphere); all smaller volumes are wire fragments.
    """
    gmsh.model.occ.synchronize()

    # Identify the domain sphere (last added) and wire tori (all others)
    all_vols = gmsh.model.getEntities(3)
    vol_tags = [v[1] for v in all_vols]

    domain_tag = vol_tags[-1]   # _add_domain_sphere is always called last
    wire_vol_tags = [t for t in vol_tags if t != domain_tag]

    if not wire_vol_tags:
        raise RuntimeError("No wire volumes found after geometry build")

    log.debug(f"Fragment: domain={domain_tag}, wires={wire_vol_tags}")

    # Fragment: cut wire volumes out of the domain sphere, keeping both
    wire_dimtags = [(3, t) for t in wire_vol_tags]
    gmsh.model.occ.fragment([(3, domain_tag)], wire_dimtags)
    gmsh.model.occ.synchronize()

    # Re-query all volumes after the boolean operation
    all_vols_after = gmsh.model.getEntities(3)
    final_vol_tags = [v[1] for v in all_vols_after]

    # Classify: domain = the largest bounding-box volume (the sphere remainder).
    # Wire = everything else.  Using > 50 % of the max extent as threshold.
    extents = {}
    for tag in final_vol_tags:
        bb = gmsh.model.getBoundingBox(3, tag)          # (xmin,ymin,zmin,xmax,ymax,zmax)
        extents[tag] = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])

    max_extent = max(extents.values())
    domain_vols = [t for t, e in extents.items() if e > 0.5 * max_extent]
    wire_vols   = [t for t, e in extents.items() if e <= 0.5 * max_extent]

    # Safety fallback: if nothing classified as wire, largest = domain, rest = wire
    if not wire_vols:
        domain_tag_new = max(extents, key=extents.get)
        domain_vols = [domain_tag_new]
        wire_vols   = [t for t in final_vol_tags if t != domain_tag_new]

    # Assign physical groups with EXPLICIT tags to match solver constants
    if wire_vols:
        gmsh.model.addPhysicalGroup(3, wire_vols,   tag=1,  name="coil_wire")
    if domain_vols:
        gmsh.model.addPhysicalGroup(3, domain_vols, tag=2,  name="air_domain")

    # Outer boundary surfaces
    outer_surfs = _get_outer_boundary_surfaces()
    if outer_surfs:
        gmsh.model.addPhysicalGroup(2, outer_surfs, tag=10, name="boundary_sphere")


def _get_outer_boundary_surfaces() -> list[int]:
    """Find the outermost boundary surfaces (those belonging to only 1 volume)."""
    surfaces = gmsh.model.getEntities(2)
    outer = []
    for _, stag in surfaces:
        up, _ = gmsh.model.getAdjacencies(2, stag)
        if len(up) == 1:  # Only one adjacent volume → boundary
            outer.append(stag)
    return outer


def _finalize_and_mesh(
    params: CoilParams,
    domain_radius_m: float,
    mesh_resolution: str = "medium",
) -> Path:
    """
    Set mesh characteristic lengths based on resolution preset and generate.
    Returns path to .msh file.

    Mesh sizing strategy
    --------------------
    Wire region (lc_coil):
      The wire cross-section needs ≥ 2 elements through its radius for J to be
      physically meaningful.  We scale lc_coil with the resolution preset.
      IMPORTANT: lc_min must stay ≥ wire_radius / 2 to prevent Gmsh from
      generating millions of elements and crashing (the domain-to-wire aspect
      ratio is already 200:1 for typical parameters).

    Domain region (lc_domain):
      Scales with the coil radius (not domain radius), which keeps the near-field
      resolution proportional to the coil geometry regardless of domain size.

    Resolution presets
    ------------------
      coarse  — fast preview   (~5 k–30 k tets,  < 5 s solve)
      medium  — good results   (~30 k–150 k tets, < 60 s solve)
      fine    — accurate       (~150 k+ tets,     minutes)
    """
    gmsh.model.occ.synchronize()

    # Wire characteristic length — controls resolution in/around the wire.
    #
    # CRITICAL: lc_coil must ensure ≥ 6 elements around the minor circumference of
    # the wire torus (2π·wire_radius / lc_coil ≥ 6).  With fewer segments the 2D
    # surface mesher produces degenerate ("equivalent") triangles → "Invalid boundary
    # mesh (overlapping facets)" → 3D mesh failure.
    #   coarse → lc_coil = wire_radius × 1.0  →  ~6 elems/minor circle
    #   medium → lc_coil = wire_radius × 0.5  → ~12 elems/minor circle
    #   fine   → lc_coil = wire_radius × 0.25 → ~25 elems/minor circle
    _WIRE_FACTORS  = {"coarse": 1.0, "medium": 0.5, "fine": 0.25}
    lc_coil = params.wire_radius_m * _WIRE_FACTORS.get(mesh_resolution, 0.5)

    # Domain characteristic length — controls resolution away from the wire.
    # Based on coil radius (the natural length scale of the near-field).
    _DOMAIN_FACTORS = {"coarse": 1.5, "medium": 0.5, "fine": 0.15}
    lc_domain = params.radius_m * _DOMAIN_FACTORS.get(mesh_resolution, 0.5)

    # lc_min is a hard lower bound.  Set to wire_radius/2 to prevent runaway
    # refinement in narrow transition zones while still resolving the wire.
    lc_min = params.wire_radius_m * 0.5

    log.info(
        f"Meshing ({mesh_resolution}): "
        f"lc_wire={lc_coil:.4g}m, lc_domain={lc_domain:.4g}m, lc_min={lc_min:.4g}m"
    )

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_domain)

    # 2D: Delaunay (avoids "equivalent triangles" on square-section wire rings).
    # 3D: Delaunay (stable for fragmented Boolean geometries;
    #     Frontal-Delaunay silently SEGV'd on sphere+cylinder Boolean geometries).
    gmsh.option.setNumber("Mesh.Algorithm",   5)   # Delaunay 2D
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay 3D

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Relocate2D")
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Relocate3D")

    # Renumber to guarantee contiguous 1..N node indices.
    # dolfinx's extract_geometry asserts contiguity; gaps arise after
    # the Boolean cut + fragment pipeline.
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    tmp = tempfile.mkdtemp(prefix="oracle_mesh_")
    msh_path = Path(tmp) / "coil.msh"
    gmsh.write(str(msh_path))
    return msh_path
