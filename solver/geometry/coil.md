# solver/geometry/coil.py

## Purpose
Parametric Gmsh coil geometry builder. Given `CoilParams` and a domain radius, produces a `.msh` file with tagged physical groups ready for FEniCSx ingestion.

## Components
- `build_coil_geometry(params, domain_radius_m)` — public entry point
- `_build_solenoid` — N stacked rings approximating helical winding
- `_build_toroid` — azimuthal winding (standard torus)
- `_build_toroid_poloidal` — poloidal winding (key EED discriminator)
- `_build_flat_spiral` — Archimedean concentric rings
- `_build_rodin` — figure-8 alternating-tilt approximation of Rodin coil
- `_add_torus` — Gmsh OCC primitive for a single wire tube
- `_tag_physical_groups` — boolean fragment + physical group tagging

## Physical Groups
- `coil_wire` — wire volume (current source J assigned here)
- `air_domain` — surrounding vacuum
- `boundary_sphere` — outer mesh boundary (BCs applied here)

## Decisions
- Helical solenoid approximated as stacked loops — true helix is topologically hard to volume-mesh; stacked loops are standard FEM practice
- Rodin coil approximated with alternating-tilt loops — full 36-point Rodin star pattern is a future improvement
- Characteristic length scales from wire radius (fine near wire) to coil radius (coarse in domain)
- `gmsh.model.occ.fragment` used to cut wire out of domain without overlap

## Contracts
- Caller must call `gmsh.finalize()` — this function handles it internally
- Returns `Path` to `.msh` file in a temp dir; caller responsible for cleanup
- Mesh always has 3D elements (tetrahedra); surface groups for BCs
