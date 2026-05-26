# solver/geometry/mesh.py

## Purpose
Bridges Gmsh `.msh` output to dolfinx mesh objects with tagged cell/facet regions. Also defines resolution presets.

## Components
- `apply_resolution_preset(resolution)` — returns mesh size scaling factor; consumed by coil.py before meshing
- `load_mesh_from_file(msh_path)` — reads `.msh` → `(mesh, cell_tags, facet_tags)` via dolfinx gmshio
- `get_mesh_stats(mesh)` — returns cell/node counts for SolveResult

## Decisions
- Resolution presets are scaling factors, not absolute element counts — actual count depends on coil geometry
- gmshio.read_from_msh handles the Gmsh→dolfinx conversion including physical group → MeshTags mapping
- MPI.COMM_WORLD required by dolfinx even in single-process mode

## Contracts
- `cell_tags`: physical group tag values from Gmsh (coil_wire, air_domain)
- `facet_tags`: boundary surfaces (boundary_sphere)
- Caller must be running inside the Docker container (`docker compose up solver`) — dolfinx is not in the UV venv
