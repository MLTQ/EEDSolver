"""
Mesh loading and resolution control for Oracle solver.
Bridges Gmsh output → dolfinx mesh with physical group markers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

try:
    import dolfinx
    import dolfinx.io
    from dolfinx.io import gmsh as gmshio  # renamed gmshio → gmsh in dolfinx 0.9+
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

from solver.models.params import MeshResolution


# Resolution presets: (max_elements_target, description)
RESOLUTION_PRESETS: dict[MeshResolution, dict] = {
    "coarse": {
        "mesh_size_factor": 4.0,   # Large elements → fast preview, ~5k–20k tets
        "description": "Coarse preview (<5s solve)",
    },
    "medium": {
        "mesh_size_factor": 1.0,   # Default sizing from coil.py
        "description": "Working resolution (<30s solve)",
    },
    "fine": {
        "mesh_size_factor": 0.3,   # Fine mesh for publication
        "description": "Fine mesh (minutes)",
    },
}


def apply_resolution_preset(resolution: MeshResolution) -> float:
    """
    Return the mesh size scaling factor for the given resolution preset.
    Used in coil.py to scale characteristic lengths before meshing.
    """
    return RESOLUTION_PRESETS[resolution]["mesh_size_factor"]


def load_mesh_from_file(msh_path: Path) -> tuple:
    """
    Load a .msh file produced by coil.py into dolfinx.
    Returns (mesh, cell_tags, facet_tags).

    cell_tags: MeshTags for 3D regions (coil_wire=1, air_domain=2)
    facet_tags: MeshTags for 2D boundaries (boundary_sphere=10)
    """
    if not HAS_DOLFINX:
        raise ImportError("dolfinx not available. Run: docker compose up solver (see SETUP.md)")

    comm = MPI.COMM_WORLD

    # dolfinx 0.10+: returns MeshData(mesh, cell_tags, facet_tags, ridge_tags, peak_tags, physical_groups)
    md = gmshio.read_from_msh(str(msh_path), comm, rank=0, gdim=3)
    mesh, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    return mesh, cell_tags, facet_tags


def get_mesh_stats(mesh) -> dict:
    """Return basic mesh statistics for the SolveResult."""
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    num_nodes = mesh.topology.index_map(0).size_global
    return {
        "num_cells": num_cells,
        "num_nodes": num_nodes,
    }
