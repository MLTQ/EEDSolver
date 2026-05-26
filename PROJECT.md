# Oracle — EED Field Simulator

## What This Is
A hypothesis-driven field simulation tool for modeling non-standard electromagnetic phenomena,
specifically the scalar potential (φ) and vector potential (A) formulations from EED theory
and the Deleted Degrees of Freedom paper. Coil geometries are defined parametrically, solved
via FEniCSx, and visualized as interactive 2D slice heatmaps and 3D volumetric views in a
Tauri desktop app. This is a lab instrument, not a general-purpose FEM tool.

## Current State
**Stage 3 complete.** Full React/TS frontend built and Vite build clean. Ready for `cargo tauri dev` with live solver.

## Active Work
- [x] Stage 1: `solver/` Python package — FEniCSx solver + FastAPI server
- [x] Stage 2: `src-tauri/` — Tauri shell (types.rs, solver_client.rs, commands.rs) — cargo check clean
- [x] Stage 3: `src/` — React/TS frontend — VolumeViewer (Three.js ray-march), SliceViewer (Plotly), GeometryPanel, HypothesisLog, App layout — tsc clean, Vite build clean
- [ ] Stage 4: eed_coupled UI tuning, sensor placement export, HypothesisLog diff view

Build order: solver endpoint → Tauri IPC bridge → slice viewer → 3D view.

## Architecture Overview

```
src/                        # React/TS frontend
  components/
    GeometryPanel/          # Coil parameter controls (sliders, inputs)
    SliceViewer/            # 2D heatmap slice display (Plotly)
    VolumeViewer/           # 3D volumetric view (Three.js shader)
    FieldSelector/          # Choose which field to visualize (φ, A, B, etc.)
    HypothesisLog/          # Save/load named parameter sets + results
  lib/
    api.ts                  # IPC calls to Tauri commands
    fieldTypes.ts           # Shared types: FieldResult, CoilParams, SliceData
    colormap.ts             # Heatmap colorscales

src-tauri/
  src/
    main.rs                 # Tauri app entry
    commands.rs             # Tauri commands: solve, get_slice, get_volume
    solver_client.rs        # HTTP client → Python solver server
    types.rs                # Shared Rust types mirroring fieldTypes.ts

solver/                     # Python package (runs as sidecar or standalone)
  main.py                   # FastAPI app entry
  server.py                 # Route definitions
  geometry/
    coil.py                 # Gmsh coil geometry builder (parametric)
    mesh.py                 # Mesh generation + refinement
  fields/
    formulation.py          # UFL weak forms: φ, A, coupled EED system
    solver.py               # FEniCSx solve loop
    postproc.py             # Extract slice arrays, volume arrays, field maxima
  models/
    params.py               # Pydantic models: CoilParams, SolveRequest, FieldResult
```

Data flow: Frontend sliders → Tauri command → HTTP POST to solver → FEniCSx solve →
numpy arrays → JSON response → Plotly/Three.js render.

## Decision Log
- **2025-05-25** Tauri + Python sidecar over pure Rust solver — FEniCSx is Python-native;
  wrapping in Rust adds no value. Sidecar pattern (Tauri spawns solver process) keeps
  the desktop app self-contained.
- **2025-05-25** FastAPI over raw subprocess — cleaner IPC, easier to test solver in isolation,
  aligns with Pharaoh inference server pattern.
- **2025-05-25** Plotly for 2D slices over D3 — Plotly heatmap is faster to iterate with
  and handles the array→render pipeline with less code. D3 if we need more control later.
- **2025-05-25** Three.js volume shader for 3D — browser-native, no extra deps, ray-marched
  scalar volume with viridis/plasma GLSL colormaps. **Primary view** (flex-1, full panel height).
  Plotly 2D slices are secondary (bottom panel, collapsible tab).
- **2025-05-25** Mixed FEniCSx function space for coupled (φ, A) — φ is CG1 scalar,
  A is Nédélec edge elements (N1curl). This is the correct discretization for the
  vector potential in Maxwell + EED.
- **2025-05-25** Gmsh parametric geometry over hand-coded meshes — coil params (radius,
  turns, pitch, wire gauge) are first-class inputs, not hardcoded geometry.
- **2025-05-25** HypothesisLog as local JSON store — save named runs with params +
  field maxima for lab comparison. No database needed at this scale.

## Sharp Edges
- FEniCSx must be installed in the Python environment the sidecar uses. Install via
  `uv sync` (see SETUP.md) — requires system MPI (brew install open-mpi on macOS).
- Nédélec elements require a specific mesh orientation. Gmsh produces compatible meshes
  but orientation must be verified on first solve.
- The EED scalar field φ is NOT the standard EM scalar potential. The weak form in
  `formulation.py` must implement the EED/deleted-DOF equations, not Maxwell's. This
  is the core research asset — do not conflate with standard magnetostatics.
- Tauri sidecar process lifetime: solver must be started before first solve command
  and gracefully shut down on app exit. Handle port conflicts.
- ParaView is NOT in the stack. All visualization is in-browser.
