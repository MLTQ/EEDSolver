# Oracle — Architecture Specification

## System Overview

Oracle is a desktop application for rapid field simulation iteration. A scientist
adjusts coil geometry parameters, clicks "Solve", and within seconds sees heatmap
visualizations of the predicted scalar potential (φ), vector potential (A), and
derived fields across configurable 2D cross-section planes.

The system is explicitly designed for a non-standard field theory (EED / Deleted
Degrees of Freedom) in which the scalar field φ is a physically meaningful DOF
that standard EM tools discard. This is not a Maxwell solver with a pretty UI —
the governing equations in the solver are the research artifact.

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React/TS)                   │
│  GeometryPanel │ SliceViewer │ VolumeViewer │ HypothLog  │
└────────────────────────┬────────────────────────────────┘
                         │ Tauri invoke() / IPC
┌────────────────────────▼────────────────────────────────┐
│                  TAURI SHELL (Rust)                      │
│  commands.rs  │  solver_client.rs  │  types.rs           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (localhost)
┌────────────────────────▼────────────────────────────────┐
│               SOLVER SERVER (Python/FastAPI)              │
│  geometry/  │  fields/  │  models/                       │
│  Gmsh mesh  │  FEniCSx  │  numpy arrays out              │
└─────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Solver Server (`solver/`)

**Runtime**: Python 3.11+, FEniCSx (dolfinx), Gmsh, FastAPI, uvicorn, numpy, scipy

**Startup**: Tauri spawns this as a sidecar process on app launch. Listens on
`localhost:7432` (chosen to avoid common port conflicts).

#### `solver/main.py`
Entry point. Starts uvicorn with the FastAPI app. Handles graceful shutdown on
SIGTERM from Tauri.

#### `solver/server.py`
Route definitions:
- `POST /solve` — Full solve. Accepts `SolveRequest`, returns `SolveResult`.
- `GET /health` — Liveness check. Returns `{"status": "ok"}`.
- `GET /fields` — List available field names for the current solve result.

#### `solver/models/params.py`
Pydantic models:

```python
class CoilParams(BaseModel):
    radius_m: float          # Coil radius (meters)
    turns: int               # Number of turns
    pitch_m: float           # Turn-to-turn pitch
    wire_radius_m: float     # Wire cross-section radius
    current_A: float         # Applied current (amperes)
    coil_type: str           # "solenoid" | "toroid" | "toroid_poloidal" | "flat_spiral" | "rodin"

class SliceRequest(BaseModel):
    axis: str                # "x" | "y" | "z"
    position: float          # Position along axis (normalized 0–1)
    field: str               # "phi" | "A_magnitude" | "B" | "J"
    resolution: int          # Grid points per side (default 128)

class SolveRequest(BaseModel):
    coil: CoilParams
    domain_radius_m: float   # Bounding sphere radius for simulation domain
    mesh_resolution: str     # "coarse" | "medium" | "fine"
    formulation: str         # "eed_coupled" | "maxwell_only" | "scalar_only"
    slices: list[SliceRequest]  # Which slices to return post-solve

class SliceData(BaseModel):
    axis: str
    position: float
    field: str
    shape: list[int]         # [rows, cols]
    data: list[float]        # Flattened 2D array
    x_range: list[float]     # [min, max] in meters
    y_range: list[float]     # [min, max] in meters
    field_min: float
    field_max: float

class FieldMaxima(BaseModel):
    field: str
    max_value: float
    max_location: list[float]   # [x, y, z] in meters

class SolveResult(BaseModel):
    solve_time_s: float
    mesh_nodes: int
    slices: list[SliceData]
    maxima: list[FieldMaxima]
    warnings: list[str]
```

#### `solver/geometry/coil.py`
Builds a parametric Gmsh geometry for the requested coil type. Outputs a `.msh`
file in a temp directory. Key considerations:
- Solenoid: helical wire path approximated as stacked current loops for FEM
  (true helix is topologically complex for meshing; loops are standard practice)
- Wire cross-section extruded along path
- Domain: spherical bounding box with characteristic length scaling from coil size
- Physical groups tagged: `coil_wire`, `air_domain`, `boundary_sphere`

#### `solver/geometry/mesh.py`
Calls Gmsh to generate mesh from geometry. Resolution presets:
- `coarse`: ~5k–20k elements (fast preview, <5s solve)
- `medium`: ~50k–200k elements (working resolution, <30s solve)
- `fine`: ~500k–2M elements (publication quality, minutes)

#### `solver/fields/formulation.py`
**The core research code.** Defines the UFL weak forms.

Three formulations (selected via `SolveRequest.formulation`):

**`maxwell_only`** — Standard magnetostatics baseline:
```
Find A ∈ H(curl) such that:
∫ (1/μ₀) curl(A)·curl(v) dx = ∫ J·v dx  ∀v ∈ H(curl)
```
This is the control/reference case. Scalar DOF is discarded (gauge-fixed away).

**`scalar_only`** — EED scalar field in isolation:
```
Find φ ∈ H¹ such that:
∫ ∇φ·∇ψ dx + α ∫ φ·ψ dx = ∫ S_φ·ψ dx  ∀ψ ∈ H¹
```
where S_φ is the scalar source term (derivation from EED paper — see FIELD_THEORY.md),
and α is the EED coupling constant.

**`eed_coupled`** — Full coupled system (primary research formulation):
```
Find (φ, A) ∈ H¹ × H(curl) such that:
∫ ∇φ·∇ψ dx + α ∫ φ·ψ dx + β ∫ div(A)·ψ dx = ∫ S_φ·ψ dx
∫ (1/μ₀) curl(A)·curl(v) dx + γ ∫ ∇φ·v dx = ∫ J·v dx
∀(ψ, v) ∈ H¹ × H(curl)
```
Coupling terms (β, γ) are the EED parameters. When β=γ=0 this reduces to
decoupled Maxwell + isolated scalar.

Function spaces:
- φ: `CG1` (continuous Galerkin, degree 1) — standard scalar FEM
- A: `N1curl` (Nédélec first kind, degree 1) — correct for vector potential,
  preserves tangential continuity across element boundaries

#### `solver/fields/solver.py`
Sets up boundary conditions, calls FEniCSx linear or nonlinear solver, returns
`dolfinx.fem.Function` objects for each field. Uses PETSc MUMPS direct solver
for coarse/medium meshes; iterative (GMRES + algebraic multigrid) for fine.

#### `solver/fields/postproc.py`
Extracts field data from FEniCSx functions:
- **Slice extraction**: Samples field on a regular grid in the requested plane
  using `dolfinx.fem.Expression` evaluated at grid points
- **Maxima finding**: `numpy.argmax` on the sampled volume
- **Serialization**: Converts to flat float lists for JSON transport

---

### 2. Tauri Shell (`src-tauri/`)

**Runtime**: Rust, Tauri 2

#### `src-tauri/src/commands.rs`
Tauri commands exposed to frontend:

```rust
#[tauri::command]
async fn solve(request: SolveRequest) -> Result<SolveResult, String>

#[tauri::command]  
async fn get_solver_status() -> Result<SolverStatus, String>

#[tauri::command]
async fn save_hypothesis(name: String, request: SolveRequest, result: SolveResult) -> Result<(), String>

#[tauri::command]
async fn load_hypotheses() -> Result<Vec<HypothesisEntry>, String>
```

#### `src-tauri/src/solver_client.rs`
HTTP client (reqwest) that talks to the Python solver server. Handles:
- Health check polling on startup (wait for solver to be ready)
- Request serialization / response deserialization
- Timeout handling (fine meshes can take minutes — use long timeout, surface progress)

#### `src-tauri/src/types.rs`
Rust mirror of the Pydantic models. `serde` derive for JSON ser/de.
Must stay in sync with `solver/models/params.py` — this is a sharp edge.

---

### 3. Frontend (`src/`)

**Runtime**: React 18, TypeScript, Plotly.js, Three.js, Tailwind

#### `src/components/GeometryPanel/`
Parameter controls for the coil. Sliders with live numeric display for:
- Coil radius, turns, pitch, wire radius, current
- Domain radius, mesh resolution selector
- Formulation selector (maxwell_only / scalar_only / eed_coupled)
- EED coupling constants (α, β, γ) — shown only in eed_coupled mode

Emits a `CoilParams` object upstream on change. Does NOT auto-trigger solve —
user clicks "Solve" explicitly (avoids thrashing the solver on slider drag).

#### `src/components/SliceViewer/`
2D heatmap display. Uses Plotly `heatmap` trace.

Controls:
- Axis selector (X / Y / Z)
- Slice position slider (0–1 along selected axis)
- Field selector (φ / |A| / |B| / |J|)
- Colorscale selector (viridis / plasma / diverging for signed fields)
- Overlay: field maximum marker (crosshair at max location)

The slice data is already computed by the solver for the requested slices.
If the user requests a slice that wasn't in the original solve, it triggers
a lightweight re-slice call (no re-solve needed — postproc only).

#### `src/components/VolumeViewer/`
3D scalar field visualization. Three.js canvas.

Implementation: 3D texture from the scalar field volume, ray-marched in a
GLSL fragment shader. Transfer function maps field value to color + opacity.
This is the secondary view — useful for spatial intuition but 2D slices
are the primary analysis interface.

Controls: orbit rotation, opacity threshold slider, field selector.

#### `src/components/HypothesisLog/`
Local experiment journal. Saves named runs as JSON to `~/.oracle/hypotheses/`.
Each entry: timestamp, name, CoilParams, SolveRequest, FieldMaxima, optional notes.

Table view: sort by field maximum, filter by formulation type.
Click entry: restore params to GeometryPanel (does not re-solve automatically).

#### `src/lib/api.ts`
Thin wrapper around `window.__TAURI__.invoke()`. Typed call signatures matching
`commands.rs`. All solver communication goes through here — no direct HTTP
from the frontend.

#### `src/lib/fieldTypes.ts`
TypeScript interfaces mirroring `solver/models/params.py` and `types.rs`.
Single source of truth for frontend type system.

---

## Solve Lifecycle

```
1. User adjusts params in GeometryPanel
2. User clicks "Solve"
3. Frontend: api.ts → invoke('solve', request)
4. Tauri: commands.rs receives, forwards to solver_client.rs
5. solver_client.rs: POST /solve to localhost:7432
6. solver/server.py: routes to solve handler
7. geometry/coil.py: builds Gmsh geometry
8. geometry/mesh.py: generates mesh
9. fields/formulation.py: constructs UFL weak form
10. fields/solver.py: FEniCSx solve (PETSc)
11. fields/postproc.py: extracts slice arrays + maxima
12. JSON response → Tauri → Frontend
13. SliceViewer: renders Plotly heatmap
14. VolumeViewer: uploads 3D texture, renders
15. HypothesisLog: user optionally saves run
```

---

## File Layout

```
oracle/
├── PROJECT.md
├── ARCHITECTURE.md
├── FIELD_THEORY.md             ← EED equations, sources, parameter rationale
├── SETUP.md                    ← FEniCSx install via uv, first run
├── solver/
│   ├── main.py / main.md
│   ├── server.py / server.md
│   ├── geometry/
│   │   ├── coil.py / coil.md
│   │   └── mesh.py / mesh.md
│   ├── fields/
│   │   ├── formulation.py / formulation.md
│   │   ├── solver.py / solver.md
│   │   └── postproc.py / postproc.md
│   └── models/
│       └── params.py / params.md
├── src-tauri/
│   ├── src/
│   │   ├── main.rs / main.md
│   │   ├── commands.rs / commands.md
│   │   ├── solver_client.rs / solver_client.md
│   │   └── types.rs / types.md
│   └── tauri.conf.json
├── src/
│   ├── components/
│   │   ├── GeometryPanel/ (index.tsx + index.md)
│   │   ├── SliceViewer/ (index.tsx + index.md)
│   │   ├── VolumeViewer/ (index.tsx + index.md)
│   │   ├── HypothesisLog/ (index.tsx + index.md)
│   │   └── FieldSelector/ (index.tsx + index.md)
│   └── lib/
│       ├── api.ts / api.md
│       ├── fieldTypes.ts / fieldTypes.md
│       └── colormap.ts / colormap.md
└── package.json
```
