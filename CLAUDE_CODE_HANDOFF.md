# Oracle — Claude Code Session Briefing

## Read This First

This is the session entry point for Claude Code. Read all linked docs before
writing any code. The order matters.

1. **This file** — orientation
2. **PROJECT.md** — current state, architecture overview, decision log
3. **ARCHITECTURE.md** — full component spec, data flow, file layout
4. **FIELD_THEORY.md** — governing equations (read before touching `formulation.py`)
5. **SETUP.md** — environment setup (read before running anything)

---

## What We're Building

**Oracle**: A Tauri desktop app for EED/deleted-DOF field simulation. The user
inputs coil geometry parameters, clicks Solve, and sees 2D heatmaps of the
predicted scalar field (φ) and vector potential (A) from a FEniCSx solver
running as a local Python server.

This is a lab instrument for hypothesis testing in fringe EM physics research.
The EED scalar field φ is the primary quantity of interest — it does not exist
in standard Maxwell EM solvers.

---

## Build Order

Build in this sequence. Each stage should be testable before moving to the next.

### Stage 1: Solver Server (Python)
**Goal**: `POST /solve` returns valid slice data for a solenoid coil.

Files to create:
- `solver/models/params.py` — Pydantic models (spec in ARCHITECTURE.md)
- `solver/geometry/coil.py` — Gmsh solenoid builder
- `solver/geometry/mesh.py` — Mesh generation
- `solver/fields/formulation.py` — UFL weak forms (see FIELD_THEORY.md)
- `solver/fields/solver.py` — FEniCSx solve
- `solver/fields/postproc.py` — Slice extraction
- `solver/server.py` — FastAPI routes
- `solver/main.py` — App entry

Start with `scalar_only` formulation only. Add `maxwell_only` second,
`eed_coupled` third.

Test: `curl -X POST localhost:7432/solve ...` returns JSON with slice data.

### Stage 2: Tauri Shell (Rust)
**Goal**: Frontend can invoke `solve` command and get back typed data.

Files to create:
- `src-tauri/src/types.rs` — Mirror of Pydantic models
- `src-tauri/src/solver_client.rs` — reqwest HTTP client
- `src-tauri/src/commands.rs` — Tauri commands
- `src-tauri/src/main.rs` — App entry, sidecar management

Test: Tauri dev mode, invoke solve command from browser console via
`window.__TAURI__.invoke('solve', {...})`.

### Stage 3: Frontend Core (React/TS)
**Goal**: GeometryPanel + SliceViewer working end-to-end.

Files to create:
- `src/lib/fieldTypes.ts` — TypeScript types
- `src/lib/api.ts` — Tauri invoke wrappers
- `src/lib/colormap.ts` — Heatmap color scales
- `src/components/GeometryPanel/index.tsx` — Coil parameter controls
- `src/components/SliceViewer/index.tsx` — Plotly heatmap display
- `src/App.tsx` — Root layout

Test: Full solve round-trip visible in UI. Heatmap renders with correct
axis labels and field value range.

### Stage 4: Extended Features
- `src/components/VolumeViewer/` — Three.js 3D volume
- `src/components/HypothesisLog/` — Save/load runs
- EED coupling constant controls (α, β, γ sliders)
- `eed_coupled` formulation in solver

---

## Key Constraints

1. **Do not change the weak forms in `formulation.py` without updating
   FIELD_THEORY.md**. The equations are the research artifact.

2. **Nédélec elements for A, CG1 for φ**. This is not negotiable — using
   standard CG elements for A produces wrong solutions. See FIELD_THEORY.md.

3. **No auto-solve on slider drag**. Solve is triggered by explicit button
   click only. The solver can take 30+ seconds on medium meshes.

4. **Types must stay in sync**: `params.py` (Python) ↔ `types.rs` (Rust) ↔
   `fieldTypes.ts` (TypeScript). If you change one, change all three.

5. **EED scalar φ ≠ standard EM scalar potential**. Different equation,
   different physical meaning. Do not conflate in comments or UI labels.

---

## Companion Doc Requirements

Per the modular-docs skill: every `.py`, `.rs`, and `.tsx` file gets a
companion `.md` file with Purpose, Components, Decisions, and Contracts.
Create these alongside the code files, not after.

---

## Decision Log (active)

- Chose FastAPI + uvicorn over subprocess pipes — cleaner IPC, testable in isolation
- Chose Plotly for 2D heatmaps — fastest path to correct visualization
- Chose Three.js ray-march for 3D — no server-side rendering, runs in browser
- Formulations ordered: scalar_only → maxwell_only → eed_coupled (simplest first)
- Static (magnetostatic) limit only for v1 — no time-domain

---

## Questions To Resolve During Build

These need Max's input before implementation:

1. **DDOF paper citation**: Add full reference to FIELD_THEORY.md before
   implementing `eed_coupled`. The coupling term structure (β, γ) should
   be verified against the paper's notation.

2. **EED parameter defaults**: α=0, β=γ=0.1 are placeholders. What are
   physically motivated starting values based on the theory?

3. **Sensor placement output**: Should Oracle export a "recommended sensor
   positions" list (top-N field maxima locations) as a structured output?
   Useful for lab workflow.

4. **Coil types for v1**: Solenoid is definite. Toroid and flat spiral?
   Toroid is particularly interesting for EED testing (see FIELD_THEORY.md).
