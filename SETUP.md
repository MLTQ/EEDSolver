# Oracle — Setup Guide

## Prerequisites

Oracle requires FEniCSx (dolfinx), which has non-trivial installation requirements.
The recommended path is conda. Do this before anything else.

---

## 1. Python Environment (FEniCSx via conda)

FEniCSx is not pip-installable in a reliable way. Use conda:

```bash
# Install miniforge if you don't have conda
# https://github.com/conda-forge/miniforge

conda create -n oracle python=3.11
conda activate oracle

# FEniCSx + Gmsh + petsc
conda install -c conda-forge fenics-dolfinx mpich gmsh python-gmsh

# FastAPI server
pip install fastapi uvicorn[standard] pydantic numpy scipy

# Verify FEniCSx
python -c "import dolfinx; print(dolfinx.__version__)"
# Should print 0.8.x or later

# Verify Gmsh
python -c "import gmsh; print(gmsh.__version__)"
```

**macOS note**: If on Apple Silicon, the conda-forge build works but MPI
parallelism is limited. For single-machine use this is fine.

**Arch Linux note**: AUR has `python-fenics-dolfinx` but the conda path
is more reliable and self-contained.

---

## 2. Node / Bun + Tauri

```bash
# Node 20+ or Bun
curl -fsSL https://bun.sh/install | bash

# Rust (for Tauri)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Tauri CLI
cargo install tauri-cli --version "^2.0"

# Install JS deps
bun install
```

---

## 3. First Run

### Start solver server manually (for development):
```bash
conda activate oracle
cd solver
uvicorn main:app --port 7432 --reload
```

Test it:
```bash
curl http://localhost:7432/health
# {"status": "ok"}
```

### Start Tauri app (dev mode):
```bash
# In a separate terminal
cargo tauri dev
```

### Run a test solve:
```bash
curl -X POST http://localhost:7432/solve \
  -H "Content-Type: application/json" \
  -d '{
    "coil": {
      "radius_m": 0.05,
      "turns": 10,
      "pitch_m": 0.005,
      "wire_radius_m": 0.001,
      "current_A": 1.0,
      "coil_type": "solenoid"
    },
    "domain_radius_m": 0.2,
    "mesh_resolution": "coarse",
    "formulation": "scalar_only",
    "slices": [
      {"axis": "z", "position": 0.5, "field": "phi", "resolution": 64}
    ]
  }'
```

Expected response time on coarse: < 10 seconds.

---

## 4. Tauri Sidecar Configuration

In production (packaged app), Tauri spawns the solver as a sidecar.
The Python binary must be bundled or the conda env must be on PATH.

In `tauri.conf.json`:
```json
{
  "bundle": {
    "externalBin": ["../solver/sidecar_entry"]
  }
}
```

`solver/sidecar_entry` is a shell script:
```bash
#!/bin/bash
conda activate oracle
exec uvicorn solver.main:app --port 7432
```

For development, skip sidecar — run solver manually as shown above.

---

## 5. Verify Full Stack

1. Solver health: `curl localhost:7432/health` → `{"status":"ok"}`
2. Test solve (above) returns JSON with `slices` array
3. Tauri app launches and GeometryPanel renders
4. Click Solve with default params → SliceViewer shows heatmap

---

## Common Issues

**`ImportError: No module named 'dolfinx'`**
→ You're not in the oracle conda env. `conda activate oracle`.

**`Address already in use: 7432`**
→ Another solver instance is running. `lsof -i :7432` to find it.

**`Mesh generation failed`**
→ Usually a Gmsh geometry issue. Check `solver/geometry/coil.py` logs.
   Reducing `turns` or increasing `domain_radius_m` often fixes it.

**Nédélec orientation warnings from FEniCSx**
→ Expected on first mesh. If solve diverges (NaN in output), the mesh
   may have inverted elements. Use `mesh_resolution: "medium"` to see
   if it resolves.

**Three.js volume not rendering**
→ WebGL 2 required. Check browser console for shader compilation errors.
