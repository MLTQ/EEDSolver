# Oracle — Setup Guide

## How the solver runs

The Python solver uses FEniCSx (dolfinx), which is not pip-installable — it's
distributed via conda-forge and Docker only. The clean solution:

- **Docker** runs the solver (dolfinx, Gmsh, PETSc — all pre-installed in the official image)
- **UV** manages everything else (linting, testing, the JS/Rust build chain)
- The Tauri app connects to the solver over HTTP on localhost:7432, same as always

---

## 1. Python tooling (UV)

Install UV if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the non-FEM Python deps (FastAPI, pydantic, etc.):
```bash
uv sync
```

This is used for linting, type checking, and running tests against the solver API
without the full FEM stack. The actual solver runs in Docker.

---

## 2. Node / Bun + Tauri

```bash
# Bun
curl -fsSL https://bun.sh/install | bash

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# JS deps
bun install
```

---

## 3. Start the solver (Docker)

First build the solver image (one-time, ~2–3 min to pull dolfinx base):
```bash
docker compose build solver
```

Then start it:
```bash
docker compose up solver
```

The solver mounts `./solver/` as a volume, so code changes hot-reload
without a rebuild. Check it's alive:
```bash
curl http://localhost:7432/health
# {"status":"ok","solver_version":"0.1.0"}
```

---

## 4. Start the Tauri app

```bash
# In a separate terminal
cargo tauri dev
```

---

## 5. Run a test solve

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
    "eed": { "alpha": 0.0, "beta": 0.1, "gamma": 0.1 },
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

## 6. Production / sidecar

The `solver/sidecar_entry` script for the packaged app:
```bash
#!/bin/bash
docker run --rm -p 7432:7432 \
  -v "$(dirname "$0"):/app/solver" \
  oracle-solver
```

For a fully self-contained app (no Docker dep on user machine), the alternative is
to bundle a compiled `dolfinx` binary — complex, deferred to v2.

---

## Common Issues

**`docker compose up solver` hangs on first run**
→ Pulling the `dolfinx/dolfinx:stable` base image (~2 GB). Wait it out.

**`curl localhost:7432/health` → connection refused**
→ Container isn't up yet, or port conflict. Check: `docker ps` and `lsof -i :7432`.

**Mesh generation failed in solver logs**
→ Gmsh geometry issue. Reduce `turns` or increase `domain_radius_m`.

**Nédélec orientation warnings**
→ Expected on first mesh. If solve returns NaN, try `mesh_resolution: "medium"`.

**`Address already in use: 7432`**
→ `docker ps` to find the running container, `docker stop <id>`.

**Three.js volume not rendering**
→ WebGL 2 required. Check browser console for shader compilation errors.
