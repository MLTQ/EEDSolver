# Oracle v2 — Full Architecture & Implementation Plan

> **Status**: Planning  
> **Supersedes**: `ARCHITECTURE.md` (v1, Python/dolfinx/FEM)  
> **Physics reference**: `V2_PHYSICS.md`

---

## 1. The Decision: Keep Shell, Replace Engine

### What survives from v1

| Component | Status | Rationale |
|---|---|---|
| Tauri shell (`src-tauri/`) | **Keep, extend** | Solid foundation; add wgpu dep in-process |
| React frontend (`src/`) | **Keep, adapt** | UI structure is right; add new fields/panels |
| Component patterns (SliceViewer, etc.) | **Keep** | Field-agnostic; just feed it new data |
| Type contract pattern (SolveRequest/Result) | **Keep as pattern** | Rewrite in v2 with expanded fields |
| Documentation structure (FIELD_THEORY.md etc.) | **Keep, update** | Science docs are permanent |

### What gets deleted

| Component | Reason |
|---|---|
| `solver/` (entire Python tree) | Language, paradigm, and physics all wrong for v2 |
| `solver_client.rs` HTTP proxy | No longer needed — solver is in-process |
| Gmsh geometry builder | Replaced by analytical Biot-Savart wire descriptions |
| FEniCSx/dolfinx/PETSc/MUMPS | Entire FEM stack replaced by GPU Yee FDTD |
| Python sidecar process model | GPU solver runs inside Tauri, not as a separate process |

### Architecture change summary

```
v1:
  Frontend → invoke() → Tauri → HTTP → Python/FastAPI → dolfinx/FEniCSx → PETSc/MUMPS

v2:
  Frontend → invoke() → Tauri → Rust/wgpu GPU solver (in-process)
```

The HTTP server, subprocess management, and inter-process serialization overhead
disappear entirely. The GPU solver is a Rust crate compiled into the Tauri binary.

---

## 2. Target Architecture

### 2.1 Layer diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    FRONTEND (React/TS)                        │
│  GeometryPanel │ SliceViewer │ VolumeViewer │ HypothesisLog   │
│  + GEMPanel    │ + CFieldView│ + HolonomyViz│ + EntityManager │
└───────────────────────────┬──────────────────────────────────┘
                            │ Tauri invoke() / IPC
┌───────────────────────────▼──────────────────────────────────┐
│                   TAURI SHELL (Rust)                          │
│  commands.rs  │  types.rs  │  [no more solver_client.rs]      │
└───────────────────────────┬──────────────────────────────────┘
                            │ direct function call (same process)
┌───────────────────────────▼──────────────────────────────────┐
│              GPU SOLVER CRATE  (Rust + wgpu)                  │
│                                                               │
│  biot/          Analytical Biot-Savart per coil entity        │
│  grid/          Yee grid: allocation, indexing, BC           │
│  shaders/       WGSL compute shaders                         │
│    ├── biot.wgsl          Biot-Savart sum kernel             │
│    ├── fdtd_em.wgsl       Potential-primary EM FDTD update   │
│    ├── fdtd_gem.wgsl      GEM FDTD update (same stencil)     │
│    ├── c_field.wgsl       C-field dynamics & coupling        │
│    └── postproc.wgsl      Slice extract, Poynting, holonomy  │
│  solver/        Static (elliptic CG) + time-domain FDTD      │
│  postproc/      Field extraction, slices, volume, maxima     │
└──────────────────────────────────────────────────────────────┘
                            │ wgpu
                    ┌───────▼───────┐
                    │  GPU backend  │
                    │  Metal (Mac)  │
                    │  Vulkan (Lin) │
                    │  DX12 (Win)   │
                    └───────────────┘
```

### 2.2 Two solver modes

**Static mode** (default, fast): Solves the time-independent EED + GEM equations.
Uses preconditioned conjugate gradient on GPU. Returns instantaneous field snapshot.
Appropriate for: coil design, parameter sweeps, A/B comparison.

**Time-domain mode** (FDTD): Steps the full potential-primary wave equations forward
in time. Captures SLW propagation, transient responses, Faraday cage penetration.
Appropriate for: SLW signature predictions, dynamic coupling studies.

---

## 3. Physics Implementation

### 3.1 Primary state variables (Yee grid)

```
φ   scalar potential      [V]        cell vertices   (i, j, k)
Ax  x-component of A      [V·s/m]    x-edges         (i+½, j, k)
Ay  y-component of A      [V·s/m]    y-edges         (i, j+½, k)
Az  z-component of A      [V·s/m]    z-edges         (i, j, k+½)
```

**Why potentials are primary, not fields**: The DDOF paper establishes that C = ∇·A +
(1/c²)∂φ/∂t is a physical degree of freedom. If you evolve E and B (field-primary FDTD),
C is never computed — it's structurally inaccessible. Potential-primary FDTD has C
available at every cell, every time step, for free.

### 3.2 Derived fields (computed on-GPU as needed)

```
E  = −∇φ − ∂A/∂t          electric field
B  = ∇×A                   magnetic field
C  = ∇·A + (1/c²)∂φ/∂t    EED scalar (the "deleted" field)
```

### 3.3 EED update equations (potential-primary FDTD, γ=1, m=0)

```
∂²A/∂t² = c²∇²A − ∇(∂φ/∂t) + (1/ε₀)J_e

∂²φ/∂t² = c²∇²φ − c²∂(∇·A)/∂t + ρ_e/ε₀

□C = ∂_μJ^μ_e    (= 0 for conserved currents)
```

No gauge condition is imposed. C evolves as a dynamical field. Setting C=0 would
recover Lorenz gauge and standard Maxwell — the simulation can do this optionally
as a validation baseline.

### 3.4 GEM update equations (same stencil, different constants)

```
State:    Φ_g, A_g (same Yee layout, same buffer structure)

∂²A_g/∂t² = c²∇²A_g − ∇(∂Φ_g/∂t) − (4πG/c)J_m + κ_G ∇C

∂²Φ_g/∂t² = c²∇²Φ_g + 4πGρ_m            + κ_G (∂C/∂t)
```

The κ_G term is the EED→GEM coupling via the C field (Kaluza-Klein identification:
∇·A encodes gravitational degrees of freedom). κ_G is a free parameter — set to the
KK-predicted value, the Li-Torr value, or zero for pure GEM without EM coupling.

**Note**: GEM is valid in the weak-field, slow-motion regime. The simulation is not
a general-relativity solver. The domain of validity is any device-scale EM configuration.

### 3.5 Coil sources: Biot-Savart (analytical, GPU)

Rather than meshing the wire and solving on the mesh, the wire current is treated
analytically. Each coil entity is described as an ordered list of wire segments. The
contribution to A at each grid point is:

```
A(r) = (μ₀/4π) Σ_segments  I · dl / |r − r'|
```

This is embarrassingly parallel — each grid point's A-contribution from each wire
segment is independent. GPU kernel: one thread per grid point, loop over segments.

For the GEM sector, mass currents J_m = ρ_m·v for moving bodies are handled the
same way.

**Advantages over FEM meshing**:
- Exact for thin-wire geometries (no discretization error from mesh)
- Motion is free: new coil position → recompute Biot-Savart sum (no remeshing)
- Multiple entities: linear superposition, no interaction between source terms
- Resolution-independent: fine grid doesn't mean slow Biot-Savart

### 3.6 Observables

```
Modified Poynting vector:   P = E×B − E·C          (EED energy flux)
Scalar energy density:      u = ½(E² + B² + C²)    (EED energy density)
Holonomy (AB phase):        Γ = ∮ A·dl              (line integral on grid edges)
Magnetic helicity:          H = ∫ A·B d³x           (volume integral)
Gravitomagnetic London:     B_g = −(2m_e/e)ω        (for superconducting entity)
```

---

## 4. Implementation Phases

### Phase 0 — Foundation (week 1–2)
**Goal**: wgpu device initializes inside Tauri; Yee grid allocates on GPU; Tauri command
returns a dummy solve result. Nothing simulates yet, but the plumbing is real.

- [ ] Add `wgpu` dependency to `src-tauri/Cargo.toml`
- [ ] Create `solver-gpu` crate (workspace member)
- [ ] `GpuContext`: device, queue, adapter selection (Metal on Mac)
- [ ] `YeeGrid`: define cell dimensions, allocate phi/A buffers on GPU
- [ ] Remove Python sidecar launch from Tauri startup
- [ ] Remove `solver_client.rs`; wire `commands.rs` directly to `solver-gpu`
- [ ] v2 `SolveRequest`/`SolveResult` types (Rust + TS)

### Phase 1 — Biot-Savart Engine (week 2–3)
**Goal**: Given a solenoid description, GPU computes A-field on Yee grid. Validate
against analytic formula for infinite solenoid.

- [ ] `WireEntity`: ordered segment list, current, coil type
- [ ] WGSL kernel `biot.wgsl`: A(r) = Σ μ₀/4π · I dl / |r−r'|
- [ ] CPU dispatch: launch 1 thread/grid-point, n_segments iterations
- [ ] Coil geometry builders: solenoid, toroid, toroid_poloidal, flat_spiral
- [ ] Derive B = ∇×A on GPU; validate |B| on axis vs analytic
- [ ] Multi-entity: accumulate A contributions from N coils

### Phase 2 — Static EED Solver (week 3–5)
**Goal**: Given Biot-Savart sources, solve static EED equations (C is ∇·A in static
limit). GPU preconditioned CG. Validate φ and C against v1 dolfinx reference.

- [ ] Assemble sparse Laplacian stencil on GPU (Yee finite differences)
- [ ] Preconditioned CG: Jacobi preconditioner (trivial for structured grid)
- [ ] Solve φ equation: ∇²φ + α²φ = source
- [ ] Solve A equation: ∇²A = −μ₀J + γ∇φ (with EED coupling)
- [ ] Compute C = ∇·A (static: no ∂φ/∂t term)
- [ ] Extract slices; wire to SliceViewer
- [ ] Validate against v1 FEM on solenoid test case

### Phase 3 — Time-Domain EED FDTD (week 5–8)
**Goal**: Full potential-primary FDTD. C propagates as a wave. SLW signatures visible.

- [ ] WGSL `fdtd_em.wgsl`: leapfrog update for φ^(n+1), A^(n+1)
- [ ] Stability: CFL condition (Δt ≤ Δx / (c√3))
- [ ] Absorbing boundaries: Mur ABC or basic PML
- [ ] C-field update: □C = 0 in free space
- [ ] Modified Ampere/Gauss with C-source terms
- [ ] Lorenz gauge mode (set C=0 each step) for Maxwell baseline comparison
- [ ] Time-stepping UI: run/pause/step, waveform at probe point
- [ ] Validate: SLW propagation speed = c (dispersion test)
- [ ] Validate: Faraday cage — TEM attenuated, C-mode passes

### Phase 4 — GEM Coupled Sector (week 8–11)
**Goal**: Gravitational sector fully coupled to EED. κ_G slider lets user explore
the coupling strength hypothesis.

- [ ] Allocate Φ_g, A_g buffers (same size as EM Yee grid)
- [ ] WGSL `fdtd_gem.wgsl`: GEM update (identical structure to fdtd_em.wgsl)
- [ ] C-field coupling: κ_G · ∇C source in A_g equation
- [ ] Li-Torr mode: for "superconducting" entity, add B_g = −(2m/e)ω
- [ ] GEM field visualization: E_g, B_g heatmaps in SliceViewer
- [ ] κ_G slider: 0 (no coupling) → KK value → Li-Torr value → user-defined
- [ ] Validate GEM alone: frame-dragging profile vs Gravity Probe B data

### Phase 5 — Advanced Observables & UI (week 11–14)
**Goal**: Full EED observable suite. Multiple entities. Motion.

- [ ] Modified Poynting vector P = E×B − EC (GPU kernel)
- [ ] Holonomy integral: ∮A·dl around user-defined path
- [ ] Magnetic helicity: ∫A·B d³x (GPU reduction)
- [ ] Multiple-entity manager: add/remove/move coil entities
- [ ] Quasi-static motion: recompute Biot-Savart at each user-set position
- [ ] Export: field data to HDF5 / numpy archive
- [ ] Hypothesis log updated for v2 fields

### Phase 6 — Validation & Hardening (week 14–16)
**Goal**: Every physics claim is verifiable. No silent wrong answers.

- [ ] Full validation suite: analytic tests for each physics module
- [ ] C-field conservation test (□C = 0 in vacuum — check residual)
- [ ] Energy conservation test (modified Poynting continuity)
- [ ] Newton's third law test (Whittaker force, open circuit)
- [ ] AB holonomy test: toroid geometry, field-free region, non-zero ∮A·dl
- [ ] Performance profiling: target <100ms static solve, <10ms FDTD step

---

## 5. File Structure (v2)

```
EEDSolver/
├── V2_PLAN.md                 ← this file
├── V2_PHYSICS.md              ← equations, notation, derivation notes
├── FIELD_THEORY.md            ← updated: EED + GEM theory (keep existing, extend)
├── ARCHITECTURE.md            ← archive label: "v1 reference"
│
├── solver-gpu/                ← NEW: Rust/wgpu GPU solver crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs             ← public API: solve(), SolveRequest, SolveResult
│   │   ├── context.rs         ← GpuContext (wgpu device/queue/adapter)
│   │   ├── grid/
│   │   │   ├── mod.rs
│   │   │   ├── yee.rs         ← YeeGrid: dimensions, buffer layout, indexing
│   │   │   └── buffers.rs     ← GPU buffer allocation (phi, A, E, B, C, Phi_g, A_g)
│   │   ├── biot/
│   │   │   ├── mod.rs
│   │   │   ├── wire.rs        ← WireEntity, segment list, coil builders
│   │   │   └── kernel.rs      ← dispatch biot.wgsl
│   │   ├── solver/
│   │   │   ├── mod.rs
│   │   │   ├── static_cg.rs   ← Preconditioned CG (static mode)
│   │   │   └── fdtd.rs        ← FDTD time stepper
│   │   ├── physics/
│   │   │   ├── mod.rs
│   │   │   ├── eed.rs         ← EED update dispatch
│   │   │   ├── gem.rs         ← GEM update dispatch + coupling
│   │   │   └── observables.rs ← Poynting, holonomy, helicity
│   │   └── postproc/
│   │       ├── mod.rs
│   │       ├── slice.rs       ← 2D slice extraction
│   │       ├── volume.rs      ← 3D volume extraction
│   │       └── maxima.rs      ← field maximum search
│   └── shaders/
│       ├── biot.wgsl          ← Biot-Savart sum
│       ├── fdtd_em.wgsl       ← EED potential-primary FDTD update
│       ├── fdtd_gem.wgsl      ← GEM FDTD update
│       ├── c_field.wgsl       ← C-field dynamics
│       ├── derive_fields.wgsl ← E = −∇φ − ∂A/∂t, B = ∇×A, C = ∇·A + ...
│       └── postproc.wgsl      ← slice/volume/holonomy extraction
│
├── src-tauri/
│   ├── Cargo.toml             ← add solver-gpu workspace dep; add wgpu
│   └── src/
│       ├── main.rs            ← remove sidecar spawn
│       ├── commands.rs        ← call solver-gpu directly (no HTTP)
│       ├── lib.rs
│       └── types.rs           ← v2 SolveRequest/SolveResult (Rust)
│                                 [solver_client.rs DELETED]
│
├── src/                       ← frontend (mostly unchanged)
│   ├── components/
│   │   ├── GeometryPanel/     ← add: multiple entities, GEM coupling sliders
│   │   ├── SliceViewer/       ← add: C field, E_g, B_g, Poynting display
│   │   ├── VolumeViewer/      ← unchanged
│   │   ├── HypothesisLog/     ← update for v2 fields
│   │   ├── GEMPanel/          ← NEW: gravitational sector controls + display
│   │   ├── HolonomyViewer/    ← NEW: path integral visualization
│   │   └── EntityManager/     ← NEW: add/remove/configure coil entities
│   └── lib/
│       ├── api.ts             ← update types only
│       ├── fieldTypes.ts      ← v2 field names (add C, E_g, B_g, Poynting)
│       └── colormap.ts        ← add signed diverging map for C field
│
└── solver/                    ← ARCHIVE (do not delete until Phase 2 validation done)
    └── [v1 Python code, kept as reference during validation phase]
```

---

## 6. Key Architectural Decisions

### Decision 1: Solver in-process, not sidecar
**v1**: Python FastAPI server, Tauri HTTP client, subprocess lifecycle management.
**v2**: Rust crate compiled into Tauri binary. One process, direct function call.
**Rationale**: GPU access from a subprocess via HTTP is fragile. wgpu must be called
from the same process that will own the GPU context. Eliminates 20–200ms HTTP
overhead per solve call, removes Python runtime dependency, removes port management.

### Decision 2: Potential-primary FDTD, not field-primary
**Alternative rejected**: Standard FDTD evolves (E, B). Fast, well-understood.
**Chosen**: Evolve (φ, A). C = ∇·A + (1/c²)∂φ/∂t is available at every cell.
**Rationale**: The entire point of EED is that C is a physical field. Field-primary FDTD
structurally cannot represent it — the Lorenz gauge suppression happens at the
algorithmic level, not just the gauge choice level.

### Decision 3: Biot-Savart for sources, not meshed volumes
**Alternative rejected**: Volume mesh of wire, FEM current density.
**Chosen**: Analytical wire-segment Biot-Savart on GPU.
**Rationale**: Exact for thin-wire limit. Embarrassingly parallel. Motion-compatible
(no remeshing). Eliminates entire Gmsh dependency and the 100:1 mesh ratio problem
that broke AMS. Wire-segment approximation is standard in computational magnetics.

### Decision 4: Yee grid (structured Cartesian)
**Alternative rejected**: Unstructured FEM mesh.
**Chosen**: Regular Cartesian Yee grid.
**Rationale**: GPU SIMD efficiency requires uniform memory access patterns.
Unstructured mesh requires indirect addressing (gather/scatter) that underperforms on GPU.
The Yee grid is equivalent to Nédélec elements on structured meshes — same H(curl)
conforming structure, but trivially parallelizable. Resolution tradeoff (need fine
cells globally to resolve wire) is addressed by Biot-Savart handling the wire analytically.

### Decision 5: GEM coupling is parameterized, not hardcoded
**Rationale**: The coupling constant κ_G between the C-field and the gravitational
sector is theoretically motivated (Kaluza-Klein) but experimentally unconfirmed.
Hardcoding it would either make unverified physics claims or disable the feature.
Parameterizing it lets the simulator be a prediction tool: "if coupling = X, you
would see Y." The physics is correct; the magnitude is the open question.

---

## 7. Open Questions

1. **Subgridding near wire**: Biot-Savart handles the wire source analytically, but
   the field response near the wire (high gradients in A) may need local grid refinement.
   Adaptive mesh refinement on Cartesian grids (octree) is an option for v2.1.

2. **Absorbing BCs for C-field**: Standard PML is designed for transverse EM waves.
   The C-field (scalar-longitudinal mode) propagates differently. Need to verify
   that PML doesn't reflect SLW back into the domain spuriously.

3. **Static vs time-domain for EED**: In the static limit, the EED equations reduce
   to elliptic PDEs — same as v1 but on a structured grid. Should the static solver
   use CG directly, or take the time-domain solver to steady state? CG is faster but
   requires the static form to be re-derived. Time-to-steady-state is simpler to implement
   but slower. Plan: implement CG for Phase 2, time-domain for Phase 3.

4. **κ_G physical range**: The Kaluza-Klein coupling is G/c² ≈ 7×10⁻²⁸ (dimensionless).
   The Li-Torr coupling is 2m_e/e ≈ 1.14×10⁻¹¹. NASA BPP found no effects at
   device scale. What range makes the GEM sector non-trivially visible in simulation?
   Answer: any κ_G > 0 will show *structure* in the gravitational fields; experimental
   significance is a separate question from visual interestingness.

5. **Holonomy path definition**: ∮A·dl requires a user-defined path. UX for specifying
   arbitrary closed paths in 3D is non-trivial. MVP: predefined paths (z-axis circle,
   toroidal loop, poloidal loop). User-defined paths in v2.1.

6. **Multiple entities interaction**: If two coils are close, their vector potentials
   superpose linearly in the source term. The field response to the combined source is
   nonlinear (because EED coupling terms are nonlinear). The Biot-Savart superposition
   is correct; the coupled EED solve handles the interaction automatically.

---

## 8. Physics Constants Reference

```
μ₀  = 4π × 10⁻⁷   H/m       (vacuum permeability)
ε₀  = 8.854 × 10⁻¹² F/m     (vacuum permittivity)
c   = 2.998 × 10⁸  m/s       (speed of light)
G   = 6.674 × 10⁻¹¹ m³/kg·s² (gravitational constant)
m_e = 9.109 × 10⁻³¹ kg       (electron mass)
e   = 1.602 × 10⁻¹⁹ C        (elementary charge)

EED coupling constants (free parameters, set by experiment):
α   [1/m]           scalar mass / Yukawa range
β   [dimensionless] A→φ coupling
γ   [dimensionless] φ→A coupling (γ=1 → EED; γ=0 → standard Maxwell)
κ_G [dimensionless] C-field → GEM coupling (Kaluza-Klein: ~G/c²; Li-Torr: ~m_e/e)
```
