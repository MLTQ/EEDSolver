//! Shared request/result types for the Oracle v2 GPU solver.
//!
//! These are the canonical definitions.  Tauri's `src-tauri/src/types.rs`
//! re-exports them (no duplication); the TypeScript frontend mirrors them
//! in `src/lib/fieldTypes.ts`.
//!
//! Design principles:
//!  - All physical quantities in SI units unless noted.
//!  - `SolveRequest` describes the *physics* and *output* desired, not
//!    solver internals (no "mesh resolution" — that's a grid_cells count).
//!  - GEM coupling is parameterised, never hardcoded.
//!  - The C field (∇·A + (1/c²)∂φ/∂t) is a first-class output.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Coil / entity geometry
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CoilType {
    Solenoid,
    Toroid,
    ToroidPoloidal,
    FlatSpiral,
    Rodin,
}

/// A single current-carrying entity in the simulation.
/// Multiple entities are superposed in the Biot-Savart source term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoilEntity {
    pub coil: CoilParams,
    /// Centre position [x, y, z] in metres.
    pub position_m:    [f64; 3],
    /// Orientation quaternion [x, y, z, w], normalised.  Default = identity.
    pub orientation:   [f64; 4],
    /// If true, this entity is treated as a superconducting body for the
    /// Li-Torr gravitomagnetic London moment coupling (GEM sector).
    pub superconducting: bool,
}

impl Default for CoilEntity {
    fn default() -> Self {
        Self {
            coil: CoilParams::default(),
            position_m:    [0.0, 0.0, 0.0],
            orientation:   [0.0, 0.0, 0.0, 1.0],
            superconducting: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoilParams {
    pub coil_type:     CoilType,
    pub radius_m:      f64,
    pub turns:         u32,
    pub pitch_m:       f64,
    pub wire_radius_m: f64,
    #[serde(rename = "current_A")]
    pub current_a:     f64,
}

impl Default for CoilParams {
    fn default() -> Self {
        Self {
            coil_type:     CoilType::Solenoid,
            radius_m:      0.05,
            turns:         10,
            pitch_m:       0.005,
            wire_radius_m: 0.001,
            current_a:     1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EED physics parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Extended Electrodynamics (EED) coupling constants.
///
/// Stueckelberg Lagrangian parameter γ=1, m=0 (gauge-free, massless C field).
///
/// * `alpha`  [1/m]  — Yukawa scalar mass.  λ = 1/α is the decay length.
///                      α = 0 → massless scalar, maximum predicted range.
/// * `beta`           — A→φ coupling in the φ equation (∫β·div(A)·ψ dx).
/// * `gamma`          — φ→A coupling in the A equation (∫γ·∇φ·v dx).
///                      γ=1 selects the full EED theory; γ=0 → standard Maxwell.
///
/// When β=γ=0 the system decouples to independent Maxwell + isolated scalar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EedParams {
    pub alpha: f64,
    pub beta:  f64,
    pub gamma: f64,
}

impl Default for EedParams {
    fn default() -> Self {
        // γ=1 → full EED (Stueckelberg); β=0.1 placeholder until calibrated
        Self { alpha: 0.0, beta: 0.1, gamma: 1.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GEM (gravitoelectromagnetic) coupling parameters
// ─────────────────────────────────────────────────────────────────────────────

/// GEM coupling parameters.
///
/// The C-field couples into the gravitomagnetic sector via the
/// Kaluza-Klein identification (∇·A encodes gravitational DOF).
///
/// # Coupling constant guide
///
/// | Mode            | κ_g value            | Physical basis               |
/// |-----------------|----------------------|------------------------------|
/// | Disabled        | 0.0                  | Pure EM, no GEM coupling     |
/// | KK prediction   | G/c² ≈ 7.4e-28       | Kaluza-Klein (weak)          |
/// | Li-Torr         | 2·mₑ/e ≈ 1.14e-11   | Superconductor London moment |
/// | User-defined    | any                  | Exploration / hypothesis     |
///
/// The simulation never asserts a specific value is correct — κ_g is
/// a free parameter. The physics structure (C→GEM coupling) is the claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemParams {
    /// Enable the GEM gravitational sector.
    pub enabled:       bool,
    /// C-field → GEM coupling constant κ_g [dimensionless].
    pub kappa_g:       f64,
    /// Enable Li-Torr gravitomagnetic London moment for superconducting entities.
    /// When true, rotating superconducting entities source B_g = -(2mₑ/e)·ω.
    pub li_torr_mode:  bool,
}

impl Default for GemParams {
    fn default() -> Self {
        Self {
            enabled:      false,
            kappa_g:      0.0,
            li_torr_mode: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solver configuration
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum SolverMode {
    /// Static (time-independent) solve.  Finds the steady-state fields.
    /// Uses preconditioned CG on GPU.  Fast — sub-100ms for typical grids.
    Static,
    /// Full potential-primary FDTD.  Steps forward in time.
    /// Captures SLW propagation, transient coupling, Faraday cage penetration.
    TimeDomain {
        /// Time step [s].  Must satisfy CFL: dt ≤ dx / (c·√3).
        dt_s:    f64,
        /// Number of time steps to advance per solve call.
        n_steps: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub mode:           SolverMode,
    /// Number of Yee cells per axis.  Domain is a cube:
    /// cells_per_axis³ total cells.  Typical: 64 (fast), 128, 256 (accurate).
    pub cells_per_axis: u32,
    /// Half-width of the simulation domain [m].  Cells span [−r, +r]³.
    pub domain_radius_m: f64,
    /// Impose Lorenz gauge (set C=0 each step) for Maxwell baseline comparison.
    pub lorenz_gauge:   bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            mode:            SolverMode::Static,
            cells_per_axis:  64,
            domain_radius_m: 0.2,
            lorenz_gauge:    false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output specification
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SliceAxis { X, Y, Z }

/// All field quantities the solver can output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FieldName {
    // ── EED electromagnetic potentials (primary state) ──
    /// Scalar potential φ [V]
    #[serde(rename = "phi")]             Phi,
    /// |A| vector potential magnitude [V·s/m]
    #[serde(rename = "A_magnitude")]     AMagnitude,

    // ── EED derived EM fields ──
    /// |E| = |−∇φ − ∂A/∂t|  [V/m]
    #[serde(rename = "E_magnitude")]     EMagnitude,
    /// |B| = |∇×A|  [T]
    #[serde(rename = "B_magnitude")]     BMagnitude,
    /// |J| current density magnitude [A/m²]
    #[serde(rename = "J_magnitude")]     JMagnitude,

    // ── EED scalar (the deleted degree of freedom) ──
    /// C = ∇·A + (1/c²)∂φ/∂t  [1/m]
    /// This is Maxwell's "seventh component" — zero under Lorenz gauge,
    /// dynamical in EED (γ=1).
    #[serde(rename = "C_field")]         CField,

    // ── EED energy / momentum ──
    /// Modified Poynting vector magnitude |P| = |E×B − E·C|  [W/m²]
    #[serde(rename = "poynting_mag")]    PoyntingMag,
    /// Modified energy density u = ½(E² + B² + C²)  [J/m³]
    #[serde(rename = "energy_density")]  EnergyDensity,

    // ── GEM gravitational sector ──
    /// Gravitational scalar potential Φ_g  [m²/s²]
    #[serde(rename = "phi_g")]           PhiG,
    /// |E_g| gravitoelectric field magnitude  [m/s²]
    #[serde(rename = "E_g_magnitude")]   EgMagnitude,
    /// |B_g| gravitomagnetic field magnitude  [1/s]
    #[serde(rename = "B_g_magnitude")]   BgMagnitude,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceRequest {
    pub axis:       SliceAxis,
    /// Normalised position along selected axis, 0–1.
    pub position:   f64,
    pub field:      FieldName,
    /// Grid points per side in the extracted slice.
    pub resolution: u32,
}

impl Default for SliceRequest {
    fn default() -> Self {
        Self {
            axis:       SliceAxis::Z,
            position:   0.5,
            field:      FieldName::Phi,
            resolution: 128,
        }
    }
}

/// Request a holonomy line integral ∮ A·dl around a predefined closed path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum HolonomyPath {
    /// Circle in the XY plane at the given z-coordinate and radius.
    ZCircle { z_m: f64, radius_m: f64 },
    /// Toroidal loop (around the major axis of a torus at the given position).
    ToroidalLoop { centre_m: [f64; 3], major_radius_m: f64 },
    /// Poloidal loop (around the tube of a torus).
    PoloidalLoop  { centre_m: [f64; 3], major_radius_m: f64, minor_radius_m: f64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level request
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveRequest {
    /// One or more current-carrying entities.  Their Biot-Savart contributions
    /// are superposed in the source term.
    pub entities:       Vec<CoilEntity>,
    pub eed:            EedParams,
    pub gem:            GemParams,
    pub solver:         SolverConfig,
    pub slices:         Vec<SliceRequest>,
    pub request_volume: bool,
    pub volume_field:   FieldName,
    /// Line integrals to compute (Aharonov-Bohm phases, holonomy).
    pub holonomy_paths: Vec<HolonomyPath>,
}

impl Default for SolveRequest {
    fn default() -> Self {
        Self {
            entities:       vec![CoilEntity::default()],
            eed:            EedParams::default(),
            gem:            GemParams::default(),
            solver:         SolverConfig::default(),
            slices:         vec![SliceRequest::default()],
            request_volume: true,
            volume_field:   FieldName::Phi,
            holonomy_paths: vec![],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceData {
    pub axis:        SliceAxis,
    pub position:    f64,
    pub field:       FieldName,
    pub shape:       [u32; 2],    // [rows, cols]
    pub data:        Vec<f32>,    // flattened row-major (f32 — GPU native)
    pub x_range:     [f64; 2],   // [min, max] metres
    pub y_range:     [f64; 2],   // [min, max] metres
    pub field_min:   f64,
    pub field_max:   f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeData {
    pub field:    FieldName,
    pub shape:    [u32; 3],      // [nx, ny, nz]
    pub data:     Vec<f32>,      // normalised to [0,1], flat row-major
    pub x_range:  [f64; 2],
    pub y_range:  [f64; 2],
    pub z_range:  [f64; 2],
    pub field_min: f64,
    pub field_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMaximum {
    pub field:        FieldName,
    pub max_value:    f64,
    pub max_location: [f64; 3],  // [x, y, z] metres
}

/// Result of a holonomy path integral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomyResult {
    pub path:  HolonomyPath,
    /// ∮ A·dl  [V·s/m · m = V·s]  (proportional to AB phase via e/ℏ)
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResult {
    pub solve_time_s: f64,
    /// Total Yee grid cells (cells_per_axis³).
    pub grid_cells:   u64,
    pub slices:       Vec<SliceData>,
    pub volume:       Option<VolumeData>,
    pub maxima:       Vec<FieldMaximum>,
    pub holonomies:   Vec<HolonomyResult>,
    pub warnings:     Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Solver status
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SolverState { Initialising, Ready, Solving, Error }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverStatus {
    pub state:   SolverState,
    pub message: String,
    /// GPU adapter description (e.g. "Apple M3 Pro").
    pub gpu_name: Option<String>,
}
