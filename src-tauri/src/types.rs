//! Rust mirror of solver/models/params.py.
//! Sync rule: any change here must be matched in params.py AND src/lib/fieldTypes.ts.
//!
//! serde names use snake_case to match Python/JSON convention throughout.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CoilType {
    Solenoid,
    Toroid,
    ToroidPoloidal,
    FlatSpiral,
    Rodin,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FormulationType {
    ScalarOnly,
    MaxwellOnly,
    EedCoupled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MeshResolution {
    Coarse,
    Medium,
    Fine,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SliceAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FieldName {
    #[serde(rename = "phi")]
    Phi,
    #[serde(rename = "A_magnitude")]
    AMagnitude,
    #[serde(rename = "B_magnitude")]
    BMagnitude,
    #[serde(rename = "J_magnitude")]
    JMagnitude,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoilParams {
    pub radius_m: f64,
    pub turns: u32,
    pub pitch_m: f64,
    pub wire_radius_m: f64,
    #[serde(rename = "current_A")]
    pub current_a: f64,
    pub coil_type: CoilType,
}

impl Default for CoilParams {
    fn default() -> Self {
        Self {
            radius_m: 0.05,
            turns: 10,
            pitch_m: 0.005,
            wire_radius_m: 0.001,
            current_a: 1.0,
            coil_type: CoilType::Solenoid,
        }
    }
}

/// EED coupling constants. Free parameters — constrained by experiment.
/// α [1/m]: Yukawa scalar mass. λ = 1/α decay length. α=0 → massless (maximum range).
/// β, γ [dimensionless]: cross-coupling strengths. β=γ=0 → decoupled Maxwell + scalar.
/// TODO: VERIFY β/γ term structure against DDOF paper once citation is confirmed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EedParams {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Default for EedParams {
    fn default() -> Self {
        Self { alpha: 0.0, beta: 0.1, gamma: 0.1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceRequest {
    pub axis: SliceAxis,
    pub position: f64,
    pub field: FieldName,
    pub resolution: u32,
}

impl Default for SliceRequest {
    fn default() -> Self {
        Self {
            axis: SliceAxis::Z,
            position: 0.5,
            field: FieldName::Phi,
            resolution: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveRequest {
    pub coil: CoilParams,
    pub eed: EedParams,
    pub domain_radius_m: f64,
    pub mesh_resolution: MeshResolution,
    pub formulation: FormulationType,
    pub slices: Vec<SliceRequest>,
    pub request_volume: bool,
}

impl Default for SolveRequest {
    fn default() -> Self {
        Self {
            coil: CoilParams::default(),
            eed: EedParams::default(),
            domain_radius_m: 0.2,
            mesh_resolution: MeshResolution::Coarse,
            formulation: FormulationType::ScalarOnly,
            slices: vec![SliceRequest::default()],
            request_volume: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceData {
    pub axis: SliceAxis,
    pub position: f64,
    pub field: FieldName,
    pub shape: Vec<u32>,      // [rows, cols]
    pub data: Vec<f64>,       // Flattened row-major
    pub x_range: Vec<f64>,    // [min, max] meters
    pub y_range: Vec<f64>,    // [min, max] meters
    pub field_min: f64,
    pub field_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMaximum {
    pub field: FieldName,
    pub max_value: f64,
    pub max_location: Vec<f64>,  // [x, y, z] meters
}

/// 3D scalar field on a regular grid, normalized to [0, 1].
/// Used by the Three.js ray-marching volume viewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeData {
    pub field: FieldName,
    pub shape: Vec<u32>,        // [nx, ny, nz]
    pub data: Vec<f64>,         // Normalized to [0, 1], flat row-major
    pub x_range: Vec<f64>,
    pub y_range: Vec<f64>,
    pub z_range: Vec<f64>,
    pub field_min: f64,         // Pre-normalization minimum
    pub field_max: f64,         // Pre-normalization maximum
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResult {
    pub solve_time_s: f64,
    pub mesh_nodes: u64,
    pub slices: Vec<SliceData>,
    pub volume: Option<VolumeData>,
    pub maxima: Vec<FieldMaximum>,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Solver status (reported to frontend on startup / status poll)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SolverState {
    Starting,   // Solver process launched, not yet responding
    Ready,      // Health check passed
    Solving,    // Solve in progress
    Error,      // Failed to start or crashed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverStatus {
    pub state: SolverState,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Hypothesis log entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisEntry {
    pub id: String,                       // UUID-ish: timestamp + name slug
    pub name: String,
    pub timestamp: DateTime<Utc>,
    pub request: SolveRequest,
    pub maxima: Vec<FieldMaximum>,
    pub notes: Option<String>,
}
