//! Tauri-side types.
//!
//! All solver physics types are canonical in `solver_gpu::types` and
//! re-exported here.  Only app-level types (hypothesis persistence, etc.)
//! are defined directly in this file.

#[allow(unused_imports)]
pub use solver_gpu::{
    // Geometry
    CoilEntity, CoilParams, CoilType,
    // Physics params
    EedParams, GemParams,
    // Solver config
    SolverConfig, SolverMode,
    // Output spec
    SliceAxis, FieldName, SliceRequest, HolonomyPath,
    // Request
    SolveRequest,
    // Results
    SliceData, VolumeData, FieldMaximum, HolonomyResult, SolveResult,
    // Status
    SolverState, SolverStatus,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Hypothesis log entry (Tauri-specific, persisted to ~/.oracle/)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisEntry {
    pub id:        String,              // timestamp + name slug
    pub name:      String,
    pub timestamp: DateTime<Utc>,
    pub request:   SolveRequest,
    pub maxima:    Vec<FieldMaximum>,
    pub notes:     Option<String>,
}
