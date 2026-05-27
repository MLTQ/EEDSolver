//! Tauri commands exposed to the frontend via `window.__TAURI__.invoke()`.
//!
//! All commands are async and return `Result<T, String>` — Tauri serialises
//! both sides to JSON automatically.
//!
//! # Command list
//!   `solve`             — full GPU solve pipeline
//!   `get_solver_status` — GPU readiness
//!   `save_hypothesis`   — persist a named run to ~/.oracle/hypotheses/
//!   `load_hypotheses`   — load all saved runs
//!   `delete_hypothesis` — remove a run by id

use std::fs;
use std::path::PathBuf;

use chrono::Utc;
use tauri::State;

use solver_gpu::OracleSolver;
use crate::types::{
    HypothesisEntry, SolveRequest, SolveResult, SolverState, SolverStatus,
};

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

/// Run a full GPU field solve.
/// The frontend must keep the solve button disabled and show a spinner.
#[tauri::command]
pub async fn solve(
    request:  SolveRequest,
    solver:   State<'_, OracleSolver>,
) -> Result<SolveResult, String> {
    solver.solve(&request).await.map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// get_solver_status
// ---------------------------------------------------------------------------

/// Return current GPU solver readiness.
/// Frontend polls this on launch to know when to enable the Solve button.
#[tauri::command]
pub async fn get_solver_status(
    solver: State<'_, OracleSolver>,
) -> Result<SolverStatus, String> {
    Ok(SolverStatus {
        state:    SolverState::Ready,
        message:  format!("GPU solver ready on {}", solver.gpu_name()),
        gpu_name: Some(solver.gpu_name()),
    })
}

// ---------------------------------------------------------------------------
// save_hypothesis
// ---------------------------------------------------------------------------

/// Save a named run to ~/.oracle/hypotheses/<id>.json.
/// Returns the entry id.
#[tauri::command]
pub async fn save_hypothesis(
    name:    String,
    request: SolveRequest,
    result:  SolveResult,
    notes:   Option<String>,
) -> Result<String, String> {
    let dir = hypothesis_dir()?;
    fs::create_dir_all(&dir).map_err(|e| format!("Cannot create hypothesis dir: {e}"))?;

    let timestamp = Utc::now();
    let id = format!("{}-{}", timestamp.format("%Y%m%dT%H%M%S"), slug(&name));

    let entry = HypothesisEntry {
        id:        id.clone(),
        name,
        timestamp,
        request,
        maxima:    result.maxima,
        notes,
    };

    let path = dir.join(format!("{id}.json"));
    let json = serde_json::to_string_pretty(&entry)
        .map_err(|e| format!("Serialisation failed: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("Write failed: {e}"))?;

    log::info!("Saved hypothesis '{}' → {}", entry.name, path.display());
    Ok(id)
}

// ---------------------------------------------------------------------------
// load_hypotheses
// ---------------------------------------------------------------------------

/// Load all saved hypothesis entries, sorted newest-first.
#[tauri::command]
pub async fn load_hypotheses() -> Result<Vec<HypothesisEntry>, String> {
    let dir = hypothesis_dir()?;
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut entries: Vec<HypothesisEntry> = fs::read_dir(&dir)
        .map_err(|e| format!("Cannot read hypothesis dir: {e}"))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path  = entry.path();
            if path.extension()?.to_str()? != "json" { return None; }
            let content = fs::read_to_string(&path).ok()?;
            match serde_json::from_str::<HypothesisEntry>(&content) {
                Ok(h)  => Some(h),
                Err(e) => {
                    log::warn!("Failed to parse hypothesis {}: {e}", path.display());
                    None
                }
            }
        })
        .collect();

    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(entries)
}

// ---------------------------------------------------------------------------
// delete_hypothesis
// ---------------------------------------------------------------------------

/// Remove a hypothesis entry by id.
#[tauri::command]
pub async fn delete_hypothesis(id: String) -> Result<(), String> {
    let dir  = hypothesis_dir()?;
    let path = dir.join(format!("{id}.json"));
    if path.exists() {
        fs::remove_file(&path).map_err(|e| format!("Delete failed: {e}"))?;
        log::info!("Deleted hypothesis {id}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hypothesis_dir() -> Result<PathBuf, String> {
    dirs::home_dir()
        .ok_or_else(|| "Cannot find home directory".to_string())
        .map(|h| h.join(".oracle").join("hypotheses"))
}

fn slug(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c.to_lowercase().next().unwrap() } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}
