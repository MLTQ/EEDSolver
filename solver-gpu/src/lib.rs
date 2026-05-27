//! Oracle GPU solver — public API.
//!
//! # Architecture
//! `OracleSolver` is created once at app startup and stored as Tauri state.
//! Each solve call is async and runs on the GPU.
//!
//! # Phase status
//!   Phase 0 (current): GPU init, stub solve (returns empty result)
//!   Phase 1: Biot-Savart engine
//!   Phase 2: Static EED (CG on GPU)
//!   Phase 3: Time-domain FDTD
//!   Phase 4: GEM coupled sector
//!   Phase 5: Observables (Poynting, holonomy, helicity)

pub mod biot;
pub mod context;
pub mod error;
pub mod grid;
pub mod physics;
pub mod postproc;
pub mod types;

pub use context::GpuContext;
pub use error::SolverError;
pub use types::*;

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

// ─────────────────────────────────────────────────────────────────────────────
// OracleSolver
// ─────────────────────────────────────────────────────────────────────────────

/// The Oracle GPU solver.
///
/// Create with `OracleSolver::new().await`, then call `.solve(request).await`.
/// Store as Tauri managed state — it is `Send + Sync`.
pub struct OracleSolver {
    ctx:   GpuContext,
    state: Arc<RwLock<InternalState>>,
}

#[derive(Debug, Default)]
struct InternalState {
    /// True once the solver has initialised all GPU resources.
    ready: bool,
}

impl OracleSolver {
    /// Initialise the GPU and all persistent solver resources.
    /// Call once at app startup.  Returns an error if no GPU is available.
    pub async fn new() -> Result<Self, SolverError> {
        let ctx = GpuContext::new().await?;
        log::info!("OracleSolver ready on: {}", ctx.adapter_name());

        let solver = Self {
            ctx,
            state: Arc::new(RwLock::new(InternalState { ready: true })),
        };
        Ok(solver)
    }

    /// Human-readable GPU name for status reporting.
    pub fn gpu_name(&self) -> String {
        self.ctx.adapter_name()
    }

    /// Run a solve.  Returns a `SolveResult` whose fields are populated
    /// progressively as each phase is implemented.
    pub async fn solve(&self, request: &SolveRequest) -> Result<SolveResult, SolverError> {
        let t0 = Instant::now();

        // Validate request
        if request.entities.is_empty() {
            return Err(SolverError::InvalidRequest(
                "At least one coil entity is required".into(),
            ));
        }

        let cfg   = &request.solver;
        let grid  = grid::YeeGrid::new(cfg.cells_per_axis, cfg.domain_radius_m);

        log::info!(
            "Solve: {}³ grid, dx={:.3}mm, CFL dt={:.2}ns",
            grid.n,
            grid.dx * 1e3,
            grid.cfl_dt() * 1e9,
        );

        let mut warnings = Vec::<String>::new();

        // ── Phase 1: Biot-Savart ─────────────────────────────────────────────
        // TODO: GPU Biot-Savart dispatch
        warnings.push("Phase 1 (Biot-Savart) not yet implemented — A field is zero.".into());

        // ── Phase 2: Static EED ──────────────────────────────────────────────
        // TODO: GPU CG solve for φ and A
        warnings.push("Phase 2 (static EED solver) not yet implemented.".into());

        // ── Phase 3: FDTD ────────────────────────────────────────────────────
        if matches!(cfg.mode, SolverMode::TimeDomain { .. }) {
            warnings.push("Phase 3 (FDTD time-domain) not yet implemented.".into());
        }

        // ── Phase 4: GEM ─────────────────────────────────────────────────────
        if request.gem.enabled {
            warnings.push("Phase 4 (GEM sector) not yet implemented.".into());
        }

        let solve_time = t0.elapsed().as_secs_f64();

        Ok(SolveResult {
            solve_time_s: solve_time,
            grid_cells:   grid.total_cells(),
            slices:       vec![],
            volume:       None,
            maxima:       vec![],
            holonomies:   vec![],
            warnings,
        })
    }
}

// OracleSolver can cross thread boundaries (Tauri requires Send + Sync)
// Safety: GpuContext is Arc-backed and wgpu types are Send + Sync.
unsafe impl Send for OracleSolver {}
unsafe impl Sync for OracleSolver {}
