//! Oracle GPU solver — public API.
//!
//! # Architecture
//! `OracleSolver` is created once at app startup and stored as Tauri state.
//! Each solve call is async and dispatches GPU compute shaders via wgpu.
//!
//! # Phase status
//!   Phase 0 ✓  GPU init, Tauri in-process integration
//!   Phase 1 ✓  Biot-Savart engine: A field from coil geometry
//!   Phase 2    Static EED CG solver (φ from ρ, A from J)
//!   Phase 3    Time-domain FDTD (potential-primary leapfrog)
//!   Phase 4    GEM coupled gravitational sector
//!   Phase 5    Observables: Poynting, holonomy, helicity

pub mod biot;
pub mod context;
pub mod error;
pub mod grid;
pub mod physics;
pub mod postproc;
pub mod types;

pub use context::GpuContext;
pub use error::SolverError;
pub use grid::GpuGridState;
pub use types::*;

use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::Instant;

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
    ready: bool,
}

impl OracleSolver {
    /// Initialise the GPU.  Call once at app startup.
    pub async fn new() -> Result<Self, SolverError> {
        let ctx = GpuContext::new().await?;
        log::info!("OracleSolver ready on: {}", ctx.adapter_name());
        Ok(Self {
            ctx,
            state: Arc::new(RwLock::new(InternalState { ready: true })),
        })
    }

    /// Human-readable GPU adapter name.
    pub fn gpu_name(&self) -> String { self.ctx.adapter_name() }

    /// Run a full solve and return field results.
    pub async fn solve(&self, request: &SolveRequest) -> Result<SolveResult, SolverError> {
        let t0 = Instant::now();

        if request.entities.is_empty() {
            return Err(SolverError::InvalidRequest(
                "At least one coil entity is required".into(),
            ));
        }

        let cfg  = &request.solver;
        let grid = grid::YeeGrid::new(cfg.cells_per_axis, cfg.domain_radius_m);

        log::info!(
            "Solve: {}³ grid ({} vertices/axis), dx={:.3}mm, CFL dt={:.2}ns",
            grid.n, grid.n + 1,
            grid.dx * 1e3,
            grid.cfl_dt() * 1e9,
        );

        let mut warnings = Vec::<String>::new();

        // ── Phase 1: Biot-Savart ─────────────────────────────────────────────
        // Convert all coil entities to GPU wire segments.
        let segments: Vec<biot::WireSegment> = request.entities.iter()
            .flat_map(|e| biot::entity_to_segments(e))
            .collect();

        log::info!("Total wire segments: {}", segments.len());

        if segments.is_empty() {
            warnings.push(
                "No wire segments generated — check coil parameters (radius, turns, pitch).".into()
            );
        }

        // Allocate GPU field buffers.
        let gstate = GpuGridState::new(&self.ctx, &grid);

        // Dispatch Biot-Savart → fills a_vec.
        gstate.run_biot_savart(&self.ctx, &grid, &segments)?;

        // Dispatch field derivation → fills b_vec and c_fld.
        gstate.run_derive_fields(&self.ctx, &grid)?;

        // ── Phase 2: Static EED ──────────────────────────────────────────────
        if request.eed.gamma > 0.0 {
            warnings.push(
                "Phase 2 (static EED CG solver) not yet implemented — φ is zero.".into()
            );
        }

        // ── Phase 3: FDTD ────────────────────────────────────────────────────
        if matches!(cfg.mode, SolverMode::TimeDomain { .. }) {
            warnings.push("Phase 3 (FDTD time-domain) not yet implemented.".into());
        }

        // ── Phase 4: GEM ─────────────────────────────────────────────────────
        if request.gem.enabled {
            warnings.push("Phase 4 (GEM sector) not yet implemented.".into());
        }

        // ── Post-processing ──────────────────────────────────────────────────
        let slices = postproc::extract_slices(
            &self.ctx, &gstate, &grid, &request.slices,
        )?;

        let maxima = postproc::compute_maxima(&self.ctx, &gstate, &grid)?;

        let holonomies = postproc::compute_holonomies(
            &self.ctx, &gstate, &grid, &request.holonomy_paths,
        );

        let solve_time = t0.elapsed().as_secs_f64();
        log::info!("Solve complete in {:.3}s", solve_time);

        Ok(SolveResult {
            solve_time_s: solve_time,
            grid_cells:   grid.total_cells(),
            slices,
            volume:       None,   // Phase 2+
            maxima,
            holonomies,
            warnings,
        })
    }
}

// OracleSolver must be Send + Sync for Tauri managed state.
// Safety: GpuContext is Arc-backed; wgpu types are Send + Sync.
unsafe impl Send for OracleSolver {}
unsafe impl Sync for OracleSolver {}
