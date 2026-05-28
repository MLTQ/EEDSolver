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
    #[allow(dead_code)]
    state: Arc<RwLock<InternalState>>,
}

#[derive(Debug, Default)]
struct InternalState {
    #[allow(dead_code)]
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
        // Capacitor entities produce no wire segments (φ is initialised below).
        let segments: Vec<biot::WireSegment> = request.entities.iter()
            .flat_map(|e| biot::entity_to_segments(e))
            .collect();

        // Collect lead attachment points per entity for the frontend.
        let lead_points: Vec<[[f64; 3]; 2]> = request.entities.iter()
            .map(|e| biot::entity_lead_points(e))
            .collect();

        // Collect AC-driven wire segments for J-grid computation.
        // Any entity with frequency_hz > 0 participates in AC injection.
        let ac_segments: Vec<biot::WireSegment> = request.entities.iter()
            .filter(|e| e.coil.frequency_hz > 0.0)
            .flat_map(|e| biot::entity_to_segments(e))
            .collect();

        log::info!("Total wire segments: {} (AC: {})", segments.len(), ac_segments.len());

        if segments.is_empty() && request.entities.iter().all(|e| {
            matches!(e.coil.coil_type, CoilType::CapacitorSymmetric | CoilType::CapacitorAsymmetric)
        }) {
            // All-capacitor configuration: no Biot-Savart needed, normal.
            log::info!("Capacitor-only configuration: skipping Biot-Savart");
        } else if segments.is_empty() {
            warnings.push(
                "No wire segments generated — check coil parameters (radius, turns, pitch).".into()
            );
        }

        // Allocate GPU field buffers.
        let mut gstate = GpuGridState::new(&self.ctx, &grid);

        // Dispatch Biot-Savart → fills a_vec (skipped for zero segments).
        gstate.run_biot_savart(&self.ctx, &grid, &segments)?;

        // ── Capacitor φ initialisation ────────────────────────────────────────
        // For capacitor entities, initialise φ with the plate field.
        // Multiple capacitors are superposed additively.
        for entity in &request.entities {
            match entity.coil.coil_type {
                CoilType::CapacitorSymmetric | CoilType::CapacitorAsymmetric => {
                    gstate.initialize_phi_capacitor(&self.ctx, &grid, entity);
                }
                _ => {}
            }
        }

        // ── AC J-source upload ────────────────────────────────────────────────
        // Pre-compute the normalised J₀ grid for AC injection (if any AC entities).
        let has_ac = request.entities.iter().any(|e| e.coil.frequency_hz > 0.0);
        if has_ac && !ac_segments.is_empty() {
            let n1     = (grid.n + 1) as usize;
            let origin = [-(grid.extent as f32); 3];
            let j_grid = biot::segments_to_j_grid(&ac_segments, n1, grid.dx as f32, origin);
            gstate.upload_j_source(&self.ctx, &j_grid);
            log::info!("J-source uploaded: {} AC segments", ac_segments.len());
        }

        // Dispatch field derivation → fills b_vec and c_fld.
        gstate.run_derive_fields(&self.ctx, &grid)?;

        // ── Phase 5a: EED observables (static baseline) ──────────────────────
        // Compute |P| and u from the static (pre-FDTD) fields.
        // Will be re-run after FDTD to reflect the evolved E-field if needed.
        gstate.run_observables(&self.ctx, &grid)?;

        // ── Phase 2: Static EED φ solver ─────────────────────────────────────
        // For closed-loop coils (solenoid, toroid, etc.) ∇·J = 0 everywhere,
        // so the rhs is zero and φ = 0 is the exact static EED solution.
        // This solver is here for correctness and future open-circuit / charge sources.
        let alpha_sq = (request.eed.alpha * request.eed.alpha) as f32;
        if alpha_sq > 0.0 {
            // rhs = −∇·J; zero for all current coil types (closed loops).
            let rhs = vec![0.0f32; gstate.scalar_len()];

            // Use PCG for larger grids (n > 32) — O(√κ) vs O(κ) convergence.
            // Fall back to Jacobi for very small debug grids.
            if grid.n > 32 {
                // 100 PCG iterations converge to 1e-6 relative tolerance
                // for typical EED problems on 64³–256³ grids.
                gstate.run_cg_phi(&self.ctx, &grid, &rhs, alpha_sq, 1e-6, 100)?;
                log::info!(
                    "Static EED φ (PCG): α={:.3} m⁻¹  λ={:.3} m",
                    request.eed.alpha,
                    1.0 / request.eed.alpha,
                );
            } else {
                let n_jacobi = (64u32).min(grid.n * 2);
                gstate.run_jacobi_phi(&self.ctx, &grid, &rhs, alpha_sq, n_jacobi)?;
                log::info!(
                    "Static EED φ (Jacobi): α={:.3} m⁻¹  λ={:.3} m  ({n_jacobi} iters)",
                    request.eed.alpha,
                    1.0 / request.eed.alpha,
                );
            }
        }

        // ── Phase 2b: Static EED A correction (Yukawa + γ coupling) ─────────
        // Apply only when α>0 (Yukawa) or γ≠0 with non-trivial φ.
        // Biot-Savart is exact for α=0, γ=0 — no correction needed then.
        let alpha_sq_f = (request.eed.alpha * request.eed.alpha) as f32;
        let gamma_f    = request.eed.gamma as f32;
        if alpha_sq_f > 0.0 || gamma_f != 0.0 {
            let n_jacobi_a = (64u32).min(grid.n * 2);
            gstate.run_jacobi_a_correction(
                &self.ctx, &grid, alpha_sq_f, gamma_f, n_jacobi_a,
            )?;
            // Re-derive B and C from the corrected A.
            gstate.run_derive_fields(&self.ctx, &grid)?;
            gstate.run_observables(&self.ctx, &grid)?;
        }

        // ── Phase 3: FDTD ────────────────────────────────────────────────────
        if let SolverMode::TimeDomain { dt_s, n_steps } = cfg.mode {
            let cfl_max = grid.cfl_dt() as f32;
            let dt      = (dt_s as f32).min(cfl_max);
            if dt < dt_s as f32 {
                warnings.push(format!(
                    "dt={:.3e}s clamped to CFL limit {:.3e}s (dx={:.3}mm, n={})",
                    dt_s, cfl_max, grid.dx * 1e3, grid.n,
                ));
            }
            // γ=0 → Lorenz gauge (Maxwell), γ=1 → full EED
            let gamma = if request.solver.lorenz_gauge { 0.0f32 }
                        else { request.eed.gamma as f32 };

            if has_ac {
                // Use the first AC entity's current and frequency for the source.
                // TODO: multi-entity AC superposition (different frequencies).
                let ac_entity = request.entities.iter()
                    .find(|e| e.coil.frequency_hz > 0.0)
                    .unwrap(); // safe: has_ac guarantees at least one
                let current_a    = ac_entity.coil.current_a as f32;
                let frequency_hz = ac_entity.coil.frequency_hz as f32;
                gstate.run_fdtd_ac(
                    &self.ctx, &grid, dt, n_steps, gamma, None,
                    current_a, frequency_hz, 0.0,
                )?;
                if current_a == 0.0 {
                    warnings.push(format!(
                        "AC injection: f={:.2}Hz but Current = 0 A — no source injected. \
                         Set Current > 0 in the geometry panel.",
                        frequency_hz
                    ));
                } else {
                    warnings.push(format!(
                        "AC injection: f={:.2}Hz, I₀={:.3}A over {n_steps} steps",
                        frequency_hz, current_a
                    ));
                }
            } else {
                gstate.run_fdtd(&self.ctx, &grid, dt, n_steps, gamma)?;
            }

            // Re-compute observables using evolved E = -∇φ - a_vel.
            gstate.run_observables(&self.ctx, &grid)?;
        }

        // ── Phase 4: GEM gravitational sector ────────────────────────────────
        if request.gem.enabled && request.gem.kappa_g != 0.0 {
            if let SolverMode::TimeDomain { dt_s, n_steps } = cfg.mode {
                let cfl_max = grid.cfl_dt() as f32;
                let dt      = (dt_s as f32).min(cfl_max);
                let kappa   = request.gem.kappa_g as f32;
                gstate.run_gem_fdtd(&self.ctx, &grid, dt, n_steps, kappa)?;
                log::info!("GEM: κ_G={:.3e}", kappa);
            } else {
                warnings.push(
                    "GEM sector requires time-domain mode (FDTD). Enable it in the Mode section.".into()
                );
            }
        }

        // ── Post-processing ──────────────────────────────────────────────────
        let slices = postproc::extract_slices(
            &self.ctx, &gstate, &grid, &request.slices,
        )?;

        let maxima = postproc::compute_maxima(&self.ctx, &gstate, &grid)?;

        let holonomies = postproc::compute_holonomies(
            &self.ctx, &gstate, &grid, &request.holonomy_paths,
        );

        let magnetic_helicity = postproc::compute_helicity(&self.ctx, &gstate, &grid);
        log::info!("Magnetic helicity ∫A·B d³x = {:.4e}", magnetic_helicity);

        // ── Volume extraction ────────────────────────────────────────────────
        let volume = if request.request_volume {
            // Guard: if the requested field isn't populated yet, fall back to B.
            let field = match &request.volume_field {
                f @ (FieldName::BMagnitude
                   | FieldName::AMagnitude
                   | FieldName::CField
                   | FieldName::Phi
                   | FieldName::PhiG
                   | FieldName::PoyntingMag
                   | FieldName::EnergyDensity) => f.clone(),
                _ => {
                    warnings.push(format!(
                        "Volume field {:?} not yet implemented — falling back to B_magnitude.",
                        request.volume_field
                    ));
                    FieldName::BMagnitude
                }
            };
            Some(postproc::extract_volume(&self.ctx, &gstate, &grid, &field)?)
        } else {
            None
        };

        let solve_time = t0.elapsed().as_secs_f64();
        log::info!("Solve complete in {:.3}s", solve_time);

        Ok(SolveResult {
            solve_time_s: solve_time,
            grid_cells:   grid.total_cells(),
            slices,
            volume,
            maxima,
            holonomies,
            magnetic_helicity,
            warnings,
            lead_points,
        })
    }
}

// OracleSolver must be Send + Sync for Tauri managed state.
// Safety: GpuContext is Arc-backed; wgpu types are Send + Sync.
unsafe impl Send for OracleSolver {}
unsafe impl Sync for OracleSolver {}
