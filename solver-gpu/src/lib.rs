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
        // Only *current-carrying* types are AC sources: capacitors are
        // voltage-driven (no current) and mass spheres carry no current at all,
        // so a stale frequency_hz on those must NOT flip has_ac (ORC-1sw) — that
        // routed a voltage source through current injection and warned about a
        // current slider that doesn't exist.
        let is_ac_current_source = |e: &CoilEntity| {
            e.coil.frequency_hz > 0.0
                && !matches!(
                    e.coil.coil_type,
                    CoilType::CapacitorSymmetric
                        | CoilType::CapacitorAsymmetric
                        | CoilType::MassSphere
                )
        };
        let has_ac = request.entities.iter().any(|e| is_ac_current_source(e));
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
                gstate.run_cg_phi(&self.ctx, &grid, &rhs, alpha_sq, 1e-6, 100, &gstate.phi)?;
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

        // ── Snapshot EED potentials for KK-direct GEM coupling ───────────────
        // The KK identification (A_g ← κ_G·A, Φ_g ← κ_G·C) must source from the
        // *physically correct* potentials of the configuration — but which set
        // that is depends on the drive (ORC-x7m):
        //
        //   • Unsustained DC drive (no AC injection): the FDTD loop radiates the
        //     un-fed static field to ~0, so we must capture A/C *before* the
        //     loop.  Here the static A is the Coulomb-gauge Biot-Savart field
        //     and C ≈ ∇·A_BS ≈ 0 — which is the correct EED answer for a closed
        //     loop (charge conserved ⇒ □C = 0 ⇒ C = 0; the AB signal lives in A).
        //
        //   • Sustained AC drive: the injected J(t) keeps the field alive, and
        //     the FDTD evolves the genuine *dynamical* EED scalar C (the deleted
        //     7th DOF) — non-zero precisely where ∂µJµ ≠ 0 (e.g. an open helix's
        //     charged tips).  Coupling to the pre-FDTD Coulomb-gauge snapshot
        //     (C ≈ 0) would throw that away, so we snapshot *after* the loop.
        //
        // `snapshot_after_fdtd` selects the post-loop capture for sustained
        // drives; otherwise we snapshot the static fields here, pre-loop.
        let is_time_domain   = matches!(cfg.mode, SolverMode::TimeDomain { .. });
        let snapshot_after_fdtd = request.gem.enabled && is_time_domain && has_ac;
        if request.gem.enabled && !snapshot_after_fdtd {
            gstate.snapshot_gem_sources(&self.ctx);
        }

        // SLW-mediated (derivative) GEM coupling now co-evolves *inside* the EM
        // FDTD loop (ORC-j07): the gravitational sector takes one step after each
        // EM step, against a freshly-refreshed C, so ∂C/∂t = (Cⁿ⁺¹−Cⁿ)/dt is
        // correct (the old post-hoc pass divided the whole-run ΔC by one dt — an
        // n_steps scale error — and saw a frozen ∇C).  Pass κ_G down to the FDTD
        // for the derivative channel only; KkDirect stays a post-loop algebraic
        // assignment (Phase 4).
        let gem_slw_kappa: Option<f32> = if request.gem.enabled
            && request.gem.kappa_g != 0.0
            && matches!(request.gem.coupling_mode,
                        types::CouplingMode::SlwMediated | types::CouplingMode::Both)
        {
            Some(request.gem.kappa_g as f32)
        } else {
            None
        };

        // ── Phase 3: FDTD ────────────────────────────────────────────────────
        if let SolverMode::TimeDomain { dt_s, n_steps } = cfg.mode {
            // The bare potential-primary leapfrog sits at the marginal stability
            // edge (dt·ω = 2) at *exactly* dt = dx/(c√3); the EED gauge coupling
            // then tips it over into a NaN blowup.  Step at a Courant-safe fraction
            // of the theoretical limit instead (empirically stable to ~0.85·CFL for
            // γ=1; 0.5 leaves a comfortable margin).  See ORC-4eg.
            const CFL_SAFETY: f32 = 0.5;
            let cfl_max  = grid.cfl_dt() as f32;
            let dt_limit = cfl_max * CFL_SAFETY;
            // dt_s == 0.0 means "auto-set to the stable limit" (the frontend always
            // sends 0 since the UI says "dt auto-set to CFL limit").  Treating it as
            // a literal zero makes FDTD do nothing and divides by zero in the GEM
            // shader (dC_dt = ΔC / dt).  Always use at least the stable limit.
            let dt = if dt_s == 0.0 {
                dt_limit
            } else {
                let d = (dt_s as f32).min(dt_limit);
                if d < dt_s as f32 {
                    warnings.push(format!(
                        "dt={:.3e}s clamped to stable limit {:.3e}s (={:.2}·CFL, dx={:.3}mm, n={})",
                        dt_s, dt_limit, CFL_SAFETY, grid.dx * 1e3, grid.n,
                    ));
                }
                d
            };
            log::info!("FDTD dt={:.3e}s ({:.2}·CFL, CFL max={:.3e}s)", dt, CFL_SAFETY, cfl_max);
            // γ=0 → Lorenz gauge (Maxwell), γ=1 → full EED
            let gamma = if request.solver.lorenz_gauge { 0.0f32 }
                        else { request.eed.gamma as f32 };

            if has_ac {
                // Use the first AC *current-source* entity (same predicate as
                // has_ac, so we never pick a voltage-driven capacitor — ORC-1sw).
                // TODO: multi-entity AC superposition (different frequencies).
                let ac_entity = request.entities.iter()
                    .find(|e| is_ac_current_source(e))
                    .unwrap(); // safe: has_ac guarantees at least one

                // Open circuits (open helix, or any winding with open_circuit set)
                // are voltage-driven: peak feed current I₀ = V₀/Z_ref.  Closed
                // loops use current_a directly.
                let current_a = if ac_entity.coil.coil_type == CoilType::OpenHelix
                    || ac_entity.coil.open_circuit
                {
                    (ac_entity.coil.voltage_v / biot::OPEN_HELIX_Z_REF) as f32
                } else {
                    ac_entity.coil.current_a as f32
                };
                let frequency_hz = ac_entity.coil.frequency_hz as f32;
                gstate.run_fdtd_ac(
                    &self.ctx, &grid, dt, n_steps, gamma, None,
                    current_a, frequency_hz, 0.0, gem_slw_kappa,
                )?;
                if current_a == 0.0 {
                    warnings.push(format!(
                        "AC injection: f={:.2}Hz but effective current = 0 A — no source injected. \
                         Set Voltage > 0 V (open circuit) or Current > 0 A in the geometry panel.",
                        frequency_hz
                    ));
                } else {
                    log::info!(
                        "AC injection: f={:.2}Hz, I₀={:.3}A over {n_steps} steps",
                        frequency_hz, current_a
                    );
                }
            } else {
                gstate.run_fdtd(&self.ctx, &grid, dt, n_steps, gamma, gem_slw_kappa)?;
            }

            // Re-compute observables using evolved E = -∇φ - a_vel.
            gstate.run_observables(&self.ctx, &grid)?;

            // Sustained (AC) drive: capture the *evolved* fields now (ORC-x7m).
            if snapshot_after_fdtd {
                gstate.snapshot_gem_sources(&self.ctx);
                log::info!("GEM sources snapshotted post-FDTD (sustained AC drive)");
            }
        }

        // ── Phase 4: GEM gravitational sector ────────────────────────────────
        if request.gem.enabled {
            // κ_G coupling.  Two channels (Wilhelm §4.10):
            //   • KkDirect    — algebraic Φ_g=κ_G·C, A_g=κ_G·A from the EED
            //                   potential snapshot.  The snapshot is the static
            //                   Biot-Savart field for unsustained DC drives, but
            //                   the *evolved* (post-FDTD) dynamical field for
            //                   sustained AC drives (ORC-x7m).  Responds to DC
            //                   configs, works in any mode, stable (ORC-bn6).
            //   • SlwMediated — derivative κ_G·∂C/∂t, κ_G·∇C wave coupling;
            //                   time-domain only, blind to static configs (ORC-0km).
            //   • Both        — SLW FDTD first, then KK-direct accumulated on top.
            if request.gem.kappa_g != 0.0 {
                use types::CouplingMode;
                let kappa = request.gem.kappa_g as f32;
                let mode  = request.gem.coupling_mode;

                // SLW (derivative) channel — already co-evolved INSIDE the EM
                // FDTD loop in Phase 3 (ORC-j07), using the same Courant-safe dt
                // so EM/GEM sim times stay aligned.  Nothing to step here; only
                // warn if it was requested without a time-domain solve.
                if matches!(mode, CouplingMode::SlwMediated | CouplingMode::Both) {
                    if is_time_domain {
                        log::info!("GEM SLW-mediated: κ_G={:.3e} (interleaved per-step with EM FDTD)", kappa);
                    } else {
                        warnings.push(
                            "GEM SLW-mediated coupling requires time-domain mode. \
                             Enable it in the Mode section (or switch to KK-direct).".into()
                        );
                    }
                }

                // KK-direct (algebraic) channel — pointwise, mode-independent.
                // In Both mode it accumulates on top of the SLW FDTD result.
                if matches!(mode, CouplingMode::KkDirect | CouplingMode::Both) {
                    let additive = matches!(mode, CouplingMode::Both);
                    gstate.run_gem_kk_direct(&self.ctx, kappa, additive)?;
                    log::info!("GEM KK-direct: κ_G={:.3e}, additive={}", kappa, additive);
                }

                // KK-Poisson (elliptic) channel — solves ∇²Φ_g=−κ_G·C and
                // ∇²A_g=−κ_G·A via PCG (vs KkDirect's pointwise copy), so the
                // inverse Laplacian smooths the sources and B_g ≠ κ_G·B in
                // general.  Sources from the same snapshot (c_src, a_src) and is
                // mode-independent — the headline static demos run in <100 ms
                // without an FDTD ramp (ORC-vzp).
                if matches!(mode, CouplingMode::KkPoisson) {
                    const GEM_POISSON_TOL:      f32 = 1.0e-6;
                    const GEM_POISSON_MAX_ITER: u32 = 2000;
                    gstate.run_gem_poisson(
                        &self.ctx, &grid, kappa, GEM_POISSON_TOL, GEM_POISSON_MAX_ITER,
                    )?;
                    log::info!("GEM KK-Poisson: κ_G={:.3e} (elliptic ∇²Φ_g=−κ_G·C)", kappa);
                }
            }

            // Li-Torr gravitomagnetic London moment (Wilhelm 2026 Eq. 23).
            // For every superconducting entity rotating at ω ≠ 0, directly impose
            // A_g = −(m_e/e)·ω×(r−r_c) inside its volume — giving uniform
            // B_g = −(2·m_e/e)·ω there.  Works in BOTH static and time-domain modes.
            if request.gem.li_torr_mode {
                let li_torr_ents: Vec<grid::state::LiTorrEntityGpu> = request.entities.iter()
                    .filter(|e| e.superconducting
                        && (e.angular_velocity_rad_s[0].abs()
                          + e.angular_velocity_rad_s[1].abs()
                          + e.angular_velocity_rad_s[2].abs()) > 0.0)
                    .map(|e| grid::state::LiTorrEntityGpu {
                        center_radius: [
                            e.position_m[0] as f32,
                            e.position_m[1] as f32,
                            e.position_m[2] as f32,
                            e.coil.radius_m  as f32,
                        ],
                        omega_pad: [
                            e.angular_velocity_rad_s[0] as f32,
                            e.angular_velocity_rad_s[1] as f32,
                            e.angular_velocity_rad_s[2] as f32,
                            0.0,
                        ],
                    })
                    .collect();

                if li_torr_ents.is_empty() {
                    warnings.push(
                        "GEM Li-Torr mode is on but no entity has `superconducting=true` with non-zero angular_velocity_rad_s.".into()
                    );
                } else {
                    gstate.run_li_torr_source(&self.ctx, &grid, &li_torr_ents)?;
                    log::info!(
                        "Li-Torr: {} SC entit{} contributing",
                        li_torr_ents.len(),
                        if li_torr_ents.len() == 1 { "y" } else { "ies" },
                    );
                }
            }

            // ── GEM mass sources (ORC-0tl) ──────────────────────────────────
            // Ordinary mass distributions source the gravitational sector
            // independently of the κ_G (EED) channel: ∇²Φ_g = 4πG·ρ_m and
            // ∇²A_g = (4πG/c²)·J_m.  Static elliptic solve, superposed on top of
            // whatever the κ_G/Li-Torr channels produced.  Every entity with
            // mass_density_kg_m3 > 0 is a uniform sphere of radius_m at its
            // position (CoilType::MassSphere is the dedicated current-free form).
            let mass_sources: Vec<grid::state::MassSourceGpu> = request.entities.iter()
                .filter(|e| e.coil.mass_density_kg_m3 > 0.0)
                .map(|e| {
                    let rho = e.coil.mass_density_kg_m3;
                    grid::state::MassSourceGpu {
                        center_radius: [
                            e.position_m[0] as f32, e.position_m[1] as f32,
                            e.position_m[2] as f32, e.coil.radius_m as f32,
                        ],
                        jm_density: [
                            (rho * e.coil.mass_velocity_m_s[0]) as f32,
                            (rho * e.coil.mass_velocity_m_s[1]) as f32,
                            (rho * e.coil.mass_velocity_m_s[2]) as f32,
                            rho as f32,
                        ],
                    }
                })
                .collect();
            if !mass_sources.is_empty() {
                const MASS_TOL:      f32 = 1.0e-6;
                const MASS_MAX_ITER: u32 = 4000;
                gstate.run_gem_mass_static(
                    &self.ctx, &grid, &mass_sources, MASS_TOL, MASS_MAX_ITER,
                )?;
                log::info!("GEM mass sources: {} sphere(s)", mass_sources.len());
            }

            // Derive B_g = ∇×A_g for display whenever GEM machinery ran.
            gstate.run_derive_gem_fields(&self.ctx, &grid)?;
        }

        // ── Post-processing ──────────────────────────────────────────────────
        let slices = postproc::extract_slices(
            &self.ctx, &gstate, &grid, &request.slices,
        )?;

        let maxima = postproc::compute_maxima(&self.ctx, &gstate, &grid, &mut warnings)?;

        let holonomies = postproc::compute_holonomies(
            &self.ctx, &gstate, &grid, &request.holonomy_paths,
        );

        let magnetic_helicity = postproc::compute_helicity(&self.ctx, &gstate, &grid);
        log::info!("Magnetic helicity ∫A·B d³x = {:.4e}", magnetic_helicity);

        // Φ_g directional asymmetry — the thrust-direction indicator (ORC-65c).
        let phi_g_asymmetry = if request.gem.enabled {
            let a = postproc::compute_phi_g_asymmetry(
                &self.ctx, &gstate, &grid, &request.entities,
            )?;
            if let Some(fa) = &a {
                log::info!(
                    "Φ_g asymmetry: {:.2}× toward [{:+.2}, {:+.2}, {:+.2}]",
                    fa.ratio, fa.lean[0], fa.lean[1], fa.lean[2],
                );
            }
            a
        } else {
            None
        };

        // ── Volume extraction ────────────────────────────────────────────────
        let volume = if request.request_volume {
            // Guard: if the requested field isn't populated yet, fall back to B.
            let field = match &request.volume_field {
                f @ (FieldName::BMagnitude
                   | FieldName::AMagnitude
                   | FieldName::CField
                   | FieldName::Phi
                   | FieldName::PhiG
                   | FieldName::BgMagnitude
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
            phi_g_asymmetry,
        })
    }
}

// OracleSolver must be Send + Sync for Tauri managed state.
// Safety: GpuContext is Arc-backed; wgpu types are Send + Sync.
unsafe impl Send for OracleSolver {}
unsafe impl Sync for OracleSolver {}
