//! ORC-j07: the SLW-mediated GEM channel must co-evolve *inside* the EM FDTD
//! loop, stepping (Φ_g, A_g) against the live, per-step EED scalar C.
//!
//! # The bug this guards against
//!
//! Previously the GEM FDTD ran as a *separate phase* on frozen snapshots:
//! `c_fld_prev` was captured once before the EM loop and `c_fld` updated once
//! after it, so the gravitational sector saw a constant ∇C and a ∂C/∂t equal to
//! the whole-run ΔC divided by a *single* timestep — an n_steps scale error —
//! with zero genuine co-evolution.  The fix interleaves one GEM step after each
//! EM step (refreshing C in between), making ∂C/∂t = (Cⁿ⁺¹−Cⁿ)/dt correct.
//!
//! # What this asserts
//!
//! An **AC open helix** develops a genuinely *time-varying* C (∂µJµ ≠ 0 at the
//! tips).  The derivative coupling κ_G·∂C/∂t (→Φ_g) and κ_G·∇C (→A_g→B_g) must
//! therefore drive a non-zero, finite gravitational sector — proof the channel
//! now responds to the dynamical C rather than a frozen snapshot.  Contrast
//! `gem_kappa_diagnostic` / `gem_kk_direct_validation`, where a *static* toroid
//! correctly leaves the SLW channel dark (∂C/∂t ≈ 0, ∇C ≈ 0).

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

fn slice_req(field: FieldName) -> SliceRequest {
    SliceRequest { axis: SliceAxis::Z, position: 0.5, field, resolution: 64 }
}

fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

#[tokio::test]
async fn slw_channel_coevolves_with_dynamical_c() {
    // Open helix, voltage-driven AC: open tips ⇒ ∂µJµ ≠ 0 ⇒ time-varying C,
    // hence non-zero ∂C/∂t and ∇C for the SLW derivative coupling to act on.
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::OpenHelix,
            radius_m:      0.05,
            turns:         12,
            pitch_m:       0.006,
            wire_radius_m: 0.001,
            current_a:     0.0,
            voltage_v:     10_000.0,
            frequency_hz:  1.0e9,      // AC: sustains a dynamical C through the loop
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };

    let request = SolveRequest {
        entities: vec![entity],
        eed:      EedParams { alpha: 0.0, beta: 0.0, gamma: 1.0 },
        gem:      GemParams {
            enabled:       true,
            kappa_g:       1.0,                    // unphysical-but-visible; isolates form
            li_torr_mode:  false,
            coupling_mode: CouplingMode::SlwMediated,  // ← the interleaved channel (ORC-j07)
        },
        solver: SolverConfig {
            mode:            SolverMode::TimeDomain { dt_s: 0.0, n_steps: 150 },
            cells_per_axis:  32,
            domain_radius_m: 0.10,
            lorenz_gauge:    false,
        },
        slices: vec![
            slice_req(FieldName::BMagnitude),
            slice_req(FieldName::CField),
            slice_req(FieldName::PhiG),
            slice_req(FieldName::BgMagnitude),
        ],
        request_volume: false,
        volume_field:   FieldName::PhiG,
        holonomy_paths: vec![],
    };

    let solver = OracleSolver::new().await.expect("GPU init failed");
    let result = solver.solve(&request).await.expect("AC open-helix SLW solve failed");

    let b_peak  = global_max(&result, FieldName::BMagnitude);
    let c_peak  = global_max(&result, FieldName::CField);
    let pg_peak = global_max(&result, FieldName::PhiG);
    let bg_peak = global_max(&result, FieldName::BgMagnitude);

    println!("\n── ORC-j07 interleaved SLW-GEM (AC open helix, κ_G=1, global 3-D max) ──");
    println!("  |B|   peak = {:.4e} T", b_peak);
    println!("  |C|   peak = {:.4e} 1/m   (live, post-FDTD)", c_peak);
    println!("  Φ_g   peak = {:.4e} m²/s² (SLW κ_G·∂C/∂t)", pg_peak);
    println!("  |B_g| peak = {:.4e} 1/s   (SLW κ_G·∇C)", bg_peak);

    // (1) The AC drive must produce an EM field at all.
    assert!(b_peak > 1e-9, "open helix produced no B field — AC drive broken");

    // (2) A genuinely dynamical C must exist for the SLW channel to couple to.
    assert!(c_peak.is_finite() && c_peak > 1e-6,
        "live C is ~0/non-finite ({c_peak:.3e}) — no dynamical 7th DOF to couple");

    // (3) The interleaved SLW channel must now RESPOND: ∂C/∂t and ∇C drive a
    // non-zero, finite gravitational sector.  With the old frozen-snapshot pass
    // the per-step ∂C/∂t was wrong by ~n_steps and ∇C was constant; co-evolution
    // is what makes this a faithful derivative coupling.
    assert!(pg_peak.is_finite() && bg_peak.is_finite(),
        "GEM sector non-finite — interleaved SLW blew up (Φ_g={pg_peak:.3e}, B_g={bg_peak:.3e})");
    assert!(pg_peak > 0.0 || bg_peak > 0.0,
        "interleaved SLW produced an entirely dark GEM sector for a *dynamical* drive \
         (Φ_g={pg_peak:.3e}, B_g={bg_peak:.3e}) — ∂C/∂t / ∇C coupling is not firing");
}
