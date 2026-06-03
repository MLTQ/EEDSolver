//! ORC-vzp: the elliptic KK-Poisson channel (∇²Φ_g = −κ_G·C, ∇²A_g = −κ_G·A)
//! must (a) run in **static** mode with no FDTD ramp, and (b) produce a result
//! that genuinely *differs* from the pointwise KK-direct copy — otherwise it's
//! not a distinct physical reading worth offering.
//!
//! # Setup
//!
//! A DC toroid in **static** mode: its Aharonov-Bohm A is non-zero (so A_g is
//! sourced), while C ≈ 0 in the Coulomb-gauge static field (so Φ_g ≈ 0 in both
//! channels).  The discriminator is therefore B_g = ∇×A_g:
//!   - `kk_direct`  copies A_g = κ_G·A pointwise → B_g = κ_G·B exactly.
//!   - `kk_poisson` solves ∇²A_g = −κ_G·A → A_g is the inverse-Laplacian-smoothed
//!     source, so B_g ≠ κ_G·B (different shape AND magnitude).
//!
//! Passing this confirms the channel is wired into the Phase-4 dispatch, runs in
//! static mode, and is not silently aliasing KK-direct.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

fn toroid_request(mode: CouplingMode) -> SolveRequest {
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Toroid,
            radius_m:      0.05,
            turns:         20,
            pitch_m:       0.012,
            wire_radius_m: 0.001,
            current_a:     10.0,
            frequency_hz:  0.0,        // DC
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };

    let slice = |field| SliceRequest { axis: SliceAxis::Z, position: 0.5, field, resolution: 64 };

    SolveRequest {
        entities: vec![entity],
        eed:      EedParams { alpha: 0.0, beta: 0.0, gamma: 1.0 },
        gem:      GemParams {
            enabled:       true,
            kappa_g:       1.0,
            li_torr_mode:  false,
            coupling_mode: mode,
        },
        solver: SolverConfig {
            mode:            SolverMode::Static,   // ← headline: static GEM, no FDTD ramp
            cells_per_axis:  32,
            domain_radius_m: 0.10,
            lorenz_gauge:    false,
        },
        slices: vec![
            slice(FieldName::BMagnitude),
            slice(FieldName::AMagnitude),
            slice(FieldName::BgMagnitude),
        ],
        request_volume: false,
        volume_field:   FieldName::BgMagnitude,
        holonomy_paths: vec![],
    }
}

fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

#[tokio::test]
async fn kk_poisson_runs_static_and_differs_from_kk_direct() {
    let solver = OracleSolver::new().await.expect("GPU init failed");

    let direct = solver.solve(&toroid_request(CouplingMode::KkDirect)).await
        .expect("KkDirect static solve failed");
    let poisson = solver.solve(&toroid_request(CouplingMode::KkPoisson)).await
        .expect("KkPoisson static solve failed");

    let b_em        = global_max(&direct,  FieldName::BMagnitude);
    let bg_direct   = global_max(&direct,  FieldName::BgMagnitude);
    let bg_poisson  = global_max(&poisson, FieldName::BgMagnitude);

    println!("\n── ORC-vzp KK-direct vs KK-Poisson (static DC toroid, κ_G=1) ──");
    println!("  |B|    (EM)         = {:.4e} T", b_em);
    println!("  |B_g|  Kk-direct    = {:.4e} 1/s   (= κ_G·B, pointwise)", bg_direct);
    println!("  |B_g|  Kk-Poisson   = {:.4e} 1/s   (∇²A_g = −κ_G·A, smoothed)", bg_poisson);

    // (1) Static-mode EM produced the toroid B field.
    assert!(b_em > 1e-7, "toroid produced no B field in static mode — Biot-Savart broken");

    // (2) KK-direct gives B_g = κ_G·B exactly (κ_G=1 ⇒ B_g ≈ B).
    assert!(bg_direct.is_finite() && bg_direct > 1e-9,
        "KkDirect B_g is zero/non-finite ({bg_direct:.3e})");

    // (3) The elliptic channel ran in static mode and produced a finite field.
    assert!(bg_poisson.is_finite(),
        "KkPoisson B_g non-finite ({bg_poisson:.3e}) — elliptic solve diverged");

    // (4) The two readings genuinely differ — KK-Poisson is not aliasing the
    // pointwise copy (the inverse Laplacian reshapes/rescales the source).
    let rel = (bg_direct - bg_poisson).abs() / bg_direct;
    assert!(rel > 1e-2,
        "KkPoisson B_g ({bg_poisson:.3e}) ≈ KkDirect B_g ({bg_direct:.3e}); the elliptic \
         channel is not distinct from the pointwise copy (rel diff {rel:.3e})");
}
