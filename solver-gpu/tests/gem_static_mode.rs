//! ORC-vzp: GEM must light up in *Static* mode (no FDTD ramp).
//!
//! The paper's flagship configurations — the toroid producing A in a field-free
//! region, Maxwell-Lodge, VPT — are static / quasi-static.  ORC-vzp asked for a
//! static-mode GEM path so "turn on a torus, see Φ_g/B_g track the A-field" works
//! in <100 ms instead of a seconds-long FDTD ramp.
//!
//! That goal is delivered by ORC-bn6's *algebraic* KK-direct coupling
//! (A_g = κ_G·A, Φ_g = κ_G·C): it is a pointwise assignment with no time
//! stepping, so it runs unconditionally — including in SolverMode::Static, which
//! skips the FDTD loop entirely.  As a bonus, static mode never drains a_vec
//! (no FDTD), so the displayed A field is also correct here.
//!
//! This test runs the DC toroid in Static mode and asserts the GEM sector lights
//! up with B_g = κ_G·B (the KK identity), confirming ORC-vzp's intent without a
//! separate elliptic (Poisson) solve.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

#[tokio::test]
async fn static_mode_toroid_lights_up_gem() {
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Toroid,
            radius_m:      0.05,
            turns:         20,
            pitch_m:       0.012,
            wire_radius_m: 0.001,
            current_a:     10.0,
            frequency_hz:  0.0,
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
            kappa_g:       1.0,
            li_torr_mode:  false,
            coupling_mode: CouplingMode::KkDirect,
        },
        solver: SolverConfig {
            mode:            SolverMode::Static,   // ← the whole point: no FDTD
            cells_per_axis:  32,
            domain_radius_m: 0.10,
            lorenz_gauge:    false,
        },
        slices: vec![SliceRequest {
            axis: SliceAxis::Z, position: 0.5, field: FieldName::AMagnitude, resolution: 64,
        }],
        request_volume: false,
        volume_field:   FieldName::BgMagnitude,
        holonomy_paths: vec![],
    };

    let solver = OracleSolver::new().await.expect("GPU init failed");
    let result = solver.solve(&request).await.expect("static solve failed");

    let b    = global_max(&result, FieldName::BMagnitude);
    let a    = global_max(&result, FieldName::AMagnitude);
    let bg   = global_max(&result, FieldName::BgMagnitude);

    println!("\n── ORC-vzp static-mode GEM (DC toroid, κ_G=1, SolverMode::Static) ──");
    println!("  |B|   = {:.4e} T", b);
    println!("  |A|   = {:.4e} V·s/m", a);
    println!("  |B_g| = {:.4e} 1/s", bg);

    // EM source present.
    assert!(b > 1e-7, "toroid produced no B field in static mode");

    // Static mode does NOT run FDTD, so a_vec is never drained — the A field is
    // visible here (unlike the time-domain DC case in gem_kk_direct_validation).
    assert!(a > 1e-9, "static-mode A field missing — KK-direct has no source");

    // The payoff: GEM lights up in static mode, with B_g = κ_G·B (κ_G=1 ⇒ |B_g|=|B|).
    assert!(bg > 1e-9, "static-mode KK-direct produced no B_g — GEM dark in static mode");
    let ratio = bg / b;
    assert!(
        (0.5..2.0).contains(&ratio),
        "static-mode |B_g|={:.3e} should equal |B|={:.3e} at κ_G=1 (ratio={:.3})",
        bg, b, ratio,
    );

    println!("\n✓ GEM lights up in Static mode (no FDTD): |B_g|={:.2e}, |A|={:.2e} visible.", bg, a);
}
