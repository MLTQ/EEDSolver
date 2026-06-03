//! Validation (ORC-bn6): KK-direct coupling lights up the GEM sector for a
//! static potential configuration that the derivative coupling cannot see.
//!
//! # Physics
//!
//! Wilhelm 2026 (§4.10 / Fig. 9) identifies the EM four-potential A_μ with the
//! off-diagonal Kaluza-Klein 5D metric component.  This is a *direct algebraic*
//! identification: the gravitomagnetic potential A_g tracks A, and the
//! gravitational scalar Φ_g tracks ∇·A = C — even for a *static* field.
//!
//! The headline test case is the toroid: B is confined to the tube, but the
//! vector potential A is non-zero everywhere (the Aharonov-Bohm region).  The
//! legacy SLW-mediated coupling (κ_G·∂C/∂t, κ_G·∇C) is blind to this static
//! configuration (proven by ORC-0km: B_g ≡ 0 even at κ_G = 1).
//!
//! # What this test asserts
//!
//! Running the *identical* DC toroid in both modes at κ_G = 1:
//!   - SlwMediated → B_g stays at floating-point zero (regression guard).
//!   - KkDirect    → B_g becomes substantial (A_g sourced by κ_G·A).
//! The KK-direct result must dwarf the SLW result by many orders of magnitude.

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
            frequency_hz:  0.0,   // static / DC
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };

    let slice = |field| SliceRequest {
        axis: SliceAxis::Z, position: 0.5, field, resolution: 64,
    };

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
            mode:            SolverMode::TimeDomain { dt_s: 0.0, n_steps: 200 },
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

/// Peak |field| on the returned Z=0.5 slice (max of |min|, |max|).
fn slice_peak(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.slices.iter()
        .find(|s| s.field == f)
        .map(|s| s.field_max.abs().max(s.field_min.abs()))
        .unwrap_or(0.0)
}

/// Global max of a field across the whole volume (from the maxima report).
fn global_max(result: &solver_gpu::types::SolveResult, f: FieldName) -> f64 {
    result.maxima.iter().find(|m| m.field == f).map(|m| m.max_value).unwrap_or(0.0)
}

#[tokio::test]
async fn kk_direct_lights_up_static_toroid() {
    let solver = OracleSolver::new().await.expect("GPU init failed");

    let slw = solver.solve(&toroid_request(CouplingMode::SlwMediated)).await
        .expect("SLW solve failed");
    let kk  = solver.solve(&toroid_request(CouplingMode::KkDirect)).await
        .expect("KK solve failed");

    // Global maxima of B and B_g (slice peaks miss off-plane structure; the
    // displayed A-slice is unreliable here because the DC FDTD drains the
    // unsustained static a_vec to ~0 — KK-direct instead sources from the
    // a_src snapshot taken before the FDTD loop, so we validate against B_g).
    let b_glob   = global_max(&kk, FieldName::BMagnitude);
    let b_peak   = slice_peak(&kk, FieldName::BMagnitude);
    let bg_slw   = global_max(&slw, FieldName::BgMagnitude);
    let bg_kk    = global_max(&kk,  FieldName::BgMagnitude);

    println!("\n── ORC-bn6 KK-direct validation (DC toroid, κ_G=1) ──");
    println!("  EM  |B| global    = {:.4e} T", b_glob);
    println!("  EM  |B| slice     = {:.4e} T", b_peak);
    println!("  |B_g| SlwMediated = {:.4e} 1/s   (legacy: expected ≈ 0)", bg_slw);
    println!("  |B_g| KkDirect    = {:.4e} 1/s   (new: expected ≈ |B|)", bg_kk);

    // Sanity: the toroid produces a magnetic field at all.
    assert!(b_glob > 1e-7, "toroid produced no B field");

    // Regression guard: derivative coupling remains blind to the static toroid.
    assert!(
        bg_slw < 1e-12,
        "SlwMediated |B_g| = {:.3e} should be ~0 for a static toroid (ORC-0km)",
        bg_slw,
    );

    // The fix: KK-direct coupling produces a real gravitomagnetic field.
    assert!(
        bg_kk > 1e-9,
        "KkDirect |B_g| = {:.3e} — GEM sector still dark; κ_G·A coupling not working",
        bg_kk,
    );

    // And it must dominate the (zero) legacy channel by a wide margin.
    assert!(
        bg_kk > 1e6 * bg_slw.max(1e-30),
        "KkDirect (|B_g|={:.3e}) should dwarf SlwMediated (|B_g|={:.3e})",
        bg_kk, bg_slw,
    );

    // Physics check on the KK identity itself: A_g = κ_G·A ⇒ B_g = ∇×A_g
    // = κ_G·(∇×A) = κ_G·B.  At κ_G = 1, |B_g| must equal |B| (same curl
    // operator, same source A snapshot), confirming the algebraic coupling is
    // correct and not merely "some non-zero number".
    let ratio = bg_kk / b_glob;
    assert!(
        (0.5..2.0).contains(&ratio),
        "KkDirect |B_g|={:.3e} should equal |B|={:.3e} at κ_G=1 (B_g=κ_G·B), ratio={:.3}",
        bg_kk, b_glob, ratio,
    );

    println!(
        "\n✓ KK-direct coupling lights up the static toroid: |B_g| went from \
         {:.2e} (SLW) → {:.2e} (KK).",
        bg_slw, bg_kk,
    );
    println!("  → the AB potential A now sources a gravitomagnetic field (Wilhelm §4.10).");
}
