//! Diagnostic (ORC-0km): κ_G = 1 toroid sanity check.
//!
//! # Purpose
//!
//! Before changing the GEM coupling *physics*, isolate the coupling *form*
//! from the coupling *magnitude*.  We run a toroidal coil in time-domain FDTD
//! with κ_G = 1.0 — an unphysically huge value (the real KK constant is
//! ~7e-28, Li-Torr ~1e-11).  If the existing derivative coupling
//!
//!     ∂Φ_g/∂t² ⊃ κ_G · ∂C/∂t        (sources Φ_g)
//!     ∂A_g/∂t² ⊃ κ_G · ∇C           (sources A_g)
//!
//! were able to produce gravitomagnetic effects from a static magnetic
//! configuration, κ_G = 1 would make them blatantly visible.
//!
//! # Expected outcome (the diagnosis under test)
//!
//! A DC toroid is a *static* magnetic configuration: A is time-independent and
//! C = ∇·A + (1/c²)∂φ/∂t is near-constant in time, so ∂C/∂t ≈ 0.  The only
//! surviving source is κ_G·∇C — the tiny spatial divergence-gradient left over
//! from Biot-Savart discretisation.  We therefore expect Φ_g and B_g to remain
//! negligible relative to the EM field |B| even at κ_G = 1.
//!
//! This confirms the coupling channel is *structurally blind* to static
//! configurations — motivating the direct KK coupling (Φ_g ← κ_G·C,
//! A_g ← κ_G·A) in ORC-bn6 rather than the derivative form.
//!
//! If instead Φ_g/B_g "light up" here, the coupling form is fine and the
//! problem is purely f32 precision × tiny-κ_G underflow — i.e. only the
//! rescale (ORC-gru) would be needed.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, CouplingMode, EedParams, FieldName,
        GemParams, SliceAxis, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

fn slice_req(field: FieldName) -> SliceRequest {
    SliceRequest {
        axis:       SliceAxis::Z,
        position:   0.5,
        field,
        resolution: 64,
    }
}

#[tokio::test]
async fn diagnostic_kappa1_toroid_stays_dark() {
    // ── Toroid: DC current, no AC drive ─────────────────────────────────────
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Toroid,
            radius_m:      0.05,
            turns:         20,
            pitch_m:       0.012,
            wire_radius_m: 0.001,
            current_a:     10.0,
            frequency_hz:  0.0,   // ← static / DC: this is the whole point
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };

    // ── κ_G = 1.0 (unphysical, isolates coupling FORM from MAGNITUDE) ────────
    let request = SolveRequest {
        entities: vec![entity],
        eed:      EedParams { alpha: 0.0, beta: 0.0, gamma: 1.0 },
        gem:      GemParams {
            enabled:       true,
            kappa_g:       1.0,
            li_torr_mode:  false,
            // This diagnostic specifically exercises the LEGACY derivative
            // coupling — pin SlwMediated so it keeps testing that channel
            // even though the default is now KkDirect (ORC-bn6).
            coupling_mode: CouplingMode::SlwMediated,
        },
        solver: SolverConfig {
            mode:            SolverMode::TimeDomain { dt_s: 0.0, n_steps: 200 },
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
        volume_field:   FieldName::BgMagnitude,
        holonomy_paths: vec![],
    };

    let solver = OracleSolver::new().await.expect("GPU init failed");
    let result = solver.solve(&request).await.expect("solve failed");

    // ── Collect the peak magnitude of each slice on Z=0.5 ───────────────────
    let peak = |f: FieldName| -> f64 {
        result.slices.iter()
            .find(|s| s.field == f)
            .map(|s| s.field_max.abs().max(s.field_min.abs()))
            .unwrap_or(0.0)
    };

    let b_peak  = peak(FieldName::BMagnitude);
    let c_peak  = peak(FieldName::CField);
    let pg_peak = peak(FieldName::PhiG);
    let bg_peak = peak(FieldName::BgMagnitude);

    println!("\n── ORC-0km κ_G=1 toroid diagnostic (Z=0.5 slice peaks) ──");
    println!("  |B|   peak = {:.4e} T", b_peak);
    println!("  |C|   peak = {:.4e} 1/m", c_peak);
    println!("  Φ_g   peak = {:.4e} m²/s²", pg_peak);
    println!("  |B_g| peak = {:.4e} 1/s", bg_peak);
    if b_peak > 0.0 {
        println!("  ratio |B_g| / |B| = {:.3e}", bg_peak / b_peak);
    }

    // Also surface whatever the global maxima report (covers off-slice peaks).
    for m in &result.maxima {
        if matches!(m.field, FieldName::PhiG | FieldName::BgMagnitude) {
            println!(
                "  max {:?} = {:.4e} at [{:+.3}, {:+.3}, {:+.3}] m",
                m.field, m.max_value,
                m.max_location[0], m.max_location[1], m.max_location[2],
            );
        }
    }

    // ── Sanity: the EM field actually exists ────────────────────────────────
    assert!(b_peak > 1e-7, "toroid produced no B field — Biot-Savart broken");

    // ── The diagnosis: GEM stays dark even at κ_G = 1 ───────────────────────
    // Expect Φ_g and B_g to be utterly negligible vs. the EM field.
    // Threshold is generous (1e-3 of |B|) — if the derivative coupling worked
    // for static configs, κ_G=1 would push this to O(1).
    let bg_ratio = if b_peak > 0.0 { bg_peak / b_peak } else { 0.0 };
    assert!(
        bg_ratio < 1e-3,
        "UNEXPECTED: |B_g|/|B| = {:.3e} at κ_G=1 — derivative coupling is NOT \
         blind to static configs; revisit the ORC-jlu diagnosis (only rescale \
         ORC-gru may be needed, not the direct-coupling rewrite ORC-bn6).",
        bg_ratio,
    );

    println!(
        "\n✓ Confirmed: at κ_G=1 the GEM sector stays dark (|B_g|/|B| = {:.2e}).",
        bg_ratio,
    );
    println!(
        "  → derivative coupling (κ_G·∂C/∂t, κ_G·∇C) is structurally blind to\n\
         \x20   a static toroidal A-field.  Direct KK coupling (ORC-bn6) is required."
    );
}
