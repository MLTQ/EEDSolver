//! Integration test: validate Biot-Savart implementation against the analytic
//! on-axis B field for a finite solenoid.
//!
//! # Analytic reference (finite solenoid, on-axis)
//!
//! For a solenoid with N turns, radius R, total length L, carrying current I:
//!
//!   B_z(0,0,z) = (μ₀·N·I) / (2L) × [cos θ₁ − cos θ₂]
//!
//! where:
//!   cos θ₁ = (z + L/2) / √(R² + (z + L/2)²)
//!   cos θ₂ = (z − L/2) / √(R² + (z − L/2)²)
//!
//! This is exact for a uniformly wound solenoid; our Biot-Savart discretises
//! each turn into `ndiv` elements so we expect ~1% agreement for 10+ turns.
//!
//! Test parameters chosen to keep runtime < 5 s on CPU/GPU debug build.

use solver_gpu::{
    biot::{WireSegment, entity_to_segments},
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
    types::{CoilEntity, CoilParams, CoilType},
};

const MU0: f32 = 1.2566370614e-6;  // [H/m]

/// Analytic on-axis B_z for a finite solenoid.
fn analytic_bz(z_m: f32, radius: f32, turns: u32, pitch: f32, current: f32) -> f32 {
    let n_per_m = 1.0 / pitch;              // turns per metre
    let l       = turns as f32 * pitch;     // total axial length
    let cos1    = (z_m + l * 0.5) / (radius * radius + (z_m + l * 0.5).powi(2)).sqrt();
    let cos2    = (z_m - l * 0.5) / (radius * radius + (z_m - l * 0.5).powi(2)).sqrt();
    MU0 * n_per_m * current * 0.5 * (cos1 - cos2)
}

/// Extract on-axis B_z from the simulation at a given z index.
/// On-axis: ix = iy = n1/2 (centre of grid).
fn get_bz_on_axis(b_data: &[f32], n1: usize, iz: usize) -> f32 {
    let ix   = n1 / 2;
    let iy   = n1 / 2;
    let base = (ix + iy * n1 + iz * n1 * n1) * 4;
    b_data[base + 2]   // Bz component (index 2 in [Bx, By, Bz, 0])
}

#[tokio::test]
async fn test_biot_savart_solenoid_on_axis() {
    // ── Coil parameters ─────────────────────────────────────────────────────
    let radius_m      = 0.04_f32;   // 4 cm
    let turns         = 10_u32;
    let pitch_m       = 0.005_f32;  // 5 mm → L = 5 cm
    let current_a     = 100.0_f32;  // 100 A
    let wire_radius_m = 0.001_f32;

    let solenoid_entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Solenoid,
            radius_m:      radius_m as f64,
            turns,
            pitch_m:       pitch_m as f64,
            wire_radius_m: wire_radius_m as f64,
            current_a:     current_a as f64,
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
    };

    // ── Grid parameters ─────────────────────────────────────────────────────
    let cells      = 32_u32;       // small grid for speed
    let domain_r   = 0.1_f64;     // 10 cm half-extent (solenoid L=5cm fits)
    let grid       = YeeGrid::new(cells, domain_r);
    let n1         = (cells + 1) as usize;

    // ── GPU init ─────────────────────────────────────────────────────────────
    let ctx = GpuContext::new().await.expect("GPU init failed");
    let gstate = GpuGridState::new(&ctx, &grid);

    let segments: Vec<WireSegment> = entity_to_segments(&solenoid_entity);
    assert!(!segments.is_empty(), "No wire segments generated");

    gstate.run_biot_savart(&ctx, &grid, &segments)
        .expect("Biot-Savart dispatch failed");
    gstate.run_derive_fields(&ctx, &grid)
        .expect("Field derivation failed");

    let b_data = gstate.readback(&ctx, &gstate.b_vec, gstate.vec_len())
        .expect("B-field readback failed");

    // ── Comparison ─────────────────────────────────────────────────────────
    // Test at several on-axis z positions, converting from vertex index to
    // physical coordinate: z_m = -domain_r + iz * dx
    let dx   = grid.dx as f32;
    let r_f  = domain_r as f32;

    let mut max_rel_err = 0.0_f32;
    let mut test_points = 0_u32;

    for iz in (n1 / 4)..(3 * n1 / 4) {
        let z_m     = -r_f + iz as f32 * dx;
        let bz_sim  = get_bz_on_axis(&b_data, n1, iz);
        let bz_anal = analytic_bz(z_m, radius_m, turns, pitch_m, current_a);

        // Only compare where the analytic field is large enough to be meaningful.
        if bz_anal.abs() > 1e-6 {
            let rel_err = ((bz_sim - bz_anal) / bz_anal).abs();
            if rel_err > max_rel_err { max_rel_err = rel_err; }
            test_points += 1;

            // Print for visibility (cargo test -- --nocapture)
            if iz % 4 == 0 {
                println!("iz={iz:3}  z={z_m:+.4}m  Bz_sim={bz_sim:.4e}T  Bz_anal={bz_anal:.4e}T  rel_err={rel_err:.2}");
            }
        }
    }

    assert!(test_points > 0, "No valid comparison points found");
    println!("\nMax relative error: {:.2}% over {test_points} points", max_rel_err * 100.0);

    // Allow up to 5% error — Biot-Savart is analytic for thin wires but we
    // have finite-length discretization effects near the solenoid ends and
    // the on-axis point sits between grid vertices.
    assert!(
        max_rel_err < 0.05,
        "Biot-Savart max relative error {:.2}% exceeds 5% tolerance",
        max_rel_err * 100.0,
    );
}
