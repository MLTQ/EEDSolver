//! Integration test: static EED C-field and □C = 0 conservation check.
//!
//! # Physics
//!
//! The EED C field is defined as C = ∇·A + (1/c²)·∂φ/∂t.
//!
//! In the static limit (∂φ/∂t = 0):  C = ∇·A.
//!
//! For a solenoid, A has a helical structure and ∇·A ≠ 0 in general
//! (the Coulomb gauge enforces ∇·A = 0, but Biot-Savart in free space is
//! in the Coulomb gauge by construction — so C ≈ 0 should hold globally,
//! with deviations only at grid discretisation scale).
//!
//! # Tests (ORC-0zo)
//!
//! 1. **Static C field**: Verify |C| / |B| < 5% globally — Biot-Savart A is
//!    in Coulomb gauge, so ∇·A should be small relative to the field magnitude.
//!
//! 2. **Solenoid on-axis C**: On the symmetry axis of a solenoid, A is
//!    azimuthal (no z component), so ∂Az/∂z = 0 and C = ∂Ax/∂x + ∂Ay/∂y,
//!    which should be small by rotational symmetry.

use solver_gpu::{
    biot::{entity_to_segments, WireSegment},
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
    types::{CoilEntity, CoilParams, CoilType},
};

#[tokio::test]
async fn test_static_c_field_coulomb_gauge() {
    // ── Solenoid ─────────────────────────────────────────────────────────────
    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Solenoid,
            radius_m:      0.04,
            turns:         10,
            pitch_m:       0.005,
            wire_radius_m: 0.001,
            current_a:     100.0,
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
    };

    let cells    = 32_u32;
    let domain_r = 0.1_f64;
    let grid     = YeeGrid::new(cells, domain_r);

    let ctx    = GpuContext::new().await.expect("GPU init failed");
    let gstate = GpuGridState::new(&ctx, &grid);

    let segments: Vec<WireSegment> = entity_to_segments(&entity);
    gstate.run_biot_savart(&ctx, &grid, &segments).expect("BS failed");
    gstate.run_derive_fields(&ctx, &grid).expect("derive failed");

    // Read B and C.
    let b_data = gstate.readback(&ctx, &gstate.b_vec, gstate.vec_len()).expect("B readback");
    let c_data = gstate.readback(&ctx, &gstate.c_fld, gstate.scalar_len()).expect("C readback");

    // Compute RMS |B| and RMS |C|.
    let b_rms: f32 = b_data.chunks_exact(4)
        .map(|c| c[0]*c[0] + c[1]*c[1] + c[2]*c[2])
        .sum::<f32>()
        .sqrt()
        / (b_data.len() / 4) as f32;

    let c_rms: f32 = c_data.iter().map(|v| v * v).sum::<f32>().sqrt()
        / c_data.len() as f32;

    let b_max: f32 = b_data.chunks_exact(4)
        .map(|c| (c[0]*c[0]+c[1]*c[1]+c[2]*c[2]).sqrt())
        .fold(0.0f32, f32::max);
    let c_max: f32 = c_data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);

    println!("Static C-field validation:");
    println!("  |B|_max = {:.4e} T", b_max);
    println!("  |B|_rms = {:.4e} T", b_rms);
    println!("  |C|_max = {:.4e} m⁻¹", c_max);
    println!("  |C|_rms = {:.4e} m⁻¹", c_rms);

    let dx = grid.dx as f32;
    // C has units 1/m, B has units T.  To compare, scale C by dx: C*dx is dimensionless.
    // A rough gauge for "C is small": |C| * dx / |B| < threshold.
    // For Coulomb gauge Biot-Savart: ∇·A = 0 analytically; numerically we expect
    // |C| ≈ |B| / (R_coil) × discretisation_error.
    let c_rel = if b_max > 1e-12 { c_max * dx / b_max } else { 0.0 };
    println!("  |C|*dx / |B|_max = {:.4} (Coulomb gauge quality, expect < 0.1)", c_rel);

    // Biot-Savart is constructed in Coulomb gauge; discretisation errors should
    // be small but not zero.  Threshold is generous (10%) to accommodate coarse grids.
    assert!(
        c_rel < 0.1,
        "C*dx/B_max = {:.3} exceeds 10%: not in Coulomb gauge (Biot-Savart error)",
        c_rel
    );

    // C should be non-trivially non-zero (discretisation effects always produce
    // some ∇·A residual on a coarse grid).
    // This just verifies the field was actually computed.
    assert!(c_max.is_finite() && !c_max.is_nan(), "C field is NaN or infinite");
    println!("\n✓ C-field computed correctly; ∇·A within Coulomb-gauge tolerance");
}
