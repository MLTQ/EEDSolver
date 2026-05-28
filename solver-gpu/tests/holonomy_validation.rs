//! Integration test: validate holonomy ∮ A·dl against Stokes' theorem
//! for the vector potential of a long solenoid.
//!
//! # Analytic reference
//!
//! For an infinite solenoid of radius R carrying current I with n turns/m:
//!   B_z = μ₀·n·I  (inside)   and  0 (outside).
//!
//! By Stokes' theorem, ∮ A·dl along a circle of radius ρ in the XY plane at z=0:
//!   ∮ A·dl = ∫∫ B·dS = B_z · π·ρ²    for ρ < R   (flux through the loop)
//!   ∮ A·dl = B_z · π·R²               for ρ > R   (flux is capped at solenoid cross-section)
//!
//! We use a finite solenoid (10 turns) and choose ρ well inside and at ρ = R.
//! The finite-solenoid correction at z=0 is ~ B_z(centre) × π·ρ² which matches
//! the full Stokes integral to the same accuracy as the on-axis Biot-Savart.
//!
//! Test parameters keep runtime < 5 s on CPU/GPU debug build.

use solver_gpu::{
    biot::{WireSegment, entity_to_segments},
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
    postproc::compute_holonomies,
    types::{CoilEntity, CoilParams, CoilType, HolonomyPath},
};

const MU0: f64 = 1.2566370614e-6;

/// Approximate ∮ A·dl via Stokes: B_z(0) × π ρ²  (finite solenoid, ρ ≤ R).
///
/// This uses the on-axis B_z as a proxy for the average interior field.
/// It is only a ~5–10% approximation for finite solenoids where B(ρ) varies
/// radially (lower near the boundary).  Paths well inside (ρ ≪ R) are most
/// accurate because B is approximately uniform there.
fn analytic_holonomy(rho_m: f64, radius: f64, turns: u32, pitch: f64, current: f64) -> f64 {
    let n_per_m = 1.0 / pitch;
    let l       = turns as f64 * pitch;
    let cos1    = (l * 0.5) / (radius * radius + (l * 0.5).powi(2)).sqrt();
    let cos2    = (-l * 0.5) / (radius * radius + (l * 0.5).powi(2)).sqrt();
    let bz      = MU0 * n_per_m * current * 0.5 * (cos1 - cos2);
    bz * std::f64::consts::PI * rho_m * rho_m
}

#[tokio::test]
async fn test_holonomy_solenoid() {
    // ── Coil parameters ─────────────────────────────────────────────────────
    let radius_m      = 0.04_f64;   // 4 cm
    let turns         = 10_u32;
    let pitch_m       = 0.005_f64;  // 5 mm → L = 5 cm
    let current_a     = 100.0_f64;  // 100 A

    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:     CoilType::Solenoid,
            radius_m,
            turns,
            pitch_m,
            wire_radius_m: 0.001,
            current_a,
            ..Default::default()
        },
        position_m:    [0.0, 0.0, 0.0],
        orientation:   [0.0, 0.0, 0.0, 1.0],
        superconducting: false,
        angular_velocity_rad_s: [0.0; 3],
    };

    // ── Grid ─────────────────────────────────────────────────────────────────
    let cells    = 32_u32;
    let domain_r = 0.1_f64;
    let grid     = YeeGrid::new(cells, domain_r);

    // ── GPU: Biot-Savart ─────────────────────────────────────────────────────
    let ctx    = GpuContext::new().await.expect("GPU init failed");
    let gstate = GpuGridState::new(&ctx, &grid);

    let segments: Vec<WireSegment> = entity_to_segments(&entity);
    gstate.run_biot_savart(&ctx, &grid, &segments)
        .expect("Biot-Savart failed");
    gstate.run_derive_fields(&ctx, &grid)
        .expect("Field derivation failed");

    // ── Holonomy paths ────────────────────────────────────────────────────────
    // Two circles well inside the solenoid (field is more uniform here).
    // ρ = R/4 and ρ = R/2.  At ρ = R the finite-solenoid radial B gradient
    // makes the Stokes estimate unreliable (> 10% error expected), so we
    // stay away from the wall.
    let rho_a = radius_m * 0.25;   // 1 cm
    let rho_b = radius_m * 0.50;   // 2 cm

    let paths = vec![
        HolonomyPath::ZCircle { z_m: 0.0, radius_m: rho_a },
        HolonomyPath::ZCircle { z_m: 0.0, radius_m: rho_b },
    ];

    let results = compute_holonomies(&ctx, &gstate, &grid, &paths);

    assert_eq!(results.len(), 2, "Expected 2 holonomy results");

    // ── Comparison ────────────────────────────────────────────────────────────
    println!("\nHolonomy validation (∮ A·dl vs. Stokes):");
    for (i, res) in results.iter().enumerate() {
        let rho  = if i == 0 { rho_a } else { rho_b };
        let anal = analytic_holonomy(rho, radius_m, turns, pitch_m, current_a);
        let sim  = res.value;
        let rel  = ((sim - anal) / anal).abs();
        println!(
            "  ρ={:.3} m  ∮A·dl_sim={:.4e}  ∮A·dl_anal={:.4e}  rel_err={:.1}%",
            rho, sim, anal, rel * 100.0
        );
        // 10 % tolerance — covers:
        //   • Biot-Savart discretisation of turns
        //   • Finite-solenoid B vs. analytic on-axis approximation
        //   • 512-segment midpoint rule quadrature error (O(dx²))
        //   • Trilinear interpolation bias on a 32³ grid
        assert!(
            rel < 0.10,
            "Holonomy rel. err. {:.1}% > 10% for ρ = {:.3} m",
            rel * 100.0, rho
        );
    }
    println!();
}
