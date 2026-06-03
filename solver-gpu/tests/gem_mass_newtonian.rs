//! ORC-0tl: a uniform-density mass sphere must source the Newtonian
//! gravitational potential Φ_g = −GM/r outside itself.
//!
//! This validates the GEM **mass** source (∇²Φ_g = 4πG·ρ_m) end-to-end in
//! *static* mode — the gravitational-sourcing path that, before ORC-0tl, had no
//! representation at all (only the EED→GEM κ_G channel existed).
//!
//! # Why a two-point difference
//!
//! The elliptic solve uses Dirichlet Φ_g=0 on the cube boundary, but the true
//! −GM/r is non-zero there, so the numerical Φ_g differs from −GM/r by a smooth
//! harmonic offset.  Comparing Φ_g(r₁) − Φ_g(r₂) cancels the (near-constant)
//! offset and isolates the physical 1/r law, letting us recover GM and check it
//! against G·(4/3)πR³ρ.  Sign is the headline: gravity attracts ⇒ Φ_g < 0.

use solver_gpu::{
    OracleSolver,
    types::{
        CoilEntity, CoilParams, CoilType, EedParams, FieldName, GemParams,
        SliceAxis, SliceData, SliceRequest, SolveRequest, SolverConfig, SolverMode,
    },
};

const G: f64 = 6.674e-11;

/// Sample a Z-slice (plane z=const) at world (x, y) by nearest-pixel lookup.
fn sample(slice: &SliceData, x: f64, y: f64) -> f64 {
    let [rows, cols] = slice.shape;
    let fx = (x - slice.x_range[0]) / (slice.x_range[1] - slice.x_range[0]);
    let fy = (y - slice.y_range[0]) / (slice.y_range[1] - slice.y_range[0]);
    let col = ((fx * (cols - 1) as f64).round() as i64).clamp(0, cols as i64 - 1) as usize;
    let row = ((fy * (rows - 1) as f64).round() as i64).clamp(0, rows as i64 - 1) as usize;
    slice.data[row * cols as usize + col] as f64
}

#[tokio::test]
async fn mass_sphere_sources_newtonian_potential() {
    let radius_m: f64 = 0.025;
    let density:  f64 = 1.0e9;                // kg/m³ (large: keeps Φ_g in a comfy f32 range; solve is linear)
    let mass      = 4.0 / 3.0 * std::f64::consts::PI * radius_m.powi(3) * density;
    let gm_true   = G * mass;

    let entity = CoilEntity {
        coil: CoilParams {
            coil_type:          CoilType::MassSphere,
            radius_m,
            current_a:          0.0,          // pure mass: no current, no EM
            mass_density_kg_m3: density,
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
            kappa_g:       0.0,               // no EED coupling — isolate the mass source
            li_torr_mode:  false,
            coupling_mode: solver_gpu::types::CouplingMode::KkDirect,
        },
        solver: SolverConfig {
            mode:            SolverMode::Static,   // <100 ms, no FDTD ramp
            cells_per_axis:  64,
            domain_radius_m: 0.10,
            lorenz_gauge:    false,
        },
        slices: vec![SliceRequest {
            axis: SliceAxis::Z, position: 0.5, field: FieldName::PhiG, resolution: 128,
        }],
        request_volume: false,
        volume_field:   FieldName::PhiG,
        holonomy_paths: vec![],
    };

    let solver = OracleSolver::new().await.expect("GPU init failed");
    let result = solver.solve(&request).await.expect("mass-sphere static solve failed");

    let slice = result.slices.iter().find(|s| s.field == FieldName::PhiG)
        .expect("no Φ_g slice");

    // Radial profile along +x through the centre (y=0).  Both radii are outside
    // the sphere (>R) and comfortably inside the boundary (<0.08 of 0.10).
    let r1 = 0.040;
    let r2 = 0.070;
    let phi_r1 = sample(slice, r1, 0.0);
    let phi_r2 = sample(slice, r2, 0.0);
    let phi_center = sample(slice, 0.0, 0.0);

    // Φ_g(r) = −GM/r + C  ⇒  Φ_g(r1) − Φ_g(r2) = GM(1/r2 − 1/r1), offset C cancels.
    let gm_est = (phi_r1 - phi_r2) / (1.0 / r2 - 1.0 / r1);
    let rel_err = (gm_est - gm_true).abs() / gm_true;

    println!("\n── ORC-0tl Newtonian mass sphere (static, 64³, R={radius_m} m) ──");
    println!("  M = {mass:.4e} kg,  GM_true = {gm_true:.4e} m³/s²");
    println!("  Φ_g(center) = {phi_center:.4e}  Φ_g({r1}) = {phi_r1:.4e}  Φ_g({r2}) = {phi_r2:.4e}  m²/s²");
    println!("  GM_est (two-point 1/r fit) = {gm_est:.4e}  →  rel err = {:.2}%", rel_err * 100.0);

    // (1) Attractive: the potential is negative (this is the Option-A sign — the
    // whole reason we used source −4πG·c²·ρ_m / RHS +4πG·ρ_m, not the bead's flip).
    assert!(phi_center < 0.0 && phi_r1 < 0.0 && phi_r2 < 0.0,
        "Φ_g is not negative (center={phi_center:.3e}) — wrong sign; gravity must attract");

    // (2) Deeper closer in (|Φ_g| larger near the mass).
    assert!(phi_center < phi_r1 && phi_r1 < phi_r2,
        "Φ_g not monotonically deepening toward the mass ({phi_center:.3e} {phi_r1:.3e} {phi_r2:.3e})");

    // (3) The recovered GM matches G·(4/3)πR³ρ — the 1/r Newtonian law, within the
    // accuracy of a 64³ grid with a zero-Dirichlet box (the harmonic-offset
    // residual; tightens toward 1% on 128³ + analytic BC).
    assert!(rel_err < 0.15,
        "GM_est={gm_est:.3e} vs GM_true={gm_true:.3e} (rel err {:.1}%) — \
         mass source does not reproduce −GM/r", rel_err * 100.0);
}
