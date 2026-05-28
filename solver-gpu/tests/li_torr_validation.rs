//! Integration test: Li-Torr gravitomagnetic London moment.
//!
//! # Physics
//!
//! Wilhelm 2026 Eq. 23 (after Li & Torr 1991) predicts that a rotating
//! superconductor sources a uniform gravitomagnetic field inside its body:
//!
//!     B_g = −(2·m_e / e) · ω         (Eq. 23)
//!
//! In the solver's GEM unit convention, 2·m_e/e ≈ 1.1374×10⁻¹¹, so an
//! angular velocity ω = (0, 0, 100) rad/s should produce
//!
//!     B_g ≈ (0, 0, −1.1374×10⁻⁹)
//!
//! uniformly inside the body and zero (modulo a thin boundary layer) outside.
//!
//! # Tests (ORC-wob)
//!
//! 1. **B_g magnitude at body centre** — within 5% of |2·m_e/e·ω|.
//! 2. **B_g direction at body centre** — antiparallel to ω.
//! 3. **B_g uniformity inside the body** — ≤ 10% variation between centre
//!    and a point at half the body radius.
//! 4. **B_g vanishes well outside the body** — < 5% of the inside magnitude
//!    at 1.5·R from the centre (well past the boundary discontinuity).

use solver_gpu::{
    context::GpuContext,
    grid::{GpuGridState, YeeGrid, state::LiTorrEntityGpu},
};

/// 2 · m_e / e in the solver's GEM unit convention (matches types.rs docs).
const TWO_M_OVER_E: f32 = 1.1374e-11;

#[tokio::test]
async fn test_li_torr_london_moment_uniform_inside() {
    // ── Configuration ──────────────────────────────────────────────────────────
    let body_radius = 0.05_f32;
    let omega_z     = 100.0_f32;          // rad/s
    let target_bgz  = -TWO_M_OVER_E * omega_z;
    let expected_mag = target_bgz.abs();

    // Domain: 3× the body radius — plenty of room for a clean "outside" check.
    let domain_r = 0.15_f64;
    let cells    = 64_u32;
    let grid     = YeeGrid::new(cells, domain_r);
    let dx       = grid.dx as f32;
    let n1       = (cells + 1) as usize;

    let ctx    = GpuContext::new().await.expect("GPU init failed");
    let gstate = GpuGridState::new(&ctx, &grid);

    // ── Dispatch Li-Torr source + B_g derivation ──────────────────────────────
    let ent = LiTorrEntityGpu {
        center_radius: [0.0, 0.0, 0.0, body_radius],
        omega_pad:     [0.0, 0.0, omega_z, 0.0],
    };
    gstate
        .run_li_torr_source(&ctx, &grid, std::slice::from_ref(&ent))
        .expect("Li-Torr dispatch failed");
    gstate
        .run_derive_gem_fields(&ctx, &grid)
        .expect("GEM derive dispatch failed");

    let bg = gstate
        .readback(&ctx, &gstate.b_g_vec, gstate.vec_len())
        .expect("B_g readback failed");

    // ── Helper: B_g at grid index (ix, iy, iz) ────────────────────────────────
    let at = |ix: usize, iy: usize, iz: usize| -> [f32; 3] {
        let base = (ix + iy * n1 + iz * n1 * n1) * 4;
        [bg[base], bg[base + 1], bg[base + 2]]
    };
    let centre_idx = n1 / 2;

    // ── Check 1: |B_g| at centre matches |2·m_e/e·ω| ──────────────────────────
    let b_c = at(centre_idx, centre_idx, centre_idx);
    let mag_c = (b_c[0]*b_c[0] + b_c[1]*b_c[1] + b_c[2]*b_c[2]).sqrt();
    let rel_err_mag = ((mag_c - expected_mag) / expected_mag).abs();

    println!("Li-Torr validation:");
    println!("  dx = {:.4} m   body_radius = {:.3} m   ~{} cells across body",
        dx, body_radius, (body_radius / dx) as i32);
    println!("  ω = (0, 0, {}) rad/s  →  target B_g_z = {:.4e}", omega_z, target_bgz);
    println!("  At centre: B_g = ({:.4e}, {:.4e}, {:.4e})", b_c[0], b_c[1], b_c[2]);
    println!("  |B_g|_centre = {:.4e}   expected = {:.4e}   rel_err = {:.2}%",
        mag_c, expected_mag, rel_err_mag * 100.0);

    assert!(
        rel_err_mag < 0.05,
        "B_g magnitude at centre off by {:.2}% (expected within 5%)",
        rel_err_mag * 100.0,
    );

    // ── Check 2: direction antiparallel to ω ──────────────────────────────────
    // ω is +z, so B_g should be predominantly -z.
    assert!(
        b_c[2] < 0.0,
        "B_g_z = {:.4e} should be negative (antiparallel to ω = +z)", b_c[2],
    );
    let alignment = b_c[2] / mag_c;
    assert!(
        alignment < -0.95,
        "B_g direction not aligned with -ẑ:  B_g_z/|B_g| = {:.4}", alignment,
    );

    // ── Check 3: uniformity — sample at half the body radius along +x ─────────
    let half_r_cells = ((body_radius * 0.5) / dx).round() as usize;
    let b_half = at(centre_idx + half_r_cells, centre_idx, centre_idx);
    let mag_half = (b_half[0]*b_half[0] + b_half[1]*b_half[1] + b_half[2]*b_half[2]).sqrt();
    let uniformity_err = ((mag_half - mag_c) / mag_c).abs();
    println!("  |B_g| at r=R/2 (+x) = {:.4e}  vs centre {:.4e}  (Δ = {:.2}%)",
        mag_half, mag_c, uniformity_err * 100.0);
    assert!(
        uniformity_err < 0.10,
        "B_g not uniform inside body: centre={:.4e}, r=R/2={:.4e} ({:.2}% off)",
        mag_c, mag_half, uniformity_err * 100.0,
    );

    // ── Check 4: B_g vanishes well outside the body (r = 1.5·R along +x) ──────
    let outside_cells = ((body_radius * 1.5) / dx).round() as usize;
    let b_out = at(centre_idx + outside_cells, centre_idx, centre_idx);
    let mag_out = (b_out[0]*b_out[0] + b_out[1]*b_out[1] + b_out[2]*b_out[2]).sqrt();
    let outside_ratio = mag_out / mag_c;
    println!("  |B_g| at r=1.5·R (+x) = {:.4e}  ({:.2}% of inside magnitude)",
        mag_out, outside_ratio * 100.0);
    assert!(
        outside_ratio < 0.05,
        "B_g should vanish well outside body but is {:.2}% of inside magnitude",
        outside_ratio * 100.0,
    );

    println!("\n✓ Li-Torr B_g = −(2·m_e/e)·ω reproduced uniformly inside the SC body.");
}
