//! Integration test: SLW (scalar longitudinal wave) propagates at speed c (ORC-aq7).
//!
//! # Physics
//!
//! In EED (γ=0 Maxwell as base), the C-field satisfies □C = 0 with propagation
//! speed c.  Starting from a compact Gaussian φ pulse at the origin (A=0, ∂φ/∂t=0),
//! the time-derivative C = ∂φ/∂t / c² radiates outward as a spherical wave.
//!
//! After N FDTD steps of size dt, the C wavefront should be at:
//!
//!   R_expected = c · N · dt
//!
//! # Test strategy
//!
//! 1. Initialise φ = Gaussian(σ≈2 cells) at grid centre; A=0; ∂φ/∂t=0.
//! 2. Run N=15 FDTD steps (γ=0, no sponge).
//! 3. Compute the |C|-weighted mean radial distance from centre.
//! 4. Assert: |<r_C> − R_expected| / R_expected < 15%.
//!
//! The 15% tolerance accounts for:
//!   - Gaussian pulse width (σ≈2 cells ≈ 23% of R_expected)
//!   - Finite-difference dispersion (O(dx²) phase velocity error)
//!   - The weighted-mean estimator's systematic bias for a spherical shell

use wgpu::util::DeviceExt;
use solver_gpu::{
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
};

const C_LIGHT: f64 = 2.998e8;

fn upload_scalar(ctx: &GpuContext, dst: &wgpu::Buffer, data: &[f32]) {
    let dev = ctx.device();
    let stg = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("upload_staging"),
        contents: bytemuck::cast_slice(data),
        usage:    wgpu::BufferUsages::COPY_SRC,
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&stg, 0, dst, 0, (data.len() as u64) * 4);
    ctx.queue().submit([enc.finish()]);
    dev.poll(wgpu::MaintainBase::Wait);
}

#[tokio::test]
async fn test_slw_propagation_speed() {
    // 32³ grid, 10 cm half-extent → dx=6.25 mm
    let cells    = 32_u32;
    let domain_r = 0.10_f64;
    let grid     = YeeGrid::new(cells, domain_r);
    let n1       = (grid.n + 1) as usize;
    let dt       = grid.cfl_dt();

    let ctx    = GpuContext::new().await.expect("GPU init");
    let gstate = GpuGridState::new(&ctx, &grid);

    // Narrow Gaussian φ: σ ≈ 2 cells so the wavefront separates from the source quickly.
    let center  = (n1 / 2) as f32;
    let sigma_c = 2.0_f32;           // cells
    let phi_init: Vec<f32> = (0..n1 * n1 * n1).map(|idx| {
        let ix = (idx % n1) as f32 - center;
        let iy = ((idx / n1) % n1) as f32 - center;
        let iz = (idx / (n1 * n1)) as f32 - center;
        (-( ix*ix + iy*iy + iz*iz ) / (2.0 * sigma_c * sigma_c)).exp()
    }).collect();
    upload_scalar(&ctx, &gstate.phi, &phi_init);

    // Run N steps (γ=0, no sponge) so the wavefront propagates cleanly.
    let n_steps = 15_u32;
    gstate.run_fdtd_sponge(&ctx, &grid, dt as f32, n_steps, 0.0, Some(0.0))
        .expect("FDTD failed");

    // Expected wavefront radius (in physical metres and grid cells).
    let r_expected_m     = C_LIGHT * n_steps as f64 * dt;
    let r_expected_cells = r_expected_m / grid.dx;
    println!("SLW propagation test:");
    println!("  {} steps × dt={:.4e}s → expected r = {:.4e}m ({:.2} cells)",
             n_steps, dt, r_expected_m, r_expected_cells);

    // Read C = phi_vel/c² and compute |C|-weighted mean radius.
    let c_data = gstate.readback(&ctx, &gstate.c_fld, gstate.scalar_len())
        .expect("c_fld readback");

    let mut weight_r = 0.0_f64;
    let mut weight   = 0.0_f64;
    for iz in 0..n1 {
        for iy in 0..n1 {
            for ix in 0..n1 {
                let ix_f = ix as f64 - (n1 / 2) as f64;
                let iy_f = iy as f64 - (n1 / 2) as f64;
                let iz_f = iz as f64 - (n1 / 2) as f64;
                let r_cells = (ix_f*ix_f + iy_f*iy_f + iz_f*iz_f).sqrt();
                let c_abs = c_data[ix + iy * n1 + iz * n1 * n1].abs() as f64;
                weight_r += r_cells * c_abs;
                weight   += c_abs;
            }
        }
    }

    assert!(weight > 0.0, "C field is identically zero after {} steps", n_steps);
    let r_mean_cells = weight_r / weight;
    let r_err = (r_mean_cells - r_expected_cells).abs() / r_expected_cells;

    println!("  |C|-weighted mean r = {:.3} cells  (expected {:.3} cells)", r_mean_cells, r_expected_cells);
    println!("  Relative error = {:.2}%", r_err * 100.0);

    // The weighted mean underestimates the shell radius slightly because the
    // inner part of the Gaussian still has amplitude; allow 15% tolerance.
    assert!(
        r_err < 0.15,
        "SLW wavefront at {:.3} cells but expected {:.3} — error {:.1}% > 15%",
        r_mean_cells, r_expected_cells, r_err * 100.0
    );
    println!("✓ SLW propagates at c (wavefront within {:.1}% of c·N·dt)", r_err * 100.0);
}
