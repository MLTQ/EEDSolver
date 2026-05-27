//! Integration test: C-field satisfies the wave equation □C = 0 in vacuum (ORC-dhe).
//!
//! With γ=0 (Maxwell, decoupled), A=0 always, so C = ∂φ/∂t / c².
//! Since □φ = 0 (vacuum Maxwell), □C = □(∂φ/∂t)/c² = (∂/∂t □φ)/c² = 0. ✓
//!
//! The residual □C_h = ∂²C/∂t²_h − c²(∇²_h C) should be O(dt²,dx²) small.
//! We measure RMS(∂²C/∂t²_h − c²∇²_h C) / RMS(∂²C/∂t²_h) < 20%.
//!
//! (For the 2nd-order FDTD on a 32³ grid with σ≈5 cells, leading truncation
//!  is O(1/σ_cells²) ≈ 4%, so 20% gives comfortable margin.)

use wgpu::util::DeviceExt;
use solver_gpu::{
    context::GpuContext,
    grid::{GpuGridState, YeeGrid},
};

const C_LIGHT: f32 = 2.998e8;

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

/// Compute (sum_of_6_neighbours − 6·center) for each interior cell.
/// NOT divided by dx²; caller applies the scaling.
fn fd_lap_raw(field: &[f32], n1: usize) -> Vec<f32> {
    let mut lap = vec![0.0_f32; n1 * n1 * n1];
    for iz in 1..n1 - 1 {
        for iy in 1..n1 - 1 {
            for ix in 1..n1 - 1 {
                let g = |x: usize, y: usize, z: usize| field[x + y * n1 + z * n1 * n1];
                lap[ix + iy * n1 + iz * n1 * n1] =
                    g(ix+1,iy,iz) + g(ix-1,iy,iz)
                  + g(ix,iy+1,iz) + g(ix,iy-1,iz)
                  + g(ix,iy,iz+1) + g(ix,iy,iz-1)
                  - 6.0 * g(ix,iy,iz);
            }
        }
    }
    lap
}

#[tokio::test]
async fn test_c_field_wave_equation() {
    let cells    = 32_u32;
    let domain_r = 0.10_f64;
    let grid     = YeeGrid::new(cells, domain_r);
    let n1       = (grid.n + 1) as usize;
    let dt       = grid.cfl_dt() as f32;
    let dx       = grid.dx as f32;

    let ctx    = GpuContext::new().await.expect("GPU init");
    let gstate = GpuGridState::new(&ctx, &grid);

    // Smooth Gaussian φ with σ ≈ 5 cells (minimal Nyquist content).
    let center = (n1 / 2) as f32;
    let sigma  = n1 as f32 * 0.15;   // ≈ 5 cells for 32³
    let phi_init: Vec<f32> = (0..n1 * n1 * n1).map(|idx| {
        let ix = (idx % n1) as f32 - center;
        let iy = ((idx / n1) % n1) as f32 - center;
        let iz = (idx / (n1 * n1)) as f32 - center;
        (-( ix*ix + iy*iy + iz*iz ) / (2.0 * sigma * sigma)).exp()
    }).collect();
    upload_scalar(&ctx, &gstate.phi, &phi_init);

    // 5-step warmup (no sponge) so C = phi_vel/c² becomes non-trivial.
    gstate.run_fdtd_sponge(&ctx, &grid, dt, 5, 0.0, Some(0.0)).unwrap();

    // Record C at three consecutive time levels k-1, k, k+1.
    // run_fdtd updates c_fld = div(A) + phi_vel/c² = phi_vel/c² (A≡0).
    let c_km1 = gstate.readback(&ctx, &gstate.c_fld, gstate.scalar_len()).unwrap();
    gstate.run_fdtd_sponge(&ctx, &grid, dt, 1, 0.0, Some(0.0)).unwrap();
    let c_k   = gstate.readback(&ctx, &gstate.c_fld, gstate.scalar_len()).unwrap();
    gstate.run_fdtd_sponge(&ctx, &grid, dt, 1, 0.0, Some(0.0)).unwrap();
    let c_kp1 = gstate.readback(&ctx, &gstate.c_fld, gstate.scalar_len()).unwrap();

    let c_rms_k: f64 = c_k.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()
        / c_k.len() as f64;
    assert!(c_rms_k > 0.0, "C is zero after warmup — FDTD warmup failed");
    println!("□C wave test (γ=0, 32³ grid, σ={:.1} cells):", sigma);
    println!("  dt={:.3e}s  dx={:.3e}m", dt, dx);
    println!("  RMS(C_k) = {:.4e}", c_rms_k);

    let lap_ck    = fd_lap_raw(&c_k, n1);
    let inv_dt2   = (dt as f64).powi(-2);
    let c2_per_dx2 = (C_LIGHT as f64).powi(2) / (dx as f64).powi(2);

    let mut d2t_ss  = 0.0_f64;
    let mut res_ss  = 0.0_f64;
    let mut count   = 0_usize;

    for iz in 1..n1 - 1 {
        for iy in 1..n1 - 1 {
            for ix in 1..n1 - 1 {
                let flat = ix + iy * n1 + iz * n1 * n1;
                // ∂²C/∂t² (centred in time)
                let d2t = ((c_kp1[flat] - 2.0 * c_k[flat] + c_km1[flat]) as f64) * inv_dt2;
                // c²∇²C (centred in space): lap_raw already has the dx² baked in?
                // lap_raw = (sum_nbrs - 6*center), so c²∇²C = lap_raw * c²/dx²
                let d2x = (lap_ck[flat] as f64) * c2_per_dx2;
                let res = d2t - d2x;
                d2t_ss += d2t * d2t;
                res_ss += res * res;
                count  += 1;
            }
        }
    }

    let rms_d2t = (d2t_ss / count as f64).sqrt();
    let rms_res = (res_ss / count as f64).sqrt();
    // Normalise residual by the magnitude of the wave terms (not by C itself).
    let rel = if rms_d2t > 1e-60 { rms_res / rms_d2t } else { 0.0 };

    println!("  RMS(∂²C/∂t²)      = {:.4e}", rms_d2t);
    println!("  RMS(□C residual)   = {:.4e}", rms_res);
    println!("  RMS(res)/RMS(∂²t)  = {:.4} (expect < 0.20)", rel);

    // 2nd-order FDTD; dominant truncation: O(1/σ_cells²) ≈ 4% for σ≈5 cells.
    // Allow 20% to account for multi-step accumulation and boundary effects.
    assert!(
        rel < 0.20,
        "□C residual is {:.2}% of wave term — C violates wave equation beyond truncation tolerance",
        rel * 100.0
    );
    println!("✓ □C = 0 holds to {:.2}% accuracy", rel * 100.0);
}
